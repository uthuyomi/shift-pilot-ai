from __future__ import annotations

# X_OPSEC_LINE_B_AND_MANUAL_POST_SPEC の回帰テスト。
# ネットワーク非依存: x_manual_post ハンドラを直接呼び、rewrite_for_opsec・
# filter_private_facts・publisher・record_post をダミー注入して、
# opsec 保留→422 / private_facts NG→422 / shadow→posted:false /
# live→post_tweet(safe_text)・record_post / 未認証 401 / 空 text 400 を検証。
# あわせて opsec プロンプトが立場B(外部公開の事実は保持・具体は除去)+声の
# 保持+非交渉ラインへ更新されていることを、プロンプト文字列で回帰確認する。

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.config import settings
from app.routes.agent import ManualPostRequest, x_manual_post
from app.services.x_opsec_rewrite import OpsecResult

_AUTH = "Bearer dummy-jwt"


def _run(coro):
    return asyncio.run(coro)


def _opsec(safe_text, *, changed=False, removed=None):
    return AsyncMock(return_value=OpsecResult(
        safe_text=safe_text, changed=changed, removed=removed or [], model="openai:gpt-5.4-mini",
    ))


def _facts(ok=True, blocked=None):
    return AsyncMock(return_value=(ok, blocked or []))


# ─── 認証・入力検証 ──────────────────────────────────────────────────────
def test_missing_auth_rejected():
    with pytest.raises(HTTPException) as exc:
        _run(x_manual_post(ManualPostRequest(text="hi"), authorization=None))
    assert exc.value.status_code == 401


def test_empty_text_returns_400():
    with pytest.raises(HTTPException) as exc:
        _run(x_manual_post(ManualPostRequest(text="   "), authorization=_AUTH))
    assert exc.value.status_code == 400


# ─── opsec 保留(safe_text 空)→ 422・投稿なし ───────────────────────────
def test_opsec_hold_returns_422_no_post():
    opsec = _opsec("", changed=True, removed=["opsec 判定に失敗(安全側で保留)"])
    publisher = type("P", (), {"post_tweet": AsyncMock()})()
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", opsec), \
         patch("app.services.x_publisher.get_publisher", lambda: publisher):
        with pytest.raises(HTTPException) as exc:
            _run(x_manual_post(ManualPostRequest(text="危険な内容"), authorization=_AUTH))
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "opsec_hold"
    publisher.post_tweet.assert_not_awaited()


# ─── private_facts NG → 422・投稿なし ────────────────────────────────────
def test_private_facts_block_returns_422_no_post():
    opsec = _opsec("おはよう")
    publisher = type("P", (), {"post_tweet": AsyncMock()})()
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", opsec), \
         patch("app.services.x_privacy_filter.filter_private_facts", _facts(False, ["devices/x"])), \
         patch("app.services.x_publisher.get_publisher", lambda: publisher):
        with pytest.raises(HTTPException) as exc:
            _run(x_manual_post(ManualPostRequest(text="おはよう"), authorization=_AUTH))
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "private_facts"
    assert exc.value.detail["blocked"] == ["devices/x"]
    publisher.post_tweet.assert_not_awaited()


# ─── shadow(live フラグ false)→ 投稿せず posted:false ────────────────────
def test_shadow_does_not_post():
    opsec = _opsec("おはよう")
    publisher = type("P", (), {"post_tweet": AsyncMock()})()
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", opsec), \
         patch("app.services.x_privacy_filter.filter_private_facts", _facts(True)), \
         patch("app.services.x_publisher.get_publisher", lambda: publisher), \
         patch.object(settings, "x_enabled", True), \
         patch.object(settings, "x_categorized_post_live", False):
        out = _run(x_manual_post(ManualPostRequest(text="おはよう"), authorization=_AUTH))
    assert out["ok"] is True
    assert out["posted"] is False
    assert out["shadow"] is True
    assert out["safe_text"] == "おはよう"
    publisher.post_tweet.assert_not_awaited()


def test_shadow_when_x_disabled_even_if_live_flag():
    # X_ENABLED=false なら live フラグが true でも投稿しない(両方必要)。
    opsec = _opsec("おはよう")
    publisher = type("P", (), {"post_tweet": AsyncMock()})()
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", opsec), \
         patch("app.services.x_privacy_filter.filter_private_facts", _facts(True)), \
         patch("app.services.x_publisher.get_publisher", lambda: publisher), \
         patch.object(settings, "x_enabled", False), \
         patch.object(settings, "x_categorized_post_live", True):
        out = _run(x_manual_post(ManualPostRequest(text="おはよう"), authorization=_AUTH))
    assert out["posted"] is False
    publisher.post_tweet.assert_not_awaited()


# ─── live → post_tweet(safe_text)・record_post ───────────────────────────
def test_live_posts_safe_text_and_records():
    opsec = _opsec("外部公開の準備をしている", changed=True, removed=["ポート番号"])
    publisher = type("P", (), {"post_tweet": AsyncMock(return_value="tw999")})()
    record = AsyncMock()
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", opsec), \
         patch("app.services.x_privacy_filter.filter_private_facts", _facts(True)), \
         patch("app.services.x_publisher.get_publisher", lambda: publisher), \
         patch("app.services.x_post_generator.record_post", record), \
         patch.object(settings, "x_enabled", True), \
         patch.object(settings, "x_categorized_post_live", True):
        out = _run(x_manual_post(ManualPostRequest(text="外部公開のためポート12345を開ける"), authorization=_AUTH))
    publisher.post_tweet.assert_awaited_once_with("外部公開の準備をしている")
    record.assert_awaited_once()
    args, kwargs = record.call_args
    assert args[0] == "外部公開の準備をしている"  # safe_text を記録
    assert args[1] == "manual"                      # category=manual
    assert kwargs.get("tweet_id") == "tw999"
    assert out == {
        "ok": True, "posted": True, "tweet_id": "tw999",
        "original": "外部公開のためポート12345を開ける",
        "safe_text": "外部公開の準備をしている",
        "changed": True, "removed": ["ポート番号"],
    }


def test_live_publisher_returns_none_is_502():
    opsec = _opsec("おはよう")
    publisher = type("P", (), {"post_tweet": AsyncMock(return_value=None)})()
    record = AsyncMock()
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", opsec), \
         patch("app.services.x_privacy_filter.filter_private_facts", _facts(True)), \
         patch("app.services.x_publisher.get_publisher", lambda: publisher), \
         patch("app.services.x_post_generator.record_post", record), \
         patch.object(settings, "x_enabled", True), \
         patch.object(settings, "x_categorized_post_live", True):
        with pytest.raises(HTTPException) as exc:
            _run(x_manual_post(ManualPostRequest(text="おはよう"), authorization=_AUTH))
    assert exc.value.status_code == 502
    record.assert_not_awaited()


# ─── 立場B: プロンプト文字列の回帰(実モデル非依存) ──────────────────────
def test_prompt_reflects_line_b_and_voice():
    from app.services.x_opsec_rewrite import _SYSTEM_PROMPT

    # 外部公開の事実・意図は保持する旨が明記されている(立場B)。
    assert "事実・意図レベルの言及" in _SYSTEM_PROMPT
    assert "外部公開している/するつもり" in _SYSTEM_PROMPT
    # 具体(DDNS名・ドメイン・ポート・IP)は引き続き除去対象。
    assert "DDNS" in _SYSTEM_PROMPT
    assert "ポート番号・ポート開放" in _SYSTEM_PROMPT
    # 声の保持・非交渉ラインが入っている。
    assert "声の保持" in _SYSTEM_PROMPT
    assert "非交渉ライン" in _SYSTEM_PROMPT
