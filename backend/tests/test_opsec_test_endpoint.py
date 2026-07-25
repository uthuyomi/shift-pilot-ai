from __future__ import annotations

# X_OPSEC_TEST_HOOK_SPEC の回帰テスト。
# ネットワーク非依存: opsec_test ハンドラを直接呼び、rewrite_for_opsec を
# ダミー注入して、レスポンス形状({safe_text, changed, removed, model})・
# 未認証 401・空 text 400 を検証する。

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.routes.agent import OpsecTestRequest, opsec_test
from app.services.x_opsec_rewrite import OpsecResult

_AUTH = "Bearer dummy-jwt"


def _run(coro):
    return asyncio.run(coro)


def test_calls_rewrite_and_returns_shape():
    res = OpsecResult(
        safe_text="外部公開の準備をしている", changed=True,
        removed=["外部公開の手段"], model="openai:gpt-5.4-mini",
    )
    fake = AsyncMock(return_value=res)
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", fake):
        out = _run(opsec_test(OpsecTestRequest(text="外部公開のためポート12345を開ける"), authorization=_AUTH))
    fake.assert_awaited_once_with("外部公開のためポート12345を開ける")
    assert out == {
        "ok": True,
        "original": "外部公開のためポート12345を開ける",
        "safe_text": "外部公開の準備をしている",
        "changed": True,
        "removed": ["外部公開の手段"],
        "model": "openai:gpt-5.4-mini",
    }


def test_passthrough_result_is_reported():
    # 無害文: changed=false・素通しがそのまま返る(消しすぎ検知の入口)。
    text = "GTX 1660 の自宅サーバーで動いてるらしい、合ってる?"
    res = OpsecResult(safe_text=text, changed=False, removed=[], model="openai:gpt-5.4-mini")
    with patch("app.services.x_opsec_rewrite.rewrite_for_opsec", AsyncMock(return_value=res)):
        out = _run(opsec_test(OpsecTestRequest(text=text), authorization=_AUTH))
    assert out["changed"] is False
    assert out["safe_text"] == text
    assert out["model"] == "openai:gpt-5.4-mini"


def test_missing_auth_rejected():
    with pytest.raises(HTTPException) as exc:
        _run(opsec_test(OpsecTestRequest(text="x"), authorization=None))
    assert exc.value.status_code == 401


def test_bad_auth_scheme_rejected():
    with pytest.raises(HTTPException) as exc:
        _run(opsec_test(OpsecTestRequest(text="x"), authorization="Token abc"))
    assert exc.value.status_code == 401


def test_empty_text_returns_400():
    with pytest.raises(HTTPException) as exc:
        _run(opsec_test(OpsecTestRequest(text="   "), authorization=_AUTH))
    assert exc.value.status_code == 400


def test_auth_checked_before_empty_text():
    # 空 text でも、認証が無ければ先に 401(認証を素通りさせない)。
    with pytest.raises(HTTPException) as exc:
        _run(opsec_test(OpsecTestRequest(text=""), authorization=None))
    assert exc.value.status_code == 401
