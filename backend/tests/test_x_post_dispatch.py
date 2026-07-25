from __future__ import annotations

# X_POST_SELF_TIMING_SPEC の配信ディスパッチャ(フェーズC)の回帰テスト。
# X_OPSEC_LLM_REWRITE_SPEC 反映後: opsec は正規表現ではなく rewrite_for_opsec()
# の LLM 判定＋書き直しで、safe_text を投稿する。
# ネットワーク/LLM 非依存: store・opsec・facts・publisher・Gate をすべてモック
# して、_x_post_dispatch_check() の分岐(shadow ログのみ/opsec 保留で skipped/
# facts・Gate で skipped/live で post→posted、safe_text が投稿されること)を検証。

import asyncio
from unittest.mock import AsyncMock, patch

from app.config import settings
from app.services.proactive import scheduler as sched
from app.services.x_opsec_rewrite import OpsecResult


def _run(coro):
    return asyncio.run(coro)


def _patches(
    *, due, opsec=None, facts_ok=True, may_speak=True, live=False, tweet_id="tw1",
):
    """_x_post_dispatch_check が使うインライン import 先をまとめてモックする。"""
    if opsec is None:
        # 既定: 書き直しなしで素通し(元テキストがそのまま safe_text)。
        opsec = OpsecResult(safe_text="", changed=False, removed=[])  # placeholder, replaced below
    gate = AsyncMock(return_value=type("G", (), {"may_speak": may_speak, "blocked_by": None})())
    publisher = type("P", (), {"post_tweet": AsyncMock(return_value=tweet_id)})()
    return {
        "app.services.scheduled_x_post_store.get_due_pending": AsyncMock(return_value=due),
        "app.services.scheduled_x_post_store.mark_posted": AsyncMock(),
        "app.services.scheduled_x_post_store.mark_skipped": AsyncMock(),
        "app.services.x_post_generator.record_post": AsyncMock(),
        "app.services.x_opsec_rewrite.rewrite_for_opsec": AsyncMock(return_value=opsec),
        "app.services.x_privacy_filter.filter_private_facts": AsyncMock(
            return_value=(facts_ok, [] if facts_ok else ["devices/x"])
        ),
        "app.services.executive_gate.evaluate_executive_gate": gate,
        "app.services.x_publisher.get_publisher": lambda: publisher,
        "_publisher": publisher,
    }


DUE = [{"id": "p1", "text": "おはよう", "category": "A_spontaneous_remark", "score": 1.0, "scheduled_at": "2026-07-23T09:00:00+00:00"}]


def _apply(mock_map, live):
    marks = {}
    ctxs = []
    for target, m in mock_map.items():
        if target.startswith("_"):
            continue
        p = patch(target, m)
        p.start()
        ctxs.append(p)
        marks[target] = m
    p_live = patch.object(settings, "x_categorized_post_live", live)
    p_live.start()
    ctxs.append(p_live)
    return ctxs, marks


def test_shadow_logs_only_and_marks_posted():
    # opsec.model は shadow の would-post ログに出す(X_OPSEC_MODEL_WIRING_FIX)。
    opsec = OpsecResult(safe_text="おはよう", changed=False, removed=[], model="openai:gpt-5.4-mini")
    m = _patches(due=DUE, opsec=opsec, live=False)
    ctxs, marks = _apply(m, live=False)
    try:
        with patch("app.services.proactive.scheduler.get_sigmaris_jwt", AsyncMock(return_value="jwt")):
            _run(sched._x_post_dispatch_check())
        # shadow: 実送信せず posted にする。
        marks["app.services.scheduled_x_post_store.mark_posted"].assert_awaited_once()
        m["_publisher"].post_tweet.assert_not_awaited()
        marks["app.services.scheduled_x_post_store.mark_skipped"].assert_not_awaited()
    finally:
        for c in ctxs:
            c.stop()


def test_opsec_hold_marks_skipped():
    # safe_text が空(パース/呼び出し失敗の安全側)なら素通しさせず skipped。
    opsec = OpsecResult(safe_text="", changed=True, removed=["opsec 判定に失敗(安全側で保留)"])
    m = _patches(due=DUE, opsec=opsec, live=False)
    ctxs, marks = _apply(m, live=False)
    try:
        with patch("app.services.proactive.scheduler.get_sigmaris_jwt", AsyncMock(return_value="jwt")):
            _run(sched._x_post_dispatch_check())
        marks["app.services.scheduled_x_post_store.mark_skipped"].assert_awaited_once()
        marks["app.services.scheduled_x_post_store.mark_posted"].assert_not_awaited()
    finally:
        for c in ctxs:
            c.stop()


def test_facts_block_marks_skipped():
    opsec = OpsecResult(safe_text="おはよう", changed=False, removed=[])
    m = _patches(due=DUE, opsec=opsec, facts_ok=False, live=False)
    ctxs, marks = _apply(m, live=False)
    try:
        with patch("app.services.proactive.scheduler.get_sigmaris_jwt", AsyncMock(return_value="jwt")):
            _run(sched._x_post_dispatch_check())
        marks["app.services.scheduled_x_post_store.mark_skipped"].assert_awaited_once()
        marks["app.services.scheduled_x_post_store.mark_posted"].assert_not_awaited()
    finally:
        for c in ctxs:
            c.stop()


def test_gate_block_marks_skipped():
    opsec = OpsecResult(safe_text="おはよう", changed=False, removed=[])
    m = _patches(due=DUE, opsec=opsec, may_speak=False, live=False)
    ctxs, marks = _apply(m, live=False)
    try:
        with patch("app.services.proactive.scheduler.get_sigmaris_jwt", AsyncMock(return_value="jwt")):
            _run(sched._x_post_dispatch_check())
        marks["app.services.scheduled_x_post_store.mark_skipped"].assert_awaited_once()
        m["_publisher"].post_tweet.assert_not_awaited()
    finally:
        for c in ctxs:
            c.stop()


def test_live_posts_safe_text_and_records():
    # 書き直し後の safe_text が投稿・記録されること(元テキストではなく)。
    opsec = OpsecResult(safe_text="外部公開の準備をしている", changed=True, removed=["ポート番号"])
    m = _patches(due=DUE, opsec=opsec, live=True, tweet_id="tw123")
    ctxs, marks = _apply(m, live=True)
    try:
        with patch("app.services.proactive.scheduler.get_sigmaris_jwt", AsyncMock(return_value="jwt")):
            _run(sched._x_post_dispatch_check())
        m["_publisher"].post_tweet.assert_awaited_once_with("外部公開の準備をしている")
        marks["app.services.x_post_generator.record_post"].assert_awaited_once()
        args, _kwargs = marks["app.services.x_post_generator.record_post"].call_args
        assert args[0] == "外部公開の準備をしている"  # record_post も safe_text
        marks["app.services.scheduled_x_post_store.mark_posted"].assert_awaited_once()
    finally:
        for c in ctxs:
            c.stop()


def test_no_due_does_nothing():
    opsec = OpsecResult(safe_text="おはよう", changed=False, removed=[])
    m = _patches(due=[], opsec=opsec, live=False)
    ctxs, marks = _apply(m, live=False)
    try:
        with patch("app.services.proactive.scheduler.get_sigmaris_jwt", AsyncMock(return_value="jwt")):
            _run(sched._x_post_dispatch_check())
        marks["app.services.scheduled_x_post_store.mark_posted"].assert_not_awaited()
        marks["app.services.scheduled_x_post_store.mark_skipped"].assert_not_awaited()
    finally:
        for c in ctxs:
            c.stop()
