from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.services.heartbeat import heartbeat_tick
from app.services.research_agent import run_research
from app.services.proactive.jwt_manager import get_sigmaris_jwt

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _heartbeat() -> None:
    try:
        await heartbeat_tick()
    except Exception:
        logger.exception("Heartbeat job raised unexpectedly")


async def _research() -> None:
    try:
        result = await run_research()
        logger.info("Research job done: %s", result)
    except Exception:
        logger.exception("Research job raised unexpectedly")


async def _memory_validate() -> None:
    from app.services.memory_validator import validate_all_facts  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        result = await validate_all_facts(jwt)
        logger.info("Memory validate job done: %s", result)
    except Exception:
        logger.exception("Memory validate job raised unexpectedly")


async def _memory_embed() -> None:
    from app.services.memory_search import update_fact_embeddings  # noqa: PLC0415
    from app.services.supabase_rest import get_current_user  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        user = await get_current_user(jwt)
        user_id = user.get("id")
        if not isinstance(user_id, str):
            logger.warning("Memory embed job skipped: authenticated user id is missing")
            return
        result = await update_fact_embeddings(user_id, jwt=jwt)
        logger.info("Memory embed job done: %s", result)
    except Exception:
        logger.exception("Memory embed job raised unexpectedly")


async def _health_data_sync() -> None:
    if not settings.health_sync_enabled:
        return
    from app.services.health_data import HealthDataCollector  # noqa: PLC0415
    from datetime import date, timedelta  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        google_token = settings.sigmaris_google_access_token
        if not google_token:
            logger.info("health_data_sync: SIGMARIS_GOOGLE_ACCESS_TOKEN not set, skipping")
            return
        yesterday = date.today() - timedelta(days=1)
        collector = HealthDataCollector()
        summary = await collector.fetch_daily_summary(yesterday, google_token)
        stored = await collector.store_to_fact_memory(jwt, summary)
        logger.info(
            "health_data_sync: stored %d items for %s", len(stored), yesterday.isoformat()
        )
    except Exception:
        logger.exception("health_data_sync job raised unexpectedly")


async def _trend_analyze() -> None:
    from app.services.trend_analyzer import analyze_trends  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        result = await analyze_trends(jwt)
        logger.info("Trend analyze job done: %s", result)
    except Exception:
        logger.exception("Trend analyze job raised unexpectedly")


async def _narrative_generate() -> None:
    from app.services.self_narrative import generate_narrative_chapter  # noqa: PLC0415
    try:
        chapter = await generate_narrative_chapter()
        if chapter:
            logger.info(
                "Narrative generate job done: chapter=%s title=%s",
                chapter.get("chapter"), chapter.get("title"),
            )
        else:
            logger.warning("Narrative generate job: returned None")
    except Exception:
        logger.exception("Narrative generate job raised unexpectedly")


async def _curiosity_search() -> None:
    from app.services.curiosity_engine import execute_curiosity_search  # noqa: PLC0415
    try:
        result = await execute_curiosity_search()
        logger.info("Curiosity search job done: %s", result)
    except Exception:
        logger.exception("Curiosity search job raised unexpectedly")


async def _self_interest_queries() -> None:
    from app.services.curiosity_engine import generate_self_interest_queries  # noqa: PLC0415
    try:
        result = await generate_self_interest_queries()
        logger.info("Self-interest query job done: generated=%d", len(result))
    except Exception:
        logger.exception("Self-interest query job raised unexpectedly")


async def _experience_analyze() -> None:
    from app.services.experience_layer import analyze_patterns  # noqa: PLC0415
    try:
        result = await analyze_patterns()
        logger.info("Experience analyze job done: patterns=%s", list(result.keys()) if result else None)
    except Exception:
        logger.exception("Experience analyze job raised unexpectedly")


async def _decision_analyze() -> None:
    from app.services.decision_log import analyze_decision_patterns  # noqa: PLC0415
    try:
        result = await analyze_decision_patterns()
        logger.info("Decision analyze job done: keys=%s", list(result.keys()) if result else None)
    except Exception:
        logger.exception("Decision analyze job raised unexpectedly")


async def _preference_pattern_extract() -> None:
    from app.services.decision_log import extract_preference_patterns  # noqa: PLC0415
    from app.services.supabase_rest import get_current_user  # noqa: PLC0415
    # Unlike knowledge_graph's job, this one previously had zero dependency
    # on fetching a user JWT (extract_preference_patterns() only read/wrote
    # service-role-only tables). jwt/user_id are now used for an *optional*
    # relevant-facts search, so a failure here must degrade to the old
    # no-context behavior rather than taking the whole job down with it —
    # hence this fetch gets its own try/except, separate from the main call.
    jwt: str | None = None
    user_id: str | None = None
    try:
        jwt = await get_sigmaris_jwt()
        user = await get_current_user(jwt)
        resolved_id = user.get("id")
        user_id = resolved_id if isinstance(resolved_id, str) else None
    except Exception:
        logger.warning(
            "Preference pattern extraction: could not fetch jwt/user_id, "
            "proceeding without relevant-facts context",
            exc_info=True,
        )
    try:
        result = await extract_preference_patterns(jwt=jwt, user_id=user_id)
        logger.info("Preference pattern extraction job done: %s", result)
    except Exception:
        logger.exception("Preference pattern extraction job raised unexpectedly")


async def _adoption_count_recompute() -> None:
    from app.services.decision_log import recompute_adoption_counts  # noqa: PLC0415
    try:
        result = await recompute_adoption_counts()
        logger.info("Adoption count recompute job done: %s", result)
    except Exception:
        logger.exception("Adoption count recompute job raised unexpectedly")


async def _episode_consolidate() -> None:
    from app.services.experience_layer import consolidate_episodic_memory  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        result = await consolidate_episodic_memory(jwt)
        logger.info("Episode consolidation job done: %s", result)
    except Exception:
        logger.exception("Episode consolidation job raised unexpectedly")


async def _goal_alignment_extract() -> None:
    from app.services.goal_alignment import extract_goal_alignment_flags  # noqa: PLC0415
    from app.services.supabase_rest import get_current_user  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        user = await get_current_user(jwt)
        user_id = user.get("id")
        if not isinstance(user_id, str):
            logger.warning("Goal alignment job skipped: authenticated user id is missing")
            return
        result = await extract_goal_alignment_flags(user_id, jwt=jwt)
        logger.info("Goal alignment extraction job done: %s", result)
    except Exception:
        logger.exception("Goal alignment extraction job raised unexpectedly")


async def _knowledge_graph_extract() -> None:
    from app.services.knowledge_graph import extract_entities_and_relations  # noqa: PLC0415
    from app.services.supabase_rest import get_current_user  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        user = await get_current_user(jwt)
        user_id = user.get("id")
        if not isinstance(user_id, str):
            logger.warning("Knowledge graph job skipped: authenticated user id is missing")
            return
        result = await extract_entities_and_relations(user_id, jwt=jwt)
        logger.info("Knowledge graph extraction job done: %s", result)
    except Exception:
        logger.exception("Knowledge graph extraction job raised unexpectedly")


async def _memory_snapshot_generate() -> None:
    from app.services.memory_snapshot import generate_memory_snapshot  # noqa: PLC0415
    from app.services.supabase_rest import get_current_user  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        user = await get_current_user(jwt)
        user_id = user.get("id")
        if not isinstance(user_id, str):
            logger.warning("Memory snapshot job skipped: authenticated user id is missing")
            return
        result = await generate_memory_snapshot(user_id)
        logger.info("Memory snapshot generation job done: %s", result)
    except Exception:
        logger.exception("Memory snapshot generation job raised unexpectedly")


# ── Phase Vis-1の前提条件対応(docs/sigmaris/phase_vis_report.md): RC-1〜
# RC-5・Phase G指標・Safety Governanceは、いずれもVis-1時点では手動CLI
# 実行のみで、この定期実行の仕組みに一切登録されていなかった。以下3件は、
# 既存のscripts/run_cycle_health.py・scripts/run_grounding_health.py・
# scripts/scan_safety_critical_files.pyが、既に確立した計測ロジック
# (cycle_health_runner.py/grounding_health_runner.py/safety_critical_
# files_scan.py)を、そのまま呼び出すだけであり、新しい計測ロジックは
# 一切追加していない——既存の`_memory_validate`等と全く同じ、
# try/except一段構えのfire-and-forgetパターンを踏襲した。 ─────────────

async def _cycle_health_measure() -> None:
    from app.services.cycle_health_runner import run_cycle_health  # noqa: PLC0415
    from app.services.cycle_health_runs_store import record_cycle_health_run  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        result = await run_cycle_health(jwt=jwt)
        rc1 = result["rc1_cycle_completion"]
        rc2 = result["rc2_temporal_consistency"]
        rc3 = result["rc3_belief_stability"]
        rc4 = result["rc4_policy_belief_alignment"]
        rc5 = result["rc5_cycle_break"]
        safety_gov = result["safety_governance"]
        run_id = await record_cycle_health_run(
            window_days=result["window_days"],
            period_from=result["period_from"],
            period_to=result["period_to"],
            rc1={
                "total_experiences": rc1["total_experiences"],
                "reached_count": rc1["reached_count"],
                "raw_completion_rate": rc1["raw_completion_rate"],
                "eligible_count": rc1["eligible_count"],
                "eligible_completion_rate": rc1["eligible_completion_rate"],
            },
            rc2={
                "score": rc2["score"],
                "chat_pairs_checked": rc2["chat_pairs_checked"],
                "chat_order_violation_count": rc2["chat_order_violation_count"],
                "event_experience_checked": rc2["event_experience_checked"],
                "event_experience_violation_count": rc2["event_experience_violation_count"],
            },
            rc3={
                "score": rc3["score"],
                "comparable_pattern_count": rc3["comparable_pattern_count"],
                "flip_count": rc3["flip_count"],
                "unsupported_flip_count": rc3["unsupported_flip_count"],
            },
            rc4={"score": rc4["score"], "flags_evaluated": rc4["flags_evaluated"]},
            rc5={"status": rc5["status"], "broke_metrics": rc5["broke_metrics"]},
            safety_governance={
                "status": safety_gov["status"],
                "unregistered_count": safety_gov["unregistered_count"],
            },
            notes="scheduled:cycle_health",
            details=result["details_for_persistence"],
        )
        logger.info(
            "Cycle health job done: run_id=%s rc5_status=%s safety_governance_status=%s",
            run_id, rc5["status"], safety_gov["status"],
        )
        if rc5["status"] == "break_detected":
            logger.warning("Cycle health job: RC-5 detected a break (broke_metrics=%s)", rc5["broke_metrics"])
        if safety_gov["status"] == "gap_detected":
            logger.warning(
                "Cycle health job: safety governance gap detected (unregistered_files=%s)",
                safety_gov["unregistered_files"],
            )
    except Exception:
        logger.exception("Cycle health job raised unexpectedly")


async def _grounding_health_measure() -> None:
    from app.services.grounding_health_runner import run_grounding_health  # noqa: PLC0415
    from app.services.grounding_health_runs_store import record_grounding_health_run  # noqa: PLC0415
    try:
        jwt = await get_sigmaris_jwt()
        result = await run_grounding_health(jwt=jwt)
        run_id = await record_grounding_health_run(
            window_days=result["window_days"],
            period_from=result["period_from"],
            period_to=result["period_to"],
            citation_precision=result["citation_precision"],
            search_trigger_rate=result["search_trigger_rate"],
            contradiction_rate=result["contradiction_rate"],
            notes="scheduled:grounding_health",
            details=result["details_for_persistence"],
        )
        logger.info(
            "Grounding health job done: run_id=%s citation_precision=%s contradiction_rate=%s",
            run_id, result["citation_precision"]["precision"], result["contradiction_rate"]["rate"],
        )
    except Exception:
        logger.exception("Grounding health job raised unexpectedly")


async def _safety_governance_scan() -> None:
    # Safety-3(safety_critical_files_scan.py)のスキャンロジックを直接
    # 呼び出すのみ。DB書き込みは行わない——毎日のcycle_health_measureが
    # 既にsafety_governance_status(同じスキャン結果)をsigmaris_cycle_
    # health_runsへ記録しているため、二重に記録しない(判断根拠、
    # docs/sigmaris/phase_vis_report.md参照)。本ジョブの意義は、
    # cycle_health_measureが(RC-1〜5計測の途中で例外を投げる等の理由で)
    # 失敗した日でも、このスキャンだけは独立して実行され続けることに
    # ある——依頼書「状況確認」という言葉通り、ログへの記録のみで完結する。
    from pathlib import Path  # noqa: PLC0415
    from app.services.safety_critical_files_scan import find_unregistered_gate_files  # noqa: PLC0415
    try:
        backend_root = Path(__file__).resolve().parents[3]  # proactive/ -> services/ -> app/ -> backend/
        result = find_unregistered_gate_files(backend_root)
        logger.info(
            "Safety governance scan job done: scanned=%d gate_pattern=%d unregistered=%d",
            result.scanned_file_count, result.gate_pattern_file_count, result.unregistered_count,
        )
        if not result.coverage_complete:
            logger.warning(
                "Safety governance scan job: unregistered candidates detected: %s",
                [c.relative_path for c in result.unregistered_candidates],
            )
    except Exception:
        logger.exception("Safety governance scan job raised unexpectedly")


# ── Self-1〜3「自己認識の自動更新」(docs/sigmaris/self_awareness_report.md)
# Self-1(コードベースの機能洗い出し)・Self-2(一人称の日本語要約)を、
# 週次でまとめて再実行する。
#
# 【依頼書2章「頻度は、機能が頻繁に追加される日々の開発のペースと
# コストのバランスを考慮し、提案すること」への対応】週1回(日曜)を
# 選んだ判断根拠:
#   1. 新しいPhaseの完了は、このセッションの実績を見ても数日〜1週間に
#      1件程度のペースであり、毎日再スキャンしても差分がほとんど無い
#      日が大半になる。一方、月1回では「今日、何ができるか」との乖離が
#      1か月近く放置されうる。既存のexperience_analyze・decision_
#      analyze・knowledge_graph_extract等、Phase Rの週次バッチ群と
#      同じ「振り返り系は週次」という、このコードベース既存の頻度
#      規約にも合致する。
#   2. コスト面: Self-2のLLM呼び出しは、nano tierで1回あたり8ドメイン
#      分(現時点)——毎日実行しても軽微ではあるが、「頻繁な開発ペースに
#      対して、過剰でも過少でもない」バランスとして、既存の週次バッチと
#      同じ頻度に揃える方が、運用上の一貫性・説明のしやすさで勝ると判断
#      した。
#
# 【時間帯の判断根拠(既存ジョブとの重複回避)】日曜早朝バッチ
# (4:00 experience_analyze 〜 5:45 safety_governance_scan)の直後、
# 6:15のcuriosity_searchより前という、既存の空き時間帯(5:45〜6:15の
# 30分)に、5:55として配置した——safety_governance_scanから10分の余裕、
# curiosity_searchまで20分の余裕を残す。
async def _self_awareness_update() -> None:
    from pathlib import Path  # noqa: PLC0415
    from app.services.capability_summary import generate_capability_summaries  # noqa: PLC0415
    from app.services.capability_summary_store import record_capability_summary  # noqa: PLC0415
    from app.services.orchestrator.service import invalidate_capability_summary_cache  # noqa: PLC0415
    try:
        backend_root = Path(__file__).resolve().parents[3]  # proactive/ -> services/ -> app/ -> backend/
        summaries = await generate_capability_summaries(backend_root)
        recorded = 0
        for summary in summaries:
            row_id = await record_capability_summary(
                domain=summary.domain,
                summary_text=summary.summary_text,
                file_count=summary.file_count,
                wired_file_count=summary.wired_file_count,
                unwired_file_count=summary.unwired_file_count,
                source_files=summary.source_files,
            )
            if row_id:
                recorded += 1
        # Self-3: invalidate the orchestrator's long-TTL(24h)in-process
        # cache immediately, so the next matching turn in this same running
        # process picks up the freshly-regenerated summaries without
        # waiting for the TTL to lapse naturally.
        invalidate_capability_summary_cache()
        logger.info(
            "Self-awareness update job done: domains=%d recorded=%d",
            len(summaries), recorded,
        )
    except Exception:
        logger.exception("Self-awareness update job raised unexpectedly")


# ── 旧X投稿システムの廃止、及び、新7カテゴリシステムへの実際の接続
# (docs/sigmaris/phase_h_report.md)。旧proactive/actions.py::
# _try_smart_x_post()(should_post_todayの固定スロット判定でX投稿を
# 直接実行)は削除済み——本ジョブが、シグマリスの投稿を実際にXへ送る、
# 唯一の経路になった。
#
# 【依頼書「投稿のタイミングが、固定スケジュールではなく、Executive
# Gateの判定に基づくこと」への対応】本ジョブ自体は、1日4回、決まった
# 時刻に「今、投稿してよいか確認する」という、機会を作るだけである
# ——旧システムのように「この時間帯には、このタイプを投稿する」という
# 対応関係は、一切存在しない。実際に投稿するかどうか・何を投稿するかは、
# 毎回100%、generate_categorized_post()内部で呼ばれるselect_post_
# category()(Executive Gate・Drive State・その日の実際の材料に基づく
# 動的な判定、H-1で確立済み)が決める。本ジョブは、その判定を仰ぐ
# "きっかけ"を、1日に4回作るだけであり、Executive Gateが「話しかけて
# よくない」と判定すれば(深夜早朝・直近の連続接触等)、4回のうち何回でも
# 空振りになりうる。
async def _categorized_x_post_check() -> None:
    """X_POST_SELF_TIMING_SPEC フェーズA(決定): 「考える機会」(1日4回の
    cron)で材料からカテゴリ・本文を生成し、内容にふさわしい未来の投稿時刻を
    AI に決めさせて予約テーブルへ pending で積む。ここでは即投稿しない——
    実際の配信は _x_post_dispatch_check()(高頻度 interval)が予約時刻に行う。"""
    from datetime import datetime, timezone  # noqa: PLC0415

    from app.services.scheduled_x_post_store import (  # noqa: PLC0415
        count_today_scheduled_or_posted,
        get_pending_scheduled_ats,
        insert_scheduled_post,
    )
    from app.services.x_post_category_selector import MAX_DAILY_CATEGORY_POSTS  # noqa: PLC0415
    from app.services.x_post_generator import generate_categorized_post  # noqa: PLC0415
    from app.services.x_post_self_timing import decide_scheduled_at  # noqa: PLC0415

    try:
        # 1日上限(予約込み): その日の pending+posted 合計で判定する。
        today_count = await count_today_scheduled_or_posted()
        if today_count >= MAX_DAILY_CATEGORY_POSTS:
            logger.info(
                "Categorized X post decide: daily cap reached (%d/%d), skipping",
                today_count, MAX_DAILY_CATEGORY_POSTS,
            )
            return

        gp = await generate_categorized_post()
        if gp is None:
            logger.info("Categorized X post decide: no post generated this cycle")
            return

        now = datetime.now(timezone.utc)
        existing = await get_pending_scheduled_ats()
        decision = await decide_scheduled_at(
            category=gp.post_type, text=gp.text, now=now, existing_scheduled_ats=existing,
        )
        if decision is None:
            logger.info(
                "Categorized X post decide: timing rejected by guards (category=%s)",
                gp.post_type,
            )
            return

        row = await insert_scheduled_post(
            text=gp.text, category=gp.post_type, score=gp.score,
            scheduled_at=decision.scheduled_at,
        )
        logger.info(
            "Categorized X post scheduled: category=%s scheduled_at=%s reason=%s id=%s text=%s",
            gp.post_type, decision.scheduled_at.isoformat(), decision.reason,
            (row or {}).get("id"), gp.text,
        )
    except Exception:
        logger.exception("Categorized X post decide raised unexpectedly")


async def _x_post_dispatch_check() -> None:
    """X_POST_SELF_TIMING_SPEC フェーズC(配信ディスパッチャ): 予約時刻に達した
    (scheduled_at<=now)pending 投稿を、配信直前に opsec 判定＋書き直し
    (X_OPSEC_LLM_REWRITE_SPEC)・記憶 private 値チェック・Executive Gate を
    再適用の上で実際に出す。shadow ではログのみ、live では post_tweet→
    record_post。弾かれたら skipped。高頻度 interval で回る。

    opsec は正規表現マッチをやめ、rewrite_for_opsec() の LLM 一回で「actionable
    な具体だけを書き直して除去」する(機材/自己言及/記憶確認は保持)。素通し
    ではなく安全な投稿へ直すため、changed でも skip せず safe_text を投稿する。
    正規表現バックストップは置かない(フィルタ重ねがけが目詰まりの原因だった
    ため)。記憶 private 値の逐語一致(filter_private_facts)と Gate は別責務
    として残す。"""
    from datetime import datetime, timezone  # noqa: PLC0415

    from app.services.executive_gate import evaluate_executive_gate  # noqa: PLC0415
    from app.services.scheduled_x_post_store import (  # noqa: PLC0415
        get_due_pending,
        mark_posted,
        mark_skipped,
    )
    from app.services.x_opsec_rewrite import rewrite_for_opsec  # noqa: PLC0415
    from app.services.x_post_generator import record_post  # noqa: PLC0415
    from app.services.x_privacy_filter import filter_private_facts  # noqa: PLC0415
    from app.services.x_publisher import get_publisher  # noqa: PLC0415

    try:
        now = datetime.now(timezone.utc)
        due = await get_due_pending(now=now)
        if not due:
            return

        jwt = await get_sigmaris_jwt()
        publisher = get_publisher()

        for post in due:
            post_id = str(post.get("id"))
            text = str(post.get("text") or "")
            category = str(post.get("category") or "")
            score = float(post.get("score") or 0.0)

            # 配信直前の opsec 判定＋書き直し(決定時から状況が変わっている
            # 場合の安全)。actionable な具体だけを safe_text から除去する。
            opsec = await rewrite_for_opsec(text)
            safe_text = opsec.safe_text
            # パース/呼び出し失敗の安全側(safe_text 空)は、素通しさせず保留。
            if not safe_text.strip():
                await mark_skipped(post_id, reason=f"opsec_hold: {', '.join(opsec.removed)}")
                logger.info("X dispatch: skipped id=%s opsec_hold=%s", post_id, opsec.removed)
                continue

            # 記憶 private 値の逐語一致は opsec とは別責務として残す(書き直し
            # 後のテキストに対して確認する)。
            facts_ok, blocked = await filter_private_facts(safe_text, jwt)
            if not facts_ok:
                await mark_skipped(post_id, reason=f"private_facts: {', '.join(blocked)}")
                logger.info("X dispatch: skipped id=%s private_facts=%s", post_id, blocked)
                continue

            gate = await evaluate_executive_gate(jwt)
            if not gate.may_speak:
                await mark_skipped(post_id, reason=f"executive_gate: {gate.blocked_by}")
                logger.info("X dispatch: skipped id=%s gate=%s", post_id, gate.blocked_by)
                continue

            if not settings.x_categorized_post_live:
                logger.info(
                    "[shadow mode] would post now (scheduled_at=%s, category=%s, "
                    "opsec_model=%s, opsec_changed=%s, removed=%s, original=%s, safe_text=%s)",
                    post.get("scheduled_at"), category, opsec.model, opsec.changed,
                    opsec.removed, text, safe_text,
                )
                await mark_posted(post_id)
                continue

            tweet_id = await publisher.post_tweet(safe_text)
            if tweet_id:
                # 実際に投稿した書き直し後テキストを履歴に残す(類似度判定・
                # 表示が、投稿された内容と一致するように)。
                await record_post(safe_text, category, score=score, tweet_id=tweet_id)
                await mark_posted(post_id, tweet_id=tweet_id)
                logger.info(
                    "X dispatch: posted id=%s category=%s tweet_id=%s", post_id, category, tweet_id,
                )
            else:
                await mark_skipped(post_id, reason="publisher_returned_none")
                logger.warning("X dispatch: publisher returned None id=%s", post_id)
    except Exception:
        logger.exception("X post dispatch raised unexpectedly")


# ── Phase H-2「返信の検知、及び、フィルタリング」(docs/sigmaris/
# phase_h_report.md)。依頼書1章「既存の定期実行の仕組み(scheduler.py)
# に相乗りすること」への対応——新しい定期実行基盤は作らず、既存の
# AsyncIOScheduler/CronTriggerのパターンをそのまま使う。
#
# 【Phase H-2.5追記】検知直後に、フィルタを通過した返信の返信案を生成する
# generate_pending_drafts()(x_reply_generator.py)も、同じジョブの中で
# 続けて呼ぶことにした。新しいcronエントリを4件追加する代わりに、
# 「検知→(通過分の)生成」を1つのジョブ実行内で完結させた判断根拠:
# 生成対象は、このジョブが検知したばかりの行(またはこれまでに未生成の
# 行)に限られるため、検知と生成を別々のスケジュールで走らせても、
# 生成側は結局この検知ジョブの完了を待つだけになり、cronエントリを
# 追加する実益がない。
#
# 【本ジョブが行わないこと】返信案の生成までは行うが、生成された返信案を
# 実際にXへ投稿する処理(x_publisher.post_tweet())は、本ジョブにも
# generate_pending_drafts()にも一切実装しない——生成された案は
# "pending_post"(投稿待ち)としてDBに保存されるのみ(次のタスクH-3が、
# 承認フローとともに実際の投稿を実装する)。
async def _x_reply_detection_check() -> None:
    from app.services.x_reply_detector import run_reply_detection  # noqa: PLC0415
    from app.services.x_reply_generator import generate_pending_drafts  # noqa: PLC0415
    try:
        result = await run_reply_detection()
        logger.info(
            "X reply detection check done: scanned=%d matched=%d new=%d",
            result["scanned_mentions"], result["matched_replies"], result["new_replies_processed"],
        )
        ignored = [r for r in result["results"] if r["filter_outcome"] == "ignored"]
        if ignored:
            logger.info(
                "X reply detection check: %d replies ignored (reasons in x_reply_log)", len(ignored),
            )
    except Exception:
        logger.exception("X reply detection check raised unexpectedly")

    try:
        draft_result = await generate_pending_drafts()
        logger.info(
            "X reply draft generation done: candidates=%d generated=%d failed=%d",
            draft_result["candidates"], draft_result["generated"], draft_result["failed"],
        )
    except Exception:
        logger.exception("X reply draft generation raised unexpectedly")


def startup_scheduler() -> None:
    global _scheduler

    if not settings.proactive_enabled:
        logger.info("Proactive scheduler disabled (PROACTIVE_ENABLED=false)")
        return

    tz = settings.sigmaris_timezone
    _scheduler = AsyncIOScheduler(timezone=tz)

    _scheduler.add_job(_heartbeat,       CronTrigger(minute="*/1",                       timezone=tz), id="heartbeat",       replace_existing=True)
    _scheduler.add_job(_memory_embed,    CronTrigger(hour=3,  minute=0,                  timezone=tz), id="memory_embed",    replace_existing=True)
    _scheduler.add_job(_research,        CronTrigger(hour=7,  minute=0,                  timezone=tz), id="research",        replace_existing=True)
    _scheduler.add_job(_memory_validate, CronTrigger(hour=6,  minute=30,                 timezone=tz), id="memory_validate", replace_existing=True)
    _scheduler.add_job(_health_data_sync,CronTrigger(hour=6,  minute=45,                 timezone=tz), id="health_sync",     replace_existing=True)
    _scheduler.add_job(_trend_analyze,   CronTrigger(day_of_week="mon", hour=6, minute=0, timezone=tz), id="trend_analyze",  replace_existing=True)
    _scheduler.add_job(_narrative_generate, CronTrigger(day_of_week="sun", hour=5,  minute=0,  timezone=tz), id="narrative_generate", replace_existing=True)
    _scheduler.add_job(_curiosity_search,     CronTrigger(hour=6,  minute=15,                    timezone=tz), id="curiosity_search",     replace_existing=True)
    _scheduler.add_job(_experience_analyze,   CronTrigger(day_of_week="sun", hour=4,  minute=0,  timezone=tz), id="experience_analyze",   replace_existing=True)
    _scheduler.add_job(_decision_analyze,     CronTrigger(day_of_week="sun", hour=4,  minute=30, timezone=tz), id="decision_analyze",     replace_existing=True)
    _scheduler.add_job(_goal_alignment_extract, CronTrigger(day_of_week="sun", hour=4, minute=35, timezone=tz), id="goal_alignment_extract", replace_existing=True)
    _scheduler.add_job(_preference_pattern_extract, CronTrigger(day_of_week="sun", hour=4, minute=45, timezone=tz), id="preference_pattern_extract", replace_existing=True)
    _scheduler.add_job(_adoption_count_recompute, CronTrigger(day_of_week="sun", hour=4, minute=50, timezone=tz), id="adoption_count_recompute", replace_existing=True)
    _scheduler.add_job(_episode_consolidate, CronTrigger(day_of_week="sun", hour=4, minute=55, timezone=tz), id="episode_consolidate", replace_existing=True)
    _scheduler.add_job(_knowledge_graph_extract, CronTrigger(day_of_week="sun", hour=5, minute=15, timezone=tz), id="knowledge_graph_extract", replace_existing=True)
    _scheduler.add_job(_memory_snapshot_generate, CronTrigger(day_of_week="sun", hour=5, minute=25, timezone=tz), id="memory_snapshot_generate", replace_existing=True)
    _scheduler.add_job(_self_interest_queries,CronTrigger(day_of_week="sun", hour=5,  minute=30, timezone=tz), id="self_interest_queries",replace_existing=True)

    # Phase Vis-1の前提条件対応(docs/sigmaris/phase_vis_report.md参照)。
    # cycle_health_measure: 毎日3:20(memory_embedの3:00から20分後、6:15の
    # curiosity_searchまで約3時間の空きがあった深夜帯へ配置)。
    # grounding_health_measure・safety_governance_scan: 日曜早朝、B2週次
    # バッチ(4:00〜5:30)の直後、5:40/5:45(B2本体とは重ならず、6:15の
    # curiosity_searchまで30分以上の余裕を残した)。
    _scheduler.add_job(_cycle_health_measure, CronTrigger(hour=3, minute=20, timezone=tz), id="cycle_health_measure", replace_existing=True)
    _scheduler.add_job(_grounding_health_measure, CronTrigger(day_of_week="sun", hour=5, minute=40, timezone=tz), id="grounding_health_measure", replace_existing=True)
    _scheduler.add_job(_safety_governance_scan, CronTrigger(day_of_week="sun", hour=5, minute=45, timezone=tz), id="safety_governance_scan", replace_existing=True)

    # Self-1〜3「自己認識の自動更新」(docs/sigmaris/self_awareness_report.md
    # 参照、_self_awareness_update()のコメントに時間帯・頻度の判断根拠を
    # 記載済み)。日曜早朝バッチの末尾、safety_governance_scan(5:45)の
    # 10分後・curiosity_search(6:15)の20分前に配置。
    _scheduler.add_job(_self_awareness_update, CronTrigger(day_of_week="sun", hour=5, minute=55, timezone=tz), id="self_awareness_update", replace_existing=True)

    # 旧X投稿システムの廃止、及び、新7カテゴリシステムへの実際の接続
    # (docs/sigmaris/phase_h_report.md)。1日4回、朝・昼・夕方・夜の
    # 時間帯に分散させ、既存の全ジョブと重ならない時刻を選んだ
    # (当時存在した8:00 morning_briefing・22:00 evening_checkinの間に
    # 収まっていた——両ジョブはPhase S-6で完全廃止済みだが、この4時刻
    # 自体を変更する理由はないため、時刻決定の経緯としてそのまま残す)。
    # 4回はあくまで"確認の機会"であり、実際に投稿するかどうかは毎回
    # Executive Gateとselect_post_category()が動的に判定する
    # (_categorized_x_post_check()のdocstring参照、固定スケジュールでは
    # ない)。
    # X_POST_SELF_TIMING_SPEC: この4回の cron は「考える機会」(決定フェーズ)
    # に流用する——材料からカテゴリ・本文を生成し、AI が内容にふさわしい未来
    # 時刻を決めて予約テーブルへ pending で積む(即投稿はしない)。
    _scheduler.add_job(_categorized_x_post_check, CronTrigger(hour=9,  minute=30, timezone=tz), id="categorized_x_post_check_1", replace_existing=True)
    _scheduler.add_job(_categorized_x_post_check, CronTrigger(hour=13, minute=30, timezone=tz), id="categorized_x_post_check_2", replace_existing=True)
    _scheduler.add_job(_categorized_x_post_check, CronTrigger(hour=17, minute=30, timezone=tz), id="categorized_x_post_check_3", replace_existing=True)
    _scheduler.add_job(_categorized_x_post_check, CronTrigger(hour=21, minute=30, timezone=tz), id="categorized_x_post_check_4", replace_existing=True)

    # X_POST_SELF_TIMING_SPEC フェーズC(配信ディスパッチャ): 高頻度 interval で
    # 回り、予約時刻(scheduled_at<=now)に達した pending 投稿を、opsec/Gate 再
    # 適用の上で実際に配信する(shadow ではログのみ)。実送信経路はここに一本化。
    _scheduler.add_job(
        _x_post_dispatch_check,
        IntervalTrigger(minutes=settings.x_post_dispatch_interval_min, timezone=tz),
        id="x_post_dispatch_check", replace_existing=True,
    )

    # Phase H-2「返信の検知、及び、フィルタリング」(docs/sigmaris/
    # phase_h_report.md)。categorized_x_post_check(9:30/13:30/17:30/
    # 21:30)と30分ずらし、既存の全ジョブと重ならない4時刻に配置した
    # (22:15は、当時存在した22:00のevening_checkinと15分の余裕を持たせた
    # ——同ジョブはPhase S-6で完全廃止済みだが、この時刻自体を変更する
    # 理由はないため、時刻決定の経緯としてそのまま残す)。
    _scheduler.add_job(_x_reply_detection_check, CronTrigger(hour=10, minute=0,  timezone=tz), id="x_reply_detection_check_1", replace_existing=True)
    _scheduler.add_job(_x_reply_detection_check, CronTrigger(hour=14, minute=0,  timezone=tz), id="x_reply_detection_check_2", replace_existing=True)
    _scheduler.add_job(_x_reply_detection_check, CronTrigger(hour=18, minute=0,  timezone=tz), id="x_reply_detection_check_3", replace_existing=True)
    _scheduler.add_job(_x_reply_detection_check, CronTrigger(hour=22, minute=15, timezone=tz), id="x_reply_detection_check_4", replace_existing=True)

    _scheduler.start()
    logger.info("Proactive scheduler started (tz=%s)", tz)


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Proactive scheduler shut down")
