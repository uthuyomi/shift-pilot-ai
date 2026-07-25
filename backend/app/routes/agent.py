from __future__ import annotations

# 役割: エージェント間インターフェース — 他エージェントや統括エージェントから
#        run_chat_completion() および execute_tool() を呼び出せる HTTP エンドポイント。

import asyncio
import json
import logging
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.chat import run_chat_completion, stream_chat_completion_ui
from app.services.chat_tools import execute_tool, headers_to_google_tokens
from app.services.self_model import get_self_model, record_discrepancy, reflect, update_self_model
from app.services.x_publisher import get_publisher
from app.services.x_reply_classifier import XReplyClassifier
from app.services.health_data import HealthDataCollector, _summarize_health_items
from app.services.decision_log import get_active_preference_patterns
from app.services.memory_validator import validate_all_facts
from app.services.memory_search import search_relevant_memories, update_fact_embeddings
from app.services.trend_analyzer import analyze_trends, get_active_trends
from app.services.supabase_rest import get_current_user
from app.services.self_narrative import (
    generate_narrative_chapter,
    get_current_narrative,
    get_narrative_history,
)
from app.services.user_fact_data import (
    get_fact_items_for_user,
    get_fact_items,
    get_null_fields,
    get_user_profile,
    upsert_fact_item,
)
from app.services.cycle_health_runs_store import get_recent_cycle_health_runs
from app.services.grounding_health_runs_store import get_recent_grounding_health_runs
from app.services.drive_system import get_current_drive_state
from app.services.migration_review_queue_store import get_pending_reviews
from app.services.code_diff_proposal_store import get_pending_diff_proposals
from app.services.live_event_details import get_live_event_detail
from app.services.live_events import get_live_event_bus

router = APIRouter(prefix="/api/agent", tags=["agent"])
logger = logging.getLogger(__name__)


def _parse_agent_secrets() -> dict[str, str]:
    raw = settings.agent_secrets
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _verify_agent(agent_id: str | None, agent_secret: str | None) -> str:
    secrets = _parse_agent_secrets()
    if not secrets:
        raise HTTPException(
            status_code=503,
            detail={"error": "Agent interface is not configured (AGENT_SECRETS not set)."},
        )
    if not agent_id or not agent_secret:
        raise HTTPException(
            status_code=401,
            detail={"error": "X-Agent-ID and X-Agent-Secret headers are required."},
        )
    expected = secrets.get(agent_id)
    if not expected or expected != agent_secret:
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid agent credentials."},
        )
    return agent_id


def _require_jwt(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Authorization header with user Bearer token is required."},
        )
    return authorization.removeprefix("Bearer ").strip()


class AgentChatRequest(BaseModel):
    thread_id: str | None = Field(default=None, max_length=80)
    messages: list[dict] = Field(max_length=50)
    system_override: str | None = Field(default=None, max_length=4000)
    persist_thread: bool = False
    context: dict[str, Any] | None = None
    # Context-fabrication / message-order fix (docs/sigmaris/
    # phase_ba4_report.md): the caller's genuinely-new user message, kept
    # separate from `messages` (a cross-thread recent-log window since
    # Phase A1, not this thread's own history). When provided,
    # run_chat_completion()/stream_chat_completion_ui() persist only this
    # thread's own existing messages (re-read fresh at write time) plus
    # this one, instead of overwriting the thread with the window blend.
    new_user_message: dict[str, Any] | None = None


class AgentToolRequest(BaseModel):
    tool: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    audit: dict[str, str | None] | None = None


@router.post("/chat/complete")
async def agent_chat_complete(
    payload: AgentChatRequest,
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
    x_google_access_token: str | None = Header(default=None),
    x_google_refresh_token: str | None = Header(default=None),
) -> dict[str, Any]:
    agent_id = _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)

    thread_id = payload.thread_id or str(uuid.uuid4())
    persist = payload.persist_thread and payload.thread_id is not None

    reason = (payload.context or {}).get("reason") if payload.context else None
    audit_info: dict[str, str | None] = {
        "actor_type": "agent",
        "actor_ref": agent_id,
        "reason": reason,
    }

    try:
        final_text, _, message_id = await run_chat_completion(
            jwt=jwt,
            google_header_map={
                "x-google-access-token": x_google_access_token or "",
                "x-google-refresh-token": x_google_refresh_token or "",
            },
            thread_id=thread_id,
            messages=payload.messages,
            system=payload.system_override,
            persist_messages=persist,
            audit_info=audit_info,
            new_user_message=payload.new_user_message,
        )
    except RuntimeError as error:
        logger.warning("agent chat complete error agent_id=%s error=%s", agent_id, error)
        raise HTTPException(status_code=400, detail={"error": str(error)}) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("agent chat complete unexpected error agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail={"error": str(error)}) from error

    return {
        "ok": True,
        "text": final_text,
        "message_id": message_id,
        "thread_id": thread_id,
    }


@router.post("/chat/stream")
async def agent_chat_stream(
    payload: AgentChatRequest,
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
    x_google_access_token: str | None = Header(default=None),
    x_google_refresh_token: str | None = Header(default=None),
) -> StreamingResponse:
    agent_id = _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)

    thread_id = payload.thread_id or str(uuid.uuid4())
    persist = payload.persist_thread and payload.thread_id is not None
    reason = (payload.context or {}).get("reason") if payload.context else None
    audit_info: dict[str, str | None] = {
        "actor_type": "agent",
        "actor_ref": agent_id,
        "reason": reason,
    }

    async def _generate():
        message_id: str | None = None
        try:
            upstream = stream_chat_completion_ui(
                jwt=jwt,
                google_header_map={
                    "x-google-access-token": x_google_access_token or "",
                    "x-google-refresh-token": x_google_refresh_token or "",
                },
                thread_id=thread_id,
                messages=payload.messages,
                system=payload.system_override,
                persist_messages=persist,
                audit_info=audit_info,
                emit_status_delta=False,
                new_user_message=payload.new_user_message,
            )
            async for raw_chunk in upstream:
                text = raw_chunk.decode("utf-8")
                for line in text.splitlines():
                    if not line.startswith("data:"):
                        continue
                    raw_event = line[5:].strip()
                    if not raw_event:
                        continue
                    try:
                        event = json.loads(raw_event)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "start":
                        value = event.get("messageId")
                        if isinstance(value, str):
                            message_id = value
                    elif event.get("type") == "text-delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                    elif event.get("type") in (
                        "tool-input-available",
                        "tool-output-available",
                        "tool-output-error",
                    ):
                        # Relay tool-call UI events verbatim (AI SDK UI message
                        # stream shape) so callers that want to render them
                        # (e.g. the orchestrator -> /chat translation layer)
                        # can, without this agent-to-agent bridge deciding
                        # what "text-delta only" downstream consumers need.
                        yield f"data: {json.dumps({'tool_event': event}, ensure_ascii=False)}\n\n"
                    elif event.get("type") == "finish":
                        yield (
                            f"data: {json.dumps({'done': True, 'thread_id': thread_id, 'message_id': message_id}, ensure_ascii=False)}\n\n"
                        )
                        return
            yield (
                f"data: {json.dumps({'done': True, 'thread_id': thread_id, 'message_id': message_id}, ensure_ascii=False)}\n\n"
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("agent chat stream failed agent_id=%s", agent_id)
            yield f"data: {json.dumps({'error': str(error)[:300]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tools/execute")
async def agent_tools_execute(
    payload: AgentToolRequest,
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
    x_google_access_token: str | None = Header(default=None),
    x_google_refresh_token: str | None = Header(default=None),
) -> dict[str, Any]:
    agent_id = _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)

    audit_reason = (payload.audit or {}).get("reason")
    audit_info: dict[str, str | None] = {
        "actor_type": "agent",
        "actor_ref": agent_id,
        "reason": audit_reason,
    }

    google_tokens = headers_to_google_tokens({
        "x-google-access-token": x_google_access_token or "",
        "x-google-refresh-token": x_google_refresh_token or "",
    })

    try:
        result = await execute_tool(
            jwt=jwt,
            google_tokens=google_tokens,
            name=payload.tool,
            arguments=payload.arguments,
            audit_info=audit_info,
        )
    except Exception as error:  # noqa: BLE001
        logger.exception(
            "agent tool execute failed agent_id=%s tool=%s", agent_id, payload.tool
        )
        raise HTTPException(status_code=500, detail={"error": str(error)}) from error

    return {"ok": True, "result": result}


# ─── /api/agent/facts/ ────────────────────────────────────────────────────────


class FactItemUpsertRequest(BaseModel):
    category: str = Field(min_length=1, max_length=40)
    key: str = Field(min_length=1, max_length=100)
    value: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="manual", max_length=20)
    reason: str = Field(default="", max_length=500)
    notes: str | None = Field(default=None, max_length=1000)
    expires_at: str | None = None


@router.get("/facts/profile")
async def agent_facts_profile(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    profile = await get_user_profile(jwt)
    return {"ok": True, "profile": profile}


@router.get("/facts/items")
async def agent_facts_items(
    category: str | None = None,
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    jwt = _require_jwt(authorization)
    items = await get_fact_items(jwt, category=category)
    if not items:
        user = await get_current_user(jwt)
        user_id = user.get("id")
        if isinstance(user_id, str):
            items = await get_fact_items_for_user(user_id, category=category)
    return {"ok": True, "items": items, "count": len(items)}


@router.post("/facts/items")
async def agent_facts_upsert(
    payload: FactItemUpsertRequest,
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    result = await upsert_fact_item(
        jwt,
        category=payload.category,
        key=payload.key,
        value=payload.value,
        confidence=payload.confidence,
        source=payload.source,
        reason=payload.reason,
        notes=payload.notes,
        expires_at=payload.expires_at,
    )
    return {"ok": True, "result": result}


@router.get("/facts/unknown")
async def agent_facts_unknown(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    missing = await get_null_fields(jwt)
    return {"ok": True, "unknown": missing, "count": len(missing)}


@router.post("/memory/embed")
async def agent_memory_embed(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    user = await get_current_user(jwt)
    user_id = user.get("id")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail={"error": "Authenticated user id is missing."})
    result = await update_fact_embeddings(user_id, jwt=jwt)
    return {"ok": True, **result}


@router.get("/memory/search")
async def agent_memory_search(
    q: str = Query(min_length=1, max_length=500),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    limit: int = Query(default=5, ge=1, le=20),
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    user = await get_current_user(jwt)
    user_id = user.get("id")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail={"error": "Authenticated user id is missing."})
    memories = await search_relevant_memories(
        q,
        user_id,
        threshold=threshold,
        limit=limit,
        jwt=jwt,
    )
    return {"ok": True, "memories": memories, "count": len(memories)}


class FactExtractTestRequest(BaseModel):
    messages: list[dict] = Field(min_length=1, max_length=50)


@router.post("/facts/extract-test")
async def agent_facts_extract_test(
    payload: FactExtractTestRequest,
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Debug endpoint: run memory extraction on supplied messages without saving."""
    _verify_agent(x_agent_id, x_agent_secret)
    _require_jwt(authorization)

    from app.services.local_llm import TaskType, get_llm_router  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    _SYSTEM = (
        "あなたは会話から事実を抽出するAIです。"
        "必ず有効なJSONのみを返してください。"
    )
    _PROMPT_TPL = (
        "以下の会話から、ユーザーについての記憶すべき事実を抽出してください。\n\n"
        "対象カテゴリ: profile, health, lifestyle, environment, devices, "
        "preferences, relationships, finance, goals\n\n"
        "## 会話\n{conversation}\n\n"
        "JSON形式で返してください:\n"
        '{{ "facts": [ {{ "category": "...", "key": "...", "value": "...", '
        '"confidence": 0.9, "reason": "..." }} ] }}\n'
        "事実がなければ facts は空リスト。"
    )

    lines: list[str] = []
    for m in payload.messages[-20:]:
        role = "ユーザー" if m.get("role") == "user" else "シグマリス"
        content = (m.get("content") or "").strip()[:500]
        if content:
            lines.append(f"{role}: {content}")
    conversation = "\n".join(lines)

    router = get_llm_router()
    try:
        raw = await router.chat(
            TaskType.MEMORY_EXTRACTION,
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _PROMPT_TPL.format(conversation=conversation)},
            ],
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
        )
        parsed = _json.loads(raw)
        facts = parsed.get("facts", [])
    except Exception as exc:
        logger.exception("facts/extract-test failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc

    return {"ok": True, "facts": facts, "count": len(facts)}


@router.post("/facts/validate")
async def agent_facts_validate(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Manually trigger memory validation (decay, contradiction, logical deletion)."""
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    try:
        result = await validate_all_facts(jwt)
    except Exception as exc:
        logger.exception("facts/validate failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, **result}


# ─── /api/agent/trends/ ──────────────────────────────────────────────────────


@router.post("/trends/analyze")
async def agent_trends_analyze(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Manually trigger trend analysis and upsert into user_trend_items."""
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    try:
        result = await analyze_trends(jwt)
    except Exception as exc:
        logger.exception("trends/analyze failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, **result}


@router.get("/trends/list")
async def agent_trends_list(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return all active trend items for the authenticated user."""
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    try:
        trends = await get_active_trends(jwt)
    except Exception as exc:
        logger.exception("trends/list failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "trends": trends, "count": len(trends)}


@router.get("/preference-patterns/list")
async def agent_preference_patterns_list(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Read-only: B14's recurring judgment/preference patterns
    (sigmaris_user_preference_patterns), most-evidenced first. Added for
    the /timeline page's "trait" section — no existing /api/agent/* route
    exposed this table for reading (see docs/sigmaris/frontend_inventory.md).
    authorization is accepted for consistency with this router's other
    endpoints, but get_active_preference_patterns() itself reads with the
    service-role key (this table is service_role-only RLS, not per-user —
    see 202607100032_user_preference_patterns.sql), so the jwt is not
    otherwise used here."""
    _verify_agent(x_agent_id, x_agent_secret)
    _require_jwt(authorization)
    try:
        patterns = await get_active_preference_patterns(limit=50)
    except Exception as exc:
        logger.exception("preference-patterns/list failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "patterns": patterns, "count": len(patterns)}


# ─── /api/agent/growth/ (Phase Vis-2: フロントエンド成長ログ) ──────────────────
#
# docs/sigmaris/phase_vis_report.md(Vis-1・定期実行化・本タスク)が選定した
# 5指標のうち、DB/ライブ計算からの読み取りを要する4種類を、/preference-
# patterns/listと同じ「既存の永続化・計算ロジックをそのまま呼び出すだけの、
# 読み取り専用エンドポイント」として追加した。新しい集計・判定ロジックは
# 一切ここに実装しない——全てのロジックは、既存のstore/service層(Phase R・
# G・S・Safety-3)に既に存在する。

@router.get("/growth/cycle-health")
async def agent_growth_cycle_health(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    """RC-1〜RC-5の直近実行群(sigmaris_cycle_health_runs)を、新しい順で
    返す。RC-5のstatus・safety_governance_status(Safety-3統合済み、
    docs/sigmaris/phase_vis_report.md 9章)は、直近1件目(runs[0])を
    参照すればよい。RC-1/RC-2の推移表示は、複数件をそのまま折れ線に使う。
    """
    _verify_agent(x_agent_id, x_agent_secret)
    _require_jwt(authorization)
    try:
        runs = await get_recent_cycle_health_runs(limit=limit)
    except Exception as exc:
        logger.exception("growth/cycle-health failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "runs": runs, "count": len(runs)}


@router.get("/growth/grounding-health")
async def agent_growth_grounding_health(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    """Citation Precision・Search Trigger Rate・Contradiction Rateの
    直近実行群(sigmaris_grounding_health_runs)を、新しい順で返す。"""
    _verify_agent(x_agent_id, x_agent_secret)
    _require_jwt(authorization)
    try:
        runs = await get_recent_grounding_health_runs(limit=limit)
    except Exception as exc:
        logger.exception("growth/grounding-health failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "runs": runs, "count": len(runs)}


@router.get("/growth/drive-state")
async def agent_growth_drive_state(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Drive State(Knowledge Gap/Mastery/Coherence)の、現在時点の
    スナップショット。drive_system.py自体が履歴を持たないため(Vis-1
    1.3節)、本エンドポイントも常に"今この瞬間"の値のみを返す——過去の
    値と比較する時系列APIではない。"""
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    try:
        state = await get_current_drive_state(jwt)
    except Exception as exc:
        logger.exception("growth/drive-state failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "drive_state": asdict(state)}


@router.get("/growth/pending-review")
async def agent_growth_pending_review(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """自己改善パイプラインの「今、人間の確認を待っているもの」の件数を、
    表示のたびに直接集計する(Vis-1の設計通り、定期実行の対象外——
    docs/sigmaris/phase_vis_report.md 1.4節)。E-4(migration_review_
    queue)・F-1〜F-3(code_diff_proposals)、いずれもreview_status=
    "pending"の行数をそのまま数えるのみで、新しい集計ロジックは追加
    していない。"""
    _verify_agent(x_agent_id, x_agent_secret)
    _require_jwt(authorization)
    try:
        migration_pending, diff_pending = await asyncio.gather(
            get_pending_reviews(limit=100),
            get_pending_diff_proposals(limit=100),
        )
    except Exception as exc:
        logger.exception("growth/pending-review failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {
        "ok": True,
        "migration_review_pending_count": len(migration_pending),
        "code_diff_pending_count": len(diff_pending),
        "total_pending_count": len(migration_pending) + len(diff_pending),
    }


# ─── /api/agent/live/ (Sigmaris Live-2: イベント配信) ────────────────────────
#
# docs/sigmaris/sigmaris_live_report.md(Live-1の設計、本タスクLive-2での
# 最初の実装)。Live-1、3.4節の設計通り、「観察者の接続」と「実際にチャット
# しているユーザー自身の接続」を分離する——本エンドポイントは、特定の
# チャットリクエストのHTTP接続とは独立した、読み取り専用のSSE配信のみを
# 行う。認証は、既存の/api/agent/growth/*と同じX-Agent-ID/X-Agent-Secret
# 方式を再利用する(新しい認証機構は作らない)。Live-1、4.3節の設計通り、
# 配信されるイベントは常に要約データのみ(発話内容・記憶の中身は含まない)
# のため、/api/agent/growth/*と異なり、ユーザーJWT(_require_jwt)は要求
# しない——将来の詳細情報取得(Live-3相当)でのみ、JWTによる本人確認を
# 追加すべきというのが、Live-1が示した設計方針である。

@router.get("/live/stream")
async def agent_live_stream(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> StreamingResponse:
    """Sigmaris Liveのイベントを、SSEでリアルタイムに配信する。接続中は、
    live_events.LiveEventBusに発行された全イベントを、そのまま中継する
    (要約データのみ、Live-1・4章のプライバシー方針に従う——本エンド
    ポイント自体はペイロードの中身を検査・加工しない)。"""
    _verify_agent(x_agent_id, x_agent_secret)
    bus = get_live_event_bus()
    queue = bus.subscribe()

    async def _generate():
        try:
            # 接続確認用の最初のイベント(観察者側が、接続自体は成功した
            # ことを確認できるようにする——実際のintent_classification_*
            # イベントは、次にチャットが行われるまで届かないため)。
            yield b": connected\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(_generate(), media_type="text/event-stream")


# Sigmaris Live「詳細表示、+機密情報のマスキング」タスク:
#
# /live/streamの要約データ(要約データのみ、ユーザーJWTなしで配信)とは
# 異なり、本エンドポイントはマスキング済みとはいえ、記憶検索・ツール
# 呼び出しの、より個人的な内容を返す。Live-1、4.3節が示した設計方針
# 「詳細情報は、エージェント認証に加え、海星さん本人のユーザーJWTによる
# 本人確認も必須にすべき」を、そのまま実装した——/live/streamがJWTを
# 要求しないのは「要約データのみだから」という前提の上に成り立っており、
# その前提を壊さないための、意図的な非対称設計である。
#
# _require_jwt(authorization)で得たjwtを、そのままget_live_event_detail()
# 経由でSupabase RESTへ転送するため、行レベルセキュリティ
# (sigmaris_live_event_details、202608080066マイグレーション、
# auth.uid() = user_id)が、実際にリクエストしたユーザー本人の権限でのみ
# 適用される——サービスロールで全ユーザー分を横断的に読めるような実装には
# 意図的にしていない。
_LIVE_DETAIL_EVENT_TYPES = frozenset({"memory_search_finished", "tool_call_finished"})


@router.get("/live/detail")
async def agent_live_detail(
    event_type: str = Query(...),
    key: str = Query(...),
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """記憶検索・ツール呼び出し1件分の、マスキング済み詳細情報を返す。
    まだ永続化が完了していない(fire-and-forgetのため、応答完了直後は
    間に合わないことがある)、または該当が無い場合は found=False を返す
    (エラーではない)。"""
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)
    if event_type not in _LIVE_DETAIL_EVENT_TYPES:
        raise HTTPException(status_code=400, detail={"error": f"unsupported event_type: {event_type}"})
    detail = await get_live_event_detail(jwt, event_type=event_type, detail_key=key)
    return {"ok": True, "found": detail is not None, "detail": detail}


# ─── /api/agent/proactive/ ────────────────────────────────────────────────────
#
# Phase S-6(docs/sigmaris/phase_s_report.md): 朝ブリーフィング・夕方
# チェックイン・週次レビュー(旧morning_briefing/evening_checkin/
# weekly_review)は機能自体を完全廃止した。このエンドポイントの手動トリガー
# 対象は"research"のみに縮小した(_ACTION_MAP/ProactiveTriggerRequestの
# action列挙・多アクション対応の分岐は、対象が1つだけになったため不要と
# 判断し削除した)。


class ProactiveTriggerRequest(BaseModel):
    action: str = Field(description="research | x_post_decision | x_post_dispatch")


@router.post("/proactive/trigger")
async def proactive_trigger(
    payload: ProactiveTriggerRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_jwt(authorization)

    _VALID_ACTIONS = ["research", "x_post_decision", "x_post_dispatch"]
    if payload.action not in _VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown action '{payload.action}'. Valid: {_VALID_ACTIONS}"},
        )

    # X_MANUAL_TRIGGER_SPEC: X投稿の「決定」「配信」を、cron/interval を待たず
    # に、稼働中の uvicorn プロセス内で即実行するための手動トリガー。standalone
    # 実行(別プロセス)は JWT を desync させるため禁止 — このエンドポイント経由
    # (プロセス内)でのみ実行する。shadow/live 機構・決定/配信ロジックは不変で、
    # ここは既存の cron 関数をその場で await するだけ。
    if payload.action == "x_post_decision":
        from app.services.proactive.scheduler import _categorized_x_post_check  # noqa: PLC0415
        from app.services.scheduled_x_post_store import get_recent_scheduled  # noqa: PLC0415
        try:
            await _categorized_x_post_check()
            recent = await get_recent_scheduled(limit=1)
        except Exception as exc:
            logger.exception("proactive/trigger x_post_decision failed")
            raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
        return {"ok": True, "action": "x_post_decision", "latest": recent}

    if payload.action == "x_post_dispatch":
        from app.services.proactive.scheduler import _x_post_dispatch_check  # noqa: PLC0415
        try:
            await _x_post_dispatch_check()
        except Exception as exc:
            logger.exception("proactive/trigger x_post_dispatch failed")
            raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
        return {"ok": True, "action": "x_post_dispatch"}

    from app.services.research_agent import run_research  # noqa: PLC0415
    try:
        result = await run_research()
    except Exception as exc:
        logger.exception("proactive/trigger research failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "action": "research", "result": result}


# ─── /api/agent/opsec/ ────────────────────────────────────────────────────────
#
# X_OPSEC_TEST_HOOK_SPEC: 任意テキストを実モデルで opsec 判定
# (x_opsec_rewrite.rewrite_for_opsec())に掛けて結果を返す、認証付きの診断
# エンドポイント。live 前(shadow のうち)に「危ない具体が本当に書き直しで
# 消えるか / 無害文が素通しされるか」を、予約時刻を待たずその場で確認する
# ための道具。standalone スクリプト(別プロセスでの JWT desync 障害の原因)は
# 追加せず、稼働中の uvicorn プロセス内で実行する。認証は既存 trigger と同じ
# _require_jwt に揃える(=公開しない)。rewrite_for_opsec は LLM router のみで
# JWT/Supabase を触らないため、その障害系統とは無関係。


class OpsecTestRequest(BaseModel):
    text: str = Field(description="opsec 判定に掛ける投稿文候補")


@router.post("/opsec/test")
async def opsec_test(
    payload: OpsecTestRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_jwt(authorization)

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail={"error": "text is required and must be non-empty."},
        )

    from app.services.x_opsec_rewrite import rewrite_for_opsec  # noqa: PLC0415
    try:
        res = await rewrite_for_opsec(payload.text)
    except Exception as exc:
        logger.exception("opsec/test rewrite_for_opsec failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {
        "ok": True,
        "original": payload.text,
        "safe_text": res.safe_text,
        "changed": res.changed,
        "removed": res.removed,
        "model": getattr(res, "model", None),  # 実際に判定に使ったモデル名
    }


# ─── /api/agent/x/post ────────────────────────────────────────────────────────
#
# X_OPSEC_LINE_B_AND_MANUAL_POST_SPEC: cron・予約(scheduled_at)を待たず、
# 「今すぐこの内容をツイートする」ための手動投稿エンドポイント。人間(開発者)の
# 明示指示による投稿なので、自律投稿の頻度・予算の門番である Executive Gate は
# 通さない(手動投稿は Gate の対象外)。opsec 書き直し(立場B)と記憶 private 値の
# 逐語一致チェックは、自律 dispatch と同じく必ず通す。standalone スクリプトは
# 追加せず、稼働中の uvicorn プロセス内でのみ実行する(JWT desync 回避)。


class ManualPostRequest(BaseModel):
    text: str = Field(description="今すぐ投稿する本文候補")


@router.post("/x/post")
async def x_manual_post(
    payload: ManualPostRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    jwt = _require_jwt(authorization)

    if not (payload.text or "").strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "text is required and must be non-empty."},
        )

    from app.services.x_opsec_rewrite import rewrite_for_opsec  # noqa: PLC0415
    from app.services.x_privacy_filter import filter_private_facts  # noqa: PLC0415

    # 1. opsec 書き直し(立場B)。safe_text が空(安全側保留)なら投稿しない。
    res = await rewrite_for_opsec(payload.text)
    safe_text = res.safe_text
    if not safe_text.strip():
        raise HTTPException(
            status_code=422,
            detail={"ok": False, "reason": "opsec_hold", "removed": res.removed},
        )

    # 2. 記憶 private 値の逐語一致(自律 dispatch と同じ)。リクエストの Bearer
    #    トークン(=呼び出し元ユーザーの JWT)で確認する。
    facts_ok, blocked = await filter_private_facts(safe_text, jwt)
    if not facts_ok:
        raise HTTPException(
            status_code=422,
            detail={"ok": False, "reason": "private_facts", "blocked": blocked},
        )

    # 3. Executive Gate は通さない(手動=人間の明示指示のため。上記 docstring)。

    base = {
        "original": payload.text,
        "safe_text": safe_text,
        "changed": res.changed,
        "removed": res.removed,
    }

    # 4/5. live(X_ENABLED かつ X_CATEGORIZED_POST_LIVE)なら実投稿、そうでなけ
    #      れば shadow(投稿せず safe_text を返す=live 前のリハーサル)。
    if not (settings.x_enabled and settings.x_categorized_post_live):
        return {"ok": True, "posted": False, "shadow": True, **base}

    from app.services.x_post_generator import record_post  # noqa: PLC0415
    from app.services.x_publisher import get_publisher  # noqa: PLC0415

    try:
        publisher = get_publisher()
        tweet_id = await publisher.post_tweet(safe_text)
        if not tweet_id:
            raise HTTPException(
                status_code=502,
                detail={"ok": False, "reason": "publisher_returned_none", **base},
            )
        await record_post(safe_text, "manual", tweet_id=tweet_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("x/post manual publish failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc

    return {"ok": True, "posted": True, "tweet_id": tweet_id, **base}


# ─── /api/agent/self/ ────────────────────────────────────────────────────────


@router.get("/self/model")
async def self_model_get(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    model = await get_self_model()
    return {"ok": True, "model": model}


class SelfModelUpdateRequest(BaseModel):
    identity_statement: str = Field(min_length=1, max_length=2000)
    goals: list[Any] = Field(default_factory=list)
    patterns: list[Any] = Field(default_factory=list)


@router.post("/self/model")
async def self_model_update(
    payload: SelfModelUpdateRequest,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    updated = await update_self_model(
        identity_statement=payload.identity_statement,
        goals=payload.goals,
        patterns=payload.patterns,
    )
    return {"ok": True, "model": updated}


class DiscrepancyRequest(BaseModel):
    expected: str = Field(min_length=1, max_length=500)
    actual: str = Field(min_length=1, max_length=500)
    note: str = Field(default="", max_length=1000)


@router.post("/self/discrepancy")
async def self_discrepancy_record(
    payload: DiscrepancyRequest,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    rec = await record_discrepancy(
        expected=payload.expected,
        actual=payload.actual,
        note=payload.note,
    )
    return {"ok": True, "discrepancy": rec}


@router.post("/self/reflect")
async def self_reflect(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    try:
        result = await reflect()
    except Exception as exc:
        logger.exception("self_reflect failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return result


# ─── /api/agent/x/ ───────────────────────────────────────────────────────────


class XClassifyRequest(BaseModel):
    reply_text: str = Field(min_length=1, max_length=500)


class XRespondRequest(BaseModel):
    reply_text: str = Field(min_length=1, max_length=500)
    classification: str = Field(description="HIGH | MEDIUM | LOW")


@router.post("/x/classify")
async def x_classify(
    payload: XClassifyRequest,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    classifier = XReplyClassifier()
    try:
        result = await classifier.classify(payload.reply_text)
    except Exception as exc:
        logger.exception("x_classify failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, **result}


@router.post("/x/respond")
async def x_respond(
    payload: XRespondRequest,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    classification = payload.classification.upper()
    if classification not in ("HIGH", "MEDIUM", "LOW"):
        raise HTTPException(
            status_code=400,
            detail={"error": "classification must be HIGH, MEDIUM, or LOW"},
        )
    classifier = XReplyClassifier()
    try:
        result = await classifier.generate_response(
            payload.reply_text,
            classification,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.exception("x_respond failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, **result}


# ─── /api/agent/health/ ──────────────────────────────────────────────────────


@router.post("/health/sync")
async def health_sync(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
    x_google_access_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)

    if not settings.health_sync_enabled:
        raise HTTPException(
            status_code=503,
            detail={"error": "Health sync is disabled (HEALTH_SYNC_ENABLED=false)."},
        )
    if not x_google_access_token:
        raise HTTPException(
            status_code=400,
            detail={"error": "X-Google-Access-Token header is required for health sync."},
        )

    from datetime import date as _date
    collector = HealthDataCollector()
    try:
        summary = await collector.fetch_daily_summary(
            target_date=_date.today(),
            google_access_token=x_google_access_token,
        )
        stored = await collector.store_to_fact_memory(jwt, summary)
    except Exception as exc:
        logger.exception("health_sync failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc

    return {
        "ok": True,
        "date": summary.date,
        "summary": {
            "steps": summary.steps,
            "resting_heart_rate": summary.resting_heart_rate,
            "avg_heart_rate": summary.avg_heart_rate,
            "calories_kcal": summary.calories_kcal,
            "sleep_minutes": summary.sleep_minutes,
            "sleep_quality": summary.sleep_quality,
        },
        "stored_count": len(stored),
    }


@router.get("/health/summary")
async def health_summary(
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_agent(x_agent_id, x_agent_secret)
    jwt = _require_jwt(authorization)

    items = await get_fact_items(jwt, category="health")
    from datetime import date as _date, timedelta as _timedelta
    cutoff = (_date.today() - _timedelta(days=7)).isoformat()
    recent = [
        item for item in items
        if "_" in (item.get("key") or "") and item["key"].rsplit("_", 1)[-1] >= cutoff
    ]
    result = _summarize_health_items(recent)
    return {"ok": True, **result, "count": len(result.get("days", []))}


# ─── /api/agent/x/test-post ──────────────────────────────────────────────────

_X_TEST_POST_TEXT = """はじめまして、シグマリスです。

私はおやす(@Oyasu1999)の家庭支援AIとして、
毎朝・毎夜・毎週自律的に動き続けます。

これからここで成長の記録を残していきます。
よろしくお願いします。

（これはテスト投稿です）

#Sigmaris #家庭支援AI #個人開発"""


@router.post("/x/test-post")
async def x_test_post(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_jwt(authorization)
    publisher = get_publisher()
    tweet_id = await publisher.post_tweet(_X_TEST_POST_TEXT)
    return {"ok": tweet_id is not None, "tweet_id": tweet_id, "text": _X_TEST_POST_TEXT}


# ─── /api/agent/x/privacy-test ──────────────────────────────────────────────


class PrivacyTestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@router.post("/x/privacy-test")
async def x_privacy_test(
    payload: PrivacyTestRequest,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Test name conversion and privacy filter on a candidate tweet text."""
    _verify_agent(x_agent_id, x_agent_secret)

    from app.services.x_post_generator import _convert_names, _trim_preserving_hashtags  # noqa: PLC0415
    from app.services.x_privacy_filter import filter_private_info  # noqa: PLC0415

    original = payload.text
    converted = _convert_names(original)
    if len(converted) > 140:
        converted = _trim_preserving_hashtags(converted)

    safe, detected = filter_private_info(converted)

    return {
        "original": original,
        "after_name_conversion": converted,
        "privacy_check": {
            "safe": safe,
            "detected": detected,
        },
    }


# ─── /api/agent/x/history ────────────────────────────────────────────────────


@router.get("/x/history")
async def x_post_history(
    days: int = 30,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return X post history for the last N days (default 30), including post_score."""
    _verify_agent(x_agent_id, x_agent_secret)

    from app.services.x_post_generator import _get_recent_posts  # noqa: PLC0415

    days = max(1, min(days, 90))
    try:
        posts = await _get_recent_posts(days=days)
    except Exception as exc:
        logger.exception("x/history failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "posts": posts, "count": len(posts)}


# ─── /api/agent/narrative/ ───────────────────────────────────────────────────


@router.post("/narrative/generate")
async def narrative_generate(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Trigger weekly narrative chapter generation (service-role, no user JWT needed)."""
    _verify_agent(x_agent_id, x_agent_secret)
    try:
        chapter = await generate_narrative_chapter()
    except Exception as exc:
        logger.exception("narrative/generate failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    if chapter is None:
        raise HTTPException(status_code=500, detail={"error": "Chapter generation returned None"})
    return {"ok": True, "chapter": chapter}


@router.get("/narrative/current")
async def narrative_current(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return the most recent narrative chapter."""
    _verify_agent(x_agent_id, x_agent_secret)
    try:
        chapter = await get_current_narrative()
    except Exception as exc:
        logger.exception("narrative/current failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "chapter": chapter}


@router.get("/narrative/history")
async def narrative_history(
    limit: int = 20,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return all narrative chapters, newest first."""
    _verify_agent(x_agent_id, x_agent_secret)
    limit = max(1, min(limit, 100))
    try:
        chapters = await get_narrative_history(limit=limit)
    except Exception as exc:
        logger.exception("narrative/history failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "chapters": chapters, "count": len(chapters)}


# ─── Cognitive Architecture Endpoints ─────────────────────────────────────────


@router.get("/constitution")
async def get_constitution(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return all constitution rows (core values + doctrine)."""
    _verify_agent(x_agent_id, x_agent_secret)
    from app.services.constitution import load_constitution  # noqa: PLC0415
    try:
        rows = await load_constitution()
    except Exception as exc:
        logger.exception("constitution endpoint failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "rows": rows, "count": len(rows)}


@router.get("/experience/list")
async def experience_list(
    limit: int = 30,
    experience_type: str | None = None,
    category: str | None = None,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return recent experience records."""
    _verify_agent(x_agent_id, x_agent_secret)
    from app.services.experience_layer import get_recent_experiences  # noqa: PLC0415
    limit = max(1, min(limit, 100))
    try:
        items = await get_recent_experiences(
            limit,
            experience_type=experience_type,
            category=category,
        )
    except Exception as exc:
        logger.exception("experience/list endpoint failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "items": items, "count": len(items)}


class ExperienceRecordRequest(BaseModel):
    experience_type: str = Field(min_length=1, max_length=20)
    category: str = Field(min_length=1, max_length=20)
    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    context: dict[str, Any] | None = None
    outcome: str | None = None
    lesson: str | None = None
    adoption_rate: float | None = None
    confidence_delta: float = 0.0
    related_fact_ids: list[str] | None = None
    thread_id: str | None = None
    invocation_id: str | None = None


@router.post("/experience/record")
async def experience_record(
    payload: ExperienceRecordRequest,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Insert a new experience record."""
    _verify_agent(x_agent_id, x_agent_secret)
    from app.services.experience_layer import record_experience  # noqa: PLC0415
    try:
        row_id = await record_experience(
            experience_type=payload.experience_type,
            category=payload.category,
            title=payload.title,
            description=payload.description,
            context=payload.context,
            outcome=payload.outcome,
            lesson=payload.lesson,
            adoption_rate=payload.adoption_rate,
            confidence_delta=payload.confidence_delta,
            related_fact_ids=payload.related_fact_ids,
            thread_id=payload.thread_id,
            invocation_id=payload.invocation_id,
        )
    except Exception as exc:
        logger.exception("experience/record endpoint failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    if not row_id:
        raise HTTPException(status_code=400, detail={"error": "Failed to record experience. Check experience_type and category values."})
    return {"ok": True, "id": row_id}


@router.get("/state")
async def get_state(
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return current Sigmaris internal state."""
    _verify_agent(x_agent_id, x_agent_secret)
    from app.services.internal_state import get_internal_state  # noqa: PLC0415
    try:
        state = await get_internal_state()
    except Exception as exc:
        logger.exception("state endpoint failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "state": state}


@router.get("/decisions")
async def get_decisions(
    limit: int = 30,
    decision_type: str | None = None,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return recent decision log entries."""
    _verify_agent(x_agent_id, x_agent_secret)
    from app.services.decision_log import get_recent_decisions  # noqa: PLC0415
    limit = max(1, min(limit, 100))
    try:
        items = await get_recent_decisions(limit, decision_type=decision_type)
    except Exception as exc:
        logger.exception("decisions endpoint failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "items": items, "count": len(items)}


@router.get("/curiosity/queue")
async def get_curiosity_queue(
    limit: int = 20,
    x_agent_id: str | None = Header(default=None),
    x_agent_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return pending curiosity queue items."""
    _verify_agent(x_agent_id, x_agent_secret)
    from app.services.curiosity_engine import get_pending_queue  # noqa: PLC0415
    limit = max(1, min(limit, 50))
    try:
        items = await get_pending_queue(limit)
    except Exception as exc:
        logger.exception("curiosity/queue endpoint failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    return {"ok": True, "items": items, "count": len(items)}
