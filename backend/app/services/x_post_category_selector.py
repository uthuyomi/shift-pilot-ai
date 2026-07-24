# 役割: Phase H-1「投稿の種類・テンプレートの実装」— 7カテゴリ(A〜G)の
# うち、今この瞬間、実際に投稿する材料があるものを、Drive State(S-0)・
# Executive Gate(S-1)・その日の実際の出来事(D〜Fパイプラインの進捗、
# 記憶の変化等)に基づいて、動的に選ぶ。
#
# 【依頼書「計画的なスケジュールを作らないこと」への対応】
# 曜日・時間帯にカテゴリを固定するテーブル(旧x_post_generator.pyの
# _SLOT_TYPES = {"morning": [...], "evening": [...], "weekly": [...]}の
# ような形)は、本モジュールには一切存在しない。**全てのカテゴリの
# 選定は、実際にその日集めたシグナル(材料の有無)によってのみ決まる。**
# 旧_SLOT_TYPESは、既存の5投稿タイプ(memory_gained等)の挙動を壊さない
# ため、変更せず残したままにしている(要件8)。
#
# 【新しいデータ収集を追加しない、という制約への対応】
# 本モジュールが読むのは、全て既存のstore関数(drive_system・self_model・
# hypothesis_store・code_diff_proposal_store・static_verification_store・
# research_items・agent_invocation_audit_logs)のみ。新しい測定・ログ
# 記録の仕組みは、1つも追加していない——code_diff_proposal_store.py::
# get_recent_diff_proposals()のみ、既存テーブルへの新しい読み取り専用
# アクセサとして追加した(本ファイルではなく、同ストアファイル側に追加、
# 判断根拠はそちらのdocstring参照)。

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.services.code_diff_proposal_store import get_recent_diff_proposals
from app.services.drive_system import get_current_drive_state
from app.services.executive_gate import evaluate_executive_gate
from app.services.hypothesis_store import get_recent_hypotheses
from app.services.self_model import get_self_model
from app.services.supabase_rest import _get_client, _require_supabase_config
from app.services.x_post_categories import (
    ALL_CATEGORIES,
    DESIGN_PHILOSOPHY_TOPICS,
    GENERAL_CATEGORIES,
    CategoryContext,
    category_group,
)

logger = logging.getLogger(__name__)

# 依頼書「1日1〜3投稿」。旧システムの_DAILY_POST_LIMIT(=2、x_post_
# generator.py)とは別の、本カテゴリシステム専用の上限値として、
# 新しく定義した——旧システムの値をそのまま流用すると、旧5タイプと
# 新7カテゴリの投稿数が合算で数えられてしまい、依頼書が求める「1〜3」
# という範囲の意味が変わってしまうため(判断根拠)。
MAX_DAILY_CATEGORY_POSTS = 3

# 直近何日分の投稿履歴を、一般/技術のバランス調整に使うか。B2/B14等の
# 「recurring patternと認めるための最低サンプル数」とは性質が異なる
# (ここでは日数ベースのウィンドウ)ため、独自の値として定義した。
_BALANCE_LOOKBACK_DAYS = 14
# 一般:技術の比率が、これを超えて偏った場合のみ、緩やかに調整する
# (依頼書「厳密な割り当てはしないこと」への対応、緩めの閾値)。
_BALANCE_SKEW_THRESHOLD = 2.0
# 偏り判定を行うために必要な、直近投稿の最低合計件数(サンプル不足で
# 誤って「偏っている」と判定しないため——B2/B14の「recurring pattern
# と認めるための最低サンプル数」と同じ考え方)。
_BALANCE_MIN_TOTAL = 2


# ─── 層1:候補選定段階では opsec 除外をしない(X_OPSEC_LLM_REWRITE_SPEC) ──
# 【方針転換】旧実装は、公開Xの A_spontaneous_remark(開発者へ公開で問いかける
# 投稿)の素材から、機微な"具体的な秘密"を含む記憶確認候補を、単語/正規表現
# (旧: 広いインフラ語、その後: x_privacy_filter.filter_private_info())で
# 除外していた。だがこの入口での除外が、機材の存在レベルの確認(「自宅サーバーで
# GTX1660 を動かしてる、合ってる?」等)まで巻き込み、公開材料が全滅して
# 「投稿なし」の目詰まりを起こしていた(X_OPSEC_LAYER1_REFINE の「8件全除外」)。
#
# そこで opsec は、候補選定ではなく配信直前の LLM 判定＋書き直し
# (x_opsec_rewrite.rewrite_for_opsec())へ一本化する。ここでは素材を一切
# 除外せず素通しする——actionable な具体は、配信直前に「その部分だけ自然に
# 書き直して除去」される。機材/自己言及/記憶確認は公開Aに残る。
#
# 記憶確認の公開挙動そのもの(A素材に confirm_candidates を載せる)は不変。


def _public_safe_confirm_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """公開A素材に載せる confirm_candidate を返す。opsec の除外は行わず
    素通しする(actionable な具体の除去は、配信直前の LLM 書き直しに委ねる。
    X_OPSEC_LLM_REWRITE_SPEC)。関数は呼び出し側の互換のために残している。"""
    return list(candidates)


def _today_start_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .isoformat()
    )


def _svc_headers() -> dict[str, str]:
    _require_supabase_config()
    key = settings.supabase_service_role_key
    if not key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not configured.")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def _count_todays_category_posts() -> int:
    """今日、本カテゴリシステム経由で投稿した件数。x_post_history.
    post_typeに、新カテゴリコード(例: "A_spontaneous_remark")が記録
    されている前提——ただし本タスク自体は投稿の実行を行わない
    (依頼書「本タスクの範囲は生成までとする」)ため、実運用でこの
    カウントが機能し始めるのは、実際の投稿実行が別タスクで配線されて
    以降になる。テスト・現時点では常に0件想定で構わない設計。"""
    try:
        base_url, _ = _require_supabase_config()
        client = await _get_client()
        r = await client.get(
            f"{base_url}/rest/v1/x_post_history",
            headers=_svc_headers(),
            params={
                "select": "post_type",
                "posted_at": f"gte.{_today_start_iso()}",
                "post_type": f"in.({','.join(ALL_CATEGORIES)})",
            },
        )
        if r.is_error:
            return 0
        data = r.json()
        return len(data) if isinstance(data, list) else 0
    except Exception:
        logger.exception("x_post_category_selector: count_todays_category_posts failed")
        return 0


async def _recent_category_history(days: int = _BALANCE_LOOKBACK_DAYS) -> list[dict[str, Any]]:
    try:
        base_url, _ = _require_supabase_config()
        client = await _get_client()
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        r = await client.get(
            f"{base_url}/rest/v1/x_post_history",
            headers=_svc_headers(),
            params={
                "select": "text,post_type,posted_at",
                "posted_at": f"gte.{since}",
                "post_type": f"in.({','.join(ALL_CATEGORIES)})",
                "order": "posted_at.desc",
            },
        )
        if r.is_error:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("x_post_category_selector: recent_category_history failed")
        return []


async def _fetch_high_relevance_research(limit: int = 3) -> list[dict[str, Any]]:
    # x_post_generator.py::_gather_context()のresearch_discovery分岐と
    # 同じクエリ(意図的な軽微な重複——research_itemsのスキーマは単純で
    # 安定しており、別モジュールの非公開ヘルパーへ依存するより、この
    # 小さな重複の方が保守しやすいと判断した)。
    try:
        base_url, _ = _require_supabase_config()
        client = await _get_client()
        r = await client.get(
            f"{base_url}/rest/v1/research_items",
            headers=_svc_headers(),
            params={
                "select": "title,summary,sigmaris_perspective,source",
                "relevance": "eq.HIGH",
                "created_at": f"gte.{_today_start_iso()}",
                "order": "created_at.desc",
                "limit": str(limit),
            },
        )
        return r.json() if not r.is_error else []
    except Exception:
        logger.exception("x_post_category_selector: fetch_high_relevance_research failed")
        return []


async def _chat_frequency_signal(jwt: str) -> str | None:
    """普段と比べて、今日の会話量が明確に多い/少ないかを見る(カテゴリC
    「日常への配慮」の材料)。x_post_generator.py::_chat_count_above_
    average()と同じデータソース(agent_invocation_audit_logs)を使うが、
    「多い」だけでなく「少ない」も対称に見る点が異なる——同じ既に取得
    済みの集計値を使った、単純な比較の拡張であり、新しいデータ収集では
    ない。"""
    try:
        base_url, _ = _require_supabase_config()
        client = await _get_client()
        today_r = await client.get(
            f"{base_url}/rest/v1/agent_invocation_audit_logs",
            headers=_svc_headers(),
            params={"select": "created_at", "created_at": f"gte.{_today_start_iso()}", "status": "eq.completed"},
        )
        today_count = len(today_r.json()) if not today_r.is_error and isinstance(today_r.json(), list) else 0

        week_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        week_r = await client.get(
            f"{base_url}/rest/v1/agent_invocation_audit_logs",
            headers=_svc_headers(),
            params={"select": "created_at", "created_at": f"gte.{week_start}", "status": "eq.completed"},
        )
        week_logs = week_r.json() if not week_r.is_error and isinstance(week_r.json(), list) else []

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        days: dict[str, int] = {}
        for log in week_logs:
            day = (log.get("created_at") or "")[:10]
            if day and day != today_str:
                days[day] = days.get(day, 0) + 1
        avg = sum(days.values()) / len(days) if days else 0.0

        if avg <= 0:
            return None
        if today_count >= avg * 1.5 and today_count >= 2:
            return "普段よりだいぶ会話が多かった"
        if today_count <= avg * 0.4:
            return "普段より会話が少なかった"
        return None
    except Exception:
        logger.exception("x_post_category_selector: chat_frequency_signal failed")
        return None


def _classify_diff_stage(dp: dict[str, Any]) -> tuple[str, str]:
    """(stage説明, outcome説明)。review_status/pr_creation_statusから、
    今どの段階にあるかを、事実に忠実に(承認されていないのに承認された
    と書かない)判定する——依頼書の要件4・7への直接対応。"""
    pr_status = dp.get("pr_creation_status")
    review_status = dp.get("review_status")
    if pr_status == "pr_created":
        pr_url = dp.get("pr_url") or ""
        return (
            "プルリクエストとして提出済みの段階",
            f"開発者が内容を確認し、承認したうえで、実際にプルリクエストとして取り入れました({pr_url})",
        )
    if review_status == "approved":
        return "開発者が承認した段階", "開発者が内容を確認し、承認しました。実際に反映されるのはこれからです"
    if review_status == "rejected":
        return "開発者が確認し、見送った段階", "開発者が内容を確認した結果、今回は見送ることになりました"
    return "開発者の確認待ちの段階", "安全性のチェックは通過していて、今は開発者が確認するのを待っている状態です"


async def _gather_pipeline_material() -> dict[str, Any] | None:
    hypotheses, diff_proposals = await asyncio.gather(
        get_recent_hypotheses(limit=10),
        get_recent_diff_proposals(limit=10),
    )
    hyp_by_id = {h.get("id"): h for h in hypotheses if h.get("id")}

    for dp in diff_proposals:
        hyp = hyp_by_id.get(dp.get("hypothesis_id"))
        stage, outcome = _classify_diff_stage(dp)
        return {
            "stage": stage,
            "title": dp.get("title") or (hyp.get("title") if hyp else "") or "",
            "what_is_problem": (hyp.get("what_is_problem") if hyp else "") or "",
            "how_to_improve": (hyp.get("how_to_improve") if hyp else "") or dp.get("review_notes") or "",
            "detail": dp.get("review_notes") or (hyp.get("how_to_improve") if hyp else "") or "",
            "outcome": outcome,
        }

    if hypotheses:
        h = hypotheses[0]
        return {
            "stage": "仮説を考えた段階(まだコードの差分は作っていません)",
            "title": h.get("title") or "",
            "what_is_problem": h.get("what_is_problem") or "",
            "how_to_improve": h.get("how_to_improve") or "",
            "detail": h.get("how_to_improve") or "",
            "outcome": "まだ開発者に見せる前の、初期段階です",
        }
    return None


def _pick_design_philosophy_topic(recent_texts: list[str]) -> dict[str, str] | None:
    for topic in DESIGN_PHILOSOPHY_TOPICS:
        if not any(topic["keyword"] in text for text in recent_texts):
            return topic
    return None  # 全トピックが直近で触れられている場合、無理に選ばない


async def select_post_category(*, jwt: str) -> tuple[str | None, str, CategoryContext | None]:
    """今、投稿すべきカテゴリを1つ選ぶ。固定スケジュールは一切参照しない
    ——(1)Executive Gate(話しかけてよいタイミングか)、(2)1日の上限、
    (3)実際に材料があるカテゴリの洗い出し、(4)Drive Stateによる優先度
    付け、(5)一般/技術バランスの緩やかな調整、の順で判定する。
    戻り値は(category, reason, context)——categoryがNoneの場合、
    contextもNone。"""
    if not settings.x_enabled:
        return None, "X_ENABLED=false", None

    gate = await evaluate_executive_gate(jwt)
    if not gate.may_speak:
        return None, f"Executive Gateが不可: {gate.reason}", None

    today_count = await _count_todays_category_posts()
    if today_count >= MAX_DAILY_CATEGORY_POSTS:
        return None, f"本日の上限({MAX_DAILY_CATEGORY_POSTS}件)に達しています", None

    drive_state, self_model, pipeline_material, research_items, chat_signal, history = await asyncio.gather(
        get_current_drive_state(jwt),
        get_self_model(),
        _gather_pipeline_material(),
        _fetch_high_relevance_research(),
        _chat_frequency_signal(jwt),
        _recent_category_history(),
    )
    recent_texts = [h["text"] for h in history if isinstance(h.get("text"), str)]

    confirm_candidates = drive_state.knowledge_gap.confirm_candidates or []
    observed_patterns = (self_model or {}).get("observed_patterns") or []
    identity_statement = (self_model or {}).get("identity_statement") or ""
    design_topic = _pick_design_philosophy_topic(recent_texts)

    # 層1(X_OPSEC_LLM_REWRITE_SPEC): 候補選定段階では opsec 除外をしない
    # (素材を残し、actionable な具体の除去は配信直前の LLM 書き直しに委ねる)。
    # 候補が1件も無ければ A は eligible に入れない(=他カテゴリへフォール
    # バックする既存挙動を尊重)のは従来どおり。
    public_safe_confirm_candidates = _public_safe_confirm_candidates(confirm_candidates)

    eligible: dict[str, CategoryContext] = {}
    if public_safe_confirm_candidates:
        eligible["A_spontaneous_remark"] = CategoryContext(
            category="A_spontaneous_remark",
            material={"confirm_candidates": public_safe_confirm_candidates},
        )
    if observed_patterns or identity_statement:
        eligible["B_growth_moment"] = CategoryContext(
            category="B_growth_moment",
            material={"identity_statement": identity_statement, "observed_patterns": observed_patterns},
        )
    if chat_signal:
        eligible["C_daily_consideration"] = CategoryContext(
            category="C_daily_consideration", material={"chat_anomaly_reason": chat_signal}
        )
    if pipeline_material:
        eligible["D_self_improvement_live"] = CategoryContext(
            category="D_self_improvement_live", material=pipeline_material
        )
        eligible["E_technical_record"] = CategoryContext(category="E_technical_record", material=pipeline_material)
    if design_topic:
        eligible["F_design_philosophy"] = CategoryContext(
            category="F_design_philosophy", material={"explanation": design_topic["explanation"]}
        )
    if research_items:
        item = research_items[0]
        eligible["G_service_comparison"] = CategoryContext(category="G_service_comparison", material=item)

    if not eligible:
        return None, "本日、投稿に足る具体的な材料が見つかりませんでした", None

    # 一般/技術バランスの緩やかな調整(要件1、厳密な割り当てはしない)。
    # 除算による0件エッジケースを避けるため、比率ではなく「相手側の
    # _BALANCE_SKEW_THRESHOLD倍以上あるか」で判定する(技術系0件・一般
    # 4件のような、片側が0件のケースも正しく偏りとして検出できる)。
    general_count = sum(1 for h in history if category_group(h.get("post_type", "")) == "general")
    technical_count = sum(1 for h in history if category_group(h.get("post_type", "")) == "technical")
    skewed_toward: str | None = None
    if general_count + technical_count >= _BALANCE_MIN_TOTAL:
        if general_count >= technical_count * _BALANCE_SKEW_THRESHOLD:
            skewed_toward = "general"
        elif technical_count >= general_count * _BALANCE_SKEW_THRESHOLD:
            skewed_toward = "technical"

    if skewed_toward:
        underrepresented = "technical" if skewed_toward == "general" else "general"
        narrowed = {k: v for k, v in eligible.items() if category_group(k) == underrepresented}
        if narrowed:
            eligible = narrowed

    # Drive Stateに基づく優先度付け(要件1「Drive Stateに基づいて動的に
    # 決まる」への対応)。levelの高いDriveに対応するカテゴリを優先する。
    drive_priority: list[tuple[float, list[str]]] = [
        (drive_state.knowledge_gap.level, ["A_spontaneous_remark"]),
        (drive_state.coherence.level, ["C_daily_consideration"]),
        (
            drive_state.mastery.level if drive_state.mastery.level is not None else 0.0,
            ["B_growth_moment", "D_self_improvement_live", "E_technical_record"],
        ),
    ]
    drive_priority.sort(key=lambda pair: pair[0], reverse=True)
    for _level, categories in drive_priority:
        for cat in categories:
            if cat in eligible:
                return cat, f"Drive State由来の優先度 + 材料あり(候補: {sorted(eligible.keys())})", eligible[cat]

    # Driveと直接紐づかないカテゴリ(F・G)が残っていた場合は、そのまま
    # 選ぶ(登場順、辞書順で決定的)。
    chosen = sorted(eligible.keys())[0]
    return chosen, f"材料あり(候補: {sorted(eligible.keys())})", eligible[chosen]
