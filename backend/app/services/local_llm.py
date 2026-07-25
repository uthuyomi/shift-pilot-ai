from __future__ import annotations

import enum
import logging
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class TaskType(str, enum.Enum):
    ROUTING = "routing"
    MEMORY_EXTRACTION = "memory_extraction"
    SELF_REFLECT = "self_reflect"
    SUMMARIZE = "summarize"
    COMPLEX_REASONING = "complex_reasoning"
    DECISION_DETECTION = "decision_detection"
    EPISODE_DETECTION = "episode_detection"
    TOPIC_DETECTION = "topic_detection"
    QUERY_DECOMPOSITION = "query_decomposition"
    MEMORY_RERANK = "memory_rerank"
    ABSTENTION_REACTION_DETECTION = "abstention_reaction_detection"
    EVAL_GENERATION = "eval_generation"
    # chat_routing.py::classify_chat_intent() — a dedicated type rather than
    # reusing TaskType.ROUTING (see docs/sigmaris/
    # incident_response_latency_investigation.md 11.1 for the rationale):
    # this call is a hot-path, per-turn classification that was the subject
    # of its own multi-section latency investigation, so it gets the same
    # "one TaskType per distinct classification concern" treatment already
    # given to DECISION_DETECTION/EPISODE_DETECTION/TOPIC_DETECTION/etc.,
    # rather than folding into ROUTING's existing grab-bag of unrelated
    # one-off callers (active_inquiry.py, memory_validator.py, x_*.py).
    CHAT_INTENT_CLASSIFICATION = "chat_intent_classification"
    # Phase C-full: LLM-as-a-Judge for LongMemEval/LoCoMo public-benchmark
    # answer grading (see docs/sigmaris/phase_c_full_report.md section 3).
    # Deliberately NOT the same TaskType as EVAL_GENERATION (Phase C-mini's
    # internal-testset question generation, and this phase's own benchmark
    # answer synthesis — see bench_pipeline.py): the judge's correctness
    # verdict *is* the reported score, so a bad judgment silently invalidates
    # the whole benchmark run in a way a bad generated question or answer
    # does not (those just produce one wrong data point, still visible in
    # the per-question detail). That asymmetry — "this call's own quality is
    # the deliverable" — is the same reasoning goal_alignment.py's nuanced
    # decision-vs-goal comparison used to justify TaskType.SELF_REFLECT
    # (the advanced tier) over a cheaper classifier tier.
    EVAL_JUDGE = "eval_judge"
    # Phase G-2(docs/sigmaris/phase_g_report.md): condenses a web-search
    # result (already fetched via a direct client.responses.create() call
    # with the web_search tool — see evidence_search.py) into structured
    # claim/source_url/source_title entries. A dedicated TaskType rather
    # than reusing SUMMARIZE: SUMMARIZE condenses a single known-good text
    # into prose, while this task must also preserve a strict per-claim
    # mapping back to the citation it came from and enforce a JSON output
    # shape — a distinct enough contract to warrant its own type, per the
    # "one TaskType per distinct classification concern" precedent
    # CHAT_INTENT_CLASSIFICATION already established above.
    EVIDENCE_STRUCTURING = "evidence_structuring"
    # Phase G-3(docs/sigmaris/phase_g_report.md): verifies whether an
    # already-generated response contradicts G-2's structured Evidence, and
    # — if so — rewrites only the affected portion into hedged phrasing.
    # One TaskType covers both the critique (verdict classification) and
    # the conditional rewrite step, rather than splitting into two types:
    # unlike CHAT_INTENT_CLASSIFICATION/EVIDENCE_STRUCTURING (each a
    # standalone call with its own independent callers), critique and
    # rewrite here are two tightly-coupled steps of one feature that only
    # ever fire together — see self_critique.py's module docstring for the
    # full rationale.
    SELF_CRITIQUE = "self_critique"
    # Phase G-4(docs/sigmaris/phase_g_report.md): the second, finer-grained
    # audit layer -- for each of G-2's Evidence claims, whether the
    # response's *use* of that claim (direct quote/paraphrase/indirect
    # reflection/not used) faithfully represents what the claim actually
    # says, rather than exaggerating or misconstruing it. A distinct
    # TaskType from SELF_CRITIQUE (not a shared one, unlike SELF_CRITIQUE's
    # own critique+rewrite pairing): this check is independent of and runs
    # alongside G-3's whole-response contradiction check rather than being
    # a tightly-coupled second step of the same call, so it gets its own
    # type per the "one TaskType per distinct classification concern"
    # precedent (CHAT_INTENT_CLASSIFICATION/EVIDENCE_STRUCTURING) rather
    # than SELF_CRITIQUE's "tightly-coupled steps share one type" precedent.
    # Still nano-tier -- no new model hierarchy, per the task's constraint.
    CITATION_AUDIT = "citation_audit"
    # Phase D-2(docs/sigmaris/phase_d_report.md): generates one concrete
    # improvement hypothesis (what/why/how-direction + expected metric
    # impact) from a single prioritized Phase D-1 EvidenceItem. Deliberately
    # advanced tier, NOT nano like G-1~G-4's classification-shaped calls --
    # this is genuine architectural reasoning about the Sigmaris codebase
    # itself (what change, and why it would help), matching openai_advanced_
    # model's existing "自己反省・設計・週次レビュー" role (config.py) rather
    # than the nano tier's "記憶抽出・要約・分類" role. Also deliberately NOT
    # reusing COMPLEX_REASONING/SELF_REFLECT: this call's output shape
    # (structured JSON with a self-declared safety-mechanism flag feeding
    # Constitution review) and its cost profile (offline CLI, not a hot
    # conversational path) are distinct enough to warrant its own type, per
    # the "one TaskType per distinct classification concern" precedent.
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    # Phase D-2: an independent "critic" check (same shape as G-3's
    # SELF_CRITIQUE) of whether a single generated hypothesis's stated
    # problem/reasoning actually follows from the EvidenceItem it claims to
    # be based on -- not a full re-derivation of the hypothesis, so nano-tier
    # like G-3's own critique step.
    HYPOTHESIS_CRITIQUE = "hypothesis_critique"
    # Phase F-1(docs/sigmaris/phase_f_report.md): generates a unified-diff
    # code proposal for a single target file, from one D-3-prioritized/
    # E-1-verified hypothesis. Deliberately reuses D-2's HYPOTHESIS_
    # GENERATION model TIER (advanced) rather than introducing a new tier --
    # the task's explicit constraint ("D-2の仮説生成に使ったモデル階層を、
    # そのまま踏襲すること。新しいモデル階層は、追加しないこと"). Still gets
    # its own TaskType value (not literally HYPOTHESIS_GENERATION) per the
    # "one TaskType per distinct classification concern" precedent: this
    # call's input (hypothesis + actual current file source) and output
    # shape (a unified diff, not a JSON verdict) are a different contract
    # from hypothesis generation itself. NOT local-eligible, same reasoning
    # as HYPOTHESIS_GENERATION -- an offline CLI step where correctness
    # matters far more than latency.
    CODE_DIFF_GENERATION = "code_diff_generation"
    # Phase H-2(docs/sigmaris/phase_h_report.md): 開発者(@Oyasu1999)以外
    # からの、シグマリスの投稿への返信1件について、「対話意図があるか」
    # 「システムプロンプト等を探ろうとする、指示上書きの試みでないか」
    # 「スパム・荒らし・攻撃的な内容でないか」の3点を、1回のLLM呼び出しで
    # まとめて判定する。CHAT_INTENT_CLASSIFICATION・CITATION_AUDIT等と
    # 同じ、nano-tierの「分類・軽量判定」の性質を持つ、独立した判定対象
    # (入力=X返信テキスト、出力=3種のbool+理由のJSON)であるため、
    # 「one TaskType per distinct classification concern」の前例に従い、
    # 専用のTaskTypeとした。既存のCHAT_INTENT_CLASSIFICATION等を転用
    # しなかった判断根拠: 入力(ユーザーの発話ではなく、外部の第三者の
    # X投稿テキスト)・出力形状(3種の判定を同時に返す)のいずれも異なる、
    # 別の契約であるため。
    X_REPLY_FILTER = "x_reply_filter"
    # Self-2(docs/sigmaris/self_awareness_report.md): Self-1が洗い出した
    # 能力候補(1グループ=関連ファイル群)を、シグマリスが一人称で語る
    # 2〜3文の日本語説明へ要約する。入力(ファイルパス・関数名・配線状態
    # の一覧)・出力形状(短い一人称の説明文、JSONではなくプレーンテキスト)
    # のいずれも、既存のTaskType.SUMMARIZE(research_agent.py、Web検索結果
    # 1件の要約)とは異なる契約であるため、「one TaskType per distinct
    # classification concern」の前例に従い専用の型とした。依頼書「既存の
    # nano-tierの軽量なLLM呼び出しを使うこと。新しいモデル階層は追加しない
    # こと」に対応し、SUMMARIZEと同じnano tierをそのまま踏襲する(下記
    # _LOCAL_TASK_TYPES・_openai_model_for_task両方に、SUMMARIZEと同じ扱いで
    # 追加)。
    CAPABILITY_SUMMARIZATION = "capability_summarization"
    # X_OPSEC_MODEL_WIRING_FIX_SPEC: 配信直前の opsec 判定＋書き直し
    # (x_opsec_rewrite.rewrite_for_opsec())専用のタスク。これは "安いルー
    # ティング判断" ではなく "安全判断" なので、ROUTING(最安の nano ティア)
    # には乗せない。_openai_model_for_task() で生成モデル相当(openai_model)へ
    # 解決し、_LOCAL_TASK_TYPES にも入れない(安全判断を Ollama 任せにせず
    # OpenAI 固定)。X_OPSEC_LLM_MODEL / rewrite_for_opsec(model=) の明示上書き
    # は、LLMRouter.chat(override_model=) 経由でこの既定より優先される。
    X_OPSEC_REWRITE = "x_opsec_rewrite"


_LOCAL_TASK_TYPES = {
    TaskType.ROUTING,
    TaskType.MEMORY_EXTRACTION,
    TaskType.SELF_REFLECT,
    TaskType.SUMMARIZE,
    TaskType.DECISION_DETECTION,
    TaskType.EPISODE_DETECTION,
    TaskType.TOPIC_DETECTION,
    TaskType.QUERY_DECOMPOSITION,
    TaskType.MEMORY_RERANK,
    TaskType.ABSTENTION_REACTION_DETECTION,
    TaskType.EVAL_GENERATION,
    TaskType.CHAT_INTENT_CLASSIFICATION,
    TaskType.EVIDENCE_STRUCTURING,
    TaskType.SELF_CRITIQUE,
    TaskType.CITATION_AUDIT,
    TaskType.HYPOTHESIS_CRITIQUE,
    TaskType.X_REPLY_FILTER,
    TaskType.CAPABILITY_SUMMARIZATION,
}


class LocalLLMClient:
    """Thin async client for Ollama /api/chat with persistent connection pool."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        r = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    async def is_available(self) -> bool:
        """Whether Ollama is not just reachable, but actually has the
        configured chat model (self._model) installed.

        A bare "does /api/tags respond" check used to be enough to mark
        Ollama "available" and route chat-eligible tasks to it — but a
        server can have Ollama running with only an embedding model
        installed (e.g. nomic-embed-text) and no chat model at all. That
        looked "available" here, so every local-eligible TaskType got
        routed to Ollama and then hit a 404 on POST /api/chat, with no
        fallback (LLMRouter.chat() picks a backend once via this check and
        does not retry on failure). Checking the model list directly here
        means the OpenAI fallback triggers up front instead.
        """
        try:
            r = await self._client.get(f"{self._base_url}/api/tags")
            if not r.is_success:
                return False
            data = r.json()
            models = data.get("models", [])
            installed = {
                str(entry.get(field))
                for entry in models
                if isinstance(entry, dict)
                for field in ("name", "model")
                if entry.get(field)
            }
            if self._model in installed:
                return True
            # Fall back to a base-name match (ignoring the ":tag" suffix) in
            # case the configured model omits a tag that Ollama fills in
            # (e.g. ":latest") or vice versa.
            base_name = self._model.split(":", 1)[0]
            return any(name.split(":", 1)[0] == base_name for name in installed)
        except Exception:
            return False


def _openai_model_for_task(task: TaskType) -> str:
    """Map TaskType to the appropriate OpenAI model tier."""
    if task in {
        TaskType.SELF_REFLECT,
        TaskType.COMPLEX_REASONING,
        TaskType.HYPOTHESIS_GENERATION,
        TaskType.CODE_DIFF_GENERATION,
    }:
        return settings.sigmaris_reflect_model or settings.openai_advanced_model
    if task is TaskType.EVAL_JUDGE:
        # Same "sigmaris_reflect_model or openai_advanced_model" pattern as
        # SELF_REFLECT/COMPLEX_REASONING above, with its own override knob
        # (eval_judge_model) rather than sharing sigmaris_reflect_model, so
        # the operator can retune judge cost/quality independently of
        # self-reflection — relevant once a full-scale run (500+ LongMemEval
        # + ~2000 LoCoMo questions) makes the judge's per-call cost add up.
        return settings.eval_judge_model or settings.openai_advanced_model
    if task is TaskType.EVAL_GENERATION:
        # Pinned to gpt-5.4-mini per an explicit operator decision (Phase
        # C-mini testset generation) — deliberately kept separate from
        # openai_nano_model so it doesn't drift if that tier's model choice
        # changes for unrelated reasons.
        return settings.eval_generation_model
    if task in {
        TaskType.ROUTING,
        TaskType.MEMORY_EXTRACTION,
        TaskType.SUMMARIZE,
        TaskType.DECISION_DETECTION,
        TaskType.EPISODE_DETECTION,
        TaskType.TOPIC_DETECTION,
        TaskType.QUERY_DECOMPOSITION,
        TaskType.MEMORY_RERANK,
        TaskType.ABSTENTION_REACTION_DETECTION,
        TaskType.CHAT_INTENT_CLASSIFICATION,
        TaskType.EVIDENCE_STRUCTURING,
        TaskType.SELF_CRITIQUE,
        TaskType.CITATION_AUDIT,
        TaskType.HYPOTHESIS_CRITIQUE,
        TaskType.X_REPLY_FILTER,
        TaskType.CAPABILITY_SUMMARIZATION,
    }:
        return settings.openai_nano_model
    if task is TaskType.X_OPSEC_REWRITE:
        # X_OPSEC_MODEL_WIRING_FIX_SPEC: opsec の安全判断は nano(最安)に
        # 乗せず、生成モデル相当(openai_model)を既定にする。X_OPSEC_LLM_MODEL
        # で明示上書き可(override_model 経由でこれより優先される)。
        return settings.openai_model
    return settings.openai_model


class _OpenAIAdapter:
    """Wraps AsyncOpenAI to expose the same chat() interface as LocalLLMClient."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
        task: TaskType | None = None,
        override_model: str | None = None,
    ) -> str:
        # override_model(明示指定)を最優先。無ければ task のティア、さらに
        # 無ければ通常会話モデル。X_OPSEC_MODEL_WIRING_FIX_SPEC: これにより
        # X_OPSEC_LLM_MODEL / rewrite_for_opsec(model=) が実際に効く。
        model = override_model or (
            _openai_model_for_task(task) if task is not None else settings.openai_model
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class LLMRouter:
    """
    Routes task types to LocalLLMClient or OpenAI.

    Uses local LLM when LOCAL_LLM_ENABLED=true AND Ollama is reachable
    AND the task type is in the local-eligible set. Falls back to OpenAI
    automatically if Ollama is unavailable.
    """

    def __init__(self) -> None:
        self._local: LocalLLMClient | None = None
        self._openai = _OpenAIAdapter()
        self._local_available: bool | None = None  # None = not yet probed

        if settings.local_llm_enabled:
            self._local = LocalLLMClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )

    async def _get_backend(self, task: TaskType) -> LocalLLMClient | _OpenAIAdapter:
        if not settings.local_llm_enabled or self._local is None:
            return self._openai

        if task not in _LOCAL_TASK_TYPES:
            return self._openai

        # Probe availability once per router lifetime (lazy).
        if self._local_available is None:
            self._local_available = await self._local.is_available()
            if not self._local_available:
                logger.warning(
                    "Ollama not reachable at %s — falling back to OpenAI for all local tasks.",
                    settings.ollama_base_url,
                )

        return self._local if self._local_available else self._openai

    async def chat(
        self,
        task: TaskType,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
        override_model: str | None = None,
    ) -> str:
        # override_model が指定された場合は、そのモデル名(OpenAI モデルの
        # 想定)で必ず OpenAI 経路を使う。Ollama にローカルモデル名でない値を
        # 渡しても解釈できないため、明示上書きは OpenAI 固定にする
        # (X_OPSEC_MODEL_WIRING_FIX_SPEC)。
        backend = self._openai if override_model else await self._get_backend(task)
        backend_name = "local" if isinstance(backend, LocalLLMClient) else "openai"
        logger.debug(
            "LLMRouter task=%s backend=%s override_model=%s",
            task.value, backend_name, override_model,
        )
        if isinstance(backend, _OpenAIAdapter):
            return await backend.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                task=task,
                override_model=override_model,
            )
        return await backend.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    async def resolve_model_name(
        self, task: TaskType, *, override_model: str | None = None
    ) -> str:
        """chat() が実際に使うモデル/バックエンドの表示名を返す(可観測性用)。

        override_model があれば必ず OpenAI 経路(そのモデル名)。無ければ
        task の解決先——ローカル eligible かつ Ollama 到達可能なら Ollama、
        そうでなければ OpenAI のティア。chat() と同じ判定順序をなぞる。
        """
        if override_model:
            return f"openai:{override_model}"
        backend = await self._get_backend(task)
        if isinstance(backend, LocalLLMClient):
            return f"ollama:{settings.ollama_model}"
        return f"openai:{_openai_model_for_task(task)}"


# Module-level singleton — created lazily to avoid import-time side effects.
_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
