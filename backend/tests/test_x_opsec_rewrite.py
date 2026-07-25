from __future__ import annotations

# X_OPSEC_LLM_REWRITE_SPEC / X_OPSEC_MODEL_WIRING_FIX_SPEC:
# rewrite_for_opsec() の構造化出力・差し込み・安全側フォールバック、および
# opsec 判定モデルの配線(override が効く/既定が nano でない)の回帰。
# LLMRouter をダミー注入し、ネットワーク/LLM に依存しない(決定的)。

import asyncio
import json
from unittest.mock import patch

from app.config import settings
from app.services import x_opsec_rewrite as mod
from app.services.local_llm import TaskType, _openai_model_for_task
from app.services.x_opsec_rewrite import OpsecResult, rewrite_for_opsec


def _run(coro):
    return asyncio.run(coro)


class _DummyRouter:
    """LLMRouter 互換のダミー。chat() が返す raw を固定(または例外)し、
    渡された task/override_model を記録する。resolve_model_name() は本物と
    同じ規則(override 有 → openai:<model>、無 → openai:<tier>)で返す。"""

    def __init__(self, *, raw: str | None = None, raises: bool = False):
        self._raw = raw
        self._raises = raises
        self.calls: list[dict] = []

    async def chat(self, task, messages, **kwargs):
        self.calls.append({"task": task, "messages": messages, "kwargs": kwargs})
        if self._raises:
            raise RuntimeError("boom")
        return self._raw

    async def resolve_model_name(self, task, *, override_model=None):
        if override_model:
            return f"openai:{override_model}"
        return f"openai:{_openai_model_for_task(task)}"


def _with_router(router):
    return patch.object(mod, "get_llm_router", lambda: router)


# ─── 通る(changed=false):機材・自己言及はそのまま ──────────────────────
def test_machine_level_passes_through_unchanged():
    text = "GTX1660の自宅Ubuntuサーバーで動いてるらしい、合ってる?"
    raw = json.dumps({"safe_text": text, "changed": False, "removed": []})
    router = _DummyRouter(raw=raw)
    with _with_router(router):
        result = _run(rewrite_for_opsec(text))
    assert isinstance(result, OpsecResult)
    assert result.safe_text == text
    assert result.changed is False
    assert result.removed == []
    # json_mode で呼ばれていること(構造化出力の要請)。
    assert router.calls[0]["kwargs"].get("json_mode") is True


# ─── 書き直される(changed=true):具体が消え文意は自然に残る ────────────
def test_actionable_is_rewritten_out():
    text = "外部公開のためポート12345を開ける"
    safe = "外部公開の準備をしている"
    raw = json.dumps({"safe_text": safe, "changed": True, "removed": ["ポート番号/外部公開の手順"]})
    router = _DummyRouter(raw=raw)
    with _with_router(router):
        result = _run(rewrite_for_opsec(text))
    assert result.safe_text == safe
    assert result.changed is True
    assert result.removed == ["ポート番号/外部公開の手順"]


# ─── 不整合(changed=false 主張だが本文が変わっている)→ changed=True に寄せる ──
def test_inconsistent_changed_false_but_text_differs_is_forced_true():
    text = "IPは192.168.0.11"
    safe = "自宅のサーバーに繋いでる"
    raw = json.dumps({"safe_text": safe, "changed": False, "removed": []})
    with _with_router(_DummyRouter(raw=raw)):
        result = _run(rewrite_for_opsec(text))
    assert result.safe_text == safe
    assert result.changed is True  # 本文が変わっているので actionable 除去とみなす


# ─── パース失敗 → 安全側(保留: safe_text 空・changed=True) ───────────────
def test_unparseable_output_fails_safe():
    router = _DummyRouter(raw="これは JSON ではありません")
    with _with_router(router):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert result.safe_text == ""
    assert result.changed is True
    assert result.removed  # 理由が入っている


def test_non_object_json_fails_safe():
    router = _DummyRouter(raw=json.dumps(["not", "an", "object"]))
    with _with_router(router):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert result.safe_text == ""
    assert result.changed is True


def test_wrong_types_fail_safe():
    # safe_text が文字列でない等、型が想定外なら安全側。
    router = _DummyRouter(raw=json.dumps({"safe_text": 123, "changed": "yes", "removed": []}))
    with _with_router(router):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert result.safe_text == ""
    assert result.changed is True


# ─── LLM 呼び出し例外 → 安全側(保留) ────────────────────────────────────
def test_llm_exception_fails_safe():
    router = _DummyRouter(raises=True)
    with _with_router(router):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert result.safe_text == ""
    assert result.changed is True


# ─── 空入力は LLM を呼ばずそのまま通す ──────────────────────────────────
def test_empty_input_short_circuits():
    router = _DummyRouter(raw="should-not-be-used")
    with _with_router(router):
        result = _run(rewrite_for_opsec("   "))
    assert result.changed is False
    assert result.safe_text == "   "
    assert router.calls == []  # LLM は呼ばれない


# ─── removed が非リストでも文字列化して受ける(頑健性) ──────────────────
def test_removed_non_list_is_coerced():
    raw = json.dumps({"safe_text": "x", "changed": True, "removed": "ポート番号"})
    with _with_router(_DummyRouter(raw=raw)):
        result = _run(rewrite_for_opsec("ポート12345"))
    assert result.removed == ["ポート番号"]


# ─── X_OPSEC_MODEL_WIRING_FIX_SPEC: モデル配線 ──────────────────────────

def _ok_raw(text="x"):
    return json.dumps({"safe_text": text, "changed": False, "removed": []})


def test_model_arg_override_reaches_router():
    # rewrite_for_opsec(model=) が override_model として router.chat に渡り、
    # X_OPSEC_REWRITE タスクで呼ばれること(nano の ROUTING ではない)。
    router = _DummyRouter(raw=_ok_raw())
    with _with_router(router):
        result = _run(rewrite_for_opsec("何かの投稿", model="gpt-5.5"))
    call = router.calls[0]
    assert call["task"] is TaskType.X_OPSEC_REWRITE
    assert call["kwargs"].get("override_model") == "gpt-5.5"
    assert result.model == "openai:gpt-5.5"  # 可観測性: 実使用モデル名


def test_env_override_reaches_router():
    # settings.x_opsec_llm_model(= X_OPSEC_LLM_MODEL)が override として効く。
    router = _DummyRouter(raw=_ok_raw())
    with _with_router(router), patch.object(settings, "x_opsec_llm_model", "gpt-5.5"):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert router.calls[0]["kwargs"].get("override_model") == "gpt-5.5"
    assert result.model == "openai:gpt-5.5"


def test_explicit_arg_beats_env():
    router = _DummyRouter(raw=_ok_raw())
    with _with_router(router), patch.object(settings, "x_opsec_llm_model", "from-env"):
        _run(rewrite_for_opsec("何かの投稿", model="from-arg"))
    assert router.calls[0]["kwargs"].get("override_model") == "from-arg"


def test_default_is_not_nano_tier():
    # 未指定時: override なし & 既定ティアが nano(最安)ではなく生成モデル相当。
    router = _DummyRouter(raw=_ok_raw())
    with _with_router(router), patch.object(settings, "x_opsec_llm_model", None):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert router.calls[0]["task"] is TaskType.X_OPSEC_REWRITE
    assert router.calls[0]["kwargs"].get("override_model") is None
    # 既定は openai_model(生成相当)であって openai_nano_model ではない。
    default_tier = _openai_model_for_task(TaskType.X_OPSEC_REWRITE)
    assert default_tier == settings.openai_model
    assert default_tier != settings.openai_nano_model
    assert result.model == f"openai:{settings.openai_model}"


def test_fallback_carries_model_name():
    # フォールバック(保留)でも実使用予定モデル名が結果に載る(shadow ログ用)。
    router = _DummyRouter(raises=True)
    with _with_router(router), patch.object(settings, "x_opsec_llm_model", "gpt-5.5"):
        result = _run(rewrite_for_opsec("何かの投稿"))
    assert result.safe_text == ""  # フォールバック不変
    assert result.changed is True
    assert result.model == "openai:gpt-5.5"
