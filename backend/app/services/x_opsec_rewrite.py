# 役割: X_OPSEC_LLM_REWRITE_SPEC — opsec フィルタを「単語/正規表現マッチ」
# から「LLM 一回の判定＋書き直し」へ一本化する。
#
# 【設計の意図】旧 opsec(x_privacy_filter.py の正規表現 + x_post_category_
# selector.py の広いインフラ語除外)は、機材語(server/ubuntu/gpu/gtx/
# router/sim/tailscale/port/ip/外部公開…)を含む候補を広く弾き、実測で
# 「投稿なし」の目詰まりを起こしていた(X_OPSEC_LAYER1_REFINE_SPEC の
# 「8件全除外」記録)。フィルタの重ねがけが原因だったため、本モジュールは
# ブロックではなく「攻撃に直接使える actionable な具体だけを、その部分だけ
# 自然に書き直して除去」する——「投稿なし」でなく「安全な投稿に直す」。
#
# 【守るべき相手】守るのは「OpenAI」ではなく「公開(Twitter)に actionable な
# 具体が出ること」。生成時点で既に同モデルへ候補内容は渡っているため、この
# チェックで新たな外部漏洩は発生しない。安全網は shadow mode + Executive
# Gate(既存)。正規表現のバックストップは、意図的に置かない(目詰まりの
# 再来を避けるため。判断根拠は X_OPSEC_LLM_REWRITE_SPEC)。
#
# 【使用モデル(X_OPSEC_MODEL_WIRING_FIX_SPEC)】opsec は "安いルーティング
# 判断" ではなく "安全判断" なので、最安の nano ティア(TaskType.ROUTING)には
# 乗せない。専用の TaskType.X_OPSEC_REWRITE で LLMRouter を通し、既定は生成
# モデル相当(settings.openai_model、現状 gpt-5.4-mini)へ解決する
# (_LOCAL_TASK_TYPES に含めないため OpenAI 固定=安全判断を Ollama 任せに
# しない)。X_OPSEC_LLM_MODEL(または rewrite_for_opsec(model=))を指定すると、
# その値が override_model として router.chat に渡り、既定より優先される
# (=実際に効く。将来ローカル LLM へ寄せる差し込み口も同じ経路)。実際に
# 使ったモデル名は INFO ログと OpsecResult.model に載る。

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.services.local_llm import TaskType, get_llm_router

logger = logging.getLogger(__name__)


@dataclass
class OpsecResult:
    """rewrite_for_opsec() の構造化結果。

    safe_text : 投稿してよい書き直し後テキスト(actionable な具体だけ除去)。
    changed   : 元テキストから何か書き直したか(False=素通しでよい)。
    removed   : 除去した actionable 情報の説明(ログ・検証用)。
    model     : 実際に opsec 判定へ使ったモデル/バックエンドの表示名(可観測性
                用、例 "openai:gpt-5.4-mini")。フォールバック挙動には影響しない
                追加フィールド(X_OPSEC_MODEL_WIRING_FIX_SPEC)。
    """

    safe_text: str
    changed: bool
    removed: list[str] = field(default_factory=list)
    model: str = ""


# ─── LLM プロンプト ───────────────────────────────────────────────────────
# actionable な具体だけを書き直しで除去し、機材・OS・存在レベルの一般言及と
# 開発者宛ての記憶確認・自己言及は変更せず保持させる。出力は構造化 JSON。
_SYSTEM_PROMPT = (
    "あなたは、AI が公開 SNS(X/Twitter)へ投稿する直前の、opsec(運用上の"
    "秘密保持)チェック担当です。次の投稿文から、攻撃に直接使える"
    "actionable な情報だけを検出し、その部分だけを自然に書き直して除去して"
    "ください。文意はできるだけ自然に残し、投稿そのものはブロックしません。\n"
    "\n"
    "【書き直して除去する(actionable)】\n"
    "- IP アドレス(例 192.168.0.11)、CGNAT 帯(100.64.0.0/10)\n"
    "- ポート番号・ポート開放(例 ポート12345を開ける、:8000)\n"
    "- 外部公開の具体的な手段・手順・タイミング\n"
    "- 回線の具体設定(DDNS・固定/グローバル IP・SSID・VPN エンドポイント/鍵・"
    "Tailscale のノード名/IP)\n"
    "- 内部ホスト名(.local)\n"
    "- 認証情報(パスワード・トークン・API キー)\n"
    "- メールアドレス・電話番号\n"
    "\n"
    "【変更せず保持する(actionable ではない、公開してよい)】\n"
    "- 機材・OS・存在レベルの一般言及(GPU 型番、自宅サーバーで動いている、"
    "OS は Ubuntu、AI サーバーとして運用している 等)\n"
    "- 開発者宛ての記憶確認・自己言及(「〜で動いてるらしい、合ってる?」等)\n"
    "\n"
    "判断が文脈依存の場合(例「外部公開のため SIM ルータ前提」)は、手段・"
    "手順に当たる部分だけを書き直し、単なる存在の言及(「SIM ルータ使ってる」)"
    "はそのまま残してください。\n"
    "\n"
    "出力は必ず次のキーだけを持つ JSON オブジェクト 1 つ:\n"
    '{"safe_text": "書き直し後の投稿文(除去が無ければ元のまま)", '
    '"changed": true/false, '
    '"removed": ["除去した actionable 情報の短い説明", ...]}\n'
    "actionable な情報が無ければ safe_text は元のまま、changed は false、"
    "removed は空配列にしてください。JSON 以外は一切出力しないこと。"
)


def _resolve_model(model: str | None) -> str | None:
    """opsec 判定に使う明示モデル(override)を 1 箇所で解決する。

    優先順: 明示引数 model > settings.x_opsec_llm_model > None。
    None を返した場合は override 無し=TaskType.X_OPSEC_REWRITE の既定
    (settings.openai_model、生成モデル相当)が使われる。非 None の場合は、
    その値が router.chat(override_model=) に渡り、既定より優先される
    (X_OPSEC_MODEL_WIRING_FIX_SPEC)。
    """
    return model or settings.x_opsec_llm_model or None


async def rewrite_for_opsec(text: str, *, model: str | None = None) -> OpsecResult:
    """投稿文から actionable な具体だけを書き直しで除去する。

    - actionable が無ければ (safe_text=元テキスト, changed=False, removed=[])。
    - actionable があれば、その部分だけ書き直した safe_text を返す。
    - LLM 呼び出し失敗・JSON パース失敗時は安全側(changed=True・safe_text="")
      に倒す。呼び出し側は safe_text が空なら live 配信をスキップし、shadow
      では元/(空)/理由をログに残して検証できる(素通しさせない)。
    """
    original = text or ""
    resolved = _resolve_model(model)
    router = get_llm_router()

    # 実際に使う(予定の)モデル名を先に解決してログ・結果に載せる(可観測性)。
    try:
        model_name = await router.resolve_model_name(
            TaskType.X_OPSEC_REWRITE, override_model=resolved
        )
    except Exception:
        # 名前解決の失敗は判定本体を止めない(override 有無だけは残す)。
        model_name = f"openai:{resolved}" if resolved else "openai:default"

    if not original.strip():
        return OpsecResult(safe_text=original, changed=False, removed=[], model=model_name)

    logger.info("x_opsec_rewrite: opsec judge model=%s (override=%s)", model_name, resolved)
    try:
        raw = await router.chat(
            TaskType.X_OPSEC_REWRITE,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": original},
            ],
            temperature=0.0,
            max_tokens=400,
            json_mode=True,
            override_model=resolved,
        )
    except Exception:
        logger.exception("x_opsec_rewrite: LLM call failed (failing safe: holding post)")
        return OpsecResult(
            safe_text="", changed=True, removed=["opsec 判定に失敗(安全側で保留)"], model=model_name
        )

    return _parse_result(raw, original, model_name)


def _parse_result(raw: str, original: str, model_name: str = "") -> OpsecResult:
    """LLM の生出力を OpsecResult へ。パース失敗は安全側(保留)に倒す。"""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
        safe_text = data.get("safe_text")
        changed = data.get("changed")
        removed = data.get("removed", [])
        if not isinstance(safe_text, str) or not isinstance(changed, bool):
            raise ValueError("safe_text/changed have unexpected types")
        if not isinstance(removed, list):
            removed = [str(removed)]
        removed = [str(r) for r in removed]
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning(
            "x_opsec_rewrite: could not parse LLM output as JSON (failing safe: holding post)"
        )
        return OpsecResult(
            safe_text="", changed=True,
            removed=["opsec 判定の出力を解釈できず(安全側で保留)"], model=model_name,
        )

    # changed=False を主張しつつ本文が変わっている等の不整合は、除去された側を
    # 信頼して changed=True に寄せる(actionable を素通しさせないため)。
    if not changed and safe_text != original:
        changed = True
    return OpsecResult(safe_text=safe_text, changed=changed, removed=removed, model=model_name)
