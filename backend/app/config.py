# 役割: FastAPI バックエンドの環境設定をまとめる。

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Sigmaris Backend"
    app_env: str = "development"
    api_prefix: str = "/api"
    frontend_origin: str = "http://localhost:3000"
    google_maps_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"           # 通常会話・ルーティング
    openai_nano_model: str = "gpt-5.4-nano"       # 記憶抽出・要約・分類
    openai_advanced_model: str = "gpt-5.5"        # 自己反省・設計・週次レビュー
    openai_import_model: str | None = None
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None
    google_calendar_id: str = "primary"
    next_public_supabase_url: str | None = None
    next_public_supabase_publishable_key: str | None = None
    supabase_service_role_key: str | None = None
    agent_secrets: str | None = None
    schedule_agent_base_url: str = "http://127.0.0.1:8000"
    schedule_agent_id: str = "sigmaris-orchestrator"
    schedule_agent_secret: str | None = None
    sigmaris_persona_path: str = "../docs/persona.md"
    sigmaris_rewrite_model: str | None = None
    sigmaris_guard_model: str | None = None
    sigmaris_reflect_model: str | None = None
    sigmaris_timezone: str = "Asia/Tokyo"
    sigmaris_recent_message_window: int = 40
    sigmaris_user_jwt: str | None = None
    sigmaris_refresh_token: str | None = None
    sigmaris_google_access_token: str | None = None
    sigmaris_surface_inquiry_questions: bool = False
    pushover_user_key: str | None = None
    pushover_app_token: str | None = None
    proactive_enabled: bool = True
    local_llm_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_embed_model: str = "nomic-embed-text"
    openai_embedding_model: str = "text-embedding-3-small"
    eval_generation_model: str = "gpt-5.4-mini"   # Phase C-mini testset question generation (OpenAI fallback)
    eval_judge_model: str | None = None           # Phase C-full LLM-as-a-Judge (None -> openai_advanced_model)
    # Phase C-full: dedicated Supabase account for LongMemEval/LoCoMo ingestion,
    # deliberately separate from sigmaris_user_jwt/sigmaris_refresh_token (the
    # real user's own credentials) so public-benchmark conversation data can
    # never land in the same user_fact_items rows as 海星さん's real memories.
    # See backend/app/services/bench_auth.py.
    sigmaris_eval_bench_refresh_token: str | None = None
    sigmaris_eval_bench_user_jwt: str | None = None
    x_api_key: str | None = None
    x_api_secret: str | None = None
    x_access_token: str | None = None
    x_access_token_secret: str | None = None
    x_enabled: bool = False
    # 旧X投稿システムの廃止、及び、新7カテゴリシステムへの実際の接続
    # (docs/sigmaris/phase_h_report.md): X_ENABLED=trueの状態でも、この
    # フラグがFalse(デフォルト)の間は、7カテゴリシステムは生成・
    # Executive Gate判定・全フィルタを実際に通した上で、"投稿する
    # つもりだった内容"をログに記録するのみで、実際にはx_publisher.
    # post_tweet()を呼ばない(shadow mode)。運用者が、ログを確認して
    # 問題ないと判断した場合にのみ、明示的にTrueへ切り替えることを
    # 想定した、移行期の安全策(依頼書「最初の数回の投稿は、実際の投稿前
    # にログに記録し確認できるようにする」への対応)。
    x_categorized_post_live: bool = False
    # X_POST_SELF_TIMING_SPEC: 投稿内容に応じて AI が未来の投稿時刻を決め、
    # 予約→その時刻に配信する仕組みの制御値。
    x_post_dispatch_interval_min: int = 10   # 配信ディスパッチャの実行間隔(分)
    x_post_schedule_horizon_h: int = 24      # 予約できる最長の地平線(時間先)
    x_post_min_interval_min: int = 90        # 予約同士の最小間隔(分)
    # X_OPSEC_LLM_REWRITE_SPEC: 配信直前の opsec 判定＋書き直し(actionable な
    # 具体だけを除去)に使う LLM モデルの明示上書き。None(既定)の場合は生成と
    # 同じモデル階層(LLMRouter の TaskType.ROUTING → openai_model、または
    # Ollama)を使う。将来ローカル LLM へ寄せるための差し込み口。
    x_opsec_llm_model: str | None = None
    sigmaris_launch_date: str | None = None
    github_token: str | None = None  # research_agent.py's GitHub trending-repo search (rate-limit headers only)
    # Phase F-3 (docs/sigmaris/phase_f_report.md): 承認後のPR作成専用の、
    # 独立した書き込み権限クレデンシャル。github_token(上記、読み取り専用の
    # トレンド検索用)とは意図的に別の変数名にした——書き込み権限を持つ
    # クレデンシャルと、読み取り専用のそれを、運用者が env 上で混同しない
    # ようにするための判断(詳細、レポート参照)。github_pr_publisher.py
    # 以外のいかなるモジュールからも参照されない。
    sigmaris_pr_github_token: str | None = None
    sigmaris_pr_github_repo: str | None = None  # "owner/repo" 形式
    health_sync_enabled: bool = False
    news_api_key: str | None = None
    research_enabled: bool = False
    agent_registry_json: str | None = None

    model_config = SettingsConfigDict(
        env_file=(".env", "../frontend/.env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
