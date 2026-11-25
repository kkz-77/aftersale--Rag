"""Environment-backed settings with a cached accessor."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed runtime configuration.

    Secrets stay in SecretStr so accidental str() / logs do not leak keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "testing", "production"] = "development"
    app_name: str = "adaptive-rag-cs"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    deepseek_model: str = "qwen-flash"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    # 模型的思考控制参数不同，两项只选一项配置，不要同时设置。
    # qwen-flash：在 .env 配置 LLM_ENABLE_THINKING=false，明确关闭思考。
    # None 表示不发送该参数，避免影响不支持此参数的模型。
    llm_enable_thinking: bool | None = None
    # qwen3.8-omni-flash 对照实验用此项；字符串 "none" 关闭思考。
    # 本轮换 qwen-flash，默认不再发送 reasoning_effort。
    llm_reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None

    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = ""
    embedding_model: str = "bge-small-zh-v1.5"
    embedding_dim: int = 512
    local_embedding_enabled: bool = True
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    local_embedding_device: str = ""
    local_embedding_force: bool = False

    cross_encoder_enabled: bool = True
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    cross_encoder_device: str = ""
    cross_encoder_max_length: int = 512
    cross_encoder_force: bool = False

    database_url: str = "sqlite+aiosqlite:///./data/local/app.db"
    redis_url: str = ""
    qdrant_url: str = ""
    qdrant_collection: str = "cs_knowledge"

    retrieval_top_k_simple: int = 3
    retrieval_top_k_complex: int = 8
    rrf_k: int = 60
    rerank_top_n: int = 5
    max_retrieval_hops: int = 3

    token_budget: int = 128_000
    # 记忆改造第2步：两项开关用于同模型、同数据的A/B/C对照，不更改.env。
    # A=False/False；B=True/False；C=True/True（生成/决策）。
    memory_in_generation: bool = True
    memory_in_decisions: bool = True
    # 第3步：近期窗口是目标，不是立即删除旧消息的硬上限；与128k总预算分开。
    memory_summary_trigger_tokens: int = Field(default=3328, ge=1)
    memory_summary_target_tokens: int = Field(default=2867, ge=1)
    memory_summary_input_tokens: int = Field(default=8192, ge=1, le=8192)
    memory_summary_max_tokens: int = Field(default=512, ge=1, le=512)
    memory_summary_timeout_seconds: float = Field(default=15.0, gt=0)
    compress_soft_ratio: float = 0.50
    compress_hard_ratio: float = 0.70
    auto_compact_ratio: float = 0.85

    max_agent_steps: int = 8
    max_tool_calls: int = 6

    rate_limit_per_minute: int = 60
    retrieval_cache_ttl_seconds: int = 60
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    admin_api_token: SecretStr = SecretStr("")
    knowledge_dir: str = "knowledge"
    unstructured_enabled: bool = True
    unstructured_strategy: str = "auto"
    unstructured_ocr_languages: str = "chi_sim+eng"
    unstructured_force: bool = False
    mcp_enabled: bool = True
    mcp_remote_url: str = ""

    @property
    def llm_configured(self) -> bool:
        return bool(self.deepseek_api_key.get_secret_value())

    @property
    def embedding_configured(self) -> bool:
        return bool(self.embedding_api_key.get_secret_value() and self.embedding_base_url)

    @property
    def local_embedding_configured(self) -> bool:
        """True when production will attempt to load the local dense model."""
        return self.local_embedding_enabled and (
            self.app_env != "testing" or self.local_embedding_force
        )

    @property
    def cross_encoder_configured(self) -> bool:
        """True when production path will attempt to load the Cross-Encoder.

        Tests stay on lexical fallback unless CROSS_ENCODER_FORCE is set.
        """
        return self.cross_encoder_enabled and (
            self.app_env != "testing" or self.cross_encoder_force
        )

    @property
    def unstructured_configured(self) -> bool:
        """True when ingest will attempt Unstructured partition.

        Tests stay on markdown heading split unless UNSTRUCTURED_FORCE is set.
        """
        return self.unstructured_enabled and (self.app_env != "testing" or self.unstructured_force)

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_url)

    @property
    def qdrant_configured(self) -> bool:
        return bool(self.qdrant_url)

    @property
    def use_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Used by tests after monkeypatching environment variables."""
    get_settings.cache_clear()
