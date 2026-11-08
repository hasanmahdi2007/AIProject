"""One ``Settings`` object for the whole system, nested by concern.

Read from the environment and from ``.env``, with ``__`` as the nesting
delimiter: ``BEACON_POSTGRES__HOST=localhost`` sets ``settings.postgres.host``.

Everything configurable lives here, so an ablation is a config change rather
than an edit. In particular ``retrieval.query_count`` is the N that the Day 3
sweep tunes over {1, 3, 5, 10} - it is deliberately not a literal in the
query-generation prompt.

Secrets are ``SecretStr``: printing a ``Settings`` never leaks the API key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AnthropicSettings",
    "KafkaSettings",
    "PostgresSettings",
    "QdrantSettings",
    "RedisSettings",
    "RetrievalSettings",
    "Settings",
    "get_settings",
]

REPO_ROOT = Path(__file__).resolve().parents[4]


class PostgresSettings(BaseModel):
    host: str = "localhost"
    port: int = 15432
    """Not 5432. The dev machine runs a native PostgreSQL 18 service on 5432, so
    deploy/compose.yaml publishes 15432 instead and this default matches it - one
    source of truth, no env override needed for a normal local run. Containers
    still talk to each other on 5432 internally."""

    user: str = "beacon"
    password: SecretStr = SecretStr("beacon")
    database: str = "beacon"

    def dsn(self, *, driver: str = "postgresql+asyncpg") -> str:
        """SQLAlchemy URL. Alembic runs against the same async driver, so there
        is no second connection string to keep in step."""
        pw = self.password.get_secret_value()
        return f"{driver}://{self.user}:{pw}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 16379
    """Not 6379. Another compose project on the dev machine publishes its Redis
    there; see PostgresSettings.port for the same reasoning."""

    db: int = 0
    conversation_ttl_seconds: int = 60 * 60 * 24

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "trials"

    # Named vectors on one point, so a single query hits both and the hits
    # refer to the same trial.
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = 384
    dense_max_tokens: int = 512
    """bge-small silently truncates past 512 tokens, so the overview text is
    built most-discriminative-field-first and the truncation rate is logged."""

    sparse_model: str = "Qdrant/bm25"
    sparse_idf_modifier: bool = True
    """BM25 needs corpus-level IDF, and Qdrant only computes it when the sparse
    field is declared with ``modifier=IDF``. Omitting it raises no error and
    silently degrades rare-term matching - the entire reason BM25 is here."""

    reranker_model: str = "BAAI/bge-reranker-base"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class KafkaSettings(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "beacon-match-workers"
    max_retries: int = 3
    """After this many failures the message goes to ``match.dlq``."""


class AnthropicSettings(BaseSettings):
    """Its own settings class with an ``ANTHROPIC_`` prefix, so the key lives in
    ``.env`` once, under the name the Anthropic SDK itself already reads."""

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_", env_file=(".env",), env_file_encoding="utf-8", extra="ignore"
    )

    api_key: SecretStr = SecretStr("")

    # Verified against platform.claude.com on 2026-11-07.
    cheap_model: str = "claude-haiku-5-5"
    headline_model: str = "claude-sonnet-5-5"
    judge_model: str = "claude-opus-5-5"
    """The judge must differ from the model being judged, or it is grading
    itself. The headline run is Sonnet 5.5, so the judge is Opus 5.5."""

    max_tokens: int = 16_000

    haiku_prompt_token_ceiling: int = 100_000
    """Haiku 5.5 prices jump 5x above 100K *prompt* tokens, counting cache reads
    and writes. The wrapper counts tokens before the call and refuses above this
    ceiling, so a "batch all 20 trials into one call" refactor fails loudly
    instead of quintupling cost in silence."""

    min_cacheable_prefix_tokens: int = 512
    """Below this, prompt caching silently does nothing and returns no error."""


class RetrievalSettings(BaseModel):
    base_url: str = "http://localhost:8001"

    query_count: int = 3
    """N generated queries. Default 3 - facet-diverse, not paraphrases. Tuned
    by the Day 3 sweep over {1, 3, 5, 10}; N=1 doubles as the
    "does LLM query generation help at all?" ablation row."""

    fuse_top_k: int = 100
    rerank_top_k: int = 20
    rrf_k: int = 60
    use_rerank: bool = True
    apply_age_sex_filters: bool = True
    """Its own ablation row, and allowed to come out negative: the filter is
    only as good as the age parsing."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BEACON_",
        env_nested_delimiter="__",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    ir_datasets_home: Path | None = None
    """Where the TREC corpus is cached. Point it outside the repo so ingestion and
    evals agree on one copy.

    Measured 8 Nov: the corpus is **1.79 GB** of archives, about 2.6 GB once
    cached - not the ">=25 GB" the project guide estimates. Allow a few GB, not
    twenty-five."""

    otel_endpoint: str = "http://localhost:6006/v1/traces"
    llm_fixtures_dir: Path = REPO_ROOT / "tests" / "fixtures" / "llm"
    """Recorded Claude responses, keyed by prompt hash. This is what makes CI
    free and deterministic."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()
