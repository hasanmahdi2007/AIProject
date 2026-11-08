"""Worker-local knobs.

``beacon_core.settings`` is frozen and read-only, so everything that is the
*worker's* policy rather than the system's contract lives here: budgets,
retry counts, fan-out width, timeouts. Prefix ``BEACON_WORKER_``, so a demo
run can lower the budget from the shell without a code change.

Two of these are load-bearing rather than cosmetic:

* ``max_cost_usd`` / ``max_tokens`` are the per-match budget. On breach the run
  returns what it has, marked partial - see :mod:`match_worker.budget`.
* ``trial_attempts`` is 2 on purpose. One retry absorbs a transient API blip;
  a second failure is treated as "this trial cannot be evaluated" and the other
  nineteen carry on. Raising it does not make a malformed trial parse.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["WorkerConfig"]


class WorkerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BEACON_WORKER_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------- budgets --
    max_cost_usd: float = Field(default=0.50, gt=0)
    """Dollar ceiling for one match. 20 Haiku rule checks on a cached prefix
    land near a cent, so the default is ~50x headroom: it stops a runaway, it
    does not shape normal runs."""

    max_tokens: int = Field(default=2_000_000, gt=0)
    """Token ceiling for one match, counting prompt (incl. cache reads) and
    output. The dollar ceiling is the real guard; this one catches a model
    priced lower than expected still burning the context window."""

    # ------------------------------------------------------------- retries --
    trial_attempts: int = Field(default=2, ge=1)
    """Attempts per trial inside ``check_rules`` before it is marked
    "could not evaluate"."""

    trial_retry_base_seconds: float = Field(default=0.5, ge=0)
    trial_retry_max_seconds: float = Field(default=8.0, ge=0)

    message_attempts: int = Field(default=3, ge=1)
    """Kafka-message attempts before the message goes to ``match.dlq``.
    Defaults to ``settings.kafka.max_retries`` semantics, kept separately so a
    worker can be made more patient without touching the frozen settings."""

    message_retry_base_seconds: float = Field(default=1.0, ge=0)
    message_retry_max_seconds: float = Field(default=30.0, ge=0)

    # -------------------------------------------------------------- limits --
    fanout_concurrency: int = Field(default=5, ge=1)
    """Simultaneous in-flight Claude calls per node. Kept modest: the first
    call writes the shared prefix to cache and the rest read it, so firing all
    twenty at once races the cache write and pays full price twenty times."""

    throttle_seconds: float = Field(default=0.0, ge=0)
    """Pause before each Claude call. A crude client-side rate limiter for a
    laptop demo, and the knob the crash test uses to make a run long enough to
    be killed at a reproducible point. Default 0: off in normal operation."""

    retrieval_timeout_seconds: float = Field(default=30.0, gt=0)
    retrieval_attempts: int = Field(default=3, ge=1)

    explain_top_k: int = Field(default=5, ge=0)
    """Plain-language summaries are written for the top K trials only; the rest
    carry the code-derived verdict without an LLM call."""

    verify_with_llm: bool = True
    """The citation check is always code. This toggles the extra cheap LLM pass
    that asks whether the cited sentence actually supports the label - the
    faithfulness number. Off makes the pipeline cheaper, not more correct."""

    relay_enabled: bool = True
    relay_poll_seconds: float = Field(default=0.5, gt=0)
    relay_batch_size: int = Field(default=100, ge=1)

    poll_timeout_ms: int = Field(default=1_000, ge=1)
