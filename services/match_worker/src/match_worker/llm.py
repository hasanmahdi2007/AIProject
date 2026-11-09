"""The worker's one way to call Claude: budget, then the seam, then the audit row.

LangGraph does not build a prompt anywhere in this service and never sees a
model. Every call goes:

    node -> build_*_prompt() -> LlmGateway.call() -> beacon_llm.AsyncLlmClient

which keeps three things in our hands rather than a framework's: the cache
ordering (``build_cached_prompt`` renders the blocks, and only it can),
the cost accounting (metered from the returned ``usage``), and the prompt text
itself.

The gateway adds exactly three things to ``beacon_llm`` and deliberately
nothing else:

1. **Refuse before spending.** The budget is checked before the request, so a
   breach costs nothing and the node can return partial results.
2. **Meter after spending.** The real ``usage`` is folded into the meter.
3. **Write the audit row.** Every call lands in ``llm_calls``, which is where
   every cost figure in the README comes from.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from beacon_core.logging import get_logger
from beacon_llm import CachedPrompt, CallResult, Effort
from match_worker.budget import BudgetMeter

__all__ = ["LlmGateway", "SupportsCall", "SupportsLlmAudit"]

log = get_logger(__name__)


class SupportsCall(Protocol):
    """The slice of ``beacon_llm.AsyncLlmClient`` the worker uses.

    A Protocol so a test can inject a deterministic double without subclassing
    the real client - but note that the default double in
    :mod:`match_worker._testing` is the *real* client in replay mode, because
    replay already guarantees zero network calls and exercises the real
    pricing, guard and validation paths.
    """

    async def call[OutputT: BaseModel](
        self,
        *,
        prompt: CachedPrompt,
        output_model: type[OutputT],
        model: str | None = ...,
        effort: Effort | None = ...,
        max_tokens: int | None = ...,
        verify_capabilities: bool = ...,
    ) -> CallResult[OutputT]: ...


class SupportsLlmAudit(Protocol):
    async def record_llm_call(self, row: dict[str, Any]) -> None: ...


@dataclass
class LlmGateway:
    """One run's Claude calls: budgeted, metered and audited."""

    client: SupportsCall
    budget: BudgetMeter
    match_id: str
    audit: SupportsLlmAudit | None = None
    trace_id: str | None = None
    model: str | None = None
    """``None`` means ``settings.anthropic.cheap_model``, which is the default
    for every node. A node that wants a stronger model passes it per call."""

    calls_by_step: dict[str, int] = field(default_factory=dict)
    throttle_seconds: float = 0.0
    """See ``WorkerConfig.throttle_seconds``: a client-side pause before each
    call, off by default."""

    async def call[OutputT: BaseModel](
        self,
        *,
        step: str,
        prompt: CachedPrompt,
        output_model: type[OutputT],
        effort: Effort | None = None,
        model: str | None = None,
    ) -> CallResult[OutputT]:
        """Make one structured call, or raise ``BudgetExhausted`` before it.

        The ``BudgetExhausted`` raised here has never spent anything: the check
        precedes the request. That is what lets a node treat it as "stop and
        mark partial" rather than "we may or may not have been billed".
        """
        self.budget.check(step=step)
        if self.throttle_seconds:
            await asyncio.sleep(self.throttle_seconds)
        result = await self.client.call(
            prompt=prompt,
            output_model=output_model,
            model=model or self.model,
            effort=effort,
        )
        self.budget.record(result.cost, step=step)
        self.calls_by_step[step] = self.calls_by_step.get(step, 0) + 1

        if self.audit is not None:
            await self.audit.record_llm_call(
                result.audit_row(match_run_id=self.match_id, step=step, trace_id=self.trace_id)
            )
        if result.cost.cache_read_tokens == 0 and prompt.maybe_too_short_to_cache:
            log.debug(
                "worker.llm.no_cache_read",
                step=step,
                estimated_prefix_tokens=prompt.estimated_prefix_tokens,
            )
        return result
