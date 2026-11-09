"""Exponential backoff with jitter, and the retryable/not-retryable split.

Jitter is not decoration. Twenty parallel rule checks that hit one 429 and all
back off for exactly two seconds retry in the same millisecond and get 429
again; the jitter is what breaks that lockstep. It is full jitter - a uniform
draw over the whole window, not the window plus a small wobble - because that
is the variant that actually decorrelates a fan-out.

The classification matters more than the timing. A refusal and an
over-ceiling prompt are *outcomes*: retrying the identical prompt produces the
identical answer, so retrying them only spends money and delays the run. A
429, a 5xx and a dropped connection are transport, and retrying them is the
whole point.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anthropic
import httpx

from beacon_core.logging import get_logger
from beacon_llm import PromptTooLargeError, RefusalError, StructuredOutputError
from match_worker.budget import BudgetExhausted

__all__ = ["RetryPolicy", "is_retryable", "retry_async"]

log = get_logger(__name__)


def is_retryable(exc: BaseException) -> bool:
    """Whether trying the same call again could plausibly succeed."""
    if isinstance(exc, BudgetExhausted | RefusalError | PromptTooLargeError):
        # Outcomes, not blips. A retry changes nothing except the bill.
        return False
    if isinstance(exc, StructuredOutputError):
        # A truncated or invalid JSON body is worth exactly one more go: the
        # model is sampling, so the second answer is not the first one.
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 408 or exc.status_code == 429 or exc.status_code >= 500
    # httpx errors reach here from the retrieval client and from test doubles
    # that stand in for the transport; both are blips by definition.
    return isinstance(
        exc,
        anthropic.APIConnectionError
        | anthropic.APITimeoutError
        | asyncio.TimeoutError
        | httpx.TransportError,
    )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int
    base_seconds: float
    max_seconds: float

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Full jitter: uniform over ``[0, min(max, base * 2**(attempt-1))]``."""
        window = min(self.max_seconds, self.base_seconds * (2 ** max(attempt - 1, 0)))
        draw = (rng or random).random()
        return window * draw


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    what: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run ``operation``, retrying only what is worth retrying.

    Raises the last exception once the attempts are spent, so the caller - not
    this helper - decides what a final failure means. For a trial inside
    ``check_rules`` it means "could not evaluate"; for a Kafka message it means
    the DLQ.
    """
    last: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation()
        except BaseException as exc:
            last = exc
            if not is_retryable(exc) or attempt == policy.attempts:
                raise
            delay = policy.delay_for(attempt, rng=rng)
            log.warning(
                "worker.retry",
                what=what,
                attempt=attempt,
                attempts=policy.attempts,
                delay_seconds=round(delay, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            await sleep(delay)
    raise AssertionError("unreachable") from last
