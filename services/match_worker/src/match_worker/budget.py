"""The per-match budget: a meter that refuses, rather than a number in a log.

One match is roughly twenty-five Claude calls. The failure this guards against
is not a single expensive call, it is a loop that retries - a malformed trial
whose rules re-split into four hundred criteria, a verifier that re-runs on
every resume. Left unmetered that is a plausible way to spend real money on a
demo, and nothing about it looks wrong while it happens.

So spend is checked *before* every call and accumulated after it, from the
``usage`` the API reported rather than from an estimate. On breach the meter
raises :class:`BudgetExhausted`; the nodes catch it, mark what they have as
partial and return. Nothing truncates silently.

The meter is in-memory and per run, and that is correct rather than a
limitation: it is re-seeded from the checkpointed ``total_cost_usd`` when a
crashed run resumes, so the budget covers the *match*, not the attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beacon_core.logging import get_logger
from beacon_llm import CostBreakdown

__all__ = ["BudgetExhausted", "BudgetMeter", "BudgetSnapshot"]

log = get_logger(__name__)


class BudgetExhausted(Exception):
    """The per-match budget is spent; the caller must return partial results.

    Carries the numbers so the marker written into ``MatchState.error`` says
    what was spent and against which ceiling, not merely "partial".
    """

    def __init__(self, *, reason: str, spent_usd: float, limit_usd: float, tokens: int) -> None:
        self.reason = reason
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.tokens = tokens
        super().__init__(
            f"per-match budget exhausted ({reason}): spent ${spent_usd:.6f} of "
            f"${limit_usd:.6f} over {tokens:,} tokens"
        )


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    spent_usd: float
    tokens: int
    calls: int
    limit_usd: float
    limit_tokens: int

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.limit_usd or self.tokens >= self.limit_tokens


@dataclass
class BudgetMeter:
    """Dollars and tokens spent by one match run.

    ``spent_usd`` starts non-zero when a run resumes after a crash: the cost of
    the nodes that already completed is in the checkpoint, and a resumed run
    that started its meter at zero could spend the whole budget again.
    """

    limit_usd: float
    limit_tokens: int
    spent_usd: float = 0.0
    tokens: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    _deltas: list[float] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------- checking --

    def check(self, *, step: str) -> None:
        """Raise if there is no budget left for another call."""
        if self.spent_usd >= self.limit_usd:
            raise BudgetExhausted(
                reason=f"dollar ceiling reached before {step}",
                spent_usd=self.spent_usd,
                limit_usd=self.limit_usd,
                tokens=self.tokens,
            )
        if self.tokens >= self.limit_tokens:
            raise BudgetExhausted(
                reason=f"token ceiling reached before {step}",
                spent_usd=self.spent_usd,
                limit_usd=self.limit_usd,
                tokens=self.tokens,
            )

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.limit_usd or self.tokens >= self.limit_tokens

    # ------------------------------------------------------------ recording --

    def record(self, cost: CostBreakdown, *, step: str) -> float:
        """Add one call's metered cost. Returns the dollar delta.

        The delta is what a node puts in its state update: ``total_cost_usd``
        carries an ``operator.add`` reducer precisely so that twenty parallel
        tasks can each report their own spend without overwriting each other.
        """
        delta = cost.cost_usd
        self.spent_usd += delta
        self.tokens += cost.prompt_tokens + cost.output_tokens
        self.input_tokens += cost.input_tokens
        self.output_tokens += cost.output_tokens
        self.cached_tokens += cost.cache_read_tokens
        self.calls += 1
        self._deltas.append(delta)
        log.debug(
            "worker.budget.record",
            step=step,
            delta_usd=delta,
            spent_usd=self.spent_usd,
            limit_usd=self.limit_usd,
            tokens=self.tokens,
            cache_read_tokens=cost.cache_read_tokens,
        )
        return delta

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            spent_usd=self.spent_usd,
            tokens=self.tokens,
            calls=self.calls,
            limit_usd=self.limit_usd,
            limit_tokens=self.limit_tokens,
        )
