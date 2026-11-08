"""The price table, as data.

Every rate in this file is a row in ``PRICE_TABLE``. Nothing downstream
multiplies a hardcoded number: :mod:`beacon_llm.cost` reads these rows, so a
price change is a one-line edit here and the cost meter's tests keep passing.

Two facts this file exists to encode, both of which are silent money leaks if
got wrong:

1. **Haiku 5.5 has a pricing cliff at 100,000 prompt tokens.** Above it every
   rate jumps 5x. The threshold counts *all* input tokens - fresh input, cache
   reads and cache writes together - and each request is priced on its own, so
   twenty small calls and one batched call are not the same money.
2. **The cache-read discount is not a flat rate.** It is 0.1x on Haiku 5.5 but
   0.05x on Opus 5.5 and Sonnet 5.5. Writing "times 0.1" once in the cost
   function would overcharge the two models the project actually reports on.

Prices are taken verbatim from ``docs/AGENT_BRIEFS.md`` section A3, which
records them as verified against platform.claude.com on 2026-11-07. They are
not re-derived here and must not be "corrected" from memory.

``Decimal`` throughout: a per-call cost is often ~1e-5 USD and these are summed
over thousands of calls into a figure that goes in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

__all__ = [
    "CACHE_WRITE_MULTIPLIERS",
    "HAIKU_PROMPT_TOKEN_CLIFF",
    "MIN_CACHEABLE_PREFIX_TOKENS",
    "PRICE_TABLE",
    "BilledTier",
    "CacheTtl",
    "ModelPricing",
    "TierPrices",
    "known_models",
    "pricing_for",
]

BilledTier = Literal["standard", "haiku_base", "haiku_long_prompt"]
"""Which price tier a single request was billed at.

Stored on every ``beacon_core.models.LlmCall`` row. Without it the Haiku cliff
is invisible after the fact: two rows with identical token counts can differ 5x
in cost and nothing in the table would say why.
"""

CacheTtl = Literal["5m", "1h"]

HAIKU_PROMPT_TOKEN_CLIFF = 100_000
"""Where Haiku 5.5 rates jump 5x, counting cache reads and writes.

This is a *pricing* fact and therefore lives in the price table. The ceiling at
which Beacon *refuses* to call is a separate policy knob,
``settings.anthropic.haiku_prompt_token_ceiling``. They happen to coincide at
100,000 today; keeping them apart is what lets the refusal be tightened without
claiming the prices moved.
"""

MIN_CACHEABLE_PREFIX_TOKENS = 512
"""Shortest prefix the API will cache, on all three models.

Below it, a ``cache_control`` breakpoint is accepted, ignored, and no error is
returned: the prompt simply costs full price forever. Mirrored in
``settings.anthropic.min_cacheable_prefix_tokens`` and asserted against real
``usage`` in the live smoke test, because it cannot be detected any other way.
"""

CACHE_WRITE_MULTIPLIERS: dict[CacheTtl, Decimal] = {
    "5m": Decimal("1.25"),
    "1h": Decimal("2"),
}
"""Cache-*write* premium over the tier's input rate, per TTL.

**Not part of the verified A3 table**, which lists input, output and cache-read
rates only. These are the standard published write multipliers. They are data
for exactly that reason: if the verified figure differs, this dict is the single
place to change, and :class:`~beacon_llm.cost.CostBreakdown` reports the
cache-write cost as its own component so the uncertainty is never buried inside
a total.
"""


@dataclass(frozen=True, slots=True)
class TierPrices:
    """USD per million tokens for one model at one billing tier."""

    tier: BilledTier
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    cache_read_multiplier: Decimal
    """``cache_read_per_mtok / input_per_mtok``, held explicitly.

    Both forms are stored and ``test_pricing`` asserts they agree, so the
    0.1x-vs-0.05x asymmetry is checked by a test rather than trusted to a
    comment.
    """

    def cache_write_per_mtok(self, ttl: CacheTtl) -> Decimal:
        return self.input_per_mtok * CACHE_WRITE_MULTIPLIERS[ttl]


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """All tiers for one model, plus the prompt size at which they switch."""

    model: str
    base: TierPrices
    long_prompt: TierPrices | None = None
    prompt_token_cliff: int | None = None
    """``None`` means this model has a single flat tier at any prompt size."""

    def __post_init__(self) -> None:
        if (self.long_prompt is None) != (self.prompt_token_cliff is None):
            raise ValueError(
                f"{self.model}: a long-prompt tier and a cliff must be declared together"
            )

    def tier_for(self, prompt_tokens: int) -> TierPrices:
        """The tier a request of this prompt size is billed at.

        ``prompt_tokens`` is the *total* input for the request - fresh input
        plus cache reads plus cache writes - because that is what the threshold
        counts. Each request is priced independently; there is no averaging
        across the twenty calls made for one patient.
        """
        if (
            self.long_prompt is not None
            and self.prompt_token_cliff is not None
            and prompt_tokens > self.prompt_token_cliff
        ):
            return self.long_prompt
        return self.base


PRICE_TABLE: dict[str, ModelPricing] = {
    "claude-opus-5-5": ModelPricing(
        model="claude-opus-5-5",
        base=TierPrices(
            tier="standard",
            input_per_mtok=Decimal("4"),
            output_per_mtok=Decimal("20"),
            cache_read_per_mtok=Decimal("0.20"),
            cache_read_multiplier=Decimal("0.05"),
        ),
    ),
    "claude-sonnet-5-5": ModelPricing(
        model="claude-sonnet-5-5",
        base=TierPrices(
            tier="standard",
            input_per_mtok=Decimal("2"),
            output_per_mtok=Decimal("10"),
            cache_read_per_mtok=Decimal("0.10"),
            cache_read_multiplier=Decimal("0.05"),
        ),
    ),
    "claude-haiku-5-5": ModelPricing(
        model="claude-haiku-5-5",
        base=TierPrices(
            tier="haiku_base",
            input_per_mtok=Decimal("0.10"),
            output_per_mtok=Decimal("0.50"),
            cache_read_per_mtok=Decimal("0.01"),
            cache_read_multiplier=Decimal("0.1"),
        ),
        long_prompt=TierPrices(
            tier="haiku_long_prompt",
            input_per_mtok=Decimal("0.50"),
            output_per_mtok=Decimal("2.50"),
            cache_read_per_mtok=Decimal("0.05"),
            cache_read_multiplier=Decimal("0.1"),
        ),
        prompt_token_cliff=HAIKU_PROMPT_TOKEN_CLIFF,
    ),
}


def known_models() -> tuple[str, ...]:
    return tuple(PRICE_TABLE)


def pricing_for(model: str) -> ModelPricing:
    """Price rows for ``model``, or refuse.

    Unknown models raise rather than costing zero: a model missing from this
    table would otherwise show up in ``llm_calls`` as free.
    """
    from beacon_llm.errors import UnknownModelPriceError

    try:
        return PRICE_TABLE[model]
    except KeyError:
        raise UnknownModelPriceError(model, known_models()) from None
