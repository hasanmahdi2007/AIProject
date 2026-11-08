"""The cost meter: one ``usage`` object in, one priced breakdown out.

It reads :mod:`beacon_llm.pricing` and multiplies nothing of its own, so the
only way to change a price is to change a row in the table.

The breakdown is deliberately itemised rather than a single total. Four numbers
that sum to one figure make a wrong total locatable: if the README's cost is
off, the component that is off says which assumption broke - usually the cache
not engaging, which shows up as ``cache_read_tokens == 0`` and a full-price
input component.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from anthropic.types import Usage

from beacon_llm.pricing import BilledTier, CacheTtl, TierPrices, pricing_for

__all__ = ["CostBreakdown", "price_tokens", "price_usage"]

_PER_MILLION = Decimal(1_000_000)
_CENT_OF_A_MILLIONTH = Decimal("0.00000001")
"""Rounding unit, eight decimal places.

A single Haiku call with a cached prefix costs on the order of 1e-5 USD, so
rounding to cents per call would record every call as free. Totals are summed
from these rounded values, which keeps a sum of rows equal to the row of sums.
"""


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Everything ``beacon_core.models.LlmCall`` needs about money and tokens.

    ``prompt_tokens`` is the figure the Haiku cliff is judged on: fresh input +
    cache reads + cache writes. It is kept as a field rather than recomputed by
    callers, because the temptation to compare the cliff against
    ``input_tokens`` alone is exactly how the 5x surprise happens - on a cached
    prompt ``input_tokens`` is the *small* number.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    prompt_tokens: int
    billed_tier: BilledTier
    cache_ttl: CacheTtl

    input_cost_usd: Decimal
    output_cost_usd: Decimal
    cache_read_cost_usd: Decimal
    cache_write_cost_usd: Decimal
    total_cost_usd: Decimal

    @property
    def cost_usd(self) -> float:
        """``total_cost_usd`` as a float, for the ``Float`` column on ``LlmCall``."""
        return float(self.total_cost_usd)

    @property
    def cache_hit(self) -> bool:
        """Whether the prefix actually came from cache.

        Worth asserting after any call that marked a cache breakpoint: a prefix
        under 512 tokens is accepted, silently not cached, and reports ``0`` here.
        """
        return self.cache_read_tokens > 0


def _round(value: Decimal) -> Decimal:
    return value.quantize(_CENT_OF_A_MILLIONTH, rounding=ROUND_HALF_UP)


def price_tokens(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_ttl: CacheTtl = "5m",
) -> CostBreakdown:
    """Price a request from raw token counts.

    ``input_tokens`` must be the *uncached* input the API reported, i.e. not
    including ``cache_read_tokens`` or ``cache_write_tokens`` - that is how the
    Anthropic ``usage`` object is shaped, and these three are billed at three
    different rates.
    """
    if min(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens) < 0:
        raise ValueError("token counts cannot be negative")

    pricing = pricing_for(model)
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    tier: TierPrices = pricing.tier_for(prompt_tokens)

    input_cost = _round(Decimal(input_tokens) * tier.input_per_mtok / _PER_MILLION)
    output_cost = _round(Decimal(output_tokens) * tier.output_per_mtok / _PER_MILLION)
    cache_read_cost = _round(Decimal(cache_read_tokens) * tier.cache_read_per_mtok / _PER_MILLION)
    cache_write_cost = _round(
        Decimal(cache_write_tokens) * tier.cache_write_per_mtok(cache_ttl) / _PER_MILLION
    )

    return CostBreakdown(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        prompt_tokens=prompt_tokens,
        billed_tier=tier.tier,
        cache_ttl=cache_ttl,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        cache_read_cost_usd=cache_read_cost,
        cache_write_cost_usd=cache_write_cost,
        total_cost_usd=input_cost + output_cost + cache_read_cost + cache_write_cost,
    )


def price_usage(model: str, usage: Usage, *, cache_ttl: CacheTtl = "5m") -> CostBreakdown:
    """Price a request straight from the SDK's ``usage`` object.

    The cache fields are ``Optional[int]`` on the wire - absent, not zero, when
    no cache control was sent - so they are coalesced here rather than at every
    call site.
    """
    return price_tokens(
        model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cache_ttl=cache_ttl,
    )
