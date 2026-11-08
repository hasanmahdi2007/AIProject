"""The price table checked against itself and against the cliff.

These tests exist because every number here was copied by hand from the
verified table in ``docs/AGENT_BRIEFS.md``, and a transposed digit in a price
is not something any other test would notice.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from beacon_llm import (
    CACHE_WRITE_MULTIPLIERS,
    HAIKU_PROMPT_TOKEN_CLIFF,
    PRICE_TABLE,
    UnknownModelPriceError,
    known_models,
    pricing_for,
)
from beacon_llm.pricing import ModelPricing, TierPrices

EXPECTED_VERBATIM = {
    # model, tier -> (input/MTok, output/MTok, cache read/MTok)
    ("claude-opus-5-5", "standard"): ("4", "20", "0.20"),
    ("claude-sonnet-5-5", "standard"): ("2", "10", "0.10"),
    ("claude-haiku-5-5", "haiku_base"): ("0.10", "0.50", "0.01"),
    ("claude-haiku-5-5", "haiku_long_prompt"): ("0.50", "2.50", "0.05"),
}


def _all_tiers() -> list[tuple[str, TierPrices]]:
    tiers: list[tuple[str, TierPrices]] = []
    for model, pricing in PRICE_TABLE.items():
        tiers.append((model, pricing.base))
        if pricing.long_prompt is not None:
            tiers.append((model, pricing.long_prompt))
    return tiers


def test_table_matches_the_verified_figures() -> None:
    """Every rate equals the A3 brief's table, to the digit."""
    actual = {
        (model, tier.tier): (
            str(tier.input_per_mtok),
            str(tier.output_per_mtok),
            str(tier.cache_read_per_mtok),
        )
        for model, tier in _all_tiers()
    }
    assert actual == EXPECTED_VERBATIM


@pytest.mark.parametrize(("model", "tier"), _all_tiers(), ids=lambda v: getattr(v, "tier", v))
def test_cache_read_multiplier_agrees_with_the_cache_read_rate(
    model: str, tier: TierPrices
) -> None:
    """The multiplier and the absolute rate are two views of one fact.

    Holding both and checking them against each other is what stops the
    0.1x-vs-0.05x asymmetry becoming a comment nobody reads: Haiku discounts
    cache reads to a tenth, Opus and Sonnet to a twentieth.
    """
    assert tier.cache_read_per_mtok == tier.input_per_mtok * tier.cache_read_multiplier


def test_haiku_discounts_cache_reads_less_than_opus_and_sonnet() -> None:
    """The asymmetry, asserted directly rather than implied."""
    assert PRICE_TABLE["claude-haiku-5-5"].base.cache_read_multiplier == Decimal("0.1")
    assert PRICE_TABLE["claude-haiku-5-5"].long_prompt is not None
    assert PRICE_TABLE["claude-opus-5-5"].base.cache_read_multiplier == Decimal("0.05")
    assert PRICE_TABLE["claude-sonnet-5-5"].base.cache_read_multiplier == Decimal("0.05")


def test_the_haiku_long_prompt_tier_is_exactly_five_times_the_base() -> None:
    """The cliff is a 5x step on input, output and cache reads alike."""
    haiku = PRICE_TABLE["claude-haiku-5-5"]
    assert haiku.long_prompt is not None
    assert haiku.long_prompt.input_per_mtok == haiku.base.input_per_mtok * 5
    assert haiku.long_prompt.output_per_mtok == haiku.base.output_per_mtok * 5
    assert haiku.long_prompt.cache_read_per_mtok == haiku.base.cache_read_per_mtok * 5


@pytest.mark.parametrize(
    ("prompt_tokens", "expected"),
    [
        (0, "haiku_base"),
        (1, "haiku_base"),
        (99_999, "haiku_base"),
        (HAIKU_PROMPT_TOKEN_CLIFF, "haiku_base"),
        (HAIKU_PROMPT_TOKEN_CLIFF + 1, "haiku_long_prompt"),
        (1_000_000, "haiku_long_prompt"),
    ],
)
def test_the_cliff_is_strictly_above_the_threshold(prompt_tokens: int, expected: str) -> None:
    """Exactly 100,000 is still the base tier; 100,001 is not."""
    assert pricing_for("claude-haiku-5-5").tier_for(prompt_tokens).tier == expected


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_opus_and_sonnet_have_no_cliff(model: str) -> None:
    """A flat tier at any prompt size - so the guard leaves them alone."""
    pricing = pricing_for(model)
    assert pricing.prompt_token_cliff is None
    assert pricing.tier_for(5_000_000).tier == "standard"


def test_cache_write_premium_is_per_ttl() -> None:
    """A 1h breakpoint costs more to write than a 5m one."""
    haiku = PRICE_TABLE["claude-haiku-5-5"].base
    assert haiku.cache_write_per_mtok("5m") == haiku.input_per_mtok * CACHE_WRITE_MULTIPLIERS["5m"]
    assert haiku.cache_write_per_mtok("1h") > haiku.cache_write_per_mtok("5m")


def test_an_unpriced_model_raises_rather_than_costing_nothing() -> None:
    """Fable 5.1 is a real model with no row in the verified table.

    It must raise. Costing it at zero would make it the cheapest model in the
    ``llm_calls`` table and therefore the one an ablation picks.
    """
    with pytest.raises(UnknownModelPriceError) as excinfo:
        pricing_for("claude-fable-5-1")
    assert "claude-fable-5-1" in str(excinfo.value)
    assert set(excinfo.value.known) == set(known_models())


def test_a_long_prompt_tier_without_a_cliff_is_rejected() -> None:
    """Declaring a second tier but no threshold would silently never use it."""
    base = PRICE_TABLE["claude-haiku-5-5"].base
    with pytest.raises(ValueError, match="declared together"):
        ModelPricing(model="x", base=base, long_prompt=base)
