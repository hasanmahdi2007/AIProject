"""The cost meter against hand calculations.

Each case below is worked out in the comment above it, in the same units the
price table uses, so the test is a check on the arithmetic and not a snapshot
of whatever the code happened to return the first time it ran.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from anthropic.types import Usage

from beacon_llm import UnknownModelPriceError, price_tokens, price_usage


def test_haiku_base_tier_hand_calculation() -> None:
    """Haiku 5.5, small prompt, cache miss.

    These are the real token counts from call 1 of the live smoke test.

    ===========  ======  ===========  =======================  ============
    Component    Tokens  $/MTok       Arithmetic               USD
    ===========  ======  ===========  =======================  ============
    input            90  0.10         90 x 0.10 / 1e6          0.00000900
    output          148  0.50         148 x 0.50 / 1e6         0.00007400
    cache write    2032  0.125        2032 x 0.125 / 1e6       0.00025400
    cache read        0  0.01         0                        0.00000000
    **total**                                                  **0.00033700**
    ===========  ======  ===========  =======================  ============

    The cache-write rate is the tier's input rate times the 5-minute write
    multiplier: 0.10 x 1.25 = 0.125 per MTok.

    Prompt total is 90 + 2032 + 0 = 2122, under 100,000, so the base tier
    applies. Note that the cache *write* counts toward that total.
    """
    cost = price_tokens(
        "claude-haiku-5-5",
        input_tokens=90,
        output_tokens=148,
        cache_read_tokens=0,
        cache_write_tokens=2032,
    )
    assert cost.billed_tier == "haiku_base"
    assert cost.prompt_tokens == 2122
    assert cost.input_cost_usd == Decimal("0.00000900")
    assert cost.output_cost_usd == Decimal("0.00007400")
    assert cost.cache_write_cost_usd == Decimal("0.00025400")
    assert cost.cache_read_cost_usd == Decimal("0")
    assert cost.total_cost_usd == Decimal("0.00033700")
    assert cost.cost_usd == pytest.approx(0.000337)
    assert not cost.cache_hit


def test_haiku_cache_hit_hand_calculation() -> None:
    """Haiku 5.5, same prefix, served from cache. Call 2 of the smoke test.

    ===========  ======  ===========  =======================  ============
    Component    Tokens  $/MTok       Arithmetic               USD
    ===========  ======  ===========  =======================  ============
    input            86  0.10         86 x 0.10 / 1e6          0.00000860
    output          117  0.50         117 x 0.50 / 1e6         0.00005850
    cache read     2032  0.01         2032 x 0.01 / 1e6        0.00002032
    cache write       0  0.125        0                        0.00000000
    **total**                                                  **0.00008742**
    ===========  ======  ===========  =======================  ============

    Worth comparing with the previous test: the same 2032-token prefix costs
    0.00025400 to write and 0.00002032 to read, a 12.5x difference. That ratio
    is the whole argument for cache-ordering the prompt - nineteen of twenty
    per-trial calls pay the small number.
    """
    cost = price_tokens(
        "claude-haiku-5-5",
        input_tokens=86,
        output_tokens=117,
        cache_read_tokens=2032,
        cache_write_tokens=0,
    )
    assert cost.billed_tier == "haiku_base"
    assert cost.cache_read_cost_usd == Decimal("0.00002032")
    assert cost.total_cost_usd == Decimal("0.00008742")
    assert cost.cache_hit


def test_opus_hand_calculation_uses_the_twentieth_cache_multiplier() -> None:
    """Opus 5.5 discounts cache reads to 0.05x, not 0.1x.

    ===========  ======  ===========  =======================  ============
    Component    Tokens  $/MTok       Arithmetic               USD
    ===========  ======  ===========  =======================  ============
    input          1000  4            1000 x 4 / 1e6           0.00400000
    output          500  20           500 x 20 / 1e6           0.01000000
    cache read    10000  0.20         10000 x 0.20 / 1e6       0.00200000
    **total**                                                  **0.01600000**
    ===========  ======  ===========  =======================  ============

    Priced with Haiku's 0.1x multiplier the cache read would come to 0.004 and
    the total to 0.018 - a 12.5% error on the judge model, which is the one the
    evaluation harness runs most.
    """
    cost = price_tokens(
        "claude-opus-5-5", input_tokens=1000, output_tokens=500, cache_read_tokens=10_000
    )
    assert cost.billed_tier == "standard"
    assert cost.cache_read_cost_usd == Decimal("0.00200000")
    assert cost.total_cost_usd == Decimal("0.01600000")


def test_the_cliff_costs_exactly_five_times_the_same_tokens() -> None:
    """One request over the threshold; the identical tokens under it.

    Over the cliff - 50,000 fresh input + 60,000 cache read = 110,000 prompt
    tokens, so the ``haiku_long_prompt`` tier:

    ===========  ======  ========  =========================  ==========
    Component    Tokens  $/MTok    Arithmetic                 USD
    ===========  ======  ========  =========================  ==========
    input        50000   0.50      50000 x 0.50 / 1e6         0.025
    output        2000   2.50      2000 x 2.50 / 1e6          0.005
    cache read   60000   0.05      60000 x 0.05 / 1e6         0.003
    **total**                                                 **0.033**
    ===========  ======  ========  =========================  ==========

    The same token counts at the base tier come to 0.0066. The ratio is exactly
    5, on every component, which is what makes "batch the twenty trials into
    one call" so expensive: the prompt gets bigger *and* every token in it
    reprices.

    Note the cache read is 60,000 of the 110,000. A caller who compared the
    threshold against ``input_tokens`` alone would see 50,000 and conclude it
    was safely under.
    """
    over = price_tokens(
        "claude-haiku-5-5", input_tokens=50_000, output_tokens=2_000, cache_read_tokens=60_000
    )
    assert over.billed_tier == "haiku_long_prompt"
    assert over.prompt_tokens == 110_000
    assert over.input_cost_usd == Decimal("0.02500000")
    assert over.output_cost_usd == Decimal("0.00500000")
    assert over.cache_read_cost_usd == Decimal("0.00300000")
    assert over.total_cost_usd == Decimal("0.03300000")

    under = price_tokens(
        "claude-haiku-5-5", input_tokens=50_000, output_tokens=2_000, cache_read_tokens=40_000
    )
    assert under.billed_tier == "haiku_base"
    assert under.prompt_tokens == 90_000

    same_tokens_base_tier = Decimal("0.00500000") + Decimal("0.00100000") + Decimal("0.00060000")
    assert same_tokens_base_tier == Decimal("0.00660000")
    assert over.total_cost_usd == same_tokens_base_tier * 5


def test_components_always_sum_to_the_total() -> None:
    cost = price_tokens(
        "claude-sonnet-5-5",
        input_tokens=1234,
        output_tokens=567,
        cache_read_tokens=89_012,
        cache_write_tokens=3_456,
    )
    assert (
        cost.input_cost_usd
        + cost.output_cost_usd
        + cost.cache_read_cost_usd
        + cost.cache_write_cost_usd
        == cost.total_cost_usd
    )


def test_price_usage_coalesces_the_optional_cache_fields() -> None:
    """``usage`` omits the cache fields entirely when no cache control was sent.

    They arrive as ``None``, not ``0``, so coalescing them here keeps every
    call site from having to.
    """
    usage = Usage(input_tokens=1000, output_tokens=100)
    assert usage.cache_read_input_tokens is None
    cost = price_usage("claude-haiku-5-5", usage)
    assert cost.cache_read_tokens == 0
    assert cost.cache_write_tokens == 0
    assert cost.prompt_tokens == 1000
    # 1000 x 0.10/1e6 + 100 x 0.50/1e6 = 0.0001 + 0.00005
    assert cost.total_cost_usd == Decimal("0.00015000")


def test_a_one_hour_breakpoint_costs_more_to_write() -> None:
    """2000 write tokens: 0.10 x 1.25 vs 0.10 x 2 per MTok."""
    five_min = price_tokens(
        "claude-haiku-5-5", input_tokens=0, output_tokens=0, cache_write_tokens=2000
    )
    one_hour = price_tokens(
        "claude-haiku-5-5",
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=2000,
        cache_ttl="1h",
    )
    assert five_min.cache_write_cost_usd == Decimal("0.00025000")
    assert one_hour.cache_write_cost_usd == Decimal("0.00040000")


def test_an_unpriced_model_cannot_be_metered() -> None:
    with pytest.raises(UnknownModelPriceError):
        price_tokens("claude-fable-5-1", input_tokens=1, output_tokens=1)


def test_negative_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        price_tokens("claude-haiku-5-5", input_tokens=-1, output_tokens=0)
