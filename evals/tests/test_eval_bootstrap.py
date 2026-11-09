"""Bootstrap tests: degenerate cases by hand, the spread against theory.

A bootstrap cannot be pinned to a single exact number the way nDCG can, so it
is tested three ways instead:

1. **Degenerate cases with one right answer.** Constant scores must give a
   zero-width interval; a single topic must give a zero-width interval; two
   identical systems must come out not significant with ``p = 1.0``. Each of
   these is exactly computable and each catches a different class of bug.
2. **Against the analytic normal approximation.** For 0/1 scores split evenly
   over 100 topics the standard error is ``0.5 / sqrt(100) = 0.05``, so a 95%
   interval should land near ``0.5 +/- 0.098``. If the resampling were wrong -
   sampling without replacement, resampling the wrong axis, or using the wrong
   sample size - the width would be visibly wrong, and the test asserts the
   bounds fall in a window around the theoretical value.
3. **The percentile rule itself**, hand-indexed on a ten-element list.

The pairing is tested by construction: a variant that is exactly 0.05 better on
every topic has a *zero-width* paired interval, while each system on its own
has a wide one. That difference is the entire reason the comparison is paired.
"""

from __future__ import annotations

import pytest

from evals.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    _percentile,
    bootstrap_mean_ci,
    paired_bootstrap,
)

# Small resample counts keep the suite fast; the defaults are asserted
# separately so a change to them is a deliberate, visible edit.
FAST = 2_000


def test_defaults_are_the_documented_ones() -> None:
    assert DEFAULT_RESAMPLES == 10_000
    assert DEFAULT_CONFIDENCE == 0.95
    assert DEFAULT_SEED == 20_261_109


# --------------------------------------------------------- percentile rule --


def test_percentile_is_nearest_rank_on_a_hand_indexed_list() -> None:
    """``index = floor(q * n)``, clamped. Worked out for n = 10."""
    values = [float(i) for i in range(10)]
    assert _percentile(values, 0.0) == 0.0
    assert _percentile(values, 0.025) == 0.0  # floor(0.25) = 0
    assert _percentile(values, 0.5) == 5.0  # floor(5.0)  = 5
    assert _percentile(values, 0.975) == 9.0  # floor(9.75) = 9
    assert _percentile(values, 1.0) == 9.0  # clamped from 10


def test_percentile_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        _percentile([], 0.5)


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_percentile_rejects_q_outside_the_unit_interval(q: float) -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        _percentile([1.0], q)


# ------------------------------------------------------- degenerate cases --


def test_constant_scores_give_a_zero_width_interval() -> None:
    """Every resample of a constant is the same constant."""
    ci = bootstrap_mean_ci({f"t{i}": 0.42 for i in range(30)}, resamples=FAST)
    assert ci.point == pytest.approx(0.42)
    assert ci.low == pytest.approx(0.42)
    assert ci.high == pytest.approx(0.42)
    assert ci.width == pytest.approx(0.0)


def test_a_single_topic_gives_a_zero_width_interval() -> None:
    """With one topic there is nothing to resample, and the interval must say
    so rather than implying precision it does not have."""
    ci = bootstrap_mean_ci({"t1": 0.7}, resamples=100)
    assert (ci.point, ci.low, ci.high) == pytest.approx((0.7, 0.7, 0.7))
    assert ci.n_topics == 1


def test_point_estimate_is_the_observed_mean() -> None:
    """Not the mean of the resamples - the observed value is what is reported."""
    ci = bootstrap_mean_ci({"a": 0.0, "b": 1.0, "c": 0.5}, resamples=FAST)
    assert ci.point == pytest.approx(0.5)


def test_interval_brackets_the_point_estimate() -> None:
    ci = bootstrap_mean_ci({f"t{i}": i / 20 for i in range(20)}, resamples=FAST)
    assert ci.low <= ci.point <= ci.high


def test_same_seed_reproduces_the_interval_exactly() -> None:
    values = {f"t{i}": (i * 7 % 11) / 10 for i in range(25)}
    first = bootstrap_mean_ci(values, resamples=FAST, seed=99)
    second = bootstrap_mean_ci(values, resamples=FAST, seed=99)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_different_seed_moves_the_interval() -> None:
    """If it did not, the resampling is not actually random."""
    values = {f"t{i}": (i * 7 % 11) / 10 for i in range(25)}
    a = bootstrap_mean_ci(values, resamples=FAST, seed=1)
    b = bootstrap_mean_ci(values, resamples=FAST, seed=2)
    assert (a.low, a.high) != (b.low, b.high)


def test_interval_is_independent_of_the_insertion_order_of_topics() -> None:
    """Topic ids are sorted internally, so a seed pins one sample regardless of
    how the caller happened to build the mapping."""
    forward = {f"t{i}": i / 10 for i in range(10)}
    backward = {f"t{i}": i / 10 for i in reversed(range(10))}
    assert bootstrap_mean_ci(forward, resamples=FAST, seed=5).low == pytest.approx(
        bootstrap_mean_ci(backward, resamples=FAST, seed=5).low
    )


def test_more_topics_narrow_the_interval() -> None:
    """The basic sanity property: precision grows with the sample."""
    few = bootstrap_mean_ci({f"t{i}": i % 2 for i in range(10)}, resamples=FAST, seed=7)
    many = bootstrap_mean_ci({f"t{i}": i % 2 for i in range(200)}, resamples=FAST, seed=7)
    assert many.width < few.width


def test_width_matches_the_analytic_standard_error() -> None:
    """Balanced 0/1 scores on 100 topics: se = 0.5/sqrt(100) = 0.05.

    A 95% normal interval is 0.5 +/- 1.96 * 0.05 = [0.402, 0.598]. The
    percentile bootstrap should land close to that; the generous windows below
    allow Monte-Carlo wobble while still failing loudly if the resampling uses
    the wrong sample size or draws without replacement (either of which changes
    the width by a large factor, not a small one).
    """
    values = {f"t{i}": float(i % 2) for i in range(100)}
    ci = bootstrap_mean_ci(values, resamples=10_000, seed=3)
    assert ci.point == pytest.approx(0.5)
    assert 0.37 < ci.low < 0.45
    assert 0.55 < ci.high < 0.63
    assert 0.15 < ci.width < 0.23


def test_excludes_zero_is_false_for_an_interval_spanning_zero() -> None:
    assert not bootstrap_mean_ci({"a": -1.0, "b": 1.0}, resamples=FAST).excludes_zero
    assert bootstrap_mean_ci({"a": 1.0, "b": 1.0}, resamples=FAST).excludes_zero


def test_format_shows_point_and_bounds() -> None:
    ci = bootstrap_mean_ci({"a": 0.5}, resamples=10)
    assert ci.format() == "0.500 [0.500, 0.500]"


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no per-topic scores"):
        bootstrap_mean_ci({})


@pytest.mark.parametrize("resamples", [0, -5])
def test_bootstrap_rejects_non_positive_resamples(resamples: int) -> None:
    with pytest.raises(ValueError, match="resamples must be >= 1"):
        bootstrap_mean_ci({"a": 1.0}, resamples=resamples)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 2.0])
def test_bootstrap_rejects_an_impossible_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match=r"confidence must be in \(0, 1\)"):
        bootstrap_mean_ci({"a": 1.0}, confidence=confidence)


# ----------------------------------------------------------------- paired --


def test_identical_systems_are_not_significant_and_have_p_equal_to_one() -> None:
    """The case that matters most: no difference must read as no difference.

    Every centred draw is 0.0 and ``|0 - 0| >= |0|`` holds, so the add-one
    p-value is exactly 1.0 and the interval has zero width.
    """
    values = {f"t{i}": i / 10 for i in range(10)}
    result = paired_bootstrap(values, dict(values), resamples=FAST)
    assert result.delta.point == pytest.approx(0.0)
    assert result.delta.width == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.significant
    assert result.verdict() == "not significant"


def test_a_uniform_improvement_is_significant_with_the_minimum_p_value() -> None:
    """+0.05 on every topic: the delta cannot vary under resampling.

    p is then ``1 / (R + 1)`` - the smallest a bootstrap of R draws can
    justify - rather than a spuriously exact zero.
    """
    baseline = {f"t{i}": i / 10 for i in range(10)}
    variant = {k: v + 0.05 for k, v in baseline.items()}
    result = paired_bootstrap(baseline, variant, resamples=FAST)
    assert result.delta.point == pytest.approx(0.05)
    assert result.delta.low == pytest.approx(0.05)
    assert result.delta.high == pytest.approx(0.05)
    assert result.p_value == pytest.approx(1 / (FAST + 1))
    assert result.significant


def test_pairing_cancels_the_per_topic_variance() -> None:
    """The point of pairing, as a measurement.

    The two systems differ by exactly 0.05 on every topic but each has scores
    spread across the whole [0, 1] range. Paired, the difference has zero
    variance. Unpaired, each system's own interval is wide enough to swamp a
    0.05 effect - which is how a real improvement gets reported as noise.
    """
    baseline = {f"t{i}": i / 19 for i in range(20)}
    variant = {k: v + 0.05 for k, v in baseline.items()}

    paired = paired_bootstrap(baseline, variant, resamples=FAST, seed=11)
    assert paired.delta.width == pytest.approx(0.0)
    assert paired.significant

    own_width = bootstrap_mean_ci(baseline, resamples=FAST, seed=11).width
    assert own_width > 0.05 * 2, "unpaired intervals should swamp a 0.05 effect"


def test_paired_means_are_reported_alongside_the_delta() -> None:
    baseline = {"a": 0.2, "b": 0.4}
    variant = {"a": 0.3, "b": 0.5}
    result = paired_bootstrap(
        baseline,
        variant,
        metric="ndcg@10",
        baseline_name="bm25",
        variant_name="hybrid",
        resamples=FAST,
    )
    assert result.baseline_mean == pytest.approx(0.3)
    assert result.variant_mean == pytest.approx(0.4)
    assert result.delta.point == pytest.approx(0.1)
    assert result.metric == "ndcg@10"
    assert "hybrid vs bm25" in str(result)


def test_a_small_noisy_difference_comes_out_not_significant() -> None:
    """Ten topics, variant better on six and worse on four by the same amount.

    Exactly the situation the brief warns about: a visible point-estimate gap
    that an honest test must refuse to call an improvement.
    """
    baseline = {f"t{i}": 0.5 for i in range(10)}
    variant = {f"t{i}": 0.5 + (0.2 if i < 6 else -0.2) for i in range(10)}
    result = paired_bootstrap(baseline, variant, resamples=FAST, seed=13)
    assert result.delta.point == pytest.approx(0.04)
    assert not result.significant
    assert result.p_value > 0.05


def test_paired_bootstrap_rejects_unmatched_topics() -> None:
    with pytest.raises(ValueError, match="same topics on both sides"):
        paired_bootstrap({"a": 1.0}, {"b": 1.0}, resamples=10)


def test_paired_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no per-topic scores"):
        paired_bootstrap({}, {}, resamples=10)


def test_paired_bootstrap_is_reproducible_from_the_seed() -> None:
    baseline = {f"t{i}": (i % 5) / 5 for i in range(15)}
    variant = {f"t{i}": (i % 3) / 3 for i in range(15)}
    first = paired_bootstrap(baseline, variant, resamples=FAST, seed=21)
    second = paired_bootstrap(baseline, variant, resamples=FAST, seed=21)
    assert (first.delta.low, first.delta.high, first.p_value) == (
        second.delta.low,
        second.delta.high,
        second.p_value,
    )
