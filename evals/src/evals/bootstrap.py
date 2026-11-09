"""Bootstrap confidence intervals and paired significance tests over topics.

Why this module exists
----------------------
The held-out split has **50 topics**. A 0.02 difference in nDCG@10 between two
configurations on 50 topics is, in all likelihood, noise. A table of bare point
estimates quoted to three decimal places invites a reader to believe
differences that are not there, and the first reviewer who asks "is that
significant?" gets no answer. So every aggregate this harness publishes is an
interval, and every ablation row can be compared to its baseline with a paired
test that is allowed to come back "not significant".

What is resampled, and why it is topics
---------------------------------------
Topics, with replacement - never documents, and never ranks. The topics are the
independent sampling units: TREC drew 50 synthetic patients from a population
of possible patients, and the question an interval answers is "if the track had
drawn 50 *different* patients, how much would this number move?". Resampling
documents within a topic would answer a question nobody asked and would
understate the variance badly, because per-topic scores are the part that
varies.

Pairing matters
---------------
:func:`paired_bootstrap` resamples a set of topics **once** per iteration and
scores both systems on that same set. The two systems see identical patients in
every iteration, so the per-topic correlation between them - which is large,
since some patients are simply easier than others - cancels out. Comparing two
independent intervals instead would be much more conservative and would miss
real improvements.

Determinism
-----------
Pure ``random.Random`` with an explicit seed, not ``numpy``: the resampling is
then reproducible from the seed alone, independent of any library's RNG
implementation, and :func:`_percentile` is simple enough to check by hand.
Re-running the report with the same seed reproduces every interval digit for
digit.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "ConfidenceInterval",
    "PairedComparison",
    "bootstrap_mean_ci",
    "paired_bootstrap",
]

DEFAULT_RESAMPLES = 10_000
"""Resamples per interval.

10,000 is the usual floor for a percentile interval: the 2.5th percentile is
then estimated from the 250th order statistic rather than from a handful of
draws, and the interval's own Monte-Carlo wobble is well under the third
decimal place it would be printed to. 1,000 is visibly jittery between seeds;
100,000 costs ten times as much and moves nothing. Configurable everywhere -
``--resamples`` on the CLI - because a quick exploratory sweep does not need it.
"""

DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 20_261_109
"""Fixed so the report is reproducible. Changing it changes every interval in
the last decimal; it is recorded in the report header for that reason."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate with a percentile bootstrap interval around it.

    ``point`` is the statistic on the *observed* topics, not the mean of the
    resamples. Those differ slightly (the bootstrap mean carries a small bias)
    and the observed value is the one being reported, so it is the one printed.
    """

    point: float
    low: float
    high: float
    confidence: float
    resamples: int
    n_topics: int

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval sits on one side of zero.

        For a *difference* this is the significance test; for a plain metric
        mean it is nearly always true and means nothing.
        """
        return self.low > 0.0 or self.high < 0.0

    def format(self, places: int = 3) -> str:
        return f"{self.point:.{places}f} [{self.low:.{places}f}, {self.high:.{places}f}]"

    def __str__(self) -> str:
        return self.format()


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-quantile of an already-sorted sequence, by nearest rank.

    Deliberately the simplest defensible rule - index ``floor(q * n)`` clamped
    into range - rather than an interpolating one. With 10,000 resamples the
    difference between nearest-rank and linear interpolation is far below the
    printed precision, and this version can be checked by hand, which the
    tests do.

    Raises:
        ValueError: on an empty sequence or a ``q`` outside [0, 1].
    """
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    n = len(sorted_values)
    index = min(n - 1, max(0, math.floor(q * n)))
    return sorted_values[index]


def _as_pairs(values: Mapping[str, float]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Topic ids and their scores, in a fixed (sorted) order.

    Sorting is what makes a seed reproducible: dictionary order is stable
    within a process but the caller may have built the mapping in any order,
    and an unsorted iteration would pair a given seed with a different sample.
    """
    keys = tuple(sorted(values))
    return keys, tuple(values[k] for k in keys)


def bootstrap_mean_ci(
    values: Mapping[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for the mean of per-topic scores.

    Args:
        values: ``{topic_id: score}``. One entry per topic, already restricted
            to the topics where the metric is defined.

    Raises:
        ValueError: on empty ``values``, non-positive ``resamples``, or a
            ``confidence`` outside (0, 1).
    """
    if not values:
        raise ValueError("no per-topic scores to bootstrap")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    _, scores = _as_pairs(values)
    n = len(scores)
    point = sum(scores) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(scores, k=n)
        means.append(sum(sample) / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        point=point,
        low=_percentile(means, tail),
        high=_percentile(means, 1.0 - tail),
        confidence=confidence,
        resamples=resamples,
        n_topics=n,
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A variant against a baseline on the same topics."""

    baseline: str
    variant: str
    metric: str
    baseline_mean: float
    variant_mean: float
    delta: ConfidenceInterval
    p_value: float
    n_topics: int

    @property
    def significant(self) -> bool:
        """Whether the difference's interval excludes zero.

        This is the honest headline for an ablation row. ``False`` is a
        perfectly good result and should be printed as such rather than
        dropped.
        """
        return self.delta.excludes_zero

    def verdict(self) -> str:
        return "significant" if self.significant else "not significant"

    def __str__(self) -> str:
        return (
            f"{self.variant} vs {self.baseline} on {self.metric}: "
            f"delta {self.delta.format()} p={self.p_value:.4f} "
            f"({self.verdict()}, n={self.n_topics})"
        )


def paired_bootstrap(
    baseline: Mapping[str, float],
    variant: Mapping[str, float],
    *,
    metric: str = "",
    baseline_name: str = "baseline",
    variant_name: str = "variant",
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Paired bootstrap on ``variant - baseline``, topic by topic.

    The p-value is the bootstrap achieved significance level for the two-sided
    null "the true mean difference is zero". The resampled differences are
    *centred* on the observed difference, which converts the sampling
    distribution into a distribution under the null, and the p-value is the
    share of centred draws at least as extreme as what was observed::

        p = (1 + #{ |d_i - d_obs| >= |d_obs| }) / (resamples + 1)

    The ``1 +`` and ``+ 1`` are the standard add-one correction: it keeps the
    p-value away from an impossible exact 0 when no draw is more extreme, since
    a finite bootstrap can only bound such a p-value from above.

    Two identical systems therefore get ``p = 1.0`` and a zero-width interval,
    and a system that is better on every single topic gets ``p = 1/(R+1)``.

    Raises:
        ValueError: when the two mappings cover different topics, which would
            silently unpair the comparison.
    """
    if set(baseline) != set(variant):
        only_b = sorted(set(baseline) - set(variant))[:5]
        only_v = sorted(set(variant) - set(baseline))[:5]
        raise ValueError(
            "a paired test needs the same topics on both sides; "
            f"{len(set(baseline) - set(variant))} only in {baseline_name} (e.g. {only_b}), "
            f"{len(set(variant) - set(baseline))} only in {variant_name} (e.g. {only_v}). "
            "Scoring the two systems on different topic sets and subtracting the "
            "means is not a paired test."
        )
    if not baseline:
        raise ValueError("no per-topic scores to compare")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")

    keys, base_scores = _as_pairs(baseline)
    var_scores = tuple(variant[k] for k in keys)
    diffs = tuple(v - b for v, b in zip(var_scores, base_scores, strict=True))
    n = len(diffs)
    observed = sum(diffs) / n

    rng = random.Random(seed)
    indices = range(n)
    draws: list[float] = []
    extreme = 0
    for _ in range(resamples):
        # One index list per iteration, used for both systems: that is the
        # pairing. Scoring each system on its own resample would destroy it.
        picks = rng.choices(indices, k=n)
        mean_diff = sum(diffs[i] for i in picks) / n
        draws.append(mean_diff)
        if abs(mean_diff - observed) >= abs(observed):
            extreme += 1
    draws.sort()

    tail = (1.0 - confidence) / 2.0
    delta = ConfidenceInterval(
        point=observed,
        low=_percentile(draws, tail),
        high=_percentile(draws, 1.0 - tail),
        confidence=confidence,
        resamples=resamples,
        n_topics=n,
    )
    return PairedComparison(
        baseline=baseline_name,
        variant=variant_name,
        metric=metric,
        baseline_mean=sum(base_scores) / n,
        variant_mean=sum(var_scores) / n,
        delta=delta,
        p_value=(1 + extreme) / (resamples + 1),
        n_topics=n,
    )
