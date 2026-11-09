"""Retrieval metrics over graded TREC Clinical Trials qrels.

Every number that ends up on a resume or in ``README.md`` comes through this
module, so each metric is implemented here directly and then *cross-checked*
against ``ranx`` rather than delegated to it blindly. The hand-computed cases
that pin the conventions live in ``evals/tests/test_metrics.py``.

The five decisions this module encodes
--------------------------------------

**1. Grades are ordinal, and nDCG sees the integer.** TREC CT grades are
``2 = eligible``, ``1 = excluded``, ``0 = not relevant`` (see
:data:`evals.trec.RELEVANCE_LABELS`). Grade 1 means *right disease, wrong
eligibility*, which is partially correct retrieval: the pipeline is supposed to
surface those so the rule checker can rule them out. :func:`ndcg_at_k`
therefore takes the graded integer and never collapses it to a boolean.

**2. Binary metrics carry their threshold in their name.** Recall, precision
and MRR need a relevance cut, and a Recall@100 without a stated threshold is
uninterpretable. :data:`DEFAULT_RELEVANCE_THRESHOLD` is 1 - a grade-1
"excluded" trial *is* a retrieval hit, because retrieval is measured on
relevance and eligibility is the LLM's job. The stricter
:data:`ELIGIBLE_ONLY_THRESHOLD` of 2 is computed alongside it and reported as
its own row, so the pair is always visible and nobody has to guess which one a
number came from.

**3. Unjudged documents count as grade 0.** Standard TREC practice: the qrels
are a pool, not a census. A consequence worth stating out loud whenever recall
is quoted - this is *recall over the judged pool*, which is an upper bound on
nothing and a lower bound on nothing; it is simply the only recall the
benchmark can measure.

**4. Precision@k divides by k, not by the number retrieved.** A run that
returns 3 documents for a k=10 cutoff scores 0.1, not 1.0, if one is relevant.
This matches ``ranx`` and ``trec_eval``; the alternative silently rewards
short runs. Verified against ranx in ``test_precision_at_k_divides_by_k``.

**5. A topic with no relevant document is *undefined*, not zero.** Recall of
0/0 is not 0.0. Per-topic functions return ``None`` there and
:func:`score_run` records the topic as undefined and leaves it out of the mean,
reporting the count. ``ranx`` instead scores such a topic 0.0 and folds it into
the average, which drags the mean down by a factor that depends on how many
topics are unjudgeable - documented, and asserted, in
``test_ranx_scores_a_topic_with_no_relevant_docs_as_zero``. On TREC CT at
threshold 1 no topic is undefined, so the two agree; at threshold 2 the
distinction can bite.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "DEFAULT_RELEVANCE_THRESHOLD",
    "ELIGIBLE_ONLY_THRESHOLD",
    "HEADLINE_METRICS",
    "MRR",
    "NDCG_AT_10",
    "NDCG_AT_10_EXPONENTIAL",
    "PRECISION_AT_10",
    "RECALL_AT_100",
    "RECALL_AT_100_ELIGIBLE_ONLY",
    "CrossCheck",
    "GainScheme",
    "MetricKind",
    "MetricSpec",
    "RunScores",
    "crosscheck_with_ranx",
    "dcg",
    "gain_of",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_run",
]

GainScheme = Literal["linear", "exponential"]
"""How a graded relevance label becomes a gain.

``linear`` uses the grade itself (Jarvelin-Kekalainen); ``exponential`` uses
``2**grade - 1`` (Burges). Both are *graded*; they differ only in how much
extra credit a grade-2 document earns over a grade-1 one. ``linear`` is the
default because it is the formulation ``ranx`` calls ``ndcg`` and that
``trec_eval``'s ``ndcg_cut`` implements, which is what makes a Beacon number
comparable to published TREC CT track results. The exponential variant is
reported beside it rather than instead of it.
"""

MetricKind = Literal["recall", "precision", "ndcg", "mrr"]

DEFAULT_RELEVANCE_THRESHOLD = 1
"""A grade-1 ("excluded") trial counts as a retrieval hit.

Justification, which belongs next to every Recall number this harness prints:
the benchmark grades an "excluded" trial *above* "not relevant" on purpose. It
is the right disease for this patient and the patient merely fails a criterion
- exactly the case the downstream rule checker exists to catch. Penalising
retrieval for surfacing it would measure eligibility, which retrieval does not
decide, instead of relevance, which it does.
"""

ELIGIBLE_ONLY_THRESHOLD = 2
"""The strict cut: only grade-2 ("eligible") trials count as hits.

Reported alongside the threshold-1 number, never instead of it. It answers a
different and harder question - "did retrieval surface the trials this patient
could actually join?" - and it is the number to quote when someone objects
that threshold 1 is generous.
"""


def gain_of(grade: int, scheme: GainScheme) -> float:
    """The gain contributed by a document of this grade.

    Raises:
        ValueError: on a negative grade, which is never a TREC CT label and
            would otherwise produce a negative gain under ``linear``.
    """
    if grade < 0:
        raise ValueError(f"relevance grades must be >= 0, got {grade}")
    if scheme == "linear":
        return float(grade)
    return float(2**grade - 1)


def dcg(grades: Sequence[int], *, gain: GainScheme = "linear") -> float:
    """Discounted cumulative gain of a ranking, given its grades in rank order.

    The discount is ``log2(rank + 1)`` with ``rank`` 1-based, so rank 1 is
    undiscounted (``log2(2) == 1``).
    """
    return sum(
        gain_of(grade, gain) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1)
    )


def _check_ranking(ranking: Sequence[str]) -> None:
    """Reject a ranking containing the same document twice.

    A duplicated document id is always a bug in the run that produced it (a
    fusion step that forgot to deduplicate, typically), and it inflates
    precision and recall silently. Better to fail here than to publish it.
    """
    if len(set(ranking)) != len(ranking):
        counts = Counter(ranking)
        dupes = sorted(doc for doc, n in counts.items() if n > 1)
        raise ValueError(f"ranking contains {len(dupes)} duplicate document id(s): {dupes[:5]}")


def _relevant_ids(judged: Mapping[str, int], threshold: int) -> frozenset[str]:
    return frozenset(doc for doc, grade in judged.items() if grade >= threshold)


def recall_at_k(
    ranking: Sequence[str],
    judged: Mapping[str, int],
    k: int,
    *,
    threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Fraction of the topic's relevant documents found in the top ``k``.

    The denominator is every judged document at or above ``threshold``, not
    just the ones inside ``k``: this is the ceiling metric, so it has to notice
    what retrieval missed entirely.

    Returns:
        ``None`` when the topic has no document at or above ``threshold`` -
        recall is 0/0 there, not 0.0.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _check_ranking(ranking)
    relevant = _relevant_ids(judged, threshold)
    if not relevant:
        return None
    hits = sum(1 for doc in ranking[:k] if doc in relevant)
    return hits / len(relevant)


def precision_at_k(
    ranking: Sequence[str],
    judged: Mapping[str, int],
    k: int,
    *,
    threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Fraction of the top ``k`` positions holding a relevant document.

    Divided by ``k``, not by ``len(ranking)`` - see the module docstring.

    Returns:
        ``None`` when the topic has no document at or above ``threshold``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _check_ranking(ranking)
    relevant = _relevant_ids(judged, threshold)
    if not relevant:
        return None
    hits = sum(1 for doc in ranking[:k] if doc in relevant)
    return hits / k


def reciprocal_rank(
    ranking: Sequence[str],
    judged: Mapping[str, int],
    *,
    threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
    k: int | None = None,
) -> float | None:
    """``1 / rank`` of the first relevant document, or 0.0 if there is none.

    Args:
        k: optional cutoff. ``None`` scans the whole ranking, which is what
            plain "MRR" means.

    Returns:
        ``None`` when the topic has no document at or above ``threshold``;
        ``0.0`` when it has some but the run found none of them.
    """
    _check_ranking(ranking)
    relevant = _relevant_ids(judged, threshold)
    if not relevant:
        return None
    window = ranking if k is None else ranking[:k]
    for rank, doc in enumerate(window, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranking: Sequence[str],
    judged: Mapping[str, int],
    k: int,
    *,
    gain: GainScheme = "linear",
) -> float | None:
    """Graded nDCG@k: DCG of the run over DCG of the ideal ranking.

    Graded throughout. The ideal ranking is the topic's judged grades sorted
    descending and truncated to ``k``, so a grade-1 "excluded" trial
    contributes real gain (1 under ``linear``, 1 under ``exponential``) while a
    grade-2 "eligible" one contributes more (2 and 3 respectively). Documents
    in the run that the pool never judged contribute 0.

    Returns:
        ``None`` when the topic has no judged document above grade 0, which is
        where IDCG is 0 and the ratio is undefined.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _check_ranking(ranking)
    ideal_grades = sorted(judged.values(), reverse=True)[:k]
    ideal = dcg(ideal_grades, gain=gain)
    if ideal <= 0.0:
        return None
    actual = dcg([judged.get(doc, 0) for doc in ranking[:k]], gain=gain)
    return actual / ideal


# ------------------------------------------------------------ metric specs --


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One metric, named so that its conventions cannot be lost in transit.

    The ``name`` is the key used in reports and JSON; :attr:`label` is the
    human-readable form that states the threshold or the gain scheme, so a
    table cell is never just "Recall@100".
    """

    name: str
    kind: MetricKind
    k: int | None = None
    threshold: int = DEFAULT_RELEVANCE_THRESHOLD
    gain: GainScheme = "linear"

    def __post_init__(self) -> None:
        if self.kind != "mrr" and self.k is None:
            raise ValueError(f"{self.name}: {self.kind} needs a cutoff k")
        if self.k is not None and self.k < 1:
            raise ValueError(f"{self.name}: k must be >= 1, got {self.k}")
        if self.threshold < 1:
            raise ValueError(
                f"{self.name}: threshold must be >= 1 (0 would make every judged "
                f"document relevant), got {self.threshold}"
            )

    def score(self, ranking: Sequence[str], judged: Mapping[str, int]) -> float | None:
        """This metric's value for one topic, or ``None`` where it is undefined."""
        if self.kind == "recall":
            assert self.k is not None  # guaranteed by __post_init__
            return recall_at_k(ranking, judged, self.k, threshold=self.threshold)
        if self.kind == "precision":
            assert self.k is not None
            return precision_at_k(ranking, judged, self.k, threshold=self.threshold)
        if self.kind == "ndcg":
            assert self.k is not None
            return ndcg_at_k(ranking, judged, self.k, gain=self.gain)
        return reciprocal_rank(ranking, judged, threshold=self.threshold, k=self.k)

    @property
    def is_graded(self) -> bool:
        """Whether the metric reads the integer grade rather than a threshold."""
        return self.kind == "ndcg"

    @property
    def label(self) -> str:
        """Display name that carries the convention with it."""
        if self.kind == "ndcg":
            return f"nDCG@{self.k} (graded, {self.gain} gains)"
        cut = "grade >= 2, eligible only" if self.threshold >= 2 else "grade >= 1, incl. excluded"
        head = {
            "recall": f"Recall@{self.k}",
            "precision": f"P@{self.k}",
            "mrr": "MRR" if self.k is None else f"MRR@{self.k}",
        }[self.kind]
        return f"{head} ({cut})"

    @property
    def ranx_name(self) -> str:
        """The equivalent ``ranx`` metric string, for the cross-check.

        ``ranx`` binarises recall, precision and MRR at grade >= 1, so a
        threshold of 2 is expressed by pre-filtering the qrels rather than by
        this name; see :func:`crosscheck_with_ranx`.
        """
        if self.kind == "ndcg":
            stem = "ndcg" if self.gain == "linear" else "ndcg_burges"
            return f"{stem}@{self.k}"
        if self.kind == "mrr":
            return "mrr"
        return f"{self.kind}@{self.k}"


RECALL_AT_100 = MetricSpec(
    name="recall@100", kind="recall", k=100, threshold=DEFAULT_RELEVANCE_THRESHOLD
)
"""The metric that governs the system: 375,580 -> 100 is where a good trial is
lost for good, and nothing downstream can recover it."""

RECALL_AT_100_ELIGIBLE_ONLY = MetricSpec(
    name="recall@100_eligible_only", kind="recall", k=100, threshold=ELIGIBLE_ONLY_THRESHOLD
)

NDCG_AT_10 = MetricSpec(name="ndcg@10", kind="ndcg", k=10, gain="linear")
NDCG_AT_10_EXPONENTIAL = MetricSpec(name="ndcg@10_exp", kind="ndcg", k=10, gain="exponential")
PRECISION_AT_10 = MetricSpec(name="p@10", kind="precision", k=10)
MRR = MetricSpec(name="mrr", kind="mrr")

HEADLINE_METRICS: tuple[MetricSpec, ...] = (
    RECALL_AT_100,
    RECALL_AT_100_ELIGIBLE_ONLY,
    NDCG_AT_10,
    NDCG_AT_10_EXPONENTIAL,
    PRECISION_AT_10,
    MRR,
)
"""Recall first, because it is the ceiling on everything after it.

Both recall thresholds and both nDCG gain schemes are in the default set
deliberately: computing them together costs nothing and removes any chance of a
reader having to guess which convention produced a published figure.
"""


# -------------------------------------------------------------- scoring run --


@dataclass(frozen=True, slots=True)
class RunScores:
    """Per-topic metric values for one run, plus what could not be scored."""

    per_topic: dict[str, dict[str, float]] = field(default_factory=dict)
    """``{metric_name: {topic_id: value}}``, containing only defined values."""

    undefined: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """``{metric_name: topic_ids}`` the metric was undefined for and that are
    therefore excluded from its mean."""

    topic_ids: tuple[str, ...] = ()
    specs: tuple[MetricSpec, ...] = ()

    def values(self, metric: str) -> dict[str, float]:
        return self.per_topic[metric]

    def mean(self, metric: str) -> float | None:
        """Unweighted mean over the topics where the metric is defined."""
        scored = self.per_topic[metric]
        if not scored:
            return None
        return sum(scored.values()) / len(scored)

    def n_scored(self, metric: str) -> int:
        return len(self.per_topic[metric])


def score_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    specs: Iterable[MetricSpec] = HEADLINE_METRICS,
    *,
    require_exact_topics: bool = True,
) -> RunScores:
    """Score a run - ``{topic_id: ranked doc ids}`` - against graded qrels.

    Args:
        require_exact_topics: default ``True``, and it should stay that way.
            A run missing topics that the qrels judge is the quiet way to
            inflate every mean in a report: scoring 60 of 75 topics and
            averaging over 60 reports the easy 60. ``ranx`` guards the same
            mistake with an assertion and offers ``make_comparable=True`` to
            paper over it; this harness makes you say so explicitly.

    Raises:
        ValueError: when the topic sets differ and ``require_exact_topics``.
    """
    specs = tuple(specs)
    run_topics, qrel_topics = set(run), set(qrels)
    if require_exact_topics and run_topics != qrel_topics:
        missing = sorted(qrel_topics - run_topics)[:5]
        extra = sorted(run_topics - qrel_topics)[:5]
        raise ValueError(
            "run and qrels cover different topics: "
            f"{len(qrel_topics - run_topics)} judged topic(s) absent from the run "
            f"(e.g. {missing}), {len(run_topics - qrel_topics)} unjudged topic(s) in "
            f"the run (e.g. {extra}). Averaging over only the topics present would "
            "silently change the denominator; fix the run or pass "
            "require_exact_topics=False and read n_scored."
        )

    topics = tuple(sorted(run_topics & qrel_topics))
    per_topic: dict[str, dict[str, float]] = {s.name: {} for s in specs}
    undefined: dict[str, list[str]] = {s.name: [] for s in specs}
    for topic in topics:
        judged = qrels[topic]
        ranking = run[topic]
        for spec in specs:
            value = spec.score(ranking, judged)
            if value is None:
                undefined[spec.name].append(topic)
            else:
                per_topic[spec.name][topic] = value
    return RunScores(
        per_topic=per_topic,
        undefined={name: tuple(ids) for name, ids in undefined.items()},
        topic_ids=topics,
        specs=specs,
    )


# ----------------------------------------------------------- ranx crosscheck --


@dataclass(frozen=True, slots=True)
class CrossCheck:
    """One metric computed twice: here, and by ``ranx``."""

    metric: str
    mine: float
    ranx: float
    n_topics: int
    tolerance: float = 1e-9

    @property
    def abs_diff(self) -> float:
        return abs(self.mine - self.ranx)

    @property
    def agrees(self) -> bool:
        return self.abs_diff <= self.tolerance

    def __str__(self) -> str:
        verdict = "agree" if self.agrees else "DISAGREE"
        return (
            f"{self.metric}: mine={self.mine:.6f} ranx={self.ranx:.6f} "
            f"diff={self.abs_diff:.2e} ({verdict}, n={self.n_topics})"
        )


def _ranx_inputs(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    topics: Sequence[str],
    spec: MetricSpec,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """Build the ``ranx`` qrels/run dicts for one metric.

    Two translations happen here, and both are the reason this function exists
    rather than being inlined:

    * ``ranx`` binarises recall/precision/MRR at grade >= 1. A threshold of 2
      is therefore expressed by *dropping* grade-1 documents from the qrels,
      which is exactly equivalent: a dropped document is unjudged, and
      unjudged counts as not relevant.
    * ``ranx`` ranks by score, so a ranked list becomes strictly decreasing
      scores. Equal scores would let ``ranx`` break ties however it likes and
      the comparison would be against a different ranking than the one scored.
    """
    if spec.is_graded:
        ranx_qrels = {t: {d: g for d, g in qrels[t].items() if g > 0} for t in topics}
    else:
        ranx_qrels = {t: {d: g for d, g in qrels[t].items() if g >= spec.threshold} for t in topics}
    ranx_run = {t: {doc: float(len(run[t]) - i) for i, doc in enumerate(run[t])} for t in topics}
    return ranx_qrels, ranx_run


def crosscheck_with_ranx(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    specs: Iterable[MetricSpec] = HEADLINE_METRICS,
) -> tuple[CrossCheck, ...]:
    """Recompute each metric with ``ranx`` and compare means.

    An independent second implementation is the cheapest insurance available
    against a metric that quietly computes the wrong thing, so the report runs
    this on every scored run and prints the result.

    The comparison is restricted, per metric, to the topics where this module's
    value is *defined* and the run is non-empty. ``ranx`` scores an undefined
    topic 0.0 and averages it in (asserted in the tests), so including those
    topics would compare two different quantities and the mismatch would mean
    nothing.
    """
    from ranx import Qrels as RanxQrels
    from ranx import Run as RanxRun
    from ranx import evaluate

    scores = score_run(run, qrels, specs, require_exact_topics=False)
    checks: list[CrossCheck] = []
    for spec in scores.specs:
        defined = [t for t in scores.per_topic[spec.name] if run[t]]
        ranx_qrels, ranx_run = _ranx_inputs(run, qrels, defined, spec)
        usable = [t for t in defined if ranx_qrels[t] and ranx_run[t]]
        if not usable:
            continue
        mine = sum(scores.per_topic[spec.name][t] for t in usable) / len(usable)
        theirs = float(
            evaluate(
                RanxQrels.from_dict({t: ranx_qrels[t] for t in usable}),
                RanxRun.from_dict({t: ranx_run[t] for t in usable}),
                spec.ranx_name,
            )
        )
        checks.append(CrossCheck(metric=spec.name, mine=mine, ranx=theirs, n_topics=len(usable)))
    return tuple(checks)
