"""Metric tests, built on ranked lists whose scores were computed by hand.

Why hand-computed cases and not an end-to-end check
---------------------------------------------------
A metric tested against a live retrieval system is tested against whatever
that system returned, and such a test cannot distinguish a broken metric from
a broken retriever - it passes either way as long as the number looks
plausible. Every metric here is instead pinned to a tiny ranking whose correct
value was worked out on paper, so a regression is arithmetic, not opinion.

``ranx`` is then used as an *independent second implementation* and asserted
to agree. That ordering is deliberate: ranx verifies this module, and this
module verifies ranx, and the one thing neither can do is quietly agree on the
wrong convention, because the hand values decide which convention is right.

THE WORKED EXAMPLE
------------------
Used by most of the tests below. One topic, four judged documents, one
retrieved document the pool never judged::

    qrels   = {d1: 2, d2: 1, d3: 0, d4: 2}      (eligible, excluded, not rel, eligible)
    ranking = [d2, d5, d1, d3]                   (d5 is unjudged -> grade 0)

Grades in rank order: ``[1, 0, 2, 0]``. Discount at rank r is ``log2(r + 1)``.

*Linear gains* (gain = grade), truncated at k=3::

    DCG@3  = 1/log2(2) + 0/log2(3) + 2/log2(4) = 1 + 0 + 1 = 2.0
    ideal grades sorted desc, truncated to 3   = [2, 2, 1]
    IDCG@3 = 2/1 + 2/1.5849625 + 1/2 = 2 + 1.26185951 + 0.5 = 3.76185951
    nDCG@3 = 2.0 / 3.76185951 = 0.5316519652587917

*Exponential gains* (gain = 2^grade - 1)::

    DCG@3  = 1/1 + 0 + 3/2 = 2.5
    IDCG@3 = 3/1 + 3/1.5849625 + 1/2 = 3 + 1.89278926 + 0.5 = 5.39278926
    nDCG@3 = 2.5 / 5.39278926 = 0.4635819942403664

*Binary metrics at threshold 1* (relevant = {d1, d2, d4}, three documents)::

    top 3 = [d2, d5, d1] -> hits d2, d1 = 2
    Recall@3 = 2/3        P@3 = 2/3        first hit at rank 1 -> RR = 1.0

*Binary metrics at threshold 2* (relevant = {d1, d4}, two documents)::

    top 3 hits = {d1} = 1
    Recall@3 = 1/2        P@3 = 1/3        first hit at rank 3 -> RR = 1/3
"""

from __future__ import annotations

import math
import random

import pytest

from evals.metrics import (
    DEFAULT_RELEVANCE_THRESHOLD,
    ELIGIBLE_ONLY_THRESHOLD,
    HEADLINE_METRICS,
    MRR,
    NDCG_AT_10,
    NDCG_AT_10_EXPONENTIAL,
    PRECISION_AT_10,
    RECALL_AT_100,
    RECALL_AT_100_ELIGIBLE_ONLY,
    MetricSpec,
    crosscheck_with_ranx,
    dcg,
    gain_of,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_run,
)

# The worked example from the module docstring.
JUDGED = {"d1": 2, "d2": 1, "d3": 0, "d4": 2}
RANKING = ["d2", "d5", "d1", "d3"]

NDCG3_LINEAR = 0.5316519652587917
NDCG3_EXPONENTIAL = 0.4635819942403664


# ------------------------------------------------------------------- gains --


def test_gain_of_linear_is_the_grade_itself() -> None:
    assert gain_of(0, "linear") == 0.0
    assert gain_of(1, "linear") == 1.0
    assert gain_of(2, "linear") == 2.0


def test_gain_of_exponential_is_two_to_the_grade_minus_one() -> None:
    assert gain_of(0, "exponential") == 0.0
    assert gain_of(1, "exponential") == 1.0
    assert gain_of(2, "exponential") == 3.0


def test_both_gain_schemes_rank_excluded_above_not_relevant() -> None:
    """The point of graded qrels: grade 1 must earn more than grade 0.

    If this ever fails, nDCG has been collapsed to something binary and the
    distinction the benchmark was built to express is gone.
    """
    for scheme in ("linear", "exponential"):
        assert gain_of(2, scheme) > gain_of(1, scheme) > gain_of(0, scheme)


def test_gain_of_rejects_a_negative_grade() -> None:
    with pytest.raises(ValueError, match="grades must be >= 0"):
        gain_of(-1, "linear")


def test_dcg_discount_is_log2_of_rank_plus_one() -> None:
    """Rank 1 is undiscounted; rank 3 is divided by exactly 2."""
    assert dcg([5], gain="linear") == 5.0
    assert dcg([0, 0, 4], gain="linear") == 2.0
    # Hand: 1/log2(2) + 0/log2(3) + 2/log2(4) = 1 + 0 + 1
    assert dcg([1, 0, 2], gain="linear") == pytest.approx(2.0)
    # Hand: 1/1 + 0 + 3/2
    assert dcg([1, 0, 2], gain="exponential") == pytest.approx(2.5)


# -------------------------------------------------------------------- nDCG --


def test_ndcg_at_3_linear_matches_the_hand_computation() -> None:
    assert ndcg_at_k(RANKING, JUDGED, 3, gain="linear") == pytest.approx(NDCG3_LINEAR)


def test_ndcg_at_3_exponential_matches_the_hand_computation() -> None:
    assert ndcg_at_k(RANKING, JUDGED, 3, gain="exponential") == pytest.approx(NDCG3_EXPONENTIAL)


def test_ndcg_is_rederived_from_the_formula_independently() -> None:
    """Re-derive the expected value from the definition, not from the literal.

    Written out separately from :func:`evals.metrics.ndcg_at_k` so that a bug
    in the implementation's loop structure cannot be mirrored by the same bug
    in the expectation.
    """
    expected = (1 / math.log2(2) + 0 / math.log2(3) + 2 / math.log2(4)) / (
        2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    )
    assert ndcg_at_k(RANKING, JUDGED, 3, gain="linear") == pytest.approx(expected)


def test_ndcg_rewards_a_grade_one_hit_over_nothing() -> None:
    """An "excluded" trial at rank 1 must score above an unjudged one.

    This is the grade-1 policy stated as a test: retrieving a
    right-disease-wrong-eligibility trial is partially correct retrieval, so it
    cannot score the same as retrieving junk.
    """
    judged = {"a": 2, "b": 1}
    excluded_hit = ndcg_at_k(["b"], judged, 1, gain="linear")
    nothing = ndcg_at_k(["zzz"], judged, 1, gain="linear")
    assert excluded_hit is not None and nothing == 0.0
    assert excluded_hit > 0.0


def test_ndcg_of_the_ideal_ranking_is_one() -> None:
    judged = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], judged, 3, gain="linear") == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "c"], judged, 3, gain="exponential") == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking() -> None:
    """Hand: DCG([1,2]) / IDCG([2,1]) with linear gains."""
    judged = {"a": 2, "b": 1}
    expected = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], judged, 2, gain="linear") == pytest.approx(expected)
    assert expected < 1.0


def test_ndcg_ideal_is_truncated_to_k() -> None:
    """IDCG@1 uses only the single best grade, so one perfect hit scores 1.0.

    An IDCG computed over *all* judged documents instead of the top k would
    make a perfect top-1 ranking score far below 1.0, which is the classic
    nDCG implementation bug.
    """
    judged = {"a": 2, "b": 2, "c": 2, "d": 2}
    assert ndcg_at_k(["a"], judged, 1, gain="linear") == pytest.approx(1.0)


def test_ndcg_treats_unjudged_documents_as_grade_zero() -> None:
    judged = {"a": 2}
    assert ndcg_at_k(["unjudged", "a"], judged, 2, gain="linear") == pytest.approx(
        (0 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2))
    )


def test_ndcg_is_undefined_when_no_document_is_relevant() -> None:
    """IDCG is 0 there, and 0/0 is not 0.0."""
    assert ndcg_at_k(["a"], {"a": 0, "b": 0}, 10) is None


# ------------------------------------------------------- recall / precision --


def test_recall_at_3_threshold_one_matches_the_hand_computation() -> None:
    assert recall_at_k(RANKING, JUDGED, 3, threshold=1) == pytest.approx(2 / 3)


def test_recall_at_3_threshold_two_matches_the_hand_computation() -> None:
    assert recall_at_k(RANKING, JUDGED, 3, threshold=2) == pytest.approx(0.5)


def test_recall_default_threshold_counts_an_excluded_trial_as_a_hit() -> None:
    """The documented default: grade 1 is a retrieval hit.

    Retrieval is measured on relevance; eligibility is the LLM's job. A
    threshold of 2 would penalise retrieval for surfacing exactly the trials
    the rule checker exists to rule out.
    """
    assert DEFAULT_RELEVANCE_THRESHOLD == 1
    assert ELIGIBLE_ONLY_THRESHOLD == 2
    judged = {"excluded_trial": 1}
    assert recall_at_k(["excluded_trial"], judged, 10) == pytest.approx(1.0)
    assert recall_at_k(["excluded_trial"], judged, 10, threshold=2) is None


def test_recall_denominator_counts_relevant_docs_outside_k() -> None:
    """The ceiling metric has to notice what retrieval missed entirely."""
    judged = {"a": 2, "b": 2, "c": 2, "d": 2}
    assert recall_at_k(["a"], judged, 1) == pytest.approx(0.25)


def test_precision_at_k_divides_by_k_not_by_the_number_retrieved() -> None:
    """A 3-document run at k=10 with one hit scores 0.1, not 0.33 or 1.0.

    Dividing by ``len(ranking)`` would reward a system that returns almost
    nothing, which is the opposite of what P@10 is for. Matches ranx and
    trec_eval; cross-checked below.
    """
    judged = {"a": 2, "b": 2}
    assert precision_at_k(["a", "x", "y"], judged, 10) == pytest.approx(0.1)


def test_precision_at_3_matches_the_hand_computation_at_both_thresholds() -> None:
    assert precision_at_k(RANKING, JUDGED, 3, threshold=1) == pytest.approx(2 / 3)
    assert precision_at_k(RANKING, JUDGED, 3, threshold=2) == pytest.approx(1 / 3)


def test_reciprocal_rank_matches_the_hand_computation_at_both_thresholds() -> None:
    assert reciprocal_rank(RANKING, JUDGED, threshold=1) == pytest.approx(1.0)
    assert reciprocal_rank(RANKING, JUDGED, threshold=2) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_found() -> None:
    """Distinct from ``None``: there *were* relevant documents and the run
    missed them all, which is a real score of 0.0."""
    assert reciprocal_rank(["x", "y"], {"a": 2}) == 0.0


def test_binary_metrics_are_undefined_with_no_relevant_documents() -> None:
    judged = {"a": 0, "b": 0}
    assert recall_at_k(["a"], judged, 10) is None
    assert precision_at_k(["a"], judged, 10) is None
    assert reciprocal_rank(["a"], judged) is None


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_a_non_positive_cutoff(k: int) -> None:
    for fn in (recall_at_k, precision_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="k must be >= 1"):
            fn(RANKING, JUDGED, k)


def test_a_duplicated_document_in_a_ranking_is_rejected() -> None:
    """Deduplication bugs in fusion inflate recall and precision silently."""
    with pytest.raises(ValueError, match="duplicate document id"):
        recall_at_k(["d1", "d1"], JUDGED, 10)


# --------------------------------------------------------------- MetricSpec --


def test_headline_metrics_state_their_conventions_in_their_labels() -> None:
    """A table cell must never read just "Recall@100"."""
    assert "grade >= 1" in RECALL_AT_100.label
    assert "grade >= 2" in RECALL_AT_100_ELIGIBLE_ONLY.label
    assert "linear" in NDCG_AT_10.label
    assert "exponential" in NDCG_AT_10_EXPONENTIAL.label
    assert all("graded" in s.label for s in (NDCG_AT_10, NDCG_AT_10_EXPONENTIAL))


def test_headline_set_leads_with_recall_at_100() -> None:
    """Recall@100 is the ceiling on everything downstream, so it comes first."""
    assert HEADLINE_METRICS[0] is RECALL_AT_100


def test_headline_set_reports_both_recall_thresholds_and_both_gain_schemes() -> None:
    """Computing both removes any chance of guessing which produced a figure."""
    names = {s.name for s in HEADLINE_METRICS}
    assert {"recall@100", "recall@100_eligible_only", "ndcg@10", "ndcg@10_exp"} <= names


def test_metric_spec_requires_a_cutoff_for_cut_metrics() -> None:
    with pytest.raises(ValueError, match="needs a cutoff"):
        MetricSpec(name="bad", kind="recall")


def test_metric_spec_rejects_a_zero_threshold() -> None:
    """Threshold 0 would make every judged document relevant, including the
    ones explicitly judged not relevant."""
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        MetricSpec(name="bad", kind="recall", k=10, threshold=0)


def test_metric_spec_score_dispatches_to_the_right_metric() -> None:
    assert MetricSpec(name="r", kind="recall", k=3).score(RANKING, JUDGED) == pytest.approx(2 / 3)
    assert MetricSpec(name="p", kind="precision", k=3).score(RANKING, JUDGED) == pytest.approx(
        2 / 3
    )
    assert MetricSpec(name="n", kind="ndcg", k=3).score(RANKING, JUDGED) == pytest.approx(
        NDCG3_LINEAR
    )
    assert MetricSpec(name="m", kind="mrr").score(RANKING, JUDGED) == pytest.approx(1.0)


def test_ranx_names_encode_the_gain_scheme() -> None:
    """ranx calls the linear variant ``ndcg`` and the exponential one
    ``ndcg_burges``; getting these the wrong way round would make the
    cross-check compare two different metrics and pass anyway."""
    assert NDCG_AT_10.ranx_name == "ndcg@10"
    assert NDCG_AT_10_EXPONENTIAL.ranx_name == "ndcg_burges@10"
    assert RECALL_AT_100.ranx_name == "recall@100"
    assert PRECISION_AT_10.ranx_name == "precision@10"
    assert MRR.ranx_name == "mrr"


# ---------------------------------------------------------------- score_run --


def test_score_run_scores_every_topic_and_metric() -> None:
    """Hand values: t1 retrieves 2 of its 3 relevant docs (d4 is missed
    entirely, which is exactly what Recall@100 exists to notice); t2 retrieves
    1 of 2. Mean = (2/3 + 1/2) / 2 = 7/12."""
    run = {"t1": RANKING, "t2": ["d1"]}
    qrels = {"t1": JUDGED, "t2": {"d1": 2, "d9": 2}}
    scores = score_run(run, qrels, [RECALL_AT_100, NDCG_AT_10])
    assert scores.values("recall@100") == {
        "t1": pytest.approx(2 / 3),
        "t2": pytest.approx(0.5),
    }
    assert scores.mean("recall@100") == pytest.approx(7 / 12)
    assert scores.n_scored("recall@100") == 2


def test_score_run_rejects_a_run_missing_judged_topics() -> None:
    """The quiet way to inflate a report: score the topics that worked."""
    with pytest.raises(ValueError, match="different topics"):
        score_run({"t1": ["d1"]}, {"t1": {"d1": 2}, "t2": {"d2": 2}})


def test_score_run_rejects_a_run_with_unjudged_topics() -> None:
    with pytest.raises(ValueError, match="different topics"):
        score_run({"t1": ["d1"], "t9": ["d1"]}, {"t1": {"d1": 2}})


def test_score_run_records_undefined_topics_instead_of_scoring_them_zero() -> None:
    """A topic with no grade-2 document is undefined for eligible-only recall.

    Scoring it 0.0 would drag the mean down by a factor that depends on how
    many topics are unjudgeable - a number that moves with the pool, not with
    the system.
    """
    run = {"t1": ["a"], "t2": ["b"]}
    qrels = {"t1": {"a": 2}, "t2": {"b": 1}}
    scores = score_run(run, qrels, [RECALL_AT_100_ELIGIBLE_ONLY])
    assert scores.undefined["recall@100_eligible_only"] == ("t2",)
    assert scores.n_scored("recall@100_eligible_only") == 1
    assert scores.mean("recall@100_eligible_only") == pytest.approx(1.0)


def test_score_run_subset_is_allowed_only_when_asked_for() -> None:
    scores = score_run(
        {"t1": ["d1"]}, {"t1": {"d1": 2}, "t2": {"d2": 2}}, require_exact_topics=False
    )
    assert scores.topic_ids == ("t1",)


# ------------------------------------------------------------- ranx checks --


def test_crosscheck_agrees_with_ranx_on_the_worked_example() -> None:
    """Two independent implementations, one hand-computed truth.

    This is the test that makes the rest trustworthy: ranx is a widely used
    library, and if it and this module agree on every headline metric, a
    convention error in either would have to be the same error in both.
    """
    run = {"t1": RANKING}
    qrels = {"t1": JUDGED}
    checks = crosscheck_with_ranx(run, qrels, HEADLINE_METRICS)
    assert checks, "no metric was cross-checked"
    for check in checks:
        assert check.agrees, str(check)


def test_crosscheck_confirms_the_hand_values_through_ranx() -> None:
    """Pin the hand numbers to ranx's output, metric by metric."""
    checks = {
        c.metric: c
        for c in crosscheck_with_ranx(
            {"t1": RANKING},
            {"t1": JUDGED},
            [
                MetricSpec(name="ndcg@3", kind="ndcg", k=3, gain="linear"),
                MetricSpec(name="ndcg@3_exp", kind="ndcg", k=3, gain="exponential"),
                MetricSpec(name="recall@3", kind="recall", k=3),
                MetricSpec(name="recall@3_strict", kind="recall", k=3, threshold=2),
                MetricSpec(name="p@3", kind="precision", k=3),
            ],
        )
    }
    assert checks["ndcg@3"].ranx == pytest.approx(NDCG3_LINEAR)
    assert checks["ndcg@3_exp"].ranx == pytest.approx(NDCG3_EXPONENTIAL)
    assert checks["recall@3"].ranx == pytest.approx(2 / 3)
    assert checks["recall@3_strict"].ranx == pytest.approx(0.5)
    assert checks["p@3"].ranx == pytest.approx(2 / 3)


def test_crosscheck_agrees_with_ranx_on_a_larger_random_run() -> None:
    """Twenty synthetic topics, so agreement is not an artefact of one example.

    Seeded, so a failure is reproducible. The rankings deliberately include
    unjudged documents and topics whose relevant documents are missed
    entirely.
    """
    rng = random.Random(4242)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, list[str]] = {}
    for topic in range(20):
        topic_id = f"t{topic}"
        docs = [f"d{topic}_{i}" for i in range(40)]
        qrels[topic_id] = {doc: rng.choice([0, 0, 1, 2]) for doc in docs[:30]}
        if not any(g > 0 for g in qrels[topic_id].values()):
            qrels[topic_id][docs[0]] = 2
        ranking = docs[:]
        rng.shuffle(ranking)
        run[topic_id] = ranking[:25]
    for check in crosscheck_with_ranx(run, qrels, HEADLINE_METRICS):
        assert check.agrees, str(check)


def test_ranx_scores_a_topic_with_no_relevant_docs_as_zero() -> None:
    """Documents the one place this module deliberately differs from ranx.

    ranx folds an unjudgeable topic into the mean as 0.0; this harness leaves
    it out and reports the count. The divergence is recorded here so nobody
    later "fixes" one to match the other without knowing the trade-off.
    """
    from ranx import Qrels, Run, evaluate

    qrels = Qrels.from_dict({"t1": {"d1": 2}, "t2": {"d9": 0}})
    run = Run.from_dict({"t1": {"d1": 1.0}, "t2": {"d8": 1.0}})
    assert float(evaluate(qrels, run, "recall@3")) == pytest.approx(0.5)

    mine = score_run(
        {"t1": ["d1"], "t2": ["d8"]},
        {"t1": {"d1": 2}, "t2": {"d9": 0}},
        [MetricSpec(name="recall@3", kind="recall", k=3)],
    )
    assert mine.mean("recall@3") == pytest.approx(1.0)
    assert mine.undefined["recall@3"] == ("t2",)
