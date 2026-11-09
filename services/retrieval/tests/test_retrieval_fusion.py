"""RRF arithmetic, checked against values computed by hand.

The numbers in this file were worked out on paper before the code was written,
which is the only way the test can catch the code being wrong rather than
merely being self-consistent. Every expected value is written as the literal
sum of reciprocals it is, not as a decimal, so the arithmetic is auditable in
the diff.
"""

from __future__ import annotations

import pytest

from retrieval.fusion import (
    fuse_across_queries,
    fuse_hybrid,
    query_weight,
    rank_ids,
    rrf_scores,
)

K = 60


class TestHandComputedHybridFusion:
    """One query, two retrievers.

    dense  = [A, B, C]
    sparse = [C, A, D]

    By hand, with k=60 and 1-based ranks:

        A = 1/61 + 1/62 = 0.0163934... + 0.0161290... = 0.0325224...
        C = 1/63 + 1/61 = 0.0158730... + 0.0163934... = 0.0322664...
        B = 1/62         =                              0.0161290...
        D = 1/63         =                              0.0158730...

    so the fused order is A, C, B, D. This is the interesting case rather than a
    trivial one: A is first in one list and second in the other, C is first in
    one and *third* in the other, and A wins by 0.00026 - a margin that only
    exists because ranks, not scores, were summed.
    """

    DENSE = ["A", "B", "C"]
    SPARSE = ["C", "A", "D"]

    def test_scores_match_the_hand_computation(self) -> None:
        scores = fuse_hybrid(self.DENSE, self.SPARSE, k=K)

        assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["B"] == pytest.approx(1 / 62)
        assert scores["C"] == pytest.approx(1 / 63 + 1 / 61)
        assert scores["D"] == pytest.approx(1 / 63)

    def test_order_matches_the_hand_computation(self) -> None:
        assert rank_ids(fuse_hybrid(self.DENSE, self.SPARSE, k=K)) == ["A", "C", "B", "D"]

    def test_a_beats_c_by_the_hand_computed_margin(self) -> None:
        """The whole decision is this 0.00026, so it is asserted directly."""
        scores = fuse_hybrid(self.DENSE, self.SPARSE, k=K)

        margin = scores["A"] - scores["C"]

        assert margin == pytest.approx((1 / 61 + 1 / 62) - (1 / 63 + 1 / 61))
        assert margin > 0

    def test_rank_based_fusion_disagrees_with_adding_raw_scores(self) -> None:
        """Why RRF exists, as an executable statement rather than a comment.

        Give the same two rankings plausible raw scores - cosine similarities on
        the dense side, BM25 on the sparse side - and add them up:

            A = 0.82 + 9.0  =  9.82
            C = 0.42 + 14.3 = 14.72
            B = 0.80        =  0.80
            D =        2.0  =  2.00

        which orders C, A, D, B. Two differences from the RRF order A, C, B, D,
        and both are artefacts of scale rather than of evidence: C overtakes A
        purely because BM25 is unbounded above while cosine is capped at 1, and
        D - a single third-place sparse hit - overtakes B, a second-place dense
        hit, for the same reason. Nothing about the documents changed; only the
        units did. Min-max normalising first does not rescue it, because the
        constants would come from whatever else happened to be in the list.
        """
        dense_scores = {"A": 0.82, "B": 0.80, "C": 0.42}
        sparse_scores = {"C": 14.3, "A": 9.0, "D": 2.0}
        naive = {
            doc: dense_scores.get(doc, 0.0) + sparse_scores.get(doc, 0.0)
            for doc in ("A", "B", "C", "D")
        }

        assert rank_ids(naive) == ["C", "A", "D", "B"]
        assert rank_ids(fuse_hybrid(self.DENSE, self.SPARSE, k=K)) == ["A", "C", "B", "D"]


class TestHandComputedCrossQueryFusion:
    """Two queries, fused with weights 1/rank.

    Q1 (rank 1, weight 1.0)   = [A, B]
    Q2 (rank 2, weight 0.5)   = [B, A]

    By hand:

        A = 1.0/61 + 0.5/62 = 0.0163934... + 0.0080645... = 0.0244579...
        B = 1.0/62 + 0.5/61 = 0.0161290... + 0.0081967... = 0.0243257...

    A wins. The case is deliberately symmetric - each document is first for one
    query and second for the other - so the *only* thing that can separate them
    is the query weighting, and the test fails if the weights are dropped.
    """

    PER_QUERY = [(1, ["A", "B"]), (2, ["B", "A"])]

    def test_scores_match_the_hand_computation(self) -> None:
        scores = fuse_across_queries(self.PER_QUERY, k=K)

        assert scores["A"] == pytest.approx(1.0 / 61 + 0.5 / 62)
        assert scores["B"] == pytest.approx(1.0 / 62 + 0.5 / 61)

    def test_the_higher_priority_query_breaks_the_symmetry(self) -> None:
        assert rank_ids(fuse_across_queries(self.PER_QUERY, k=K)) == ["A", "B"]

    def test_without_the_weights_this_case_would_tie(self) -> None:
        """Proves the previous assertion is testing the weighting and not luck."""
        unweighted = rrf_scores([["A", "B"], ["B", "A"]], k=K)

        assert unweighted["A"] == pytest.approx(unweighted["B"])

    def test_query_rank_not_list_position_decides_the_weight(self) -> None:
        """A query list that skips rank 2 must still weight rank 3 as 1/3.

        The generator may emit fewer queries than requested, or a caller may
        drop one. Weighting by list index would silently promote the survivor.
        """
        skipped = fuse_across_queries([(1, ["A", "B"]), (3, ["B", "A"])], k=K)

        assert skipped["A"] == pytest.approx(1.0 / 61 + (1 / 3) / 62)
        assert skipped["B"] == pytest.approx(1.0 / 62 + (1 / 3) / 61)


class TestWeights:
    def test_reciprocal_of_the_rank(self) -> None:
        assert query_weight(1) == 1.0
        assert query_weight(2) == 0.5
        assert query_weight(3) == pytest.approx(1 / 3)

    def test_decays_slowly_enough_that_late_queries_still_count(self) -> None:
        """The property that made 1/r preferable to an exponential decay.

        At the top of the Day 3 sweep (N=10) the last query must still be able
        to pull a trial into the fused set on its own - that biomarker-only
        trial is what multi-query retrieval is for. 0.5^10 would be 0.001, which
        is indistinguishable from not running the query at all.
        """
        assert query_weight(10) == pytest.approx(0.1)
        assert query_weight(10) > 0.5**10 * 50

    def test_rank_zero_is_rejected_rather_than_dividing_by_zero(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            query_weight(0)


class TestRrfProperties:
    def test_documents_missing_from_a_ranking_contribute_nothing(self) -> None:
        """Not a special case - it is what lets the branches return unequal depths."""
        scores = rrf_scores([["A"], ["B", "A"]], k=K)

        assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["B"] == pytest.approx(1 / 61)

    def test_a_repeated_id_is_counted_once_at_its_best_rank(self) -> None:
        """A retriever returning a duplicate must not be able to inflate it."""
        scores = rrf_scores([["A", "A", "B"]], k=K)

        assert scores["A"] == pytest.approx(1 / 61)
        assert scores["B"] == pytest.approx(1 / 63)

    def test_larger_k_flattens_the_curve(self) -> None:
        """What the k ablation is actually varying.

        Large k pushes every 1/(k+r) toward 1/k, so rank position matters less
        and *agreement between retrievers* matters more. The ratio between the
        first and tenth position is the measurable form of that.
        """
        tight = rrf_scores([[f"D{i}" for i in range(10)]], k=1)
        flat = rrf_scores([[f"D{i}" for i in range(10)]], k=1000)

        assert tight["D0"] / tight["D9"] == pytest.approx(11 / 2)
        assert flat["D0"] / flat["D9"] == pytest.approx(1010 / 1001)

    def test_ties_break_on_id_so_results_are_reproducible(self) -> None:
        """An eval score that changes between identical runs is not a score."""
        tied = rrf_scores([["B", "A"], ["A", "B"]], k=K)

        assert tied["A"] == pytest.approx(tied["B"])
        assert rank_ids(tied) == ["A", "B"]

    def test_empty_input_is_empty_output(self) -> None:
        assert rrf_scores([], k=K) == {}
        assert rrf_scores([[], []], k=K) == {}

    def test_weights_must_match_the_rankings(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            rrf_scores([["A"], ["B"]], k=K, weights=[1.0])

    def test_k_below_one_is_rejected(self) -> None:
        """k=0 makes the top-ranked document's contribution 1/1 and k=-60 a
        division by zero at rank 60. The contract already bounds it at >= 1;
        this is the same bound where the arithmetic happens."""
        with pytest.raises(ValueError, match="rrf_k"):
            rrf_scores([["A"]], k=0)
