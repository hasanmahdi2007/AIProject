"""The assembly behind ``GET /v1/matches/{match_id}``, without a database.

``_rank_rows`` and ``_to_ranked_trial`` are pure functions over ORM instances,
so they are tested here directly. The route's five queries are tested against a
real Postgres in ``test_integration_match_result.py`` - faking
``session.execute`` would only assert that the fake agrees with itself.

The ORM objects below are built but never flushed, which is deliberate: column
defaults like ``missing_facts=list`` are applied by the INSERT, not by the
constructor, so an unflushed row returns ``None`` for them. That is exactly the
state a half-written run is read in, and it is why the assembly guards every
JSON column with ``or []``.
"""

from __future__ import annotations

from beacon_core.models import CriterionJudgmentRow, Trial, TrialMatch
from gateway.router import _rank_rows, _to_ranked_trial


def _pair(nct_id: str, **kwargs: object) -> TrialMatch:
    return TrialMatch(match_run_id="run-1", nct_id=nct_id, **kwargs)


# --------------------------------------------------------------- _rank_rows --


def test_the_workers_final_ranking_is_preserved_exactly() -> None:
    rows = [
        _pair("NCT003", final_rank=3),
        _pair("NCT001", final_rank=1),
        _pair("NCT002", final_rank=2),
    ]

    assert [(rank, row.nct_id) for rank, row in _rank_rows(rows)] == [
        (1, "NCT001"),
        (2, "NCT002"),
        (3, "NCT003"),
    ]


def test_unranked_rows_are_numbered_after_the_ranked_ones() -> None:
    """A run read mid-flight: ``rank_and_explain`` has not set ``final_rank``.

    ``RankedTrial.rank`` is ``ge=1``, so these rows cannot be passed through as
    NULL, and they must not be numbered 1 and 2 either - that would claim a
    final ordering the worker has not decided.
    """
    rows = [
        _pair("NCT009", retrieval_rank=2),
        _pair("NCT001", final_rank=1),
        _pair("NCT008", retrieval_rank=1),
    ]

    assert [(rank, row.nct_id) for rank, row in _rank_rows(rows)] == [
        (1, "NCT001"),
        (2, "NCT008"),
        (3, "NCT009"),
    ]


def test_every_reported_rank_is_unique_and_at_least_one() -> None:
    """``rank`` is ``ge=1`` and two trials sharing a rank is a display bug."""
    rows = [_pair(f"NCT{i:03d}") for i in range(5)] + [_pair("NCT900", final_rank=1)]

    ranks = [rank for rank, _ in _rank_rows(rows)]

    assert len(ranks) == len(set(ranks)) == 6
    assert min(ranks) >= 1


def test_a_nonsense_final_rank_is_treated_as_unranked() -> None:
    """Rather than raising out of a read-only route. ``final_rank`` has no
    CHECK constraint, so a worker bug could write 0 or a negative; the contract
    forbids it as a ``rank``. Renumbering keeps the row visible - the judgments
    are still real - instead of 500ing the whole response."""
    rows = [_pair("NCT002", final_rank=0), _pair("NCT001", final_rank=1)]

    assert [(rank, row.nct_id) for rank, row in _rank_rows(rows)] == [
        (1, "NCT001"),
        (2, "NCT002"),
    ]


def test_no_trials_yet_is_an_empty_list_not_an_error() -> None:
    assert _rank_rows([]) == []


# ---------------------------------------------------------- _to_ranked_trial --


def test_a_judged_pair_becomes_the_contract_shape() -> None:
    row = _pair(
        "NCT04179175",
        final_rank=1,
        final_score=0.91,
        rrf_score=0.031,
        rerank_score=7.4,
        retrieval_rank=3,
        category="likely_eligible",
        confirmed_fraction=0.75,
        missing_facts=["no ECOG recorded"],
        summary="Matches on histology and biomarker.",
    )
    trial = Trial(
        nct_id="NCT04179175",
        title="Osimertinib in EGFR-mutant NSCLC",
        conditions=["non-small cell lung cancer"],
        interventions=["osimertinib"],
        phase="Phase 3",
        status="recruiting",
        min_age_years=18.0,
        max_age_years=None,
        sex="all",
    )
    judged = [
        CriterionJudgmentRow(
            match_run_id="run-1",
            nct_id="NCT04179175",
            criterion_id="INC-2",
            kind="inclusion",
            label="met",
            evidence_sentence_ids=[4, 5],
            explanation="EGFR exon 19 deletion is stated.",
        )
    ]

    ranked = _to_ranked_trial(1, row, trial, judged)

    assert ranked.rank == 1
    assert ranked.score == 0.91
    assert ranked.candidate.title == "Osimertinib in EGFR-mutant NSCLC"
    assert ranked.candidate.sex == "all"
    assert ranked.candidate.max_age_years is None, "unknown bound must stay unknown"
    assert ranked.candidate.retrieval_rank == 3
    assert ranked.judgment.category == "likely_eligible"
    assert ranked.judgment.confirmed_fraction == 0.75
    assert ranked.judgment.missing_facts == ["no ECOG recorded"]
    assert [j.criterion_id for j in ranked.judgment.judgments] == ["INC-2"]
    assert ranked.judgment.judgments[0].evidence_sentence_ids == [4, 5]


def test_the_two_per_leg_scores_are_not_invented() -> None:
    """``trial_matches`` stores the fused and reranked scores only. Defaulting
    the dense and sparse scores to 0.0 would read as "scored zero" rather than
    "not recorded", and an ablation table built on that is wrong."""
    ranked = _to_ranked_trial(1, _pair("NCT001", rrf_score=0.03), None, [])

    assert ranked.candidate.dense_score is None
    assert ranked.candidate.sparse_score is None
    assert ranked.candidate.rrf_score == 0.03


def test_an_unjudged_pair_reports_insufficient_info_not_eligible() -> None:
    """``category`` is NULL until the rules are checked. The default must be the
    abstention, never a verdict: this is a medical-decision-support surface and
    an unjudged trial that renders as eligible is the worst possible default."""
    ranked = _to_ranked_trial(1, _pair("NCT001"), None, [])

    assert ranked.judgment.category == "insufficient_info"
    assert ranked.judgment.confirmed_fraction == 0.0
    assert ranked.judgment.judgments == []
    assert ranked.judgment.missing_facts == []


def test_a_missing_trial_row_is_still_reported() -> None:
    """If the trial vanished from under a finished run, dropping the pair would
    make ``trial_count`` disagree with what the worker judged. The judgments are
    the part that matters; the descriptive fields degrade to empty."""
    ranked = _to_ranked_trial(2, _pair("NCT404", final_score=0.5), None, [])

    assert ranked.candidate.nct_id == "NCT404"
    assert ranked.candidate.title == ""
    assert ranked.candidate.conditions == []
    assert ranked.candidate.min_age_years is None


def test_a_retrieval_rank_of_zero_is_dropped_rather_than_sent() -> None:
    """``TrialCandidate.retrieval_rank`` is ``ge=1``; 0 would raise mid-response."""
    ranked = _to_ranked_trial(1, _pair("NCT001", retrieval_rank=0), None, [])

    assert ranked.candidate.retrieval_rank is None
