"""Tests that pin the Wave 0 contracts.

These are not busywork. Each one guards a specific silent failure that would
otherwise only show up as a flat ablation number three days later.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from beacon_core import events
from beacon_core.schemas import (
    AgeSexFilter,
    CriterionJudgment,
    MatchState,
    NoteSentence,
    NumberedNote,
    PatientProfile,
    RankedQuery,
    SearchRequest,
    TrialCandidate,
    TrialJudgment,
)


class TestCriterionJudgment:
    """The frozen LLM output schema."""

    def test_has_exactly_the_five_frozen_fields(self) -> None:
        # Adding a field changes the JSON schema sent to Claude, busts the
        # cached prompt prefix and invalidates every recorded fixture. If this
        # test fails, that is a decision, not a refactor.
        assert set(CriterionJudgment.model_fields) == {
            "criterion_id",
            "kind",
            "label",
            "evidence_sentence_ids",
            "explanation",
        }

    def test_insufficient_info_is_a_valid_label(self) -> None:
        # Abstaining is a success state, not an error path.
        j = CriterionJudgment(
            criterion_id="NCT01:inc:01",
            kind="inclusion",
            label="insufficient_info",
            evidence_sentence_ids=[],
            explanation="The note does not mention ECOG status.",
        )
        assert j.label == "insufficient_info"

    def test_unknown_label_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriterionJudgment(
                criterion_id="NCT01:inc:01",
                kind="inclusion",
                label="probably_met",
                explanation="",
            )

    def test_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriterionJudgment(
                criterion_id="NCT01:inc:01",
                kind="inclusion",
                label="met",
                explanation="",
                confidence=0.9,  # type: ignore[call-arg]
            )


class TestNullSafety:
    """Unknown must mean possibly eligible, never excluded."""

    def test_filters_include_unknown_by_default(self) -> None:
        # ClinicalTrials.gov age fields are often absent or unparseable. A
        # wrongly excluded trial never appears anywhere for anyone to notice,
        # which makes this the more dangerous direction of the two.
        assert AgeSexFilter().include_unknown is True

    def test_trial_age_bounds_may_be_unknown(self) -> None:
        t = TrialCandidate(nct_id="NCT00000001")
        assert t.min_age_years is None
        assert t.max_age_years is None
        assert t.sex is None

    def test_patient_age_may_be_unknown(self) -> None:
        assert PatientProfile().age is None


class TestRetrievalContract:
    def test_defaults_match_the_design(self) -> None:
        req = SearchRequest(queries=[RankedQuery(rank=1, text="nsclc")])
        assert req.fuse_top_k == 100, "bi-encoder stage keeps 100"
        assert req.rerank_top_k == 20, "cross-encoder stage keeps 20"
        assert req.rrf_k == 60, "RRF constant k=60"
        assert req.use_rerank is True

    def test_at_least_one_query_is_required(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(queries=[])


class TestNumberedNote:
    def test_valid_ids_are_the_citation_targets(self) -> None:
        note = NumberedNote(
            sentences=[
                NoteSentence(sentence_id=1, text="58-year-old woman with NSCLC."),
                NoteSentence(sentence_id=2, text="EGFR exon 19 deletion."),
            ]
        )
        assert note.valid_ids == frozenset({1, 2})

    def test_a_judgment_citing_a_missing_sentence_is_detectable_by_code(self) -> None:
        # The type cannot prevent this; the verifier must. What matters is that
        # it is checkable without an LLM, and the invalid-citation rate must
        # come out at 0%.
        note = NumberedNote(sentences=[NoteSentence(sentence_id=1, text="x")])
        judgment = CriterionJudgment(
            criterion_id="NCT01:inc:01",
            kind="inclusion",
            label="met",
            evidence_sentence_ids=[1, 7],
            explanation="",
        )
        bad = [i for i in judgment.evidence_sentence_ids if i not in note.valid_ids]
        assert bad == [7]

    def test_sentence_ids_start_at_one(self) -> None:
        with pytest.raises(ValidationError):
            NoteSentence(sentence_id=0, text="x")


class TestMatchState:
    def test_starts_queued_and_empty(self) -> None:
        state = MatchState(match_id="m1", patient_note="note")
        assert state.status == "queued"
        assert state.candidates == []
        assert state.judgments == []
        assert state.total_cost_usd == 0.0

    def test_round_trips_through_json(self) -> None:
        # LangGraph checkpoints this to Postgres after every node, so it has to
        # survive serialisation exactly.
        state = MatchState(
            match_id="m1",
            patient_note="58-year-old woman with NSCLC.",
            profile=PatientProfile(age=58, sex="female", conditions=["NSCLC"]),
            queries=[RankedQuery(rank=1, text="advanced NSCLC", facet="diagnosis")],
            candidates=[TrialCandidate(nct_id="NCT00000001", title="A trial")],
            judgments=[TrialJudgment(nct_id="NCT00000001")],
            status="running",
            current_step="retrieve",
        )
        again = MatchState.model_validate_json(state.model_dump_json())
        assert again == state


class TestEvents:
    def test_topic_names_are_stable(self) -> None:
        assert events.TOPIC_MATCH_REQUESTED == "match.requested"
        assert events.TOPIC_MATCH_PROGRESS == "match.progress"
        assert events.TOPIC_MATCH_COMPLETED == "match.completed"
        assert events.TOPIC_MATCH_DLQ == "match.dlq"
        assert len(events.ALL_TOPICS) == 6

    def test_every_event_carries_an_id_for_idempotency(self) -> None:
        # Kafka is at-least-once: the same message will occasionally arrive
        # twice, and a unique constraint on this ID is what makes the repeat a
        # no-op instead of a duplicate result.
        a = events.MatchRequested(match_id="m1")
        b = events.MatchRequested(match_id="m1")
        assert a.event_id and b.event_id
        assert a.event_id != b.event_id

    def test_progress_pct_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            events.MatchProgress(match_id="m1", step="retrieve", pct=101)
