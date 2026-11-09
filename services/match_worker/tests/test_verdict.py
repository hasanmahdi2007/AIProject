"""Verdict derivation: in code, auditable, and not the model's to decide."""

from __future__ import annotations

from beacon_core.schemas import (
    CriterionJudgment,
    CriterionKind,
    CriterionLabel,
    NoteSentence,
    NumberedNote,
)
from match_worker.verdict import (
    citations_valid,
    derive_trial_judgment,
    final_score,
    missing_facts_from,
)

NOTE = NumberedNote(
    sentences=[NoteSentence(sentence_id=i, text=f"Sentence {i}.") for i in range(1, 6)]
)


def _judgment(
    criterion_id: str,
    kind: CriterionKind,
    label: CriterionLabel,
    *,
    cites: list[int] | None = None,
    why: str = "because",
) -> CriterionJudgment:
    return CriterionJudgment(
        criterion_id=criterion_id,
        kind=kind,
        label=label,
        evidence_sentence_ids=cites if cites is not None else [1],
        explanation=why,
    )


def test_all_rules_favourable_is_eligible() -> None:
    verdict = derive_trial_judgment(
        nct_id="NCT01",
        judgments=[
            _judgment("a", "inclusion", "met"),
            _judgment("b", "exclusion", "not_met"),
            _judgment("c", "exclusion", "not_applicable", cites=[]),
        ],
        note=NOTE,
    )
    assert verdict.category == "eligible"
    assert verdict.confirmed_fraction == 1.0
    assert verdict.missing_facts == []


def test_one_unmet_inclusion_excludes_the_trial() -> None:
    verdict = derive_trial_judgment(
        nct_id="NCT01",
        judgments=[_judgment("a", "inclusion", "not_met"), _judgment("b", "inclusion", "met")],
        note=NOTE,
    )
    assert verdict.category == "excluded"


def test_one_met_exclusion_excludes_the_trial() -> None:
    verdict = derive_trial_judgment(
        nct_id="NCT01",
        judgments=[_judgment("a", "inclusion", "met"), _judgment("b", "exclusion", "met")],
        note=NOTE,
    )
    assert verdict.category == "excluded"


def test_abstentions_become_likely_eligible_or_insufficient_info() -> None:
    mostly_known = derive_trial_judgment(
        nct_id="NCT01",
        judgments=[
            _judgment("a", "inclusion", "met"),
            _judgment("b", "inclusion", "met"),
            _judgment("c", "inclusion", "insufficient_info", why="LVEF is not stated."),
        ],
        note=NOTE,
    )
    assert mostly_known.category == "likely_eligible"
    assert mostly_known.missing_facts == ["LVEF is not stated."]

    mostly_unknown = derive_trial_judgment(
        nct_id="NCT01",
        judgments=[
            _judgment("a", "inclusion", "met"),
            _judgment("b", "inclusion", "insufficient_info", why="Stage is not stated."),
            _judgment("c", "inclusion", "insufficient_info", why="Prior therapy is not stated."),
        ],
        note=NOTE,
    )
    assert mostly_unknown.category == "insufficient_info"
    assert len(mostly_unknown.missing_facts) == 2


def test_an_uncitable_not_met_cannot_exclude_a_trial() -> None:
    """An assertion whose evidence cannot be located must not exclude a trial.

    The row keeps its label - the invalid-citation rate is a reportable
    number - but the verdict treats it as an abstention.
    """
    cited_nowhere = _judgment("a", "inclusion", "not_met", cites=[99])
    assert citations_valid(cited_nowhere, NOTE) is False

    verdict = derive_trial_judgment(
        nct_id="NCT01",
        judgments=[cited_nowhere, _judgment("b", "inclusion", "met")],
        note=NOTE,
    )
    assert verdict.category != "excluded"
    assert verdict.category == "likely_eligible"
    # The label itself is untouched.
    assert verdict.judgments[0].label == "not_met"


def test_no_judgments_is_insufficient_info_not_eligible() -> None:
    verdict = derive_trial_judgment(nct_id="NCT01", judgments=[], note=NOTE)
    assert verdict.category == "insufficient_info"
    assert verdict.confirmed_fraction == 0.0


def test_missing_facts_are_deduplicated_and_capped() -> None:
    judgments = [
        _judgment(f"c{i}", "inclusion", "insufficient_info", why="Same missing fact.")
        for i in range(15)
    ]
    assert missing_facts_from(judgments) == ["Same missing fact."]


def test_ranking_puts_verdict_first_and_retrieval_last() -> None:
    eligible_last_retrieved = derive_trial_judgment(
        nct_id="A", judgments=[_judgment("a", "inclusion", "met")], note=NOTE
    )
    excluded_first_retrieved = derive_trial_judgment(
        nct_id="B", judgments=[_judgment("a", "inclusion", "not_met")], note=NOTE
    )
    assert final_score(eligible_last_retrieved, retrieval_rank=20) > final_score(
        excluded_first_retrieved, retrieval_rank=1
    )
    # Retrieval order only breaks ties between equal verdicts.
    assert final_score(eligible_last_retrieved, retrieval_rank=1) > final_score(
        eligible_last_retrieved, retrieval_rank=2
    )
