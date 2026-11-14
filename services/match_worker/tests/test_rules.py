"""The eligibility splitter, against the shapes TREC actually contains.

The invariant every test here defends: one rule in, one rule out. A criterion
split mid-sentence is unanswerable, and a model asked to judge half a sentence
answers anyway.
"""

from __future__ import annotations

import pytest

from match_worker.rules import criterion_id_for, split_eligibility


def _texts(parsed: list[object]) -> list[str]:
    return [item.text for item in parsed]  # type: ignore[attr-defined]


def test_bulleted_sections_split_by_kind() -> None:
    parsed = split_eligibility(
        "NCT01",
        "Inclusion Criteria:\n"
        "- Age 18 years or older\n"
        "- Histologically confirmed metastatic breast cancer\n"
        "Exclusion Criteria:\n"
        "- Active central nervous system metastases\n",
    )
    assert [(p.criterion_id, p.kind) for p in parsed] == [
        ("NCT01:inc:01", "inclusion"),
        ("NCT01:inc:02", "inclusion"),
        ("NCT01:exc:01", "exclusion"),
    ]
    assert parsed[1].text == "Histologically confirmed metastatic breast cancer"


def test_numbered_items_and_uppercase_headers() -> None:
    parsed = split_eligibility(
        "NCT02",
        "INCLUSION CRITERIA\n"
        "1. Measurable disease\n"
        "2) At least one prior line of chemotherapy\n"
        "EXCLUSION CRITERIA\n"
        "(1) Ejection fraction below 50 percent\n",
    )
    assert len(parsed) == 3
    assert parsed[0].text == "Measurable disease"
    assert parsed[2].kind == "exclusion"
    assert parsed[2].text == "Ejection fraction below 50 percent"


def test_continuation_lines_stay_with_their_rule() -> None:
    """The whole point of the splitter: never cut a rule in half."""
    parsed = split_eligibility(
        "NCT03",
        "Inclusion Criteria:\n"
        "- Adequate organ function as defined by\n"
        "  a creatinine clearance above 60 mL/min\n"
        "- ECOG 0 or 1\n",
    )
    assert len(parsed) == 2
    assert parsed[0].text == (
        "Adequate organ function as defined by a creatinine clearance above 60 mL/min"
    )


def test_run_on_paragraph_with_inline_header_is_not_filed_as_inclusion() -> None:
    """A missed inline header files every exclusion rule as an inclusion rule,
    which inverts the verdict for that trial."""
    parsed = split_eligibility(
        "NCT04",
        "Inclusion criteria: Postmenopausal women with metastatic breast cancer. "
        "Exclusion criteria: Pregnant or nursing patients. Active infection.",
    )
    kinds = {p.kind for p in parsed}
    assert kinds == {"inclusion", "exclusion"}
    exclusions = [p.text for p in parsed if p.kind == "exclusion"]
    assert "Pregnant or nursing patients." in exclusions
    assert "Active infection." in exclusions


@pytest.mark.parametrize(
    "header",
    [
        "Inclusion Criteria/Exclusion Criteria:",
        "Inclusion/Exclusion Criteria:",
        "Inclusion and Exclusion Criteria:",
        "INCLUSION & EXCLUSION CRITERIA",
    ],
)
def test_a_combined_header_files_its_list_as_inclusion(header: str) -> None:
    """One header over one list is a wrapper, not an exclusion section.

    Regression for NCT02491775: the ``Exclusion Criteria:`` inside the combined
    header filed all nine requirements as exclusions, so "At least 18 years of
    age" - which the patient met - excluded the trial.
    """
    parsed = split_eligibility(
        "NCT09",
        f"{header}\n\n"
        "  -  Diagnosis of metastatic stage IIIB/IV lung adenocarcinoma\n\n"
        "  -  Not pregnant or breastfeeding\n\n"
        "  -  At least 18 years of age\n",
    )
    assert [p.kind for p in parsed] == ["inclusion"] * 3
    assert _texts(list(parsed)) == [
        "Diagnosis of metastatic stage IIIB/IV lung adenocarcinoma",
        "Not pregnant or breastfeeding",
        "At least 18 years of age",
    ]


def test_a_combined_header_still_yields_to_a_real_exclusion_section() -> None:
    parsed = split_eligibility(
        "NCT10",
        "Inclusion/Exclusion Criteria:\n"
        "- At least 18 years of age\n"
        "Exclusion Criteria:\n"
        "- Active infection\n",
    )
    assert [(p.kind, p.text) for p in parsed] == [
        ("inclusion", "At least 18 years of age"),
        ("exclusion", "Active infection"),
    ]


def test_unmarked_sentences_split_on_sentence_boundaries() -> None:
    parsed = split_eligibility(
        "NCT05",
        "Eligibility:\nPostmenopausal women with metastatic disease. "
        "Total bilirubin below 1.5 times the upper limit of normal.",
    )
    assert len(parsed) == 2
    assert all(p.kind == "inclusion" for p in parsed)


def test_empty_or_missing_text_yields_nothing() -> None:
    assert split_eligibility("NCT06", None) == []
    assert split_eligibility("NCT06", "   ") == []


def test_fragments_below_the_floor_are_dropped() -> None:
    parsed = split_eligibility("NCT07", "Inclusion Criteria:\n- Age:\n- Both\n- ECOG 0 or 1\n")
    assert _texts(list(parsed)) == ["ECOG 0 or 1"]


def test_criterion_ids_are_stable_and_readable() -> None:
    assert criterion_id_for("NCT01234567", "inclusion", 3) == "NCT01234567:inc:03"
    assert criterion_id_for("NCT01234567", "exclusion", 12) == "NCT01234567:exc:12"
    # Stability is what makes a redelivered message a no-op: the ID is part of
    # uq_criterion_judgments_identity.
    assert (
        split_eligibility("NCT08", "Inclusion Criteria:\n- ECOG 0 or 1\n")[0].criterion_id
        == split_eligibility("NCT08", "Inclusion Criteria:\n- ECOG 0 or 1\n")[0].criterion_id
    )
