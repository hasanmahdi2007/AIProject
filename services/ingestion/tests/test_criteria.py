"""The rule splitter, against real messy records from the corpus.

Every assertion here is about a record that exists. The point of the splitter
is that it cuts where the author cut, so the tests check two things over and
over: that the expected number of rules came out, and that no rule is a
fragment of a sentence.
"""

from __future__ import annotations

import pytest

from ingestion._testing import REAL_RECORDS, record
from ingestion.criteria import (
    MAX_CRITERIA_PER_TRIAL,
    PARSER_RULES,
    PARSER_UNSTRUCTURED,
    SplitResult,
    criterion_id,
    split_eligibility,
)


def _split(nct_id: str) -> SplitResult:
    return split_eligibility(nct_id, record(nct_id).criteria)


# --- the common shape -------------------------------------------------------


def test_dash_bullets_under_both_headers() -> None:
    """NCT00000102 -- two inclusions, two exclusions, nothing else."""
    result = _split("NCT00000102")
    assert [c.kind for c in result.criteria] == [
        "inclusion",
        "inclusion",
        "exclusion",
        "exclusion",
    ]
    assert result.criteria[0].text == "diagnosed with Congenital Adrenal Hyperplasia (CAH)"
    assert result.criteria[2].text == "history of liver disease, or elevated liver function tests"
    assert all(c.parser == PARSER_RULES for c in result.criteria)
    assert result.had_headers is True
    assert result.marker_style == "dash"


def test_wrapped_bullet_is_rejoined_not_split() -> None:
    """A rule wrapped over two source lines is one rule, not two.

    NCT00000105's fifth inclusion runs onto a continuation line. Splitting on
    the newline would produce ``"...approved by the Committee on the Use of
    Human"`` -- a rule no patient can be judged against.
    """
    result = _split("NCT00000105")
    texts = [c.text for c in result.criteria]
    assert (
        "All patients must sign informed consent approved by the Committee on the "
        "Use of Human Subjects at the University of Minnesota" in texts
    )
    assert not any(t.endswith(" Human") for t in texts)
    assert len(result.criteria) == 12


def test_criterion_ids_are_stable_and_per_kind() -> None:
    result = _split("NCT00000105")
    inclusions = [c for c in result.criteria if c.kind == "inclusion"]
    exclusions = [c for c in result.criteria if c.kind == "exclusion"]
    assert inclusions[0].criterion_id == "NCT00000105:inc:01"
    assert inclusions[-1].criterion_id == "NCT00000105:inc:05"
    assert exclusions[0].criterion_id == "NCT00000105:exc:01"
    assert exclusions[-1].criterion_id == "NCT00000105:exc:07"
    # position is a single sequence across the trial, in source order.
    assert [c.position for c in result.criteria] == list(range(12))


def test_criterion_id_helper() -> None:
    assert criterion_id("NCT01234567", "inclusion", 3) == "NCT01234567:inc:03"
    assert criterion_id("NCT01234567", "exclusion", 12) == "NCT01234567:exc:12"


# --- awkward real shapes ----------------------------------------------------


def test_numbered_list_with_no_headers() -> None:
    """NCT00000162 -- a numbered list, no headers, with a lead-in line."""
    result = _split("NCT00000162")
    assert result.had_headers is False
    assert result.marker_style == "number"
    assert len(result.criteria) == 4
    # The lead-in survives as its own item rather than being glued to rule 1.
    assert result.criteria[0].text == "Patients with three types of diagnoses were accepted:"
    assert result.criteria[1].text == "major BVO without neovascularization"
    # Un-headed criteria default to inclusion, the safe direction.
    assert {c.kind for c in result.criteria} == {"inclusion"}


def test_bulleted_section_header() -> None:
    """NCT00001231 -- the header is itself a bullet (``-  INCLUSION CRITERIA:``).

    If the header were treated as an ordinary bullet, every rule under it
    would be folded into one giant "rule" whose text began with the words
    "INCLUSION CRITERIA".
    """
    result = _split("NCT00001231")
    assert result.had_headers is True
    assert not any("INCLUSION CRITERIA" in c.text for c in result.criteria)
    assert [c.kind for c in result.criteria].count("inclusion") == 3
    assert [c.kind for c in result.criteria].count("exclusion") == 6


def test_nested_sub_bullets_stay_with_their_lead_in() -> None:
    """A sub-list belongs to the clause that announces it.

    ``Perimenopausal Subjects:`` on its own is not a rule, and its seven
    children are meaningless without knowing which population they apply to,
    so they are folded into one self-contained criterion.
    """
    result = _split("NCT00001231")
    first = result.criteria[0].text
    assert first.startswith("Perimenopausal Subjects:")
    assert "History within the last one year" in first
    assert "Plasma FSH levels" in first or "plasma FSH levels" in first
    # Not emitted separately.
    assert not any(c.text == "Perimenopausal Subjects:" for c in result.criteria)
    assert not any(c.text.startswith("Age 40-65") for c in result.criteria)


def test_mixed_bullet_characters_at_different_indents_are_siblings() -> None:
    """NCT00000306 mixes a ``•`` with ``-`` inside one exclusion list.

    The bullet character and the indent both change, but the items are peers:
    the lead-in test (does the open item end in a colon?) is what tells them
    apart from a real sub-list.
    """
    result = _split("NCT00000306")
    exclusions = [c for c in result.criteria if c.kind == "exclusion"]
    assert len(exclusions) == 6
    assert exclusions[0].text.startswith("Have a serious medical illness")
    assert exclusions[-1].text == "Have been in any outside treatment in 3 months"


def test_header_with_trailing_clause_is_still_a_header() -> None:
    """``Inclusion Criteria- Subject must:`` opens a section."""
    result = _split("NCT00000306")
    assert result.had_headers is True
    assert not any("Subject must" in c.text for c in result.criteria)
    assert [c.kind for c in result.criteria].count("inclusion") == 5


def test_lowercase_header_with_inline_first_rule() -> None:
    """NCT00000134 -- ``inclusion criteria: Males and females...`` on one line."""
    result = _split("NCT00000134")
    assert result.had_headers is True
    assert len(result.criteria) == 2
    assert result.criteria[0].kind == "inclusion"
    assert result.criteria[0].text.startswith("Males and females eligible for the CRRT")
    assert result.criteria[1].kind == "exclusion"
    assert result.criteria[1].text.startswith("history of intolerance to ganciclovir")
    # No markers anywhere, so this is the prose fallback -- and it says so.
    assert all(c.parser == PARSER_UNSTRUCTURED for c in result.criteria)
    assert result.is_fallback is True


def test_headers_without_bullets_split_on_paragraphs() -> None:
    """NCT00000244 -- one rule per paragraph under each header."""
    result = _split("NCT00000244")
    assert [(c.kind, c.text) for c in result.criteria] == [
        ("inclusion", "opiate addict between the ages of 18-55"),
        ("exclusion", "Regular abuse of other drugs, unstable medical conditions"),
    ]
    assert all(c.parser == PARSER_UNSTRUCTURED for c in result.criteria)


def test_free_prose_is_kept_whole_not_chunked() -> None:
    """NCT00000163 -- no headers, no markers, one sentence.

    The fallback keeps it whole. A 512-character chunker would cut it at
    ``"...who are between 4 and 20 weeks of"``, which reads as a rule and is
    not one.
    """
    result = _split("NCT00000163")
    assert len(result.criteria) == 1
    only = result.criteria[0]
    assert only.parser == PARSER_UNSTRUCTURED
    assert only.text.endswith("are eligible for CEOS.")
    assert only.kind == "inclusion"


def test_boilerplate_is_not_a_criterion() -> None:
    """NCT00000187's entire criteria field is "Please contact site for information."."""
    result = _split("NCT00000187")
    assert result.criteria == ()


def test_missing_criteria_and_missing_eligibility_yield_nothing() -> None:
    assert _split("NCT00000180").criteria == ()  # <eligibility> but no <criteria>
    assert _split("NCT00026039").criteria == ()  # no <eligibility> at all
    assert split_eligibility("NCT00000000", None).criteria == ()
    assert split_eligibility("NCT00000000", "   \n\n ").criteria == ()


def test_bullets_interleaved_with_unbulleted_paragraphs() -> None:
    """NCT00002558 -- a long list with prose wedged between bullets.

    The prose joins the clause above it (which ends in a colon) instead of
    becoming a headless fragment.
    """
    result = _split("NCT00002558")
    texts = [c.text for c in result.criteria]
    assert any(
        t.startswith("Patients with advanced GCT, including patients with:")
        and "measurable or evaluable disease" in t
        for t in texts
    )
    assert "measurable or evaluable disease" not in texts


def test_a_stray_shallower_bullet_does_not_swallow_the_list_above_it() -> None:
    """Taking the *minimum* indent as the top level breaks NCT00002558.

    Its list sits at one indent and a later bullet sits shallower; with the
    minimum as the top level everything before that line collapses into a
    single item. The first marker's indent is the top level instead.
    """
    result = _split("NCT00002558")
    assert len(result.criteria) > 15
    assert result.criteria[0].text.startswith("Male/Female with histologically confirmed GCT")
    assert not result.criteria[0].text.startswith("- ")


# --- invariants over every frozen record ------------------------------------


@pytest.mark.parametrize("nct_id", sorted(REAL_RECORDS))
def test_no_criterion_is_blank_or_punctuation(nct_id: str) -> None:
    for c in _split(nct_id).criteria:
        assert c.text.strip() == c.text
        assert len(c.text) >= 3
        assert any(ch.isalnum() for ch in c.text)


@pytest.mark.parametrize("nct_id", sorted(REAL_RECORDS))
def test_no_criterion_ends_mid_word(nct_id: str) -> None:
    """A rule may end without punctuation, but never inside a word.

    Because the splitter only cuts at author-made boundaries, every criterion
    ends where a line or a list item ended -- so the last character is never
    a hyphen continuation and the text never ends with an opening bracket.
    """
    for c in _split(nct_id).criteria:
        assert not c.text.endswith("-")
        assert not c.text.endswith("(")


@pytest.mark.parametrize("nct_id", sorted(REAL_RECORDS))
def test_criterion_ids_are_unique_within_a_trial(nct_id: str) -> None:
    ids = [c.criterion_id for c in _split(nct_id).criteria]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("nct_id", sorted(REAL_RECORDS))
def test_split_is_deterministic(nct_id: str) -> None:
    """Idempotency starts here: the same text must give the same rules."""
    assert _split(nct_id) == _split(nct_id)


@pytest.mark.parametrize("nct_id", sorted(REAL_RECORDS))
def test_every_kind_satisfies_the_check_constraint(nct_id: str) -> None:
    for c in _split(nct_id).criteria:
        assert c.kind in {"inclusion", "exclusion"}
        assert c.parser in {PARSER_RULES, PARSER_UNSTRUCTURED}


@pytest.mark.parametrize("nct_id", sorted(REAL_RECORDS))
def test_no_content_is_silently_dropped(nct_id: str) -> None:
    """Every alphanumeric run in the source turns up in some criterion.

    Weaker than "the texts concatenate back", because headers, markers and
    boilerplate are removed on purpose -- but strong enough to catch the real
    hazard, a branch of the splitter that quietly discards a whole section.
    """
    source = record(nct_id).criteria or ""
    result = _split(nct_id)
    if not result.criteria:
        return
    joined = " ".join(c.text for c in result.criteria)
    # Sample long words, which are content rather than list scaffolding. The
    # header vocabulary is excluded because removing it is the job.
    scaffolding = {"inclusion", "exclusion", "criteria", "eligibility"}
    words = {
        w for w in source.split() if len(w) > 8 and w.isalpha() and w.lower() not in scaffolding
    }
    missing = {w for w in words if w not in joined}
    assert not missing, f"{nct_id} dropped {sorted(missing)[:5]}"


def test_runaway_record_is_capped_and_says_so() -> None:
    text = "Inclusion Criteria:\n\n" + "\n".join(
        f"  -  rule number {n} about something" for n in range(MAX_CRITERIA_PER_TRIAL + 50)
    )
    result = split_eligibility("NCT09999999", text)
    assert len(result.criteria) == MAX_CRITERIA_PER_TRIAL
    assert result.truncated is True


def test_a_rule_mentioning_exclusion_is_not_read_as_a_header() -> None:
    text = (
        "Inclusion Criteria:\n\n"
        "  -  No exclusion criteria: all comers are eligible\n"
        "  -  Age over 18\n"
    )
    result = split_eligibility("NCT09999998", text)
    assert {c.kind for c in result.criteria} == {"inclusion"}
    assert result.criteria[0].text.startswith("No exclusion criteria:")


def test_neutral_eligibility_wrapper_is_not_a_rule() -> None:
    text = "ELIGIBILITY CRITERIA:\n\n  -  Age over 18\n\nExclusion Criteria:\n\n  -  Pregnancy\n"
    result = split_eligibility("NCT09999997", text)
    assert [c.text for c in result.criteria] == ["Age over 18", "Pregnancy"]
    assert [c.kind for c in result.criteria] == ["inclusion", "exclusion"]
