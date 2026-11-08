"""Age and sex parsing, against the shapes the real corpus contains."""

from __future__ import annotations

import pytest

from ingestion._testing import AGE_CASES, REAL_RECORDS, SEX_CASES, record
from ingestion.parsers import (
    normalise_phase,
    normalise_status,
    parse_age,
    parse_age_bounds,
    parse_sex,
)


@pytest.mark.parametrize(("raw", "expected"), AGE_CASES)
def test_parse_age(raw: str | None, expected: float | None) -> None:
    assert parse_age(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), SEX_CASES)
def test_parse_sex(raw: str | None, expected: str | None) -> None:
    assert parse_sex(raw) == expected


def test_months_are_exact_twelfths() -> None:
    """``6 Months`` is 0.5, not 0.4931 (30.44 days) and not 0 or 1."""
    assert parse_age("6 Months") == 0.5
    assert parse_age("18 Months") == 1.5


def test_sub_year_units_do_not_collapse_to_zero() -> None:
    """A neonatal bound must stay distinguishable from "no bound"."""
    for raw in ("2 Months", "20 Weeks", "90 Days", "24 Hours", "30 Minutes"):
        value = parse_age(raw)
        assert value is not None
        assert 0.0 < value < 1.0


# --- the constraint the whole brief turns on --------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "65 Years and older",
        "18-65 Years",
        "at least 18 Years",
        "adult",
        "18",
        "eighteen years",
        "N/A",
        "",
        None,
    ],
)
def test_unparseable_age_is_none_never_a_default(raw: str | None) -> None:
    """Unknown must mean "possibly eligible", so it must be ``None``.

    This is the test that matters most in this file. A parser that returned 0
    for an unreadable minimum, or 999 for an unreadable maximum, would make
    every numeric filter answer confidently and wrongly, and the only visible
    symptom would be trials that never appear. ``None`` reaches Postgres as
    ``NULL`` and Qdrant as an empty payload field, which B2's filters let
    through.
    """
    parsed = parse_age(raw)
    assert parsed is None
    assert parsed != 0
    assert parsed != 0.0


def test_unparseable_bounds_keep_the_raw_strings_and_are_reported() -> None:
    bounds = parse_age_bounds("65 Years and older", "about 80")
    assert bounds.min_years is None
    assert bounds.max_years is None
    # The evidence a parser bug is diagnosed from survives in the row.
    assert bounds.raw_min == "65 Years and older"
    assert bounds.raw_max == "about 80"
    # ...and the failure is counted rather than conflated with "N/A".
    assert bounds.unparsed == ("min", "max")


def test_not_applicable_is_not_counted_as_a_parse_failure() -> None:
    """``N/A`` is the record declining to state a bound, not a parser defect."""
    bounds = parse_age_bounds("N/A", "N/A")
    assert (bounds.min_years, bounds.max_years) == (None, None)
    assert bounds.unparsed == ()


def test_absent_eligibility_element_yields_all_nulls() -> None:
    """NCT00026039 has no ``<eligibility>`` element at all."""
    fixture = record("NCT00026039")
    assert (fixture.raw_gender, fixture.raw_min_age, fixture.raw_max_age) == (None, None, None)
    bounds = parse_age_bounds(fixture.raw_min_age, fixture.raw_max_age)
    assert (bounds.min_years, bounds.max_years, bounds.unparsed) == (None, None, ())
    assert parse_sex(fixture.raw_gender) is None


def test_inverted_bounds_are_flagged_not_repaired() -> None:
    bounds = parse_age_bounds("65 Years", "18 Years")
    assert bounds.inverted is True
    assert (bounds.min_years, bounds.max_years) == (65.0, 18.0)


def test_sane_bounds_are_not_flagged_as_inverted() -> None:
    assert parse_age_bounds("18 Years", "65 Years").inverted is False
    assert parse_age_bounds("N/A", "65 Years").inverted is False
    assert parse_age_bounds("18 Years", "N/A").inverted is False


# --- real records -----------------------------------------------------------


def test_real_paediatric_record() -> None:
    """NCT00000363 runs from 2 months to 1 year."""
    fixture = record("NCT00000363")
    bounds = parse_age_bounds(fixture.raw_min_age, fixture.raw_max_age)
    assert bounds.min_years == pytest.approx(2 / 12, abs=1e-6)
    assert bounds.max_years == 1.0


def test_real_record_with_week_bound_and_no_minimum() -> None:
    """NCT00000163: ``N/A`` minimum, ``20 Weeks`` maximum."""
    fixture = record("NCT00000163")
    bounds = parse_age_bounds(fixture.raw_min_age, fixture.raw_max_age)
    assert bounds.min_years is None
    assert bounds.max_years == pytest.approx(20 * 7 / 365.25, abs=1e-6)


def test_every_real_record_parses_without_failure() -> None:
    """None of the frozen records trips the parser's failure path."""
    for nct_id, fixture in REAL_RECORDS.items():
        bounds = parse_age_bounds(fixture.raw_min_age, fixture.raw_max_age)
        assert bounds.unparsed == (), nct_id
        assert parse_sex(fixture.raw_gender) in {None, "all", "male", "female"}, nct_id


# --- phase / status ---------------------------------------------------------


def test_phase_na_is_preserved_and_absent_is_none() -> None:
    """``N/A`` (127,333 records) and absent (82,931) are different facts."""
    assert normalise_phase("N/A") == "N/A"
    assert normalise_phase(None) is None
    assert normalise_phase("  Phase 1/Phase 2 ") == "Phase 1/Phase 2"
    assert normalise_phase("   ") is None


def test_status_is_kept_verbatim() -> None:
    assert normalise_status("Active, not recruiting") == "Active, not recruiting"
    assert normalise_status("Unknown  status") == "Unknown status"
    assert normalise_status(None) is None
