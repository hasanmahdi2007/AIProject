"""Deriving the Postgres row and the Qdrant payload from a real record."""

from __future__ import annotations

from typing import Any

from ingestion._testing import REAL_RECORDS, WordTokenizer, as_raw_trial, record
from ingestion.corpus import RawTrial
from ingestion.overview import OverviewBuilder
from ingestion.pipeline import DerivedTrial, IngestStats, derive_trial

BUILDER = OverviewBuilder(WordTokenizer(), max_tokens=512)
"""A stub tokenizer, so these tests never download the 130 MB ONNX model.

The real tokenizer is exercised by the integration path and by the ingest run
itself; what is under test here is the derivation, not BERT word-piecing.
"""


def _derive(nct_id: str, **kwargs: Any) -> DerivedTrial:
    return derive_trial(as_raw_trial(nct_id, **kwargs), BUILDER)


# --- the NULL rule, end to end ----------------------------------------------


def test_absent_age_reaches_both_stores_as_null() -> None:
    """NCT00026039 has no ``<eligibility>`` element at all.

    The row and the payload must both say "unknown", because B2 reads the
    payload and a human reads the row, and they have to agree.
    """
    derived = _derive("NCT00026039")
    row = derived.trial_row()
    payload = derived.qdrant_payload()

    assert row["min_age_years"] is None
    assert row["max_age_years"] is None
    assert row["sex"] is None
    assert payload["min_age_years"] is None
    assert payload["max_age_years"] is None
    assert payload["sex"] is None
    # Not 0, not 999, not "unknown".
    assert row["min_age_years"] != 0
    assert row["max_age_years"] != 999


def test_na_age_reaches_both_stores_as_null_but_keeps_the_raw_string() -> None:
    """NCT00000104 states ``N/A`` for both bounds."""
    derived = _derive("NCT00000104")
    row = derived.trial_row()
    assert (row["min_age_years"], row["max_age_years"]) == (None, None)
    assert (row["raw_min_age"], row["raw_max_age"]) == ("N/A", "N/A")
    assert derived.qdrant_payload()["min_age_years"] is None


def test_unparseable_age_is_null_in_the_row_not_a_default() -> None:
    """The brief's worked example: ``"65 Years and older"`` must be NULL.

    Built as a ``RawTrial`` rather than taken from the frozen set because the
    corpus happens not to contain this shape -- but a future ClinicalTrials.gov
    snapshot might, and when it does the trial must stay findable rather than
    acquire a 65-year upper bound nobody wrote.
    """
    raw = RawTrial(
        nct_id="NCT09999999",
        title="A Trial With A Prose Age Bound",
        conditions=("Hypertension",),
        interventions=(),
        phase=None,
        status="Recruiting",
        brief_summary=None,
        detailed_description=None,
        eligibility_text="Inclusion Criteria:\n\n  -  Able to consent\n",
        raw_gender="All",
        raw_min_age="65 Years and older",
        raw_max_age="N/A",
    )
    derived = derive_trial(raw, BUILDER)
    row = derived.trial_row()

    assert row["min_age_years"] is None, "an unreadable bound must not become a number"
    assert row["max_age_years"] is None
    assert row["raw_min_age"] == "65 Years and older", "the evidence must survive"
    assert derived.ages.unparsed == ("min",), "and the failure must be countable"
    assert derived.qdrant_payload()["min_age_years"] is None


def test_real_age_bounds_are_converted() -> None:
    derived = _derive("NCT00000363")  # 2 Months .. 1 Year
    row = derived.trial_row()
    assert row["min_age_years"] is not None
    assert 0.0 < row["min_age_years"] < 1.0
    assert row["max_age_years"] == 1.0
    assert row["raw_min_age"] == "2 Months"


# --- what goes where --------------------------------------------------------


def test_payload_carries_no_eligibility_text() -> None:
    """Eligibility is Postgres-only; the retrieval unit is a trial."""
    derived = _derive("NCT00000105")
    payload = derived.qdrant_payload()
    assert set(payload) == {
        "nct_id",
        "min_age_years",
        "max_age_years",
        "sex",
        "phase",
        "status",
    }
    serialised = repr(payload)
    assert "Inclusion Criteria" not in serialised
    assert "Karnofsky" not in serialised
    assert "overview" not in payload


def test_row_carries_the_eligibility_text_and_the_overview() -> None:
    derived = _derive("NCT00000105", brief_summary="A study of immune responses.")
    row = derived.trial_row()
    assert row["eligibility_text"] is not None
    assert "Karnofsky" in row["eligibility_text"]
    assert row["overview_text"].startswith(record("NCT00000105").title)
    assert "Karnofsky" not in row["overview_text"]
    assert row["overview_token_count"] > 0
    assert row["overview_truncated"] is False


def test_row_does_not_set_indexed_at() -> None:
    """``indexed_at`` is the dual-write marker and is set after Qdrant only."""
    assert "indexed_at" not in _derive("NCT00000102").trial_row()


def test_payload_nct_id_matches_the_row() -> None:
    derived = _derive("NCT00000102")
    assert derived.qdrant_payload()["nct_id"] == derived.trial_row()["nct_id"]
    assert derived.nct_id == "NCT00000102"


def test_derivation_is_deterministic() -> None:
    assert _derive("NCT00001231") == _derive("NCT00001231")


# --- the reported numbers ---------------------------------------------------


def test_stats_count_truncation_and_fallback() -> None:
    builder = OverviewBuilder(WordTokenizer(), max_tokens=8)
    stats = IngestStats()
    stats.observe_all(
        derive_trial(as_raw_trial(nct_id, brief_summary=" ".join(["word"] * 40)), builder)
        for nct_id in REAL_RECORDS
    )
    assert stats.trials == len(REAL_RECORDS)
    # Every record gets a 40-word summary, so every overview overflows 8 tokens.
    assert stats.overview_truncated == stats.trials
    assert stats.truncation_rate == 1.0
    assert stats.overview_tokens_dropped > 0

    summary = stats.as_dict()
    assert summary["trials"] == len(REAL_RECORDS)
    assert summary["truncation_rate"] == 1.0
    assert summary["parser_counts"].keys() <= {"rules", "unstructured"}


def test_stats_count_nulls_and_missing_criteria() -> None:
    stats = IngestStats()
    stats.observe(_derive("NCT00026039"))  # no eligibility element at all
    stats.observe(_derive("NCT00000187"))  # "Please contact site for information."
    assert stats.trials == 2
    # Only NCT00026039 lacks an age; NCT00000187 states 28-47 Years.
    assert stats.min_age_null == 1
    assert stats.max_age_null == 1
    assert stats.sex_null == 1
    # Both yield no rules: one has no text, the other only boilerplate. The
    # second is the interesting one -- it is why no_criteria_extracted is
    # tracked separately from no_eligibility_text.
    assert stats.no_criteria_extracted == 2
    assert stats.no_eligibility_text == 1
    assert stats.criteria_total == 0
    assert stats.age_unparsed == 0


def test_truncation_rate_is_zero_on_an_empty_run() -> None:
    stats = IngestStats()
    assert stats.truncation_rate == 0.0
    assert stats.fallback_rate == 0.0
