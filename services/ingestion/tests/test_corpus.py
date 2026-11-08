"""Parsing one ClinicalTrials.gov document, and the corpus reader's guards.

The XML below is trimmed from the real ``NCT00000105.xml`` -- same elements,
same nesting, same ``&#xD;`` line endings inside the textblocks -- so it tests
the parser against the structure the archives actually hold without needing the
2.6 GB cache. The tests that do read the cache are marked ``integration``.
"""

from __future__ import annotations

import pytest

from ingestion.corpus import (
    EXPECTED_TRIAL_COUNT,
    MAX_STORED_LOCATIONS,
    iter_raw_trials,
    parse_trial_xml,
)

TRIMMED_XML = """<clinical_study>
  <required_header>
    <url>https://clinicaltrials.gov/show/NCT00000105</url>
  </required_header>
  <id_info>
    <org_study_id>2002LS032</org_study_id>
    <nct_id>NCT00000105</nct_id>
  </id_info>
  <brief_title>Vaccination With Tetanus and KLH to Assess Immune Responses.</brief_title>
  <official_title>Vaccination With Tetanus Toxoid and Keyhole Limpet Hemocyanin</official_title>
  <brief_summary>
    <textblock>
      The purpose of this study is to learn how the immune system works.&#xD;
    </textblock>
  </brief_summary>
  <detailed_description>
    <textblock>
      Patients will receive each vaccine once only.&#xD;
    </textblock>
  </detailed_description>
  <overall_status>Terminated</overall_status>
  <phase>Phase 2</phase>
  <condition>Cancer</condition>
  <condition>Cancer</condition>
  <intervention>
    <intervention_type>Biological</intervention_type>
    <intervention_name>KLH</intervention_name>
  </intervention>
  <intervention>
    <intervention_type>Biological</intervention_type>
    <intervention_name>Tetanus toxoid</intervention_name>
  </intervention>
  <eligibility>
    <criteria>
      <textblock>
        Inclusion Criteria:&#xD;
&#xD;
          -  Patients must have a diagnosis of cancer of any histologic type.&#xD;
      </textblock>
    </criteria>
    <gender>All</gender>
    <minimum_age>18 Years</minimum_age>
    <maximum_age>N/A</maximum_age>
  </eligibility>
  <location>
    <facility>
      <name>Masonic Cancer Center</name>
      <address>
        <city>Minneapolis</city>
        <state>Minnesota</state>
        <country>United States</country>
      </address>
    </facility>
  </location>
</clinical_study>
"""


def test_parses_every_field_ingestion_needs() -> None:
    trial = parse_trial_xml(TRIMMED_XML)
    assert trial.nct_id == "NCT00000105"
    assert trial.title == "Vaccination With Tetanus and KLH to Assess Immune Responses."
    assert trial.conditions == ("Cancer",), "duplicate <condition> elements collapse"
    assert trial.interventions == ("KLH", "Tetanus toxoid")
    assert trial.phase == "Phase 2"
    assert trial.status == "Terminated"
    assert trial.raw_gender == "All"
    assert trial.raw_min_age == "18 Years"
    assert trial.raw_max_age == "N/A"
    assert (
        trial.brief_summary == "The purpose of this study is to learn how the immune system works."
    )
    assert trial.detailed_description == "Patients will receive each vaccine once only."
    assert trial.eligibility_text is not None
    assert "Inclusion Criteria:" in trial.eligibility_text
    assert trial.locations == (
        {
            "facility": "Masonic Cancer Center",
            "city": "Minneapolis",
            "state": "Minnesota",
            "country": "United States",
        },
    )


def test_these_are_the_fields_ir_datasets_drops() -> None:
    """The reason this module exists rather than reusing ``evals.trec``.

    ``ir_datasets``' ``clinicaltrials`` doc type is
    ``(doc_id, title, condition, summary, detailed_description, eligibility)``,
    and its ``eligibility`` is only the criteria textblock. Every field
    asserted here is one that reading through ``evals.trec`` alone cannot
    supply -- and four of the five are what the Qdrant payload filters on.
    """
    trial = parse_trial_xml(TRIMMED_XML)
    for value in (
        trial.interventions,
        trial.phase,
        trial.status,
        trial.raw_gender,
        trial.raw_min_age,
        trial.raw_max_age,
    ):
        assert value, "parsed from the source XML, not available via ir_datasets"


def test_blank_elements_become_none_not_empty_strings() -> None:
    trial = parse_trial_xml(
        "<clinical_study><id_info><nct_id>NCT1</nct_id></id_info>"
        "<brief_title>T</brief_title><phase>   </phase>"
        "<eligibility><gender></gender></eligibility></clinical_study>"
    )
    assert trial.phase is None
    assert trial.raw_gender is None
    assert trial.status is None
    assert trial.eligibility_text is None
    assert trial.conditions == ()
    assert trial.locations == ()


def test_official_title_is_the_fallback_when_brief_title_is_absent() -> None:
    trial = parse_trial_xml(
        "<clinical_study><id_info><nct_id>NCT1</nct_id></id_info>"
        "<official_title>The Formal Name</official_title></clinical_study>"
    )
    assert trial.title == "The Formal Name"


def test_a_document_with_no_nct_id_is_refused() -> None:
    """Guessing a key would silently merge two trials into one row."""
    with pytest.raises(ValueError, match="nct_id"):
        parse_trial_xml("<clinical_study><brief_title>T</brief_title></clinical_study>")


def test_nct_id_may_be_supplied_by_the_caller() -> None:
    trial = parse_trial_xml(
        "<clinical_study><brief_title>T</brief_title></clinical_study>", nct_id="NCT42"
    )
    assert trial.nct_id == "NCT42"


def test_locations_are_capped() -> None:
    body = "".join(
        f"<location><facility><name>Site {n}</name></facility></location>"
        for n in range(MAX_STORED_LOCATIONS + 10)
    )
    trial = parse_trial_xml(
        f"<clinical_study><id_info><nct_id>NCT1</nct_id></id_info>{body}</clinical_study>"
    )
    assert len(trial.locations) == MAX_STORED_LOCATIONS


def test_bad_slice_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="skip"):
        next(iter_raw_trials(skip=-1))
    with pytest.raises(ValueError, match="limit"):
        next(iter_raw_trials(limit=0))


def test_expected_count_matches_the_dataset_metadata() -> None:
    """``clinicaltrials/2021`` is 375,580 documents; see ``evals.trec``."""
    assert EXPECTED_TRIAL_COUNT == 375_580


# --- against the real cache -------------------------------------------------


@pytest.mark.integration
def test_reads_real_records_from_the_cached_archives() -> None:
    trials = list(iter_raw_trials(limit=5))
    assert len(trials) == 5
    assert trials[0].nct_id == "NCT00000102"
    assert all(t.nct_id.startswith("NCT") for t in trials)
    assert all(t.title for t in trials)


@pytest.mark.integration
def test_skip_is_reproducible() -> None:
    first_three = [t.nct_id for t in iter_raw_trials(limit=3)]
    skipped = [t.nct_id for t in iter_raw_trials(skip=1, limit=2)]
    assert skipped == first_three[1:]


@pytest.mark.integration
def test_cross_check_against_ir_datasets() -> None:
    """The XML reading and ``evals.trec`` must agree on what they share."""
    from ingestion.corpus import cross_check_against_ir_datasets

    counts = cross_check_against_ir_datasets(sample=200)
    assert counts["checked"] == 200
    assert counts["id_mismatch"] == 0
    assert counts["title_mismatch"] == 0
    assert counts["eligibility_mismatch"] == 0
    # ...and disagree exactly where ir_datasets drops data.
    assert counts["irds_condition_empty"] > 0
