"""Tests for the TREC Clinical Trials loaders.

Two tiers:

* **Offline** -- the dataset-ID constants, the grade scheme, argument
  validation and the ``IR_DATASETS_HOME`` wiring. No network, no downloads.
* **Online, small** -- topics and qrels. Tens of KB and under 1 MB
  respectively, cached under ``IR_DATASETS_HOME``, so the brief allows CI to
  fetch them. They skip (rather than fail) when the network is unavailable, so
  an offline runner does not produce a spurious red build.

Nothing here touches the 375,580-document collection; the only test that does
is marked ``integration`` and is excluded from CI by ``-m "not integration"``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evals import trec

OFFLINE_REASON = "TREC topics/qrels download unavailable (no network?)"


def _fetch(loader, *args):  # type: ignore[no-untyped-def]
    """Run a small loader, skipping the test if the download cannot be made."""
    try:
        return loader(*args)
    except OSError as exc:  # requests' ConnectionError subclasses OSError
        pytest.skip(f"{OFFLINE_REASON}: {exc}")


# ------------------------------------------------------------------ offline --


def test_dataset_ids_are_the_verified_registry_entries() -> None:
    """Guards the single most dangerous mistake in this module: a wrong ID.

    These exact strings were confirmed by enumerating ``ir_datasets.registry``;
    if someone "tidies" them into ``clinicaltrials/2022/...`` the corpus silently
    changes and every eval number becomes meaningless.
    """
    assert trec.TREC_CT_2021 == "clinicaltrials/2021/trec-ct-2021"
    assert trec.TREC_CT_2022 == "clinicaltrials/2021/trec-ct-2022"
    assert trec.TREC_CT_CORPUS == "clinicaltrials/2021"


def test_both_tracks_share_the_2021_corpus() -> None:
    """2022 re-used the 2021 snapshot; both IDs must sit under it."""
    assert trec.TREC_CT_2021.startswith(f"{trec.TREC_CT_CORPUS}/")
    assert trec.TREC_CT_2022.startswith(f"{trec.TREC_CT_CORPUS}/")
    assert "clinicaltrials/2022" not in trec.TREC_CT_2022


def test_dataset_ids_are_actually_registered() -> None:
    """The IDs resolve in the installed ir_datasets -- no download triggered."""
    import ir_datasets

    registered = set(ir_datasets.registry._registered)
    assert trec.TREC_CT_2021 in registered
    assert trec.TREC_CT_2022 in registered
    assert trec.TREC_CT_CORPUS in registered


def test_registry_reports_the_counts_we_recorded() -> None:
    """Metadata check: catches an ir_datasets upgrade that swaps the corpus.

    Reads the shipped metadata only; it does not fetch anything.
    """
    import ir_datasets

    corpus = ir_datasets.load(trec.TREC_CT_CORPUS).metadata()
    assert corpus["docs"]["count"] == 375_580

    md_2021 = ir_datasets.load(trec.TREC_CT_2021).metadata()
    assert md_2021["queries"]["count"] == 75
    assert md_2021["qrels"]["count"] == 35_832
    assert md_2021["qrels"]["fields"]["relevance"]["counts_by_value"] == {
        "0": 24243,
        "1": 6019,
        "2": 5570,
    }

    md_2022 = ir_datasets.load(trec.TREC_CT_2022).metadata()
    assert md_2022["queries"]["count"] == 50


def test_ir_datasets_qrel_defs_match_our_grade_scheme() -> None:
    """ir_datasets' own labels must agree with ``RELEVANCE_LABELS``.

    This is corroborating evidence that trec-ct-2021 is the right dataset: no
    other registry entry uses the Eligible/Excluded/Not Relevant scheme.
    """
    import ir_datasets

    assert ir_datasets.load(trec.TREC_CT_2021).qrels_defs() == trec.RELEVANCE_LABELS


def test_grades_are_ordinal_not_boolean() -> None:
    """Excluded (1) must rank above Not Relevant (0); graded nDCG depends on it."""
    assert trec.RELEVANCE_LABELS == {0: "Not Relevant", 1: "Excluded", 2: "Eligible"}
    excluded = next(g for g, label in trec.RELEVANCE_LABELS.items() if label == "Excluded")
    not_relevant = next(g for g, label in trec.RELEVANCE_LABELS.items() if label == "Not Relevant")
    eligible = next(g for g, label in trec.RELEVANCE_LABELS.items() if label == "Eligible")
    assert eligible > excluded > not_relevant


def test_dataset_id_lookup() -> None:
    assert trec.dataset_id(2021) == trec.TREC_CT_2021
    assert trec.dataset_id(2022) == trec.TREC_CT_2022


@pytest.mark.parametrize("year", [2020, 2023, 21, 0, -2021])
def test_dataset_id_rejects_unsupported_years(year: int) -> None:
    with pytest.raises(ValueError, match="No TREC Clinical Trials track"):
        trec.dataset_id(year)


def test_ir_datasets_home_env_var_agrees_with_our_setting() -> None:
    """The whole point of the import-order dance: both systems see one path."""
    assert os.environ["IR_DATASETS_HOME"] == str(trec.IR_DATASETS_HOME)
    assert trec.IR_DATASETS_HOME.is_absolute()
    assert trec.IR_DATASETS_HOME.exists()


def test_ir_datasets_home_is_outside_the_repo() -> None:
    """A multi-GB corpus inside the worktree would be a disaster."""
    repo_root = Path(__file__).resolve().parents[2]
    assert not trec.IR_DATASETS_HOME.resolve().is_relative_to(repo_root)


def test_posix_fallback_is_absolute_and_not_the_ir_datasets_default() -> None:
    """On Linux CI the Windows default is relative, so the fallback must hold up."""
    assert trec.POSIX_FALLBACK_IR_DATASETS_HOME.is_absolute()
    assert Path.home() / ".ir_datasets" != trec.POSIX_FALLBACK_IR_DATASETS_HOME


def test_iter_trials_is_a_lazy_generator() -> None:
    """Calling it must not read, download or materialise anything.

    If this ever starts failing by hanging, someone turned the generator into a
    function that builds a 375K-element list.
    """
    import inspect

    assert inspect.isgeneratorfunction(trec.iter_trials)
    gen = trec.iter_trials(skip=10, limit=5)
    assert inspect.isgenerator(gen)
    gen.close()


@pytest.mark.parametrize("skip", [-1, -100])
def test_iter_trials_rejects_negative_skip(skip: int) -> None:
    with pytest.raises(ValueError, match="skip must be >= 0"):
        next(trec.iter_trials(skip=skip))


@pytest.mark.parametrize("limit", [0, -1])
def test_iter_trials_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        next(trec.iter_trials(limit=limit))


def test_corpus_is_cached_is_a_cheap_filesystem_check() -> None:
    """Must answer without downloading; the integration guard relies on it."""
    assert isinstance(trec.corpus_is_cached(), bool)
    assert len(trec.CORPUS_ZIP_NAMES) == 5
    assert trec.CORPUS_ZIP_NAMES[0] == "docs_2021_1.zip"


def test_nist_qrels_fallback_only_configured_for_2022() -> None:
    with pytest.raises(ValueError, match="No NIST qrels fallback"):
        trec._nist_qrels(2021)


def test_cli_requires_an_action() -> None:
    with pytest.raises(SystemExit):
        trec.main(["--topic", "2021"])


def test_cli_rejects_an_unsupported_year() -> None:
    with pytest.raises(SystemExit):
        trec.main(["--topic", "2019", "--first"])


# ------------------------------------------------------- online, small data --


def test_load_topics_2021() -> None:
    topics = _fetch(trec.load_topics, 2021)
    assert len(topics) == 75
    assert [t.topic_id for t in topics] == [str(i) for i in range(1, 76)]
    first = topics[0]
    assert "45-year-old man" in first.text
    assert "anaplastic astrocytoma" in first.text
    # Patient notes are prose, not keyword queries.
    assert len(first.text) > 200
    assert first.text == first.text.strip()


def test_load_topics_2022() -> None:
    topics = _fetch(trec.load_topics, 2022)
    assert len(topics) == 50
    assert [t.topic_id for t in topics] == [str(i) for i in range(1, 51)]
    assert all(t.text.strip() for t in topics)


def test_load_qrels_2021_preserves_integer_grades() -> None:
    qrels = _fetch(trec.load_qrels, 2021)
    assert len(qrels) == 75
    flat = [g for docs in qrels.values() for g in docs.values()]
    assert len(flat) == 35_832
    assert all(type(g) is int for g in flat), "grades must be int, never bool"
    counts = {g: flat.count(g) for g in (0, 1, 2)}
    assert counts == {0: 24243, 1: 6019, 2: 5570}
    # All three grades present on topic 1, so nothing collapsed them.
    assert set(qrels["1"].values()) == {0, 1, 2}
    assert len(qrels["1"]) == 407


def test_load_qrels_2022_via_nist_fallback() -> None:
    """ir_datasets 0.6.3 has no 2022 qrels; we fetch the official NIST file."""
    import ir_datasets

    assert not ir_datasets.load(trec.TREC_CT_2022).has_qrels()

    qrels = _fetch(trec.load_qrels, 2022)
    assert len(qrels) == 50
    flat = [g for docs in qrels.values() for g in docs.values()]
    assert len(flat) == 35_394
    assert {g: flat.count(g) for g in (0, 1, 2)} == {0: 28419, 1: 3036, 2: 3939}


def test_qrels_doc_ids_look_like_nct_ids() -> None:
    qrels = _fetch(trec.load_qrels, 2021)
    doc_ids = list(qrels["1"])
    assert all(d.startswith("NCT") for d in doc_ids)
    assert all(len(d) == 11 for d in doc_ids)


def test_every_topic_has_judgements() -> None:
    """Topic IDs and qrels keys must line up, or ranx silently drops topics."""
    for year in (2021, 2022):
        topics = _fetch(trec.load_topics, year)
        qrels = _fetch(trec.load_qrels, year)
        assert {t.topic_id for t in topics} == set(qrels), year


def test_cli_prints_a_patient_note_and_a_qrels_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        assert trec.main(["--topic", "2021", "--first"]) == 0
    except OSError as exc:
        pytest.skip(f"{OFFLINE_REASON}: {exc}")
    out = capsys.readouterr().out
    assert trec.TREC_CT_2021 in out
    assert "--- topic 1 ---" in out
    assert "anaplastic astrocytoma" in out
    assert "qrels for topic 1: 407" in out
    assert "2 (Eligible): 47" in out


# ------------------------------------------------- the corpus (CI-excluded) --


@pytest.mark.integration
def test_iter_trials_yields_trials_and_can_skip() -> None:
    """Touches the document collection, so it is integration-only.

    Checks the resumability contract: ``skip=n`` starts where an unskipped
    iteration's element ``n`` was, which is what lets a crashed ingestion job
    restart from its last committed offset.

    Skips unless the corpus is already cached. ``pytest evals`` (the Definition
    of Done command) carries no marker filter, so without this guard a plain
    test run would silently start a ~1.8 GB download and then spend minutes
    building the document store.
    """
    if not trec.corpus_is_cached():
        pytest.skip(
            f"TREC corpus not cached under {trec.IR_DATASETS_HOME}; "
            'run `python -c "from evals.trec import iter_trials; next(iter_trials())"` '
            "to populate it (~1.8 GB download)"
        )

    first_three = list(trec.iter_trials(limit=3))
    assert len(first_three) == 3
    assert all(t.doc_id.startswith("NCT") for t in first_three)
    assert any(t.eligibility for t in first_three)

    resumed = list(trec.iter_trials(skip=1, limit=2))
    assert [t.doc_id for t in resumed] == [t.doc_id for t in first_three[1:]]
