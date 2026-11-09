"""Tests for the retrieval seam and the TREC run-file reader.

No retrieval service is involved. That is the point of the seam: these tests
prove the harness can score a run produced by anything, including a system
that does not exist yet, and that the failure modes are loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.backends import (
    CallableBackend,
    RetrievalBackend,
    RetrievalUnavailableError,
    RunFileBackend,
    ScriptedBackend,
    UnavailableBackend,
    load_run_file,
    write_run_file,
)
from evals.trec import Topic

TOPIC = Topic(topic_id="1", text="a 45-year-old man with anaplastic astrocytoma")


def test_all_backends_satisfy_the_protocol() -> None:
    """Including at runtime, so a refactor that drops ``search`` is caught."""
    for backend in (
        UnavailableBackend(),
        ScriptedBackend(rankings={"1": ["NCT1"]}),
        RunFileBackend(directory=Path.cwd()),
        CallableBackend(fn=lambda topic, config: []),
    ):
        assert isinstance(backend, RetrievalBackend)


def test_the_default_backend_refuses_loudly() -> None:
    """An empty ranking would score a legitimate-looking 0.0 and land in a
    report as a measured number; refusing is the only safe default."""
    with pytest.raises(RetrievalUnavailableError, match="no retrieval backend"):
        UnavailableBackend().search(TOPIC, object())


def test_the_default_backend_names_what_is_missing() -> None:
    backend = UnavailableBackend(reason="B2 has not written any runs yet")
    with pytest.raises(RetrievalUnavailableError, match="B2 has not written"):
        backend.search(TOPIC, object())


def test_a_scripted_backend_returns_its_ranking() -> None:
    backend = ScriptedBackend(rankings={"1": ["NCT1", "NCT2"]})
    assert backend.search(TOPIC, object()) == ["NCT1", "NCT2"]


def test_a_scripted_backend_raises_for_an_uncovered_topic() -> None:
    """Returning [] here would silently score that topic 0.0."""
    backend = ScriptedBackend(rankings={"9": ["NCT1"]})
    with pytest.raises(RetrievalUnavailableError, match="no ranking for topic 1"):
        backend.search(TOPIC, object())


def test_a_callable_backend_adapts_a_plain_function() -> None:
    """How the real service is wired in later, with no metric code changing."""
    backend = CallableBackend(fn=lambda topic, config: ["NCT7", "NCT8"], name="svc")
    assert backend.search(TOPIC, object()) == ["NCT7", "NCT8"]
    assert backend.name == "svc"


def test_a_callable_backend_rejects_a_non_callable() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        CallableBackend(fn="not a function").search(TOPIC, object())


# ------------------------------------------------------------- run files ----

RUN_TEXT = """\
# a comment line, skipped
1 Q0 NCT0000001 1 12.5 beacon-hybrid
1 Q0 NCT0000002 2 11.0 beacon-hybrid
1 Q0 NCT0000003 3 9.25 beacon-hybrid

2 Q0 NCT0000009 1 4.0 beacon-hybrid
"""


def test_load_run_file_parses_topics_in_score_order(tmp_path: Path) -> None:
    path = tmp_path / "hybrid.run"
    path.write_text(RUN_TEXT, encoding="utf-8")
    run = load_run_file(path)
    assert run == {
        "1": ["NCT0000001", "NCT0000002", "NCT0000003"],
        "2": ["NCT0000009"],
    }


def test_load_run_file_ranks_by_score_not_by_the_rank_column(tmp_path: Path) -> None:
    """A run whose rank column disagrees with its scores would otherwise be
    scored in an order no system ever produced."""
    path = tmp_path / "scrambled.run"
    path.write_text(
        "1 Q0 LOW 1 1.0 tag\n1 Q0 HIGH 2 99.0 tag\n1 Q0 MID 3 50.0 tag\n",
        encoding="utf-8",
    )
    assert load_run_file(path)["1"] == ["HIGH", "MID", "LOW"]


def test_load_run_file_breaks_ties_deterministically(tmp_path: Path) -> None:
    """Tied scores make a run ill-defined; the harness at least scores it the
    same way every time, by document id ascending."""
    path = tmp_path / "ties.run"
    path.write_text("1 Q0 NCTB 1 5.0 tag\n1 Q0 NCTA 2 5.0 tag\n", encoding="utf-8")
    assert load_run_file(path)["1"] == ["NCTA", "NCTB"]


def test_load_run_file_rejects_a_duplicate_document(tmp_path: Path) -> None:
    path = tmp_path / "dupe.run"
    path.write_text("1 Q0 NCT1 1 5.0 tag\n1 Q0 NCT1 2 4.0 tag\n", encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        load_run_file(path)


def test_load_run_file_rejects_a_short_line(tmp_path: Path) -> None:
    path = tmp_path / "short.run"
    path.write_text("1 Q0 NCT1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 6 columns"):
        load_run_file(path)


def test_load_run_file_rejects_a_non_numeric_score(tmp_path: Path) -> None:
    path = tmp_path / "bad.run"
    path.write_text("1 Q0 NCT1 1 high tag\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a number"):
        load_run_file(path)


def test_load_run_file_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.run"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no run lines"):
        load_run_file(path)


def test_load_run_file_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no run file"):
        load_run_file(tmp_path / "absent.run")


def test_a_run_survives_a_write_read_round_trip(tmp_path: Path) -> None:
    """The order must be preserved exactly, or re-scoring a saved run would
    give a different number than the run that produced it."""
    run = {"1": ["NCTa", "NCTb", "NCTc"], "2": ["NCTz"]}
    path = tmp_path / "out.run"
    write_run_file(path, run, tag="test")
    assert load_run_file(path) == run


def test_run_file_backend_lists_and_serves_configurations(tmp_path: Path) -> None:
    write_run_file(tmp_path / "bm25_only.run", {"1": ["NCT1"]}, tag="t")
    write_run_file(tmp_path / "hybrid_rrf.run", {"1": ["NCT2"]}, tag="t")
    backend = RunFileBackend(directory=tmp_path)
    assert backend.available() == ("bm25_only", "hybrid_rrf")
    assert backend.run_for("hybrid_rrf") == {"1": ["NCT2"]}


def test_run_file_backend_names_the_available_configs_when_one_is_missing(
    tmp_path: Path,
) -> None:
    write_run_file(tmp_path / "bm25_only.run", {"1": ["NCT1"]}, tag="t")
    backend = RunFileBackend(directory=tmp_path)
    with pytest.raises(RetrievalUnavailableError, match="bm25_only"):
        backend.run_for("dense_only")


def test_run_file_backend_on_an_absent_directory_lists_nothing(tmp_path: Path) -> None:
    assert RunFileBackend(directory=tmp_path / "nope").available() == ()


def test_run_file_backend_raises_for_a_topic_the_run_omits(tmp_path: Path) -> None:
    write_run_file(tmp_path / "cfg.run", {"9": ["NCT1"]}, tag="t")

    class Cfg:
        name = "cfg"

    backend = RunFileBackend(directory=tmp_path)
    with pytest.raises(RetrievalUnavailableError, match="no rows for topic 1"):
        backend.search(TOPIC, Cfg())
