"""Tests for the tune/held-out barrier.

These tests assert *mechanism*, not etiquette. The claim being defended is
"you cannot evaluate on TREC 2022 by accident", and the way to defend it is to
try: call the held-out loader with nothing, with the wrong type, with an empty
reason, and run a tuning procedure against a 2022 split. Every one of those has
to raise.

No network is needed for any of this: the guards are exercised with
hand-built :class:`~evals.split.EvalSplit` objects. The two tests that actually
download topics and qrels (small files, under 1 MB) skip rather than fail when
the network is unavailable, so an offline runner and the CI integration job
both stay green.
"""

from __future__ import annotations

import pytest

from evals.split import (
    EXPECTED_TOPIC_COUNTS,
    HELD_OUT_ACCESS_LOG,
    HELD_OUT_YEAR,
    MIN_UNLOCK_REASON_CHARS,
    TUNE_YEAR,
    EvalSplit,
    HeldOutAccessError,
    HeldOutUnlock,
    assert_tuning_only,
    load_held_out_split,
    load_tuning_split,
    unlock_held_out,
)
from evals.trec import Topic

GOOD_REASON = "tuning on 2021 is frozen; taking the single headline measurement"


def make_split(name: str, year: int, n: int = 3) -> EvalSplit:
    """A hand-built split, so the guards can be tested without downloading.

    Callers that want to exercise a *guard* rather than the topic-count
    validation pass a name outside ``FULL_SPLIT_NAMES`` (the ``-fixture``
    suffix below): a three-topic split called ``"tune"`` is correctly rejected
    as a truncated download, which is a different test.
    """
    topics = tuple(Topic(topic_id=str(i), text=f"patient {i}") for i in range(1, n + 1))
    qrels = {t.topic_id: {f"NCT{t.topic_id}": 2} for t in topics}
    return EvalSplit(name=name, year=year, topics=topics, qrels=qrels)


# ------------------------------------------------------------ the constants --


def test_the_split_is_2021_tunes_2022_tests() -> None:
    assert TUNE_YEAR == 2021
    assert HELD_OUT_YEAR == 2022
    assert EXPECTED_TOPIC_COUNTS == {2021: 75, 2022: 50}
    assert sum(EXPECTED_TOPIC_COUNTS.values()) == 125


# ----------------------------------------------------------- the unlock token --


def test_unlocking_requires_a_written_reason() -> None:
    with pytest.raises(HeldOutAccessError, match="written reason"):
        unlock_held_out("because")


def test_unlocking_rejects_a_reason_that_is_only_whitespace() -> None:
    with pytest.raises(HeldOutAccessError, match="written reason"):
        unlock_held_out(" " * 50)


def test_a_sufficient_reason_produces_a_token() -> None:
    assert len(GOOD_REASON) >= MIN_UNLOCK_REASON_CHARS
    token = unlock_held_out(GOOD_REASON)
    assert isinstance(token, HeldOutUnlock)
    assert token.reason == GOOD_REASON


def test_every_unlock_is_recorded_for_the_report() -> None:
    """The report prints this log, so a held-out measurement says so on its face."""
    before = len(HELD_OUT_ACCESS_LOG)
    unlock_held_out(GOOD_REASON)
    assert len(HELD_OUT_ACCESS_LOG) == before + 1
    assert HELD_OUT_ACCESS_LOG[-1].reason == GOOD_REASON


def test_held_out_loader_refuses_a_fabricated_token() -> None:
    """The type annotation is the first line of defence; the CLI builds its
    arguments from strings and loses the static check, so there is a runtime
    one too."""
    with pytest.raises(HeldOutAccessError, match="needs a HeldOutUnlock"):
        load_held_out_split("yes please")  # type: ignore[arg-type]


def test_held_out_loader_refuses_none() -> None:
    with pytest.raises(HeldOutAccessError, match="needs a HeldOutUnlock"):
        load_held_out_split(None)  # type: ignore[arg-type]


def test_tuning_loader_takes_no_arguments() -> None:
    """2021 is the path of least resistance, by design: there is nothing to
    pass, so nobody has to think about which split they are on."""
    import inspect

    assert not inspect.signature(load_tuning_split).parameters


# ----------------------------------------------------------- tuning-only guard --


def test_assert_tuning_only_accepts_the_tuning_split() -> None:
    """Must not raise. The guard is only allowed to reject non-2021 splits."""
    assert_tuning_only(make_split("tune-fixture", TUNE_YEAR))


def test_assert_tuning_only_rejects_the_held_out_split() -> None:
    """A sweep that chose N on 2022 would leave no trace in the numbers."""
    with pytest.raises(HeldOutAccessError, match="must run on the TREC 2021"):
        assert_tuning_only(make_split("held-out-fixture", HELD_OUT_YEAR))


def test_assert_tuning_only_names_the_procedure_it_refused() -> None:
    with pytest.raises(HeldOutAccessError, match="the query-count sweep"):
        assert_tuning_only(
            make_split("held-out-fixture", HELD_OUT_YEAR), what="the query-count sweep"
        )


# ------------------------------------------------------------ split validation --


def test_a_full_split_with_the_wrong_topic_count_is_rejected() -> None:
    """A truncated topics file would otherwise produce a plausible number."""
    with pytest.raises(ValueError, match="must have 75 topics"):
        make_split("tune", 2021, n=40)


def test_a_subset_is_exempt_from_the_count_check_but_marked_as_one() -> None:
    full = make_split("tune-source", 2021, n=5)
    subset = full.subset(["1", "2"])
    assert subset.name == "tune-source-subset"
    assert not subset.is_full_split
    assert subset.topic_ids == ("1", "2")


def test_subset_rejects_unknown_topics() -> None:
    with pytest.raises(ValueError, match="no such topics"):
        make_split("x", 2021, n=3).subset(["99"])


def test_a_topic_without_judgements_is_rejected() -> None:
    """Such a topic is silently dropped by every scorer, shrinking the
    denominator without saying so."""
    topics = (Topic(topic_id="1", text="a"), Topic(topic_id="2", text="b"))
    with pytest.raises(ValueError, match="no judgements"):
        EvalSplit(name="x", year=1999, topics=topics, qrels={"1": {"NCT1": 2}})


def test_an_empty_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="scores nothing"):
        EvalSplit(name="x", year=1999, topics=(), qrels={})


def test_grade_counts_summarise_the_qrels() -> None:
    split = EvalSplit(
        name="x",
        year=1999,
        topics=(Topic(topic_id="1", text="a"),),
        qrels={"1": {"a": 2, "b": 1, "c": 0, "d": 1}},
    )
    assert split.grade_counts() == {0: 1, 1: 2, 2: 1}


def test_is_held_out_flags_the_2022_year() -> None:
    assert make_split("held-out-fixture", 2022).is_held_out
    assert not make_split("tune-fixture", 2021).is_held_out


# ------------------------------------------------- online, small downloads --


def test_loading_the_tuning_split_gives_75_judged_topics() -> None:
    """Topics and qrels only - tens of KB and under 1 MB. Skips offline."""
    try:
        split = load_tuning_split()
    except OSError as exc:  # requests' ConnectionError subclasses OSError
        pytest.skip(f"TREC topics/qrels download unavailable: {exc}")
    assert split.year == 2021
    assert len(split.topics) == 75
    assert split.is_full_split
    assert not split.is_held_out
    # The graded scheme must survive the trip: all three grades present.
    assert set(split.grade_counts()) == {0, 1, 2}


def test_loading_the_held_out_split_requires_and_records_an_unlock() -> None:
    """The one place this suite reads the 2022 labels, and it says why."""
    try:
        split = load_held_out_split(
            unlock_held_out("test: asserting the held-out loader returns 50 topics")
        )
    except OSError as exc:
        pytest.skip(f"TREC topics/qrels download unavailable: {exc}")
    assert split.year == 2022
    assert len(split.topics) == 50
    assert split.is_held_out
    assert any("asserting the held-out loader" in u.reason for u in HELD_OUT_ACCESS_LOG)
