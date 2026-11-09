"""The crash test. A real process, a real kill, a real checkpoint.

Marked ``integration`` because it needs the compose stack's Postgres (host port
**15432**; the dev machine runs its own PostgreSQL on 5432, which is why the
default is shifted). Excluded from the fast unit run, included in the
definition of done.

The shape of it:

1. seed a patient, a ``match_runs`` row and three trials, exactly as the
   gateway would;
2. start ``crash_child.py``, which runs one match through the real graph with
   the real ``AsyncPostgresSaver``;
3. wait until a judgment lands in Postgres, then **kill the process** - no
   cleanup, no flush, no final transaction;
4. read the checkpoint it left behind and record which nodes were still
   pending;
5. start a second process and let it finish;
6. prove the result is complete and that no row was written twice.

Step 6 is the point. "It finished" is easy; "it finished without duplicating
anything" is what at-least-once delivery and crash recovery actually cost, and
the four unique constraints in ``beacon_core.models`` are what pay for it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from match_worker._testing import SAMPLE_NOTE_TEXT, TRIALS, seed_pipeline_fixtures
from match_worker.graph import checkpointer_dsn

pytestmark = pytest.mark.integration

MATCH_ID = "match-c1-crashtest"
PATIENT_ID = "patient-c1-crashtest"
TOTAL_CRITERIA = sum(len(trial.criteria) for trial in TRIALS)
CHILD = Path(__file__).with_name("crash_child.py")


def _dsn() -> str:
    # psycopg wants the plain scheme; the app uses asyncpg through the same
    # settings object, so there is still one source of truth for the port.
    return checkpointer_dsn().replace("postgresql+psycopg", "postgresql")


def _connect() -> psycopg.Connection[Any]:
    return psycopg.connect(_dsn(), connect_timeout=5)


def _one(cursor: psycopg.Cursor[Any]) -> tuple[Any, ...]:
    """``fetchone()`` that cannot be ``None``: every query here is an aggregate
    or a primary-key lookup whose row the test has just inserted."""
    row = cursor.fetchone()
    assert row is not None
    return row


@pytest.fixture(scope="module")
def postgres() -> Iterator[psycopg.Connection[Any]]:
    try:
        connection = _connect()
    except psycopg.OperationalError as exc:  # pragma: no cover - environment
        pytest.skip(f"Postgres is not reachable at {_dsn().rsplit('@', 1)[-1]}: {exc}")
    connection.autocommit = True
    yield connection
    connection.close()


def _seed(connection: psycopg.Connection[Any]) -> None:
    """Insert what the gateway and the ingestion pipeline would have left."""
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM criterion_judgments WHERE match_run_id = %s", (MATCH_ID,))
        cursor.execute("DELETE FROM trial_matches WHERE match_run_id = %s", (MATCH_ID,))
        cursor.execute("DELETE FROM llm_calls WHERE match_run_id = %s", (MATCH_ID,))
        cursor.execute("DELETE FROM outbox WHERE partition_key = %s", (MATCH_ID,))
        cursor.execute("DELETE FROM match_runs WHERE id = %s", (MATCH_ID,))
        cursor.execute("DELETE FROM patients WHERE id = %s", (PATIENT_ID,))
        for trial in TRIALS:
            cursor.execute("DELETE FROM trial_criteria WHERE nct_id = %s", (trial.nct_id,))
            cursor.execute("DELETE FROM trials WHERE nct_id = %s", (trial.nct_id,))
        # The checkpointer's own tables, so a re-run of this test starts from
        # no checkpoint rather than resuming the previous one.
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            cursor.execute(f"SELECT to_regclass('public.{table}')")
            if _one(cursor)[0] is not None:
                cursor.execute(f"DELETE FROM {table} WHERE thread_id = %s", (MATCH_ID,))

        cursor.execute(
            "INSERT INTO patients (id, source, note_text, sentences) VALUES (%s, %s, %s, %s)",
            (PATIENT_ID, "api", SAMPLE_NOTE_TEXT, json.dumps([])),
        )
        cursor.execute(
            "INSERT INTO match_runs (id, patient_id, status, config_version) "
            "VALUES (%s, %s, %s, %s)",
            (MATCH_ID, PATIENT_ID, "queued", "v1"),
        )
        for trial in TRIALS:
            cursor.execute(
                # locations, conditions and interventions are NOT NULL with a
                # Python-side default, so a raw INSERT has to supply them.
                "INSERT INTO trials (nct_id, title, conditions, interventions, locations, "
                "eligibility_text, min_age_years, sex) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    trial.nct_id,
                    trial.title,
                    json.dumps(list(trial.conditions)),
                    json.dumps([]),
                    json.dumps([]),
                    trial.eligibility_text,
                    trial.min_age_years,
                    trial.sex,
                ),
            )


def _counts(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*), count(DISTINCT (nct_id, criterion_id)) "
            "FROM criterion_judgments WHERE match_run_id = %s",
            (MATCH_ID,),
        )
        judgments, distinct_judgments = _one(cursor)
        cursor.execute(
            "SELECT count(*), count(DISTINCT nct_id) FROM trial_matches WHERE match_run_id = %s",
            (MATCH_ID,),
        )
        matches, distinct_matches = _one(cursor)
        cursor.execute("SELECT count(*) FROM llm_calls WHERE match_run_id = %s", (MATCH_ID,))
        llm_calls = _one(cursor)[0]
        cursor.execute("SELECT count(*) FROM outbox WHERE partition_key = %s", (MATCH_ID,))
        outbox = _one(cursor)[0]
        cursor.execute(
            "SELECT count(*) FROM trial_criteria WHERE nct_id = ANY(%s)",
            ([trial.nct_id for trial in TRIALS],),
        )
        criteria = _one(cursor)[0]
        cursor.execute("SELECT status, error FROM match_runs WHERE id = %s", (MATCH_ID,))
        status, error = _one(cursor)
    return {
        "criterion_judgments": judgments,
        "criterion_judgments_distinct": distinct_judgments,
        "trial_matches": matches,
        "trial_matches_distinct": distinct_matches,
        "trial_criteria": criteria,
        "llm_calls": llm_calls,
        "outbox": outbox,
        "status": status,
        "error": error,
    }


def _checkpoint_state(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    """What the checkpointer left behind, read straight from its own tables.

    Read with plain SQL rather than through ``AsyncPostgresSaver`` because
    psycopg's async mode cannot run on Windows' default event loop (see
    ``match_worker.compat``), and because the raw row is better evidence: this
    is the checkpoint a restarted worker will find, not a reconstruction of it.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM checkpoints WHERE thread_id = %s", (MATCH_ID,))
        total = _one(cursor)[0]
        cursor.execute(
            "SELECT checkpoint_id, metadata FROM checkpoints WHERE thread_id = %s "
            "ORDER BY checkpoint_id DESC LIMIT 1",
            (MATCH_ID,),
        )
        row = cursor.fetchone()
        if row is None:
            return {"checkpoints": total, "checkpoint_id": None, "step": None, "writes": []}
        checkpoint_id, metadata = row
        cursor.execute(
            "SELECT DISTINCT channel FROM checkpoint_writes "
            "WHERE thread_id = %s AND checkpoint_id = %s",
            (MATCH_ID, checkpoint_id),
        )
        channels = sorted(str(record[0]) for record in cursor.fetchall())
    meta = metadata if isinstance(metadata, dict) else {}
    return {
        "checkpoints": total,
        "checkpoint_id": str(checkpoint_id),
        "step": meta.get("step"),
        "source": meta.get("source"),
        "writes": channels,
    }


def _spawn(fixtures: Path, port: int, log: Path) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "BEACON_LLM_FIXTURES_DIR": str(fixtures),
        # Replay is already the default; set explicitly so this test cannot be
        # made to spend money by an inherited environment variable.
        "BEACON_LLM_MODE": "replay",
        # A slow run is a killable run: one pause per Claude call, one trial at
        # a time, so the kill lands inside the fan-out rather than after it.
        "BEACON_WORKER_THROTTLE_SECONDS": "0.6",
        "BEACON_WORKER_FANOUT_CONCURRENCY": "1",
        "BEACON_WORKER_EXPLAIN_TOP_K": "0",
        "BEACON_WORKER_VERIFY_WITH_LLM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    # A log file, not a pipe. The worker logs a line per node and per Claude
    # call; a child writing to a PIPE that nobody is reading blocks forever
    # once the OS buffer fills, which looks exactly like a hung pipeline.
    handle = log.open("wb")
    return subprocess.Popen(
        [sys.executable, str(CHILD), "--match-id", MATCH_ID, "--retrieval-port", str(port)],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _milestones(log: Path) -> list[dict[str, Any]]:
    """The child's own JSON milestone lines, from its log file."""
    out: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{") and '"event": "child.' in line:
            out.append(json.loads(line))
    return out


def _wait_for_judgments(
    connection: psycopg.Connection[Any], *, at_least: int, timeout: float
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(DISTINCT nct_id) FROM criterion_judgments WHERE match_run_id = %s",
                (MATCH_ID,),
            )
            trials = _one(cursor)[0]
        if trials >= at_least:
            return trials
        time.sleep(0.1)
    return 0


def test_a_killed_worker_resumes_from_its_checkpoint_without_duplicating_rows(
    postgres: psycopg.Connection[Any], tmp_path: Path
) -> None:
    seed_pipeline_fixtures(tmp_path, verify=False)
    _seed(postgres)
    port = 8731

    # --- 1. start the run and kill it mid-fan-out ------------------------
    first_log = tmp_path / "child-1.log"
    first = _spawn(tmp_path, port, first_log)
    try:
        judged_before_kill = _wait_for_judgments(postgres, at_least=1, timeout=90)
        assert judged_before_kill >= 1, (
            f"the run never reached the fan-out; nothing to kill. Child log:\n"
            f"{first_log.read_text(encoding='utf-8', errors='replace')[-4000:]}"
        )
        assert judged_before_kill < len(TRIALS), "the run finished before it could be killed"
    finally:
        # Killed in the finally so that a failed assertion above cannot leave
        # an orphan worker running against the same match.
        first.kill()
        first.wait(timeout=30)
    assert first.returncode != 0, "the process exited cleanly; that is not a crash"

    after_kill = _counts(postgres)
    checkpoint = _checkpoint_state(postgres)

    print("\n--- after the kill -------------------------------------------")
    print(f"killed with {judged_before_kill} of {len(TRIALS)} trials judged")
    print(json.dumps({"checkpoint": checkpoint, "rows": after_kill}, indent=2, default=str))

    assert checkpoint["checkpoint_id"] is not None, (
        "no checkpoint was written, so there is nothing to resume"
    )
    assert checkpoint["checkpoints"] > 1, "a checkpoint per node is the whole claim"
    assert after_kill["status"] == "running"
    assert 0 < after_kill["criterion_judgments"] < TOTAL_CRITERIA
    # parse_rules already persisted every trial's rules, before the kill: that
    # is the lazy-and-cached part, and it is why the second process does not
    # re-split anything.
    assert after_kill["trial_criteria"] == TOTAL_CRITERIA

    # --- 2. restart, and let it finish -----------------------------------
    second_log = tmp_path / "child-2.log"
    second = _spawn(tmp_path, port, second_log)
    second.wait(timeout=240)
    assert second.returncode == 0, second_log.read_text(encoding="utf-8", errors="replace")[-4000:]
    finished = next(m for m in _milestones(second_log) if m["event"] == "child.finished")

    final = _counts(postgres)
    print("\n--- after the restart ----------------------------------------")
    print(json.dumps({"child": finished, "rows": final}, indent=2, default=str))
    print(json.dumps({"checkpoint": _checkpoint_state(postgres)}, indent=2, default=str))

    assert finished["resumed"] is True, "the second process started over instead of resuming"
    assert finished["status"] == "completed"
    assert finished["trials"] == len(TRIALS)

    # --- 3. the evidence: complete, and nothing written twice ------------
    assert final["status"] == "completed"
    assert final["criterion_judgments"] == TOTAL_CRITERIA
    assert final["criterion_judgments"] == final["criterion_judgments_distinct"]
    assert final["trial_matches"] == len(TRIALS)
    assert final["trial_matches"] == final["trial_matches_distinct"]
    assert final["trial_criteria"] == TOTAL_CRITERIA
    # Every judgment row from before the kill survived untouched, and the
    # trials that had not finished were judged exactly once.
    assert final["criterion_judgments"] > after_kill["criterion_judgments"]


def test_the_same_message_run_twice_against_postgres_is_a_no_op(
    postgres: psycopg.Connection[Any], tmp_path: Path
) -> None:
    """At-least-once delivery against the real unique constraints."""
    seed_pipeline_fixtures(tmp_path, verify=False)
    _seed(postgres)
    port = 8732

    first_log = tmp_path / "delivery-1.log"
    first = _spawn(tmp_path, port, first_log)
    first.wait(timeout=240)
    assert first.returncode == 0, first_log.read_text(encoding="utf-8", errors="replace")[-4000:]
    after_first = _counts(postgres)
    assert after_first["status"] == "completed"

    second_log = tmp_path / "delivery-2.log"
    second = _spawn(tmp_path, port, second_log)
    second.wait(timeout=240)
    assert second.returncode == 0, second_log.read_text(encoding="utf-8", errors="replace")[-4000:]
    finished = next(m for m in _milestones(second_log) if m["event"] == "child.finished")
    after_second = _counts(postgres)

    print("\n--- redelivery ------------------------------------------------")
    print(json.dumps({"first": after_first, "second": after_second}, indent=2, default=str))

    assert finished["already_complete"] is True, "the redelivery re-ran the pipeline"
    assert after_second == after_first, "a redelivered message changed the database"
