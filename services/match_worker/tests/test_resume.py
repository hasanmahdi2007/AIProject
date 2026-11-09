"""Resume-from-checkpoint, in process.

This is the unit-level half of the crash test. It interrupts a run inside the
fan-out with a ``BaseException`` - which no node catches, because a process
being killed is not a condition a node should handle - and then re-runs the
same thread. The checkpoint is what makes the second run finish the job
instead of starting it again.

The other half, with a real ``SIGKILL`` of a real process and a real Postgres
checkpointer, is ``test_crash_integration.py``. It needs Docker; this one does
not, so this is the one that runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from beacon_core.events import MatchRequested
from beacon_core.settings import Settings
from beacon_llm import CachedPrompt
from match_worker._testing import (
    MATCH_ID,
    TRIALS,
    InMemoryStore,
    replay_client,
    search_response,
    seed_pipeline_fixtures,
)
from match_worker.config import WorkerConfig
from match_worker.graph import compile_graph
from match_worker.llm import SupportsCall
from match_worker.retrieval_client import RetrievalClient
from match_worker.runner import MatchWorker

RETRIEVAL_URL = "http://retrieval.invalid"
TOTAL_CRITERIA = sum(len(trial.criteria) for trial in TRIALS)


class Killed(BaseException):
    """Stands in for the process dying. Deliberately not an ``Exception``.

    ``check_rules`` catches ``Exception`` so that one trial's failure cannot
    abort the other nineteen. A kill is not that: it must not be absorbed into
    a "could not evaluate" verdict, because the trial was not evaluated *and*
    the worker is gone. Inheriting from ``BaseException`` is how this test
    tells the two apart.
    """


@dataclass
class DyingClient:
    """A replay client that dies partway through the fan-out."""

    inner: SupportsCall
    die_on: str
    calls: list[str] = field(default_factory=list)
    armed: bool = True

    async def call[OutputT: BaseModel](
        self,
        *,
        prompt: CachedPrompt,
        output_model: type[OutputT],
        model: str | None = None,
        effort: Any = None,
        max_tokens: int | None = None,
        verify_capabilities: bool = True,
    ) -> Any:
        if self.armed and self.die_on in prompt.variable:
            raise Killed(f"process killed while judging {self.die_on}")
        self.calls.append(prompt.variable[:40])
        return await self.inner.call(
            prompt=prompt,
            output_model=output_model,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            verify_capabilities=verify_capabilities,
        )


@respx.mock
async def test_a_kill_mid_fanout_resumes_and_re_judges_only_what_it_must(
    tmp_path: Path,
) -> None:
    seed_pipeline_fixtures(tmp_path)
    respx.post(f"{RETRIEVAL_URL}/search").mock(
        return_value=httpx.Response(200, json=search_response().model_dump(mode="json"))
    )
    store = InMemoryStore.seeded()
    doomed = TRIALS[2]
    client = DyingClient(inner=replay_client(tmp_path), die_on=doomed.nct_id)
    # One graph object, so one checkpointer - the stand-in for a Postgres saver
    # that two separate processes would share.
    graph = compile_graph()
    config = WorkerConfig(
        explain_top_k=0, relay_enabled=False, verify_with_llm=False, fanout_concurrency=1
    )
    worker = MatchWorker(
        store=store,
        llm_client=client,
        graph=graph,
        settings=Settings(llm_fixtures_dir=tmp_path),
        worker=config,
        retrieval=RetrievalClient(base_url=RETRIEVAL_URL),
    )
    event = MatchRequested(match_id=MATCH_ID)

    # --- the run is killed in the middle of the fan-out ------------------
    with pytest.raises(Killed):
        await worker.run_match(event)

    killed_counts = await store.count_rows(MATCH_ID)
    assert store.runs[MATCH_ID]["status"] == "running"
    assert killed_counts["criterion_judgments"] < TOTAL_CRITERIA
    assert killed_counts["criterion_judgments"] > 0, "some trials finished before the kill"
    judgment_rows_before = dict(store.judgments)
    check_calls_before = sum(1 for row in store.llm_calls if row["step"] == "check_rules")

    # --- the worker comes back -------------------------------------------
    client.armed = False
    outcome = await worker.run_match(event)

    assert outcome.resumed is True
    assert outcome.status == "completed"
    assert outcome.trials == len(TRIALS)

    # Every trial is judged exactly once: the rows that existed before the kill
    # are untouched, and only the trials that had not finished were re-asked.
    final_counts = await store.count_rows(MATCH_ID)
    assert final_counts["criterion_judgments"] == TOTAL_CRITERIA
    assert final_counts["trial_matches"] == len(TRIALS)
    for key, row in judgment_rows_before.items():
        assert store.judgments[key]["label"] == row["label"]
    check_calls_after = sum(1 for row in store.llm_calls if row["step"] == "check_rules")
    assert check_calls_after == len(TRIALS)
    assert check_calls_after > check_calls_before
    # No judgment row was written twice - the unique keys were not even needed
    # for the trials that had already finished, because they did not re-run.
    assert store.conflicts.get("criterion_judgments", 0) == 0
    # The two per-match nodes did not re-run either: the checkpoint held them.
    assert sum(1 for row in store.llm_calls if row["step"] == "profile_patient") == 1
    assert sum(1 for row in store.llm_calls if row["step"] == "generate_queries") == 1


@respx.mock
async def test_a_resumed_run_does_not_get_a_second_budget(tmp_path: Path) -> None:
    """The budget belongs to the match, not to the attempt.

    Otherwise a crash loop is a way to spend the ceiling once per restart.
    """
    seed_pipeline_fixtures(tmp_path)
    respx.post(f"{RETRIEVAL_URL}/search").mock(
        return_value=httpx.Response(200, json=search_response().model_dump(mode="json"))
    )
    store = InMemoryStore.seeded()
    client = DyingClient(inner=replay_client(tmp_path), die_on=TRIALS[2].nct_id)
    graph = compile_graph()
    worker = MatchWorker(
        store=store,
        llm_client=client,
        graph=graph,
        settings=Settings(llm_fixtures_dir=tmp_path),
        worker=WorkerConfig(
            explain_top_k=0, relay_enabled=False, verify_with_llm=False, fanout_concurrency=1
        ),
        retrieval=RetrievalClient(base_url=RETRIEVAL_URL),
    )
    event = MatchRequested(match_id=MATCH_ID)

    with pytest.raises(Killed):
        await worker.run_match(event)
    spent_before = (await graph.aget_state({"configurable": {"thread_id": MATCH_ID}})).values[
        "total_cost_usd"
    ]
    assert spent_before > 0

    client.armed = False
    outcome = await worker.run_match(event)

    # The resumed run's total includes what the first attempt spent.
    assert outcome.cost_usd > spent_before


@respx.mock
async def test_a_trial_already_judged_in_the_database_is_not_re_asked(tmp_path: Path) -> None:
    """The durable half of idempotency, without needing Docker.

    A process killed outright can lose a finished ``Send`` task's checkpoint
    write while its judgment rows are already committed. LangGraph then has no
    record that the task ran and re-runs it - so ``check_rules`` asks Postgres
    before it asks Claude. Here the checkpoint is thrown away entirely (a fresh
    saver, the same store), which is the worst case of that situation, and the
    rule checks still cost nothing.
    """
    from match_worker._testing import FlakyClient

    seed_pipeline_fixtures(tmp_path, verify=False)
    respx.post(f"{RETRIEVAL_URL}/search").mock(
        return_value=httpx.Response(200, json=search_response().model_dump(mode="json"))
    )
    store = InMemoryStore.seeded()
    config = WorkerConfig(explain_top_k=0, relay_enabled=False, verify_with_llm=False)
    event = MatchRequested(match_id=MATCH_ID)

    first = MatchWorker(
        store=store,
        llm_client=replay_client(tmp_path),
        graph=compile_graph(),
        settings=Settings(llm_fixtures_dir=tmp_path),
        worker=config,
        retrieval=RetrievalClient(base_url=RETRIEVAL_URL),
    )
    await first.run_match(event)
    judgment_rows = dict(store.judgments)
    assert judgment_rows

    # A second worker with no memory of the first: new checkpointer, same
    # database. Any rule-check call it makes would raise.
    refuses = FlakyClient(
        inner=replay_client(tmp_path),
        fail_when_variable_contains={trial.nct_id: 99 for trial in TRIALS},
    )
    second = MatchWorker(
        store=store,
        llm_client=refuses,
        graph=compile_graph(),
        settings=Settings(llm_fixtures_dir=tmp_path),
        worker=config,
        retrieval=RetrievalClient(base_url=RETRIEVAL_URL),
    )
    outcome = await second.run_match(event)

    assert outcome.status == "completed"
    assert outcome.trials == len(TRIALS)
    assert outcome.partial is False, "a restored trial must not look like a failure"
    assert refuses.attempts == [], "a trial already in the database was re-asked"
    # The restored verdicts match what the first run wrote, rule for rule.
    assert {key: row["label"] for key, row in store.judgments.items()} == {
        key: row["label"] for key, row in judgment_rows.items()
    }
