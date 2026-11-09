"""The whole pipeline, end to end, with zero network calls.

Everything here runs the *real* graph, the real nodes, the real
``beacon_llm.AsyncLlmClient`` and the real prompt builders. Two seams are
stubbed and only two: the retrieval service, at HTTP, because another agent
owns it; and Claude, through ``beacon_llm``'s replay mode against fixtures this
test authors from the production prompts. Replay has no code path to the
network, so these tests cannot spend money even if they are wrong.

What each test is actually about is in its own docstring. Between them they
cover the four behaviours the brief calls out: a trial that fails twice is
marked "could not evaluate" while the rest finish, a budget breach returns
results marked partial, a redelivered message is a no-op, and a message that
keeps failing ends in the DLQ.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from beacon_core.events import TOPIC_MATCH_COMPLETED, TOPIC_MATCH_DLQ, MatchRequested
from beacon_core.settings import Settings
from match_worker._testing import (
    MATCH_ID,
    SAMPLE_NOTE_TEXT,
    TRIALS,
    FlakyClient,
    InMemoryStore,
    replay_client,
    search_response,
    seed_explanation_fixtures,
    seed_pipeline_fixtures,
)
from match_worker.config import WorkerConfig
from match_worker.graph import compile_graph
from match_worker.progress import NODE_ORDER
from match_worker.retrieval_client import RetrievalClient
from match_worker.runner import MatchWorker
from match_worker.state import BUDGET_EXHAUSTED, COULD_NOT_EVALUATE, PARTIAL_PREFIX

RETRIEVAL_URL = "http://retrieval.invalid"
TOTAL_CRITERIA = sum(len(trial.criteria) for trial in TRIALS)


def _stub_retrieval(trials: object = TRIALS) -> None:
    respx.post(f"{RETRIEVAL_URL}/search").mock(
        return_value=httpx.Response(
            200,
            json=search_response(trials).model_dump(mode="json"),  # type: ignore[arg-type]
        )
    )


def _worker(
    store: InMemoryStore,
    fixtures: Path,
    *,
    worker: WorkerConfig | None = None,
    client: object | None = None,
) -> MatchWorker:
    return MatchWorker(
        store=store,
        llm_client=client or replay_client(fixtures),  # type: ignore[arg-type]
        graph=compile_graph(),
        settings=Settings(llm_fixtures_dir=fixtures),
        worker=worker or WorkerConfig(explain_top_k=0, relay_enabled=False),
        retrieval=RetrievalClient(base_url=RETRIEVAL_URL),
    )


# ------------------------------------------------------------- happy path --


@respx.mock
async def test_the_seven_nodes_run_and_persist_a_complete_answer(tmp_path: Path) -> None:
    """One match, three trials, every row and every event where it belongs."""
    seed_pipeline_fixtures(tmp_path)
    _stub_retrieval()
    store = InMemoryStore.seeded()
    worker = _worker(store, tmp_path)

    outcome = await worker.run_match(MatchRequested(match_id=MATCH_ID))

    assert outcome.status == "completed"
    assert outcome.partial is False
    assert outcome.error is None
    assert outcome.trials == len(TRIALS)
    assert outcome.resumed is False
    assert outcome.cost_usd > 0

    # The run row is finished, costed and attributed to the cheap model.
    run = store.runs[MATCH_ID]
    assert run["status"] == "completed"
    assert run["model"] == "claude-haiku-5-5"
    assert run["total_cost_usd"] == pytest.approx(outcome.cost_usd)
    assert run["cached_tokens"] > 0

    # One trial_matches row per trial, each ranked.
    assert len(store.trial_matches) == len(TRIALS)
    ranks = sorted(row["final_rank"] for row in store.trial_matches.values())
    assert ranks == [1, 2, 3]

    # One criterion_judgments row per parsed rule, every citation valid.
    assert len(store.judgments) == TOTAL_CRITERIA
    assert all(row["citations_valid"] for row in store.judgments.values())
    assert all(row["verifier_verdict"] == "supported" for row in store.judgments.values())

    # trial_criteria was populated lazily by parse_rules, once per rule.
    assert len(store.criteria) == TOTAL_CRITERIA

    # A progress event per node, plus the completion event, all via the outbox.
    progress_steps = [
        row["payload"]["step"] for row in store.outbox.values() if row["topic"] == "match.progress"
    ]
    assert progress_steps == list(NODE_ORDER)
    completed = [row for row in store.outbox.values() if row["topic"] == TOPIC_MATCH_COMPLETED]
    assert len(completed) == 1
    assert completed[0]["payload"]["trial_count"] == len(TRIALS)
    assert completed[0]["payload"]["error"] is None

    # The audit log: profile + queries + one check and one verify per trial.
    steps = [row["step"] for row in store.llm_calls]
    assert steps.count("profile_patient") == 1
    assert steps.count("generate_queries") == 1
    assert steps.count("check_rules") == len(TRIALS)
    assert steps.count("verify") == len(TRIALS)
    # The cache actually engaged on the calls after the first.
    assert sum(row["cache_read_tokens"] for row in store.llm_calls) > 0

    # Zero network calls, structurally rather than by hope: the SDK's HTTP
    # client is built lazily by ``AsyncLlmClient.raw``, and after eight
    # replayed calls it has never been built. (This is also why the Claude
    # side of these tests is not stubbed with respx: anthropic 2.6.0 is built
    # on httpx2, which respx does not intercept, so an respx-based assertion
    # about Claude would pass while proving nothing. Replay reads a file.)
    assert worker.llm_client._client is None  # type: ignore[attr-defined]
    # The per-step cost model, pinned so it cannot drift silently: Haiku
    # everywhere except the eligibility decision, which runs on Sonnet because
    # its output is the product rather than a paraphrase of one.
    # "standard" is Sonnet's tier; "haiku_base" is Haiku under the 100K prompt
    # cliff, where its rates jump 5x.
    by_step = {row["step"]: row["billed_tier"] for row in store.llm_calls}
    assert by_step["check_rules"] == "standard", "the decision must run on Sonnet"
    assert all(
        row["billed_tier"] == "haiku_base"
        for row in store.llm_calls
        if row["step"] != "check_rules"
    ), "every other step stays on Haiku, under the pricing cliff"
    assert "check_rules" in by_step, "the decision step must actually have run"


@respx.mock
async def test_the_verdict_follows_the_rules_not_the_model(tmp_path: Path) -> None:
    """A single met exclusion criterion excludes a trial, whatever else passed."""
    first, second, third = TRIALS
    seed_pipeline_fixtures(
        tmp_path,
        labels={first.nct_id: {f"{first.nct_id}:exc:01": "met"}},
    )
    _stub_retrieval()
    store = InMemoryStore.seeded()
    worker = _worker(store, tmp_path)

    await worker.run_match(MatchRequested(match_id=MATCH_ID))

    categories = {nct: row["category"] for (_, nct), row in store.trial_matches.items()}
    assert categories[first.nct_id] == "excluded"
    assert categories[second.nct_id] == "eligible"
    assert categories[third.nct_id] == "eligible"
    # An excluded trial is ranked last, not hidden: "we checked and here is why"
    # is an answer.
    assert store.trial_matches[(MATCH_ID, first.nct_id)]["final_rank"] == 3


@respx.mock
async def test_insufficient_info_becomes_a_missing_facts_list(tmp_path: Path) -> None:
    trial = TRIALS[0]
    seed_pipeline_fixtures(
        tmp_path,
        labels={trial.nct_id: {f"{trial.nct_id}:inc:03": "insufficient_info"}},
    )
    _stub_retrieval()
    store = InMemoryStore.seeded()
    await _worker(store, tmp_path).run_match(MatchRequested(match_id=MATCH_ID))

    row = store.trial_matches[(MATCH_ID, trial.nct_id)]
    assert row["category"] == "likely_eligible"
    assert row["missing_facts"], "an abstention must surface as a question to ask"


@respx.mock
async def test_summaries_are_written_for_the_top_trials(tmp_path: Path) -> None:
    """The explain node asks Claude for prose about a verdict already decided."""
    from match_worker._testing import judgments_for, numbered_note
    from match_worker.verdict import derive_trial_judgment

    seed_pipeline_fixtures(tmp_path)
    note = numbered_note()
    explanations = []
    for trial in TRIALS:
        criteria = [
            {"criterion_id": c.criterion_id, "kind": c.kind, "text": c.text} for c in trial.criteria
        ]
        check = judgments_for(criteria)
        derived = derive_trial_judgment(nct_id=trial.nct_id, judgments=check.judgments, note=note)
        explanations.append(
            {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "category": derived.category,
                "confirmed_fraction": derived.confirmed_fraction,
                "judgments": [
                    {
                        "criterion_id": item.criterion_id,
                        "kind": item.kind,
                        "label": item.label,
                        "evidence_sentence_ids": list(item.evidence_sentence_ids),
                        "explanation": item.explanation,
                    }
                    for item in derived.judgments
                ],
                "missing_facts": list(derived.missing_facts),
                "summary": f"Plain-language summary for {trial.nct_id}.",
            }
        )
    seed_explanation_fixtures(tmp_path, note_text=SAMPLE_NOTE_TEXT, explanations=explanations)

    _stub_retrieval()
    store = InMemoryStore.seeded()
    worker = _worker(store, tmp_path, worker=WorkerConfig(explain_top_k=3, relay_enabled=False))
    await worker.run_match(MatchRequested(match_id=MATCH_ID))

    summaries = {row["summary"] for row in store.trial_matches.values()}
    assert summaries == {f"Plain-language summary for {trial.nct_id}." for trial in TRIALS}
    assert [row["step"] for row in store.llm_calls].count("rank_and_explain") == 3


# -------------------------------------------- one trial fails, rest finish --


@respx.mock
async def test_a_trial_that_fails_twice_is_marked_and_the_others_finish(
    tmp_path: Path,
) -> None:
    """The nineteen-out-of-twenty rule, with three trials instead of twenty.

    The failure is a transport error, so the trial really does burn both of its
    attempts before being marked - it is not rejected once on a non-retryable
    error.
    """
    seed_pipeline_fixtures(tmp_path)
    _stub_retrieval()
    doomed = TRIALS[1]
    flaky = FlakyClient(
        inner=replay_client(tmp_path),
        fail_when_variable_contains={doomed.nct_id: 99},
    )
    store = InMemoryStore.seeded()
    worker = _worker(
        store,
        tmp_path,
        client=flaky,
        worker=WorkerConfig(
            explain_top_k=0,
            relay_enabled=False,
            trial_attempts=2,
            trial_retry_base_seconds=0.0,
            trial_retry_max_seconds=0.0,
        ),
    )

    outcome = await worker.run_match(MatchRequested(match_id=MATCH_ID))

    # The run completed, with every trial accounted for.
    assert outcome.status == "completed"
    assert outcome.trials == len(TRIALS)
    # Both attempts were spent on the doomed trial.
    assert flaky.attempts.count(doomed.nct_id) == 2

    rows = {nct: row for (_, nct), row in store.trial_matches.items()}
    assert rows[doomed.nct_id]["summary"].startswith(COULD_NOT_EVALUATE)
    assert rows[doomed.nct_id]["category"] == "insufficient_info"
    assert rows[TRIALS[0].nct_id]["category"] == "eligible"
    assert rows[TRIALS[2].nct_id]["category"] == "eligible"

    # No judgment rows for the trial that could not be judged, all of the
    # others' - a failure is never recorded as a verdict.
    judged_ncts = {nct for (_, nct, _) in store.judgments}
    assert doomed.nct_id not in judged_ncts
    assert len(judged_ncts) == len(TRIALS) - 1

    # The run is marked partial, because one trial's answer is missing.
    assert outcome.partial is True
    assert outcome.error is not None
    assert outcome.error.startswith(PARTIAL_PREFIX)
    assert store.runs[MATCH_ID]["error"].startswith(PARTIAL_PREFIX)


@respx.mock
async def test_a_trial_with_no_parseable_rules_is_marked_not_passed(tmp_path: Path) -> None:
    """A trial with no criteria is not a trial everyone qualifies for."""
    from match_worker._testing import TrialFixture

    blank = TrialFixture(
        nct_id="NCT04000009", title="Registry with no criteria text", eligibility_text=""
    )
    trials = (*TRIALS, blank)
    seed_pipeline_fixtures(tmp_path, trials=trials)
    _stub_retrieval(trials)
    store = InMemoryStore.seeded(trials=trials)

    outcome = await _worker(store, tmp_path).run_match(MatchRequested(match_id=MATCH_ID))

    row = store.trial_matches[(MATCH_ID, blank.nct_id)]
    assert row["summary"].startswith(COULD_NOT_EVALUATE)
    assert row["category"] == "insufficient_info"
    assert outcome.partial is True


# ------------------------------------------------------------ budget breach --


@respx.mock
async def test_a_budget_breach_returns_partial_results_clearly_marked(
    tmp_path: Path,
) -> None:
    """Never a silent truncation: what was judged is returned and labelled."""
    seed_pipeline_fixtures(tmp_path)
    _stub_retrieval()
    store = InMemoryStore.seeded()
    # Enough for the two per-match calls and the first trial, not for all three.
    # Sized against the *decision* model, not the cheap one: check_rules runs on
    # Sonnet, which is roughly 20x Haiku's input price, so a cap calibrated for
    # an all-Haiku pipeline would be spent before the first trial was judged and
    # this test would stop proving anything about partial results.
    worker = _worker(
        store,
        tmp_path,
        worker=WorkerConfig(
            explain_top_k=0,
            relay_enabled=False,
            verify_with_llm=False,
            max_cost_usd=0.009,
        ),
    )

    outcome = await worker.run_match(MatchRequested(match_id=MATCH_ID))

    assert outcome.status == "completed"
    assert outcome.partial is True
    assert outcome.error is not None and outcome.error.startswith(PARTIAL_PREFIX)
    assert BUDGET_EXHAUSTED in outcome.error

    summaries = [row["summary"] or "" for row in store.trial_matches.values()]
    judged = [s for s in summaries if not s.startswith(COULD_NOT_EVALUATE)]
    skipped = [s for s in summaries if s.startswith(COULD_NOT_EVALUATE)]
    assert judged, "the budget must not cut the run off before any trial is judged"
    assert skipped, "the breach must leave a visible mark on the trials it skipped"
    # Every trial still has a row: a skipped trial is reported, not dropped.
    assert len(store.trial_matches) == len(TRIALS)
    # The breach stops the run promptly: at most one trial's worth of overshoot
    # past the cap, never a run that quietly spends several times its budget.
    assert outcome.cost_usd <= 0.009 * 1.5


# ---------------------------------------------------- redelivery and the DLQ --


@respx.mock
async def test_the_same_message_delivered_twice_is_a_no_op(tmp_path: Path) -> None:
    """At-least-once delivery, handled by the checkpoint thread and the keys.

    The second delivery finds the thread finished, so it re-runs nothing - and
    every write it *would* have made goes through a unique key, which the
    identical row counts demonstrate.
    """
    seed_pipeline_fixtures(tmp_path)
    _stub_retrieval()
    store = InMemoryStore.seeded()
    worker = _worker(store, tmp_path)
    message = MatchRequested(match_id=MATCH_ID).model_dump(mode="json")

    first = await worker.handle_message(message)
    counts_after_first = await store.count_rows(MATCH_ID)
    calls_after_first = len(store.llm_calls)

    second = await worker.handle_message(message)
    counts_after_second = await store.count_rows(MATCH_ID)

    assert first.already_complete is False
    assert second.already_complete is True
    assert second.resumed is True
    assert counts_after_second == counts_after_first
    assert len(store.llm_calls) == calls_after_first, "a redelivery must not re-ask Claude"
    assert counts_after_first["trial_matches"] == len(TRIALS)
    assert counts_after_first["criterion_judgments"] == TOTAL_CRITERIA


@respx.mock
async def test_a_message_with_no_job_row_is_parked_immediately(tmp_path: Path) -> None:
    """A permanent condition: waiting cannot make the row appear."""
    seed_pipeline_fixtures(tmp_path)
    _stub_retrieval()
    store = InMemoryStore()  # no match_runs row at all
    worker = _worker(store, tmp_path)

    outcome = await worker.handle_message(
        MatchRequested(match_id="match-does-not-exist").model_dump(mode="json")
    )

    assert outcome.status == "failed"
    dlq = [row for row in store.outbox.values() if row["topic"] == TOPIC_MATCH_DLQ]
    assert len(dlq) == 1
    assert "has no match_runs row" in dlq[0]["payload"]["error"]
    assert store.runs["match-does-not-exist"]["status"] == "failed"


@respx.mock
async def test_a_message_that_keeps_failing_ends_in_the_dlq(tmp_path: Path) -> None:
    """After ``message_attempts`` the message is parked, never silently dropped."""
    seed_pipeline_fixtures(tmp_path)
    respx.post(f"{RETRIEVAL_URL}/search").mock(return_value=httpx.Response(503))
    store = InMemoryStore.seeded()
    worker = _worker(
        store,
        tmp_path,
        worker=WorkerConfig(
            explain_top_k=0,
            relay_enabled=False,
            message_attempts=2,
            message_retry_base_seconds=0.0,
            message_retry_max_seconds=0.0,
            retrieval_attempts=2,
        ),
    )

    outcome = await worker.handle_message(MatchRequested(match_id=MATCH_ID).model_dump(mode="json"))

    assert outcome.status == "failed"
    assert outcome.error is not None and "RetrievalError" in outcome.error
    dlq = [row for row in store.outbox.values() if row["topic"] == TOPIC_MATCH_DLQ]
    assert len(dlq) == 1
    assert store.runs[MATCH_ID]["status"] == "failed"
    # No partial answer was invented for a run that never retrieved anything.
    assert store.trial_matches == {}
