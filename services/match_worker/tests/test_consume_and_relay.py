"""Offset commits, and the outbox relay.

The assertion that matters in this file is an *ordering* one: the offset for a
message moves only after that message's work is durable. Getting it backwards
costs nothing in a happy run and silently loses jobs on a crash, which is why
it is asserted against a recording consumer rather than reasoned about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import respx
from aiokafka.structs import TopicPartition

from beacon_core.events import TOPIC_MATCH_PROGRESS, TOPIC_MATCH_REQUESTED, MatchRequested
from beacon_core.settings import Settings
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
from match_worker.relay import OutboxRelay
from match_worker.retrieval_client import RetrievalClient
from match_worker.runner import MatchWorker

RETRIEVAL_URL = "http://retrieval.invalid"
PARTITION = TopicPartition(TOPIC_MATCH_REQUESTED, 0)


@dataclass
class FakeMessage:
    value: dict[str, Any]
    offset: int
    headers: tuple[tuple[str, bytes], ...] = ()


@dataclass
class RecordingConsumer:
    """Hands out one batch, then nothing, and records every commit."""

    batches: list[dict[TopicPartition, list[FakeMessage]]]
    commits: list[dict[TopicPartition, int]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    observe: Any = None

    async def getmany(self, timeout_ms: int) -> dict[TopicPartition, list[FakeMessage]]:
        return self.batches.pop(0) if self.batches else {}

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.commits.append({partition: offset for partition, offset in offsets.items()})
        if self.observe is not None:
            self.observations.append(await self.observe())


@dataclass
class RecordingProducer:
    sent: list[tuple[str, dict[str, Any], str | None]] = field(default_factory=list)
    fail_topics: set[str] = field(default_factory=set)

    async def send_and_wait(self, topic: str, *, value: dict[str, Any], key: str | None) -> None:
        if topic in self.fail_topics:
            raise RuntimeError("broker unavailable")
        self.sent.append((topic, value, key))


def _worker(store: InMemoryStore, fixtures: Path, consumer: Any) -> MatchWorker:
    return MatchWorker(
        store=store,
        llm_client=replay_client(fixtures),
        graph=compile_graph(),
        settings=Settings(llm_fixtures_dir=fixtures),
        worker=WorkerConfig(explain_top_k=0, relay_enabled=False, poll_timeout_ms=1),
        retrieval=RetrievalClient(base_url=RETRIEVAL_URL),
        consumer=consumer,
    )


@respx.mock
async def test_the_offset_moves_only_after_the_work_is_durable(tmp_path: Path) -> None:
    """The contract of the whole service: checkpoint first, commit second."""
    seed_pipeline_fixtures(tmp_path)
    respx.post(f"{RETRIEVAL_URL}/search").mock(
        return_value=httpx.Response(200, json=search_response().model_dump(mode="json"))
    )
    store = InMemoryStore.seeded()
    message = FakeMessage(value=MatchRequested(match_id=MATCH_ID).model_dump(mode="json"), offset=7)
    consumer = RecordingConsumer(batches=[{PARTITION: [message]}])

    async def observe() -> dict[str, Any]:
        return {
            "status": store.runs[MATCH_ID]["status"],
            "judgments": len(store.judgments),
            "completion_events": sum(
                1 for row in store.outbox.values() if row["topic"] == "match.completed"
            ),
        }

    consumer.observe = observe
    worker = _worker(store, tmp_path, consumer)

    handled = await worker.consume_forever(max_messages=1)

    assert handled == 1
    # Exactly one commit, for this partition only, at offset+1.
    assert consumer.commits == [{PARTITION: 8}]
    # ...and at the moment it happened, the run was already finished and its
    # rows and completion event already written.
    assert consumer.observations == [
        {
            "status": "completed",
            "judgments": sum(len(trial.criteria) for trial in TRIALS),
            "completion_events": 1,
        }
    ]


@respx.mock
async def test_a_failed_message_still_commits_after_being_parked(tmp_path: Path) -> None:
    """A poison message must not hold its partition hostage forever.

    It is parked in the DLQ *and* the offset moves, because an uncommitted
    offset is redelivered indefinitely and the next job behind it never runs.
    """
    seed_pipeline_fixtures(tmp_path)
    store = InMemoryStore()  # no job row: a permanent failure
    message = FakeMessage(
        value=MatchRequested(match_id="match-missing").model_dump(mode="json"), offset=0
    )
    consumer = RecordingConsumer(batches=[{PARTITION: [message]}])
    worker = _worker(store, tmp_path, consumer)

    await worker.consume_forever(max_messages=1)

    assert consumer.commits == [{PARTITION: 1}]
    assert any(row["topic"] == "match.dlq" for row in store.outbox.values())


# -------------------------------------------------------------------- relay --


async def test_the_relay_publishes_outbox_rows_once_and_marks_them() -> None:
    store = InMemoryStore.seeded()
    await store.enqueue_event(
        event_id="e1", topic=TOPIC_MATCH_PROGRESS, partition_key=MATCH_ID, payload={"pct": 10}
    )
    await store.enqueue_event(
        event_id="e2", topic=TOPIC_MATCH_PROGRESS, partition_key=MATCH_ID, payload={"pct": 20}
    )
    producer = RecordingProducer()
    relay = OutboxRelay(source=store, producer=producer)

    published = await relay.drain_once()

    assert published == 2
    assert [topic for topic, _, _ in producer.sent] == [TOPIC_MATCH_PROGRESS] * 2
    # The partition key is the match ID, so one match's events stay ordered.
    assert {key for _, _, key in producer.sent} == {MATCH_ID}
    # A second pass finds nothing: published rows are marked, not re-sent.
    assert await relay.drain_once() == 0
    assert len(producer.sent) == 2


async def test_a_broker_failure_leaves_the_row_unpublished_and_counted() -> None:
    """Unpublished is the safe state: the relay will try again."""
    store = InMemoryStore.seeded()
    await store.enqueue_event(
        event_id="e1", topic=TOPIC_MATCH_PROGRESS, partition_key=MATCH_ID, payload={"pct": 10}
    )
    producer = RecordingProducer(fail_topics={TOPIC_MATCH_PROGRESS})
    relay = OutboxRelay(source=store, producer=producer)

    assert await relay.drain_once() == 0
    row = store.outbox["e1"]
    assert row["published_at"] is None
    assert row["attempts"] == 1
    assert "broker unavailable" in row["last_error"]


async def test_a_duplicate_event_id_is_never_queued_twice() -> None:
    """``outbox.event_id`` is unique: this is what makes a re-run idempotent."""
    store = InMemoryStore.seeded()
    first = await store.enqueue_event(
        event_id="same", topic=TOPIC_MATCH_PROGRESS, partition_key=MATCH_ID, payload={"pct": 10}
    )
    second = await store.enqueue_event(
        event_id="same", topic=TOPIC_MATCH_PROGRESS, partition_key=MATCH_ID, payload={"pct": 10}
    )
    assert first is True
    assert second is False
    assert len(store.outbox) == 1
