"""The Kafka -> Redis progress bridge, and the SSE stream's Postgres fallback.

Together these are the fix for "the console says Queued forever": the bridge
delivers live events, and the heartbeat re-read guarantees the stream ends even
when an event is lost. Everything here runs on fakes - no Kafka, no Redis.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from fastapi import Request

from beacon_core.redis import chan_match
from gateway.bridge import BRIDGE_TOPICS, ProgressBridge, decode_value
from gateway.router import _match_event_stream
from gateway.schemas import MatchStatusSnapshot
from gateway.testing import FakePubSub, FakeRequest

MATCH_ID = "match-1"


@dataclass
class RecordingPublisher:
    sent: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail: bool = False

    async def publish(self, channel: str, message: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        self.sent.append((channel, json.loads(message)))
        return 1


@dataclass
class Message:
    value: Any


class ScriptedConsumer:
    """Yields its messages, then blocks like a quiet topic would."""

    def __init__(self, messages: list[Any], *, start_error: Exception | None = None) -> None:
        self._messages = messages
        self._start_error = start_error
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Message]:
        for value in self._messages:
            yield Message(value)
        await asyncio.Event().wait()


def _frames(chunks: list[str]) -> list[str]:
    return [c.partition("\n")[0].removeprefix("event: ") for c in chunks if c.startswith("event:")]


# ----------------------------------------------------------------- bridge ----


def test_the_bridge_listens_to_every_event_the_console_needs() -> None:
    assert set(BRIDGE_TOPICS) == {"match.progress", "match.completed", "match.dlq"}


async def test_an_event_is_published_to_its_match_channel() -> None:
    publisher = RecordingPublisher()
    bridge = ProgressBridge(consumer_factory=lambda: ScriptedConsumer([]), publisher=publisher)
    event = {"event_type": "match.progress", "match_id": MATCH_ID, "step": "retrieve", "pct": 40}

    assert await bridge.forward(event) is True
    assert publisher.sent == [(chan_match(MATCH_ID), event)]


@pytest.mark.parametrize("value", [None, [], "text", {"no": "match id"}, {"match_id": ""}])
async def test_an_event_without_a_match_id_is_skipped(value: Any) -> None:
    publisher = RecordingPublisher()
    bridge = ProgressBridge(consumer_factory=lambda: ScriptedConsumer([]), publisher=publisher)

    assert await bridge.forward(value) is False
    assert publisher.sent == []


@pytest.mark.parametrize("raw", [b"not json", b"\xff\xfe", b"[1, 2]", None])
def test_junk_bytes_decode_to_none_instead_of_raising(raw: bytes | None) -> None:
    """A raising deserializer would kill the consumer loop on the first junk
    message - the same failure that once kept the worker from starting."""
    assert decode_value(raw) is None


def test_a_json_object_decodes() -> None:
    assert decode_value(b'{"match_id": "m"}') == {"match_id": "m"}


async def test_run_reconnects_after_kafka_is_unavailable() -> None:
    """Kafka down at startup must not kill the bridge (or the gateway)."""
    publisher = RecordingPublisher()
    event = {"match_id": MATCH_ID, "status": "completed"}
    consumers = iter(
        [
            ScriptedConsumer([], start_error=ConnectionError("no broker")),
            ScriptedConsumer([event]),
        ]
    )
    bridge = ProgressBridge(
        consumer_factory=lambda: next(consumers),
        publisher=publisher,
        retry_base_seconds=0.0,
    )

    task = asyncio.create_task(bridge.run())
    for _ in range(100):
        if publisher.sent:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert publisher.sent == [(chan_match(MATCH_ID), event)]


async def test_a_redis_failure_drops_one_event_and_keeps_consuming() -> None:
    publisher = RecordingPublisher(fail=True)
    bridge = ProgressBridge(
        consumer_factory=lambda: ScriptedConsumer([]),
        publisher=publisher,
    )
    consumer = ScriptedConsumer([{"match_id": "a"}, {"match_id": "b"}])

    task = asyncio.create_task(bridge.consume(consumer))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not task.done(), "a publish failure must not end the consume loop"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelling_the_bridge_stops_its_consumer() -> None:
    consumer = ScriptedConsumer([])
    bridge = ProgressBridge(consumer_factory=lambda: consumer, publisher=RecordingPublisher())

    task = asyncio.create_task(bridge.run())
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert consumer.started and consumer.stopped


# --------------------------------------------------- the SSE fallback ----


def _snapshot(status: str, step: str | None = "retrieve") -> MatchStatusSnapshot:
    return MatchStatusSnapshot.model_validate(
        {"match_id": MATCH_ID, "status": status, "current_step": step}
    )


async def _collect(refresh: Any, *, disconnect_after: int = 50) -> list[str]:
    pubsub = FakePubSub([])  # nothing ever arrives over pub/sub
    return [
        chunk
        async for chunk in _match_event_stream(
            MATCH_ID,
            _snapshot("queued", None),
            cast("Any", pubsub),
            cast("Request", FakeRequest(disconnect_after=disconnect_after)),
            heartbeat_seconds=0.0,
            refresh=refresh,
        )
    ]


async def test_a_lost_completed_event_still_ends_the_stream() -> None:
    """The bug this fixes: with no event on the channel, the stream used to
    heartbeat forever and the console sat on Queued."""
    states = iter([_snapshot("running", "retrieve"), _snapshot("completed", "rank_and_explain")])

    async def refresh() -> MatchStatusSnapshot:
        return next(states)

    chunks = await _collect(refresh)

    assert _frames(chunks) == ["status", "status", "status", "done"]
    assert json.loads(chunks[-1].split("data: ", 1)[1])["status"] == "completed"


async def test_an_unchanged_status_sends_no_extra_frame() -> None:
    async def refresh() -> MatchStatusSnapshot:
        return _snapshot("queued", None)

    chunks = await _collect(refresh, disconnect_after=5)

    assert _frames(chunks) == ["status"]


async def test_a_failing_refresh_is_skipped_not_fatal() -> None:
    calls = 0

    async def refresh() -> MatchStatusSnapshot:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("postgres blip")
        return _snapshot("failed", "retrieve")

    chunks = await _collect(refresh)

    assert _frames(chunks) == ["status", "status", "done"]
