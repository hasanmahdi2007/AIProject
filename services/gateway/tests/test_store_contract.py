"""The parts of the conversation store that need no Redis.

Protocol conformance belongs in the fast run: if the Redis store ever stops
satisfying ``beacon_core.protocols.ConversationStore``, the swap-in-a-Postgres
-store-later plan is already broken and CI should say so without Docker.
"""

from __future__ import annotations

from typing import Any

import pytest
from redis.asyncio import Redis

from beacon_core.protocols import ConversationStore
from beacon_core.redis import key_conversation
from gateway.conversations import RedisConversationStore
from gateway.sse import SSE_MEDIA_TYPE, format_sse, heartbeat


class RecordingRedis:
    """Records calls instead of making them. Enough to prove that an empty
    append does not reach Redis at all."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def pipeline(self, *, transaction: bool = True) -> Any:
        self.calls.append(f"pipeline(transaction={transaction})")
        raise AssertionError("no round trip should happen for an empty append")

    async def delete(self, *keys: str) -> int:
        self.calls.append(f"delete({keys!r})")
        return len(keys)


def test_the_redis_store_satisfies_the_protocol() -> None:
    store = RedisConversationStore(Redis(), ttl_seconds=60)

    assert isinstance(store, ConversationStore)


async def test_appending_nothing_is_a_no_op() -> None:
    recorder = RecordingRedis()
    store = RedisConversationStore(recorder, ttl_seconds=60)  # type: ignore[arg-type]

    await store.append("conv-1")

    assert recorder.calls == []


def test_the_key_format_is_the_contract_helper() -> None:
    assert key_conversation("abc") == "chat:abc"


def test_sse_frames_are_named_events_with_json_data() -> None:
    assert format_sse("token", {"text": "hi"}) == 'event: token\ndata: {"text":"hi"}\n\n'
    assert format_sse("done", {"ok": True}) == 'event: done\ndata: {"ok":true}\n\n'


def test_an_sse_frame_survives_a_payload_with_newlines() -> None:
    """A raw newline inside ``data:`` would split one event into two. JSON
    escaping is what prevents that, which is why every payload is an object."""
    frame = format_sse("token", {"text": "line one\nline two"})

    assert frame.count("\n\n") == 1
    assert frame.removesuffix("\n\n").splitlines() == [
        "event: token",
        'data: {"text":"line one\\nline two"}',
    ]


def test_a_heartbeat_is_a_comment_not_an_event() -> None:
    assert heartbeat().startswith(":")
    assert "event:" not in heartbeat()


def test_the_media_type_is_event_stream() -> None:
    """Defect 3's other half: the spike served ``text/plain``, so a browser
    would not treat the response as SSE at all."""
    assert SSE_MEDIA_TYPE == "text/event-stream"


@pytest.mark.parametrize("event", ["token", "error", "done"])
def test_the_three_chat_events_frame_cleanly(event: str) -> None:
    frame = format_sse(event, {"k": "v"})  # type: ignore[arg-type]

    assert frame.startswith(f"event: {event}\n")
    assert frame.endswith("\n\n")
