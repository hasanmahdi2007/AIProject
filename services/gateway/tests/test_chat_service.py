"""Defect 1 (poisoned history) and defect 3 (errors as prose), at the seam
where both were introduced.

Every test here drives :func:`gateway.service.start_reply` directly, because
that is the function the spike got wrong. The router-level consequences are in
``test_chat_route.py``.
"""

from __future__ import annotations

import json
from typing import Any, cast

import anthropic
import httpx2
import pytest

from gateway.service import MODEL, start_reply
from gateway.testing import FakeAnthropic, InMemoryConversationStore

CONVERSATION = "conv-1"


def _api_error(message: str = "upstream exploded") -> anthropic.APIError:
    """An ``APIError`` built the way the SDK builds one, without a network."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIError(message, request, body=None)


def _status_error(
    status_code: int, cls: type[anthropic.APIStatusError]
) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request, json={"error": {"message": "nope"}})
    return cls("boom", response=response, body=None)


async def _drain(events: Any) -> list[tuple[str, dict[str, Any]]]:
    """Collect a framed SSE stream into ``(event name, data)`` pairs."""
    collected: list[tuple[str, dict[str, Any]]] = []
    async for frame in events:
        if frame.startswith(":"):  # heartbeat comment
            continue
        head, _, body = frame.partition("\n")
        collected.append((head.removeprefix("event: "), json.loads(body.removeprefix("data: "))))
    return collected


def _as_client(fake: FakeAnthropic) -> anthropic.AsyncAnthropic:
    return cast("anthropic.AsyncAnthropic", fake)


# ----------------------------------------------------------- the happy path --


async def test_successful_reply_persists_both_turns_in_one_append() -> None:
    store = InMemoryConversationStore()
    fake = FakeAnthropic(chunks=["Hel", "lo"])

    events = await start_reply(CONVERSATION, "hi", store=store, client=_as_client(fake))
    frames = await _drain(events)

    assert [name for name, _ in frames] == ["token", "token", "done"]
    assert [data["text"] for name, data in frames if name == "token"] == ["Hel", "lo"]

    # One append, carrying both turns: that is the contract the Protocol spells
    # out and the reason a failure cannot leave half an exchange behind.
    assert len(store.appends) == 1
    _, turns = store.appends[0]
    assert turns == (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    )
    assert fake.calls[0]["model"] == MODEL


async def test_prior_history_is_sent_to_claude_oldest_first() -> None:
    store = InMemoryConversationStore(
        {
            CONVERSATION: [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ]
        }
    )
    fake = FakeAnthropic(chunks=["ok"])

    events = await start_reply(CONVERSATION, "second", store=store, client=_as_client(fake))
    await _drain(events)

    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]


# ------------------------------------------------- defect 1: poisoned history --


async def test_user_turn_is_not_persisted_before_the_stream_opens() -> None:
    """The exact regression. The spike appended the user turn to the *stored*
    list before the request; here the store must still be untouched at the
    moment Claude is called."""
    store = InMemoryConversationStore()
    seen_at_stream_open: list[list[dict[str, Any]]] = []

    fake = FakeAnthropic(
        chunks=["hi"],
        on_stream=lambda: seen_at_stream_open.append(store.snapshot(CONVERSATION)),
    )

    events = await start_reply(CONVERSATION, "hello", store=store, client=_as_client(fake))

    # The request has been issued (that is where a 429 would surface) and
    # nothing has been written yet.
    assert fake.calls, "the Claude request must be issued before the response"
    assert seen_at_stream_open == [[]]
    assert store.snapshot(CONVERSATION) == []
    assert store.appends == []

    await _drain(events)
    assert len(store.snapshot(CONVERSATION)) == 2


async def test_failure_opening_the_stream_leaves_history_untouched() -> None:
    store = InMemoryConversationStore({CONVERSATION: [{"role": "user", "content": "old"}]})
    fake = FakeAnthropic(raises=_status_error(429, anthropic.RateLimitError))

    with pytest.raises(anthropic.RateLimitError):
        await start_reply(CONVERSATION, "hello", store=store, client=_as_client(fake))

    assert store.snapshot(CONVERSATION) == [{"role": "user", "content": "old"}]
    assert store.appends == []


async def test_midstream_failure_leaves_history_untouched() -> None:
    """A stream that dies after two deltas must persist nothing - not the user
    turn, and not the partial reply."""
    store = InMemoryConversationStore({CONVERSATION: [{"role": "user", "content": "old"}]})
    fake = FakeAnthropic(chunks=["par", "tial"], fail_after=1, failure=_api_error())

    events = await start_reply(CONVERSATION, "hello", store=store, client=_as_client(fake))
    frames = await _drain(events)

    assert [name for name, _ in frames] == ["token", "error"]
    assert store.appends == []
    assert store.snapshot(CONVERSATION) == [{"role": "user", "content": "old"}]

    # ...and the same message can be retried against a clean history.
    retry_client = FakeAnthropic(chunks=["fine"])
    retry = await start_reply(CONVERSATION, "hello", store=store, client=_as_client(retry_client))
    await _drain(retry)
    assert [turn["content"] for turn in store.snapshot(CONVERSATION)] == [
        "old",
        "hello",
        "fine",
    ]


async def test_refusal_is_not_persisted_and_is_not_a_token() -> None:
    store = InMemoryConversationStore()
    fake = FakeAnthropic(chunks=["I "], stop_reason="refusal")

    events = await start_reply(CONVERSATION, "something off", store=store, client=_as_client(fake))
    frames = await _drain(events)

    assert frames[-1][0] == "error"
    assert frames[-1][1]["code"] == "refusal"
    assert store.appends == []


# ------------------------------------------------ defect 3: errors as prose --


async def test_error_never_leaks_into_a_token_payload() -> None:
    """The point of defect 3. A forced mid-stream failure must produce a typed
    ``error`` event, and no ``token`` may carry the failure text - otherwise a
    console renders the outage as if Claude had said it."""
    store = InMemoryConversationStore()
    fake = FakeAnthropic(
        chunks=["The answer is "],
        fail_after=1,
        failure=_api_error("overloaded_error: upstream is down"),
    )

    events = await start_reply(CONVERSATION, "hello", store=store, client=_as_client(fake))
    frames = await _drain(events)

    tokens = [data for name, data in frames if name == "token"]
    errors = [data for name, data in frames if name == "error"]

    assert len(errors) == 1
    assert errors[0]["code"] == "claude_stream_failed"
    assert "upstream is down" in errors[0]["message"]

    # Nothing resembling an error is in the prose channel.
    assert tokens == [{"text": "The answer is "}]
    for token in tokens:
        assert "upstream is down" not in token["text"]
        assert "error" not in token["text"].lower()
        assert not token["text"].startswith("[")

    # And exactly one terminal event, so a client knows it is over.
    assert [name for name, _ in frames].count("done") == 0


async def test_error_event_carries_a_machine_readable_status_code() -> None:
    store = InMemoryConversationStore()
    fake = FakeAnthropic(
        chunks=["a"],
        fail_after=1,
        failure=_status_error(529, anthropic.APIStatusError),
    )

    events = await start_reply(CONVERSATION, "hello", store=store, client=_as_client(fake))
    frames = await _drain(events)

    assert frames[-1][0] == "error"
    assert frames[-1][1]["code"] == "claude_http_529"


async def test_stream_context_is_closed_even_on_failure() -> None:
    store = InMemoryConversationStore()
    fake = FakeAnthropic(chunks=["a"], fail_after=1, failure=_api_error())

    events = await start_reply(CONVERSATION, "hello", store=store, client=_as_client(fake))
    await _drain(events)

    assert fake.managers[0].closed is True
