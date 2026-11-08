"""``POST /v1/matches/{id}/chat``.

What this file exists to protect: the Claude request is issued *before* the
``StreamingResponse`` is built, so a 429 from Claude is a 429 from the gateway
rather than a 200 whose body contains an apology. The exception chain is
specific-first; ``RateLimitError`` and ``BadRequestError`` subclass
``APIStatusError``, so reordering it would collapse 429 and 400 into 502 and
no test but this one would notice.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from beacon_core.models import MatchRun
from gateway.deps import (
    provide_anthropic_client,
    provide_conversation_store,
    provide_sessionmaker,
)
from gateway.main import create_app
from gateway.testing import FakeAnthropic, FakeSession, FakeSessionmaker, InMemoryConversationStore

MATCH_ID = "match-1"


def _status_error(
    status_code: int, cls: type[anthropic.APIStatusError]
) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request, json={"error": {"message": "nope"}})
    return cls("boom", response=response, body=None)


def _connection_error() -> anthropic.APIConnectionError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="no route to host", request=request)


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


def build_client(
    fake: FakeAnthropic,
    store: InMemoryConversationStore,
    *,
    rows: dict[tuple[str, str], Any] | None = None,
) -> TestClient:
    app = create_app()
    session = FakeSession(
        rows=rows
        if rows is not None
        else {("MatchRun", MATCH_ID): MatchRun(id=MATCH_ID, status="running")}
    )
    app.dependency_overrides[provide_sessionmaker] = lambda: FakeSessionmaker(session)
    app.dependency_overrides[provide_conversation_store] = lambda: store
    app.dependency_overrides[provide_anthropic_client] = lambda: fake
    return TestClient(app)


@pytest.fixture
def default_client(store: InMemoryConversationStore) -> Iterator[TestClient]:
    with build_client(FakeAnthropic(chunks=["hi", " there"]), store) as client:
        yield client


def test_reply_is_an_event_stream(
    default_client: TestClient, store: InMemoryConversationStore
) -> None:
    response = default_client.post(f"/v1/matches/{MATCH_ID}/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Conversation-Id"] == MATCH_ID
    assert response.headers["Cache-Control"] == "no-cache"

    body = response.text
    assert 'event: token\ndata: {"text":"hi"}' in body
    assert "event: done" in body
    assert store.snapshot(MATCH_ID) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_conversation_id_defaults_to_the_match_but_can_be_overridden(
    store: InMemoryConversationStore,
) -> None:
    with build_client(FakeAnthropic(chunks=["ok"]), store) as client:
        response = client.post(
            f"/v1/matches/{MATCH_ID}/chat",
            json={"message": "hello", "conversation_id": "thread-9"},
        )

    assert response.headers["X-Conversation-Id"] == "thread-9"
    assert store.snapshot("thread-9")
    assert store.snapshot(MATCH_ID) == []


def test_unknown_match_is_404_and_claude_is_never_called(
    store: InMemoryConversationStore,
) -> None:
    fake = FakeAnthropic()
    with build_client(fake, store, rows={}) as client:
        response = client.post(f"/v1/matches/{MATCH_ID}/chat", json={"message": "hello"})

    assert response.status_code == 404
    assert fake.calls == []


def test_blank_message_is_422(default_client: TestClient) -> None:
    assert (
        default_client.post(f"/v1/matches/{MATCH_ID}/chat", json={"message": ""}).status_code == 422
    )


@pytest.mark.parametrize(
    ("raises", "expected_status"),
    [
        (_status_error(429, anthropic.RateLimitError), 429),
        (_status_error(400, anthropic.BadRequestError), 400),
        (_status_error(529, anthropic.APIStatusError), 502),
        (_connection_error(), 503),
    ],
    ids=["rate_limit", "bad_request", "api_status", "connection"],
)
def test_specific_first_exception_chain(
    store: InMemoryConversationStore, raises: BaseException, expected_status: int
) -> None:
    """A failure before the first byte is a real status code, never a 200 with
    the error in the body - and nothing is written to history."""
    with build_client(FakeAnthropic(raises=raises), store) as client:
        response = client.post(f"/v1/matches/{MATCH_ID}/chat", json={"message": "hello"})

    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert store.appends == []


def test_a_midstream_failure_is_a_typed_event_not_a_token(
    store: InMemoryConversationStore,
) -> None:
    """End to end: the response is a 200 event stream (headers were already
    sent when the failure happened), the failure arrives as ``error``, and no
    ``token`` frame carries the message."""
    fake = FakeAnthropic(
        chunks=["Trial NCT0 ", "looks "],
        fail_after=2,
        failure=_status_error(500, anthropic.APIStatusError),
    )
    with build_client(fake, store) as client:
        response = client.post(f"/v1/matches/{MATCH_ID}/chat", json={"message": "hello"})

    assert response.status_code == 200
    frames = [frame for frame in response.text.split("\n\n") if frame]
    names = [frame.partition("\n")[0].removeprefix("event: ") for frame in frames]

    assert names == ["token", "token", "error"]
    assert "claude_http_500" in frames[-1]
    for frame in frames[:-1]:
        assert "claude_http_500" not in frame
        assert "boom" not in frame
    assert store.appends == []
