"""``GET /v1/matches/{id}/events``: typed SSE over the match pub/sub channel."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from beacon_core.events import MatchCompleted, MatchProgress
from beacon_core.models import MatchRun
from beacon_core.redis import chan_match
from gateway.deps import provide_redis, provide_sessionmaker
from gateway.main import create_app
from gateway.testing import FakeRedis, FakeSession, FakeSessionmaker

MATCH_ID = "match-1"


def _message(payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "message", "channel": chan_match(MATCH_ID), "data": json.dumps(payload)}


def _frames(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for frame in text.split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        head, _, body = frame.partition("\n")
        out.append((head.removeprefix("event: "), body.removeprefix("data: ")))
    return out


def _client(
    *,
    status: str = "running",
    messages: list[dict[str, Any] | None] | None = None,
    rows: dict[tuple[str, str], Any] | None = None,
) -> tuple[TestClient, FakeRedis]:
    app = create_app()
    session = FakeSession(
        rows=rows
        if rows is not None
        else {("MatchRun", MATCH_ID): MatchRun(id=MATCH_ID, status=status, current_step="retrieve")}
    )
    redis = FakeRedis(pubsub_messages=messages)
    app.dependency_overrides[provide_sessionmaker] = lambda: FakeSessionmaker(session)
    app.dependency_overrides[provide_redis] = lambda: redis
    return TestClient(app), redis


def test_unknown_match_is_404() -> None:
    client, redis = _client(rows={})

    with client:
        assert client.get(f"/v1/matches/{MATCH_ID}/events").status_code == 404

    # No channel was subscribed for a run that does not exist.
    assert redis.pubsubs == []


def test_a_finished_run_gets_a_snapshot_and_closes() -> None:
    """A client connecting after the run finished must not hang waiting for an
    event that will never be published."""
    client, _ = _client(status="completed")

    with client:
        response = client.get(f"/v1/matches/{MATCH_ID}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Match-Id"] == MATCH_ID
    assert [name for name, _ in _frames(response.text)] == ["status", "done"]


def test_progress_then_completion_is_forwarded_as_typed_events() -> None:
    progress = MatchProgress(match_id=MATCH_ID, step="retrieve", pct=40, message="searching")
    completed = MatchCompleted(match_id=MATCH_ID, status="completed", trial_count=12)
    client, redis = _client(
        messages=[
            _message(progress.model_dump(mode="json")),
            _message(completed.model_dump(mode="json")),
        ]
    )

    with client:
        response = client.get(f"/v1/matches/{MATCH_ID}/events")

    frames = _frames(response.text)
    assert [name for name, _ in frames] == ["status", "progress", "completed", "done"]

    snapshot = json.loads(frames[0][1])
    assert snapshot == {"match_id": MATCH_ID, "status": "running", "current_step": "retrieve"}
    assert json.loads(frames[1][1])["pct"] == 40
    assert json.loads(frames[2][1])["trial_count"] == 12

    pubsub = redis.pubsubs[0]
    assert pubsub.subscribed == [chan_match(MATCH_ID)]
    assert pubsub.unsubscribed == [chan_match(MATCH_ID)]
    assert pubsub.closed is True, "the subscription must be released with the request"


def test_an_unparseable_message_is_an_error_event_not_a_progress_event() -> None:
    completed = MatchCompleted(match_id=MATCH_ID, status="completed")
    client, _ = _client(
        messages=[
            {"type": "message", "channel": chan_match(MATCH_ID), "data": "not json"},
            _message(completed.model_dump(mode="json")),
        ]
    )

    with client:
        response = client.get(f"/v1/matches/{MATCH_ID}/events")

    frames = _frames(response.text)
    assert [name for name, _ in frames] == ["status", "error", "completed", "done"]
    assert json.loads(frames[1][1])["code"] == "malformed_event"
