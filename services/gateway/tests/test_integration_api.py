"""End-to-end against a real Postgres and a real Redis.

Marked ``integration``: these need the ad-hoc stack from the A2 brief with
``alembic upgrade head`` already applied, so they are excluded from the fast
unit run that CI gates on.

    BEACON_POSTGRES__PORT=55432 BEACON_REDIS__PORT=56379 \\
        uv run pytest services/gateway -m integration
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from beacon_core.events import (
    TOPIC_MATCH_REQUESTED,
    MatchCompleted,
    MatchProgress,
    MatchRequested,
)
from beacon_core.redis import chan_match
from beacon_core.settings import Settings
from gateway.deps import provide_redis, provide_sessionmaker
from gateway.main import create_app
from gateway.router import _match_event_stream
from gateway.schemas import MatchStatusSnapshot
from gateway.testing import INTEGRATION_HINT, FakeRequest

pytestmark = pytest.mark.integration

NOTE = "58-year-old man, stage IV NSCLC, EGFR exon 19 deletion, ECOG 1."


@pytest.fixture
async def live_redis() -> AsyncIterator[Redis]:
    redis: Redis = Redis.from_url(Settings().redis.url, decode_responses=True)
    try:
        await redis.ping()
    except Exception as exc:
        await redis.aclose()
        pytest.skip(f"Redis unreachable: {exc}. {INTEGRATION_HINT}")
    try:
        yield redis
    finally:
        await redis.aclose()


@pytest.fixture
async def live_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(Settings().postgres.dsn())
    try:
        async with engine.connect():
            pass
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Postgres unreachable: {exc}. {INTEGRATION_HINT}")
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def api(live_engine: AsyncEngine, live_redis: Redis) -> AsyncIterator[httpx.AsyncClient]:
    """An ASGI client in *this* event loop.

    Not ``TestClient``: it drives the app from a portal thread with its own
    loop, and an asyncpg pool created here cannot be used from there - the
    whole suite comes back 503. Everything in one loop instead.
    """
    maker = async_sessionmaker(live_engine, expire_on_commit=False)
    app = create_app()
    app.dependency_overrides[provide_sessionmaker] = lambda: maker
    app.dependency_overrides[provide_redis] = lambda: live_redis
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


async def test_health_reports_both_dependencies_against_real_services(
    api: httpx.AsyncClient,
) -> None:
    response = await api.get("/health")

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert body["dependencies"]["redis"]["status"] == "ok"


async def test_a_submitted_match_commits_the_run_and_its_outbox_row_together(
    api: httpx.AsyncClient, live_engine: AsyncEngine
) -> None:
    response = await api.post("/v1/matches", json={"patient_note": NOTE})

    assert response.status_code == 202
    body: dict[str, Any] = response.json()
    match_id = body["match_id"]

    # Read with a *different* session: only committed state is visible.
    maker = async_sessionmaker(live_engine, expire_on_commit=False)
    async with maker() as session:
        run = (
            await session.execute(
                text("SELECT status, patient_id FROM match_runs WHERE id = :id"),
                {"id": match_id},
            )
        ).one()
        event_row = (
            await session.execute(
                text(
                    "SELECT topic, partition_key, payload, published_at "
                    "FROM outbox WHERE event_id = :event_id"
                ),
                {"event_id": body["event_id"]},
            )
        ).one()
        note = (
            await session.execute(
                text("SELECT note_text FROM patients WHERE id = :id"),
                {"id": body["patient_id"]},
            )
        ).scalar_one()

    assert run.status == "queued"
    assert run.patient_id == body["patient_id"]
    assert note == NOTE

    assert event_row.topic == TOPIC_MATCH_REQUESTED
    assert event_row.partition_key == match_id
    assert event_row.published_at is None, "the relay publishes it, not the route"
    assert MatchRequested.model_validate(event_row.payload).match_id == match_id


async def test_a_rejected_match_leaves_no_half_written_rows(
    api: httpx.AsyncClient, live_engine: AsyncEngine
) -> None:
    """The rollback half of the same transaction: an unknown patient with no
    note must leave nothing behind - no run, and above all no outbox row that
    would start a job for a patient that does not exist."""
    maker = async_sessionmaker(live_engine, expire_on_commit=False)
    async with maker() as session:
        before = await _counts(session)

    response = await api.post("/v1/matches", json={"patient_id": "does-not-exist"})
    assert response.status_code == 404

    async with maker() as session:
        after = await _counts(session)

    assert after == before


async def _counts(session: AsyncSession) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in ("match_runs", "outbox", "patients"):
        counts[table] = (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()
    return counts


async def test_the_event_stream_forwards_a_real_redis_publish(live_redis: Redis) -> None:
    """A real pub/sub round trip through the generator the route returns.

    Driving the generator rather than the ASGI response keeps the test
    deterministic: the publishes are already queued on the subscription, which
    is only true because the route subscribes *before* it returns a response.
    """
    match_id = f"it-{uuid.uuid4()}"
    channel = chan_match(match_id)
    pubsub = live_redis.pubsub()
    await pubsub.subscribe(channel)

    await live_redis.publish(
        channel,
        MatchProgress(match_id=match_id, step="retrieve", pct=40).model_dump_json(),
    )
    await live_redis.publish(
        channel,
        MatchCompleted(match_id=match_id, status="completed", trial_count=7).model_dump_json(),
    )

    snapshot = MatchStatusSnapshot(match_id=match_id, status="running", current_step="retrieve")
    frames = [
        frame
        async for frame in _match_event_stream(
            match_id,
            snapshot,
            pubsub,
            cast("Request", FakeRequest()),
            heartbeat_seconds=5.0,
        )
    ]

    # Heartbeats are dropped: redis-py reports the subscribe confirmation as
    # "no message arrived", which the stream answers with a keep-alive comment.
    names = [
        frame.partition("\n")[0].removeprefix("event: ")
        for frame in frames
        if not frame.startswith(":")
    ]
    assert names == ["status", "progress", "completed", "done"]
    assert pubsub.subscribed is False, "the subscription must be released with the request"
