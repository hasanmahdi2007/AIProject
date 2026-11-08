"""``GET /health`` reports each dependency separately.

A probe that only answers "ok" is worse than none: it is the per-dependency
detail that tells you whether to restart Redis or look at Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from beacon_core.models import MatchRun
from gateway.deps import provide_redis, provide_sessionmaker
from gateway.main import create_app
from gateway.testing import FakeRedis, FakeSession, FakeSessionmaker


@pytest.fixture
def client_parts() -> Iterator[tuple[TestClient, FakeSession, FakeRedis]]:
    app = create_app()
    session = FakeSession(rows={("MatchRun", "known"): MatchRun(id="known", status="queued")})
    redis = FakeRedis()
    app.dependency_overrides[provide_sessionmaker] = lambda: FakeSessionmaker(session)
    app.dependency_overrides[provide_redis] = lambda: redis
    with TestClient(app) as client:
        yield client, session, redis
    app.dependency_overrides.clear()


def test_health_reports_both_dependencies_ok(
    client_parts: tuple[TestClient, FakeSession, FakeRedis],
) -> None:
    client, session, redis = client_parts

    response = client.get("/health")

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert body["dependencies"]["redis"]["status"] == "ok"
    # Both were really contacted, not assumed.
    assert len(session.executed) == 1
    assert redis.pings == 1


def test_health_is_503_and_names_the_broken_dependency() -> None:
    app = create_app()
    session = FakeSession()
    redis = FakeRedis(ping_error=ConnectionError("connection refused"))
    app.dependency_overrides[provide_sessionmaker] = lambda: FakeSessionmaker(session)
    app.dependency_overrides[provide_redis] = lambda: redis

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body: dict[str, Any] = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert body["dependencies"]["redis"]["status"] == "error"
    assert "connection refused" in body["dependencies"]["redis"]["detail"]


def test_a_broken_postgres_does_not_hide_a_healthy_redis() -> None:
    app = create_app()
    session = FakeSession()
    session.fail_on_execute = OSError("could not translate host name")
    redis = FakeRedis()
    app.dependency_overrides[provide_sessionmaker] = lambda: FakeSessionmaker(session)
    app.dependency_overrides[provide_redis] = lambda: redis

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body: dict[str, Any] = response.json()
    assert body["dependencies"]["postgres"]["status"] == "error"
    assert body["dependencies"]["redis"]["status"] == "ok"
    assert cast("float", body["dependencies"]["redis"]["latency_ms"]) >= 0
