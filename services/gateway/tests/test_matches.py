"""``POST /v1/matches``: validate, persist, enqueue - in one transaction.

The outbox pattern is pointless unless the domain row and its event are
committed together, so the structural assertion here is "exactly one
transaction, and both rows are inside it". The same claim is checked against a
real Postgres in ``test_integration_matches.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from beacon_core.events import TOPIC_MATCH_REQUESTED, MatchRequested
from beacon_core.models import Patient
from gateway.deps import provide_sessionmaker
from gateway.main import create_app
from gateway.testing import FakeSession, FakeSessionmaker

NOTE = "62-year-old woman with ER+ metastatic breast cancer, ECOG 1."


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def client(session: FakeSession) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[provide_sessionmaker] = lambda: FakeSessionmaker(session)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_accepted_with_a_match_id(client: TestClient, session: FakeSession) -> None:
    response = client.post("/v1/matches", json={"patient_note": NOTE})

    assert response.status_code == 202
    body: dict[str, Any] = response.json()
    assert body["match_id"]
    assert body["status"] == "queued"
    assert response.headers["Location"] == f"/v1/matches/{body['match_id']}"


def test_request_row_and_outbox_row_share_one_transaction(
    client: TestClient, session: FakeSession
) -> None:
    response = client.post("/v1/matches", json={"patient_note": NOTE})
    match_id = response.json()["match_id"]

    # One transaction opened, one commit, no rollback. The flush that orders
    # the patient insert ahead of its foreign key is not a second transaction.
    assert (session.begins, session.commits, session.rollbacks) == (1, 1, 0)
    assert session.flushes == 1

    runs = session.added_of("MatchRun")
    outbox = session.added_of("Outbox")
    patients = session.added_of("Patient")
    assert len(runs) == len(outbox) == len(patients) == 1

    assert runs[0].id == match_id
    assert runs[0].status == "queued"
    assert outbox[0].topic == TOPIC_MATCH_REQUESTED
    assert outbox[0].partition_key == match_id, "events for one job must stay ordered"

    # The payload is the frozen contract, not an ad-hoc dict.
    event = MatchRequested.model_validate(outbox[0].payload)
    assert event.match_id == match_id
    assert event.event_id == response.json()["event_id"]
    assert json.dumps(outbox[0].payload)  # JSON-serialisable for Kafka


def test_nothing_is_published_outside_the_transaction(
    client: TestClient, session: FakeSession
) -> None:
    """Every write went through ``session.add`` inside the one transaction;
    the route never produces to Kafka itself. That is the relay's job."""
    client.post("/v1/matches", json={"patient_note": NOTE})

    assert {type(obj).__name__ for obj in session.added} == {"Patient", "MatchRun", "Outbox"}
    assert session.begins == 1


def test_known_patient_id_is_reused_without_a_new_note(
    client: TestClient, session: FakeSession
) -> None:
    session.rows[("Patient", "trec-2021-7")] = Patient(id="trec-2021-7", note_text=NOTE)

    response = client.post("/v1/matches", json={"patient_id": "trec-2021-7"})

    assert response.status_code == 202
    assert response.json()["patient_id"] == "trec-2021-7"
    assert session.added_of("Patient") == []
    assert len(session.added_of("MatchRun")) == 1
    assert len(session.added_of("Outbox")) == 1


def test_unknown_patient_id_without_a_note_is_404(client: TestClient, session: FakeSession) -> None:
    response = client.post("/v1/matches", json={"patient_id": "nope"})

    assert response.status_code == 404
    # The transaction was opened and rolled back: nothing queued.
    assert session.commits == 0
    assert session.rollbacks == 1


def test_an_empty_body_is_422(client: TestClient) -> None:
    assert client.post("/v1/matches", json={}).status_code == 422


def test_a_blank_note_is_422(client: TestClient) -> None:
    assert client.post("/v1/matches", json={"patient_note": ""}).status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    """``extra="forbid"`` on the wire models: a misspelled field is a 422, not
    a silently ignored option."""
    response = client.post("/v1/matches", json={"patient_note": NOTE, "conig_version": "v2"})

    assert response.status_code == 422
