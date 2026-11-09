"""``GET /v1/matches/{match_id}`` against a real Postgres.

Marked ``integration``: the route is five real queries and an ``IN`` clause, so
the thing worth testing is that they return what the contract promises from
actual rows. Faking ``session.execute`` would assert that the fake agrees with
itself.

    uv run pytest services/gateway -m integration

The stack must be up (``docker compose -f deploy/compose.yaml up -d --wait``)
with ``alembic upgrade head`` applied. Postgres is published on **15432**, not
5432 - the dev machine runs its own PostgreSQL service there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from beacon_core.models import (
    CriterionJudgmentRow,
    MatchRun,
    Outbox,
    Patient,
    Trial,
    TrialMatch,
)
from beacon_core.settings import Settings
from gateway.deps import provide_sessionmaker
from gateway.main import create_app
from gateway.testing import INTEGRATION_HINT

pytestmark = pytest.mark.integration

NOTE = (
    "58-year-old woman with stage IV non-small cell lung cancer. "
    "EGFR exon 19 deletion confirmed on tissue biopsy. "
    "Completed four cycles of carboplatin and pemetrexed in March. "
    "Creatinine 0.9 mg/dL. ECOG performance status 1."
)
SENTENCES = [
    {"sentence_id": 1, "text": "58-year-old woman with stage IV non-small cell lung cancer."},
    {"sentence_id": 2, "text": "EGFR exon 19 deletion confirmed on tissue biopsy."},
    {"sentence_id": 3, "text": "Completed four cycles of carboplatin and pemetrexed in March."},
    {"sentence_id": 4, "text": "Creatinine 0.9 mg/dL."},
    {"sentence_id": 5, "text": "ECOG performance status 1."},
]


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
async def api(live_engine: AsyncEngine) -> AsyncIterator[httpx.AsyncClient]:
    """An ASGI client in *this* event loop - see ``test_integration_api``: an
    asyncpg pool created here cannot be driven from ``TestClient``'s portal
    thread. No Redis override, because this route never touches Redis."""
    maker = async_sessionmaker(live_engine, expire_on_commit=False)
    app = create_app()
    app.dependency_overrides[provide_sessionmaker] = lambda: maker
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
async def finished_run(live_engine: AsyncEngine) -> AsyncIterator[dict[str, str]]:
    """One completed run: a patient with a split note, two judged trials.

    Written straight to Postgres rather than by running the worker - this is a
    test of the read path, and the worker is C1's to test.
    """
    maker = async_sessionmaker(live_engine, expire_on_commit=False)
    ids = {
        "match_id": f"test-{uuid.uuid4()}",
        "patient_id": f"test-{uuid.uuid4()}",
        "nct_a": f"NCT{uuid.uuid4().hex[:8]}",
        "nct_b": f"NCT{uuid.uuid4().hex[:8]}",
    }

    async with maker() as session, session.begin():
        session.add(
            Patient(
                id=ids["patient_id"],
                source="api",
                note_text=NOTE,
                sentences=SENTENCES,
                profile={
                    "age": 58,
                    "sex": "female",
                    "conditions": ["non-small cell lung cancer"],
                    "biomarkers": ["EGFR exon 19 deletion"],
                    "prior_treatments": ["carboplatin", "pemetrexed"],
                    "labs": {"creatinine": "0.9 mg/dL"},
                    "performance_status": 1,
                    "location": None,
                },
            )
        )
        session.add_all(
            [
                Trial(
                    nct_id=ids["nct_a"],
                    title="Osimertinib in EGFR-mutant NSCLC",
                    conditions=["non-small cell lung cancer"],
                    interventions=["osimertinib"],
                    phase="Phase 3",
                    status="recruiting",
                    min_age_years=18.0,
                    sex="all",
                ),
                Trial(
                    nct_id=ids["nct_b"],
                    title="Pediatric neuroblastoma maintenance",
                    conditions=["neuroblastoma"],
                    interventions=["dinutuximab"],
                    min_age_years=1.0,
                    max_age_years=17.0,
                    sex="all",
                ),
            ]
        )
        await session.flush()

        session.add(
            MatchRun(
                id=ids["match_id"],
                patient_id=ids["patient_id"],
                status="completed",
                current_step="rank_and_explain",
                total_cost_usd=0.0241,
                duration_ms=88_400.0,
            )
        )
        await session.flush()

        session.add_all(
            [
                TrialMatch(
                    match_run_id=ids["match_id"],
                    nct_id=ids["nct_a"],
                    retrieval_rank=2,
                    rrf_score=0.0312,
                    rerank_score=7.41,
                    final_rank=1,
                    final_score=0.91,
                    category="likely_eligible",
                    confirmed_fraction=0.75,
                    missing_facts=["no brain imaging recorded"],
                    summary="Histology and biomarker both match.",
                ),
                TrialMatch(
                    match_run_id=ids["match_id"],
                    nct_id=ids["nct_b"],
                    retrieval_rank=1,
                    rrf_score=0.0290,
                    rerank_score=2.10,
                    final_rank=2,
                    final_score=0.08,
                    category="excluded",
                    confirmed_fraction=0.0,
                    missing_facts=[],
                    summary="Age range excludes this patient.",
                ),
            ]
        )
        session.add_all(
            [
                CriterionJudgmentRow(
                    match_run_id=ids["match_id"],
                    nct_id=ids["nct_a"],
                    criterion_id="INC-1",
                    kind="inclusion",
                    label="met",
                    evidence_sentence_ids=[1],
                    explanation="Stage IV NSCLC is stated.",
                    citations_valid=True,
                ),
                CriterionJudgmentRow(
                    match_run_id=ids["match_id"],
                    nct_id=ids["nct_a"],
                    criterion_id="INC-2",
                    kind="inclusion",
                    label="met",
                    evidence_sentence_ids=[2],
                    explanation="EGFR exon 19 deletion is confirmed.",
                    citations_valid=True,
                ),
                CriterionJudgmentRow(
                    match_run_id=ids["match_id"],
                    nct_id=ids["nct_a"],
                    criterion_id="EXC-1",
                    kind="exclusion",
                    label="insufficient_info",
                    evidence_sentence_ids=[],
                    explanation="The note does not mention brain metastases.",
                    citations_valid=True,
                ),
                CriterionJudgmentRow(
                    match_run_id=ids["match_id"],
                    nct_id=ids["nct_b"],
                    criterion_id="INC-1",
                    kind="inclusion",
                    label="not_met",
                    evidence_sentence_ids=[1],
                    explanation="Patient is 58; the trial enrols 1-17 years.",
                    citations_valid=True,
                ),
            ]
        )

    try:
        yield ids
    finally:
        async with maker() as session, session.begin():
            # match_runs cascades to trial_matches and criterion_judgments.
            await session.execute(delete(MatchRun).where(MatchRun.id == ids["match_id"]))
            await session.execute(delete(Patient).where(Patient.id == ids["patient_id"]))
            await session.execute(
                delete(Trial).where(Trial.nct_id.in_([ids["nct_a"], ids["nct_b"]]))
            )


@pytest.fixture
async def submitted(
    api: httpx.AsyncClient, live_engine: AsyncEngine
) -> AsyncIterator[dict[str, str]]:
    """A real ``POST /v1/matches``, cleaned up afterwards.

    Without the cleanup every run of this file leaves a patient, a match_run
    and an unpublished outbox row in the database, and an unpublished outbox
    row is not inert - the relay will pick it up and enqueue a job for a
    patient that only ever existed to satisfy a test.
    """
    response = await api.post("/v1/matches", json={"patient_note": NOTE})
    assert response.status_code == 202
    body: dict[str, Any] = response.json()
    ids = {
        "match_id": body["match_id"],
        "patient_id": body["patient_id"],
        "event_id": body["event_id"],
        "location": response.headers["Location"],
    }

    maker = async_sessionmaker(live_engine, expire_on_commit=False)
    try:
        yield ids
    finally:
        async with maker() as session, session.begin():
            await session.execute(delete(Outbox).where(Outbox.event_id == ids["event_id"]))
            await session.execute(delete(MatchRun).where(MatchRun.id == ids["match_id"]))
            await session.execute(delete(Patient).where(Patient.id == ids["patient_id"]))


async def test_a_finished_run_is_readable_after_the_stream_is_long_gone(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    """The gap this route closes: before it, the ``completed`` SSE frame was the
    only delivery of these results, so a reload lost them permanently."""
    response = await api.get(f"/v1/matches/{finished_run['match_id']}")

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["status"] == "completed"
    assert body["current_step"] == "rank_and_explain"
    assert body["trial_count"] == 2
    assert body["total_cost_usd"] == pytest.approx(0.0241)
    assert body["duration_ms"] == pytest.approx(88_400.0)
    assert body["error"] is None


async def test_it_is_readable_more_than_once(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    """A plain GET, not a one-shot delivery: two viewers, or one reload."""
    first = await api.get(f"/v1/matches/{finished_run['match_id']}")
    second = await api.get(f"/v1/matches/{finished_run['match_id']}")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


async def test_the_numbered_note_comes_back_with_the_results(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    """Without the note, ``evidence_sentence_ids`` index nothing. A client must
    never re-split the note itself: a different split yields different sentence
    IDs and the citation then points at the wrong sentence, silently."""
    body = (await api.get(f"/v1/matches/{finished_run['match_id']}")).json()

    assert body["note"] is not None
    assert [s["sentence_id"] for s in body["note"]["sentences"]] == [1, 2, 3, 4, 5]
    assert body["note"]["sentences"][1]["text"].startswith("EGFR exon 19 deletion")


async def test_every_cited_sentence_id_resolves_in_the_returned_note(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    """The invalid-citation rate of this response must be 0%, and it must be
    checkable from the response alone - that is the point of shipping both
    halves together."""
    body = (await api.get(f"/v1/matches/{finished_run['match_id']}")).json()

    valid = {s["sentence_id"] for s in body["note"]["sentences"]}
    cited = [
        sid
        for trial in body["results"]
        for judgment in trial["judgment"]["judgments"]
        for sid in judgment["evidence_sentence_ids"]
    ]

    assert cited, "the fixture cites sentences; an empty list would prove nothing"
    assert set(cited) <= valid, f"unresolvable citations: {sorted(set(cited) - valid)}"


async def test_the_profile_comes_back_typed(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    body = (await api.get(f"/v1/matches/{finished_run['match_id']}")).json()

    assert body["profile"]["age"] == 58, "an int, not the string '58'"
    assert body["profile"]["sex"] == "female"
    assert body["profile"]["biomarkers"] == ["EGFR exon 19 deletion"]


async def test_results_arrive_in_the_workers_final_order_with_their_trials(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    body = (await api.get(f"/v1/matches/{finished_run['match_id']}")).json()
    results = body["results"]

    assert [r["rank"] for r in results] == [1, 2]
    assert results[0]["candidate"]["nct_id"] == finished_run["nct_a"]
    assert results[0]["candidate"]["title"] == "Osimertinib in EGFR-mutant NSCLC"
    assert results[0]["candidate"]["conditions"] == ["non-small cell lung cancer"]
    assert results[0]["candidate"]["retrieval_rank"] == 2, "retrieval order, not final order"
    assert results[0]["judgment"]["category"] == "likely_eligible"
    assert results[1]["judgment"]["category"] == "excluded"


async def test_each_trials_judgments_stay_with_that_trial(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    """The judgments are fetched in one query for the whole run and grouped in
    memory, so a grouping bug would attach one trial's rules to another."""
    body = (await api.get(f"/v1/matches/{finished_run['match_id']}")).json()
    by_nct = {r["candidate"]["nct_id"]: r["judgment"] for r in body["results"]}

    assert [j["criterion_id"] for j in by_nct[finished_run["nct_a"]]["judgments"]] == [
        "INC-1",
        "INC-2",
        "EXC-1",
    ]
    assert [j["criterion_id"] for j in by_nct[finished_run["nct_b"]]["judgments"]] == ["INC-1"]
    for judgment in by_nct.values():
        for rule in judgment["judgments"]:
            assert set(rule) == {
                "criterion_id",
                "kind",
                "label",
                "evidence_sentence_ids",
                "explanation",
            }, "CriterionJudgment is the frozen LLM output schema: exactly five fields"


async def test_an_abstention_is_preserved_rather_than_coerced(
    api: httpx.AsyncClient, finished_run: dict[str, str]
) -> None:
    """``insufficient_info`` with no citation is a success state - it is how the
    system abstains instead of guessing. It must survive the round trip as
    itself, not become ``not_met``."""
    body = (await api.get(f"/v1/matches/{finished_run['match_id']}")).json()
    rules = {
        rule["criterion_id"]: rule
        for trial in body["results"]
        if trial["candidate"]["nct_id"] == finished_run["nct_a"]
        for rule in trial["judgment"]["judgments"]
    }

    assert rules["EXC-1"]["label"] == "insufficient_info"
    assert rules["EXC-1"]["evidence_sentence_ids"] == []
    assert rules["EXC-1"]["kind"] == "exclusion"


async def test_an_unknown_match_id_is_404_not_an_empty_result(
    api: httpx.AsyncClient,
) -> None:
    """An empty 200 would render as "no trials matched", which is a different
    and much worse claim than "no such run"."""
    response = await api.get(f"/v1/matches/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_the_location_header_from_the_202_now_resolves(
    api: httpx.AsyncClient, submitted: dict[str, str]
) -> None:
    """``POST /v1/matches`` has always advertised this path. Until this route
    existed the header pointed at a 404."""
    followed = await api.get(submitted["location"])

    assert followed.status_code == 200
    body: dict[str, Any] = followed.json()
    assert body["match_id"] == submitted["match_id"]
    assert body["status"] == "queued"
    assert body["results"] == [], "nothing has been matched yet"
    assert body["note"] is None, "the worker has not split the note yet"


async def test_a_queued_run_reports_no_note_rather_than_an_empty_one(
    api: httpx.AsyncClient, submitted: dict[str, str]
) -> None:
    """An empty ``NumberedNote`` would read as "a note with no sentences", which
    makes a not-yet look like a data error."""
    body = (await api.get(f"/v1/matches/{submitted['match_id']}")).json()

    assert body["note"] is None
    assert body["profile"] is None
    assert body["trial_count"] == 0
