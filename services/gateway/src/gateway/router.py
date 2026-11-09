"""The gateway's HTTP surface.

Five routes: a health probe that actually probes, a match submission that is
one database transaction, a result read that survives a page reload, an SSE
progress stream, and the migrated chat endpoint.

Two things in here were already right in the spike and are deliberately
preserved:

* the Claude stream is opened *before* the ``StreamingResponse`` is
  constructed, so an auth/rate-limit/validation failure is a real HTTP status
  code instead of a 200 with an apology in the body;
* the exception chain is **specific first**. ``RateLimitError`` and
  ``BadRequestError`` are subclasses of ``APIStatusError``, so reordering these
  blocks silently collapses 429 and 400 into 502.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Sequence
from typing import Any, Final, cast

import anthropic
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio.client import PubSub
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from beacon_core.events import TOPIC_MATCH_COMPLETED, TOPIC_MATCH_REQUESTED, MatchRequested
from beacon_core.models import (
    CriterionJudgmentRow,
    MatchRun,
    Outbox,
    Patient,
    Trial,
    TrialMatch,
)
from beacon_core.redis import chan_match
from beacon_core.schemas import (
    CriterionJudgment,
    CriterionKind,
    CriterionLabel,
    EligibilityCategory,
    MatchStatus,
    NumberedNote,
    PatientProfile,
    PipelineStep,
    RankedTrial,
    Sex,
    TrialCandidate,
    TrialJudgment,
)
from gateway import __version__
from gateway.deps import ClientDep, RedisDep, SessionmakerDep, StoreDep
from gateway.schemas import (
    ChatRequest,
    DependencyHealth,
    HealthResponse,
    MatchAccepted,
    MatchRequest,
    MatchResult,
    MatchStatusSnapshot,
)
from gateway.service import start_reply
from gateway.sse import SSE_HEADERS, SSE_MEDIA_TYPE, format_sse, heartbeat

__all__ = ["router"]

router = APIRouter()

TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed"})

HEARTBEAT_SECONDS: Final = 15.0
"""Idle gap after which the stream emits an SSE comment, so an intermediary
with a 30-60s read timeout does not kill a quiet but healthy connection."""


# ------------------------------------------------------------------ health --


async def _probe(probe: Awaitable[Any]) -> DependencyHealth:
    """Await one probe and turn any failure into a reported status.

    A health endpoint that raises is useless: the point is to say *which*
    dependency is down, which means catching everything here.
    """
    started = time.perf_counter()
    try:
        await probe
    except Exception as exc:
        return DependencyHealth(
            status="error",
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            detail=f"{type(exc).__name__}: {exc}"[:500],
        )
    return DependencyHealth(
        status="ok",
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


async def _ping_postgres(maker: async_sessionmaker[AsyncSession]) -> None:
    async with maker() as session:
        await session.execute(text("SELECT 1"))


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "A dependency is down"}},
)
async def health(maker: SessionmakerDep, redis: RedisDep) -> Response:
    """Per-dependency health for Postgres and Redis.

    Returns 503 when either is down, so an orchestrator can act on the status
    line without parsing the body - but the body still names the culprit.
    """
    dependencies = {
        "postgres": await _probe(_ping_postgres(maker)),
        "redis": await _probe(redis.ping()),
    }
    healthy = all(dep.status == "ok" for dep in dependencies.values())
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        dependencies=dependencies,
    )
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


# ----------------------------------------------------------------- matches --


@router.post("/v1/matches", status_code=status.HTTP_202_ACCEPTED, response_model=MatchAccepted)
async def create_match(
    body: MatchRequest,
    maker: SessionmakerDep,
    response: Response,
) -> MatchAccepted:
    """Validate, persist and enqueue one match run. The pipeline is C1's.

    202, not 200: nothing has been matched yet. The ``match_runs`` row and its
    ``outbox`` row are written in **one** transaction - that is the entire
    point of the outbox pattern. Committing the job and then publishing to
    Kafka are two operations that can half-fail; one transaction plus a relay
    that reads ``outbox`` makes a queued job and its event impossible to
    desynchronise.
    """
    match_id = str(uuid.uuid4())
    patient_id = body.patient_id or str(uuid.uuid4())
    event = MatchRequested(
        match_id=match_id,
        patient_id=patient_id,
        config_version=body.config_version,
    )

    async with maker() as session, session.begin():
        known = await session.get(Patient, patient_id) if body.patient_id else None
        if known is None:
            if body.patient_note is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    f"Unknown patient_id {patient_id!r}; send patient_note to create it",
                )
            session.add(
                Patient(
                    id=patient_id,
                    source="api",
                    note_text=body.patient_note,
                    sentences=[],
                )
            )
            # Flush, do not commit: SQLAlchemy orders inserts by mapper, not by
            # the order they were added, so without this the ``match_runs`` row
            # can hit ``fk_match_runs_patient`` before its patient exists.
            # Still one transaction - the outbox guarantee is untouched.
            await session.flush()

        session.add(
            MatchRun(
                id=match_id,
                patient_id=patient_id,
                status="queued",
                config_version=body.config_version,
            )
        )
        session.add(
            Outbox(
                event_id=event.event_id,
                topic=TOPIC_MATCH_REQUESTED,
                partition_key=match_id,
                payload=event.model_dump(mode="json"),
            )
        )

    response.headers["Location"] = f"/v1/matches/{match_id}"
    return MatchAccepted(
        match_id=match_id,
        patient_id=patient_id,
        status="queued",
        event_id=event.event_id,
    )


# ------------------------------------------------------------ match result --


def _rank_rows(rows: Sequence[TrialMatch]) -> list[tuple[int, TrialMatch]]:
    """Pair every stored trial with the rank to report for it.

    ``final_rank`` is NULL until ``rank_and_explain`` runs, and
    ``RankedTrial.rank`` is ``ge=1``, so an in-flight run's rows cannot simply
    be passed through. Rows the worker ranked keep its numbering exactly; the
    rest are numbered after the highest rank in use, ordered by retrieval rank.
    That keeps ranks unique and never invents a final ordering the worker has
    not decided yet.
    """
    ranked = sorted(
        (r for r in rows if r.final_rank is not None and r.final_rank >= 1),
        key=lambda r: (r.final_rank or 0, r.nct_id),
    )
    pending = sorted(
        (r for r in rows if r.final_rank is None or r.final_rank < 1),
        key=lambda r: (r.retrieval_rank is None, r.retrieval_rank or 0, r.nct_id),
    )
    out: list[tuple[int, TrialMatch]] = [(int(r.final_rank or 0), r) for r in ranked]
    next_rank = max((rank for rank, _ in out), default=0) + 1
    out.extend((next_rank + offset, row) for offset, row in enumerate(pending))
    return out


def _to_ranked_trial(
    rank: int,
    row: TrialMatch,
    trial: Trial | None,
    judged: Sequence[CriterionJudgmentRow],
) -> RankedTrial:
    """One ``trial_matches`` row plus its trial and its rules, as the contract.

    ``trial`` is ``None`` only if the trial was deleted from under a finished
    run. The pair is still reported - dropping it would make ``trial_count``
    disagree with what the worker actually judged - with empty descriptive
    fields, because the judgments are the part that matters here.

    ``dense_score`` and ``sparse_score`` stay ``None``: ``trial_matches``
    persists the fused and reranked scores, not the two per-leg scores, which
    live only inside the retrieval service's response.
    """
    return RankedTrial(
        rank=rank,
        score=row.final_score if row.final_score is not None else 0.0,
        candidate=TrialCandidate(
            nct_id=row.nct_id,
            title=trial.title if trial is not None else "",
            conditions=list(trial.conditions or []) if trial is not None else [],
            interventions=list(trial.interventions or []) if trial is not None else [],
            phase=trial.phase if trial is not None else None,
            status=trial.status if trial is not None else None,
            min_age_years=trial.min_age_years if trial is not None else None,
            max_age_years=trial.max_age_years if trial is not None else None,
            sex=cast("Sex | None", trial.sex) if trial is not None else None,
            rrf_score=row.rrf_score,
            rerank_score=row.rerank_score,
            retrieval_rank=(
                row.retrieval_rank
                if row.retrieval_rank is not None and row.retrieval_rank >= 1
                else None
            ),
        ),
        judgment=TrialJudgment(
            nct_id=row.nct_id,
            judgments=[
                CriterionJudgment(
                    criterion_id=j.criterion_id,
                    kind=cast("CriterionKind", j.kind),
                    label=cast("CriterionLabel", j.label),
                    evidence_sentence_ids=list(j.evidence_sentence_ids or []),
                    explanation=j.explanation or "",
                )
                for j in judged
            ],
            category=(
                cast("EligibilityCategory", row.category)
                if row.category is not None
                else "insufficient_info"
            ),
            confirmed_fraction=(
                row.confirmed_fraction if row.confirmed_fraction is not None else 0.0
            ),
            missing_facts=list(row.missing_facts or []),
            summary=row.summary,
        ),
    )


@router.get("/v1/matches/{match_id}", response_model=MatchResult)
async def get_match(match_id: str, maker: SessionmakerDep) -> MatchResult:
    """The whole answer for one run, readable at any time and any number of times.

    This is the route ``POST /v1/matches`` has always advertised in its
    ``Location`` header. Without it the only copy of a finished run's results
    was the ``completed`` SSE frame, which is delivered once, to whoever
    happened to be connected: reload the page and the answer was gone, even
    though every row of it was sitting in Postgres.

    Returning the note alongside the results is the point as much as the
    results are. ``evidence_sentence_ids`` index ``note.sentences``, so a
    client that renders cited evidence cannot do it from ``results`` alone.

    Five round trips, independent of the number of trials: the run, the
    patient, the pairs, their trials in one ``IN``, and all judgments for the
    run in one query, grouped in memory. Deliberately not a per-trial loop.
    """
    async with maker() as session:
        run = await session.get(MatchRun, match_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No match run {match_id!r}")

        patient = await session.get(Patient, run.patient_id) if run.patient_id else None

        pairs = list(
            (await session.execute(select(TrialMatch).where(TrialMatch.match_run_id == match_id)))
            .scalars()
            .all()
        )

        trials: dict[str, Trial] = {}
        judged: dict[str, list[CriterionJudgmentRow]] = {}
        if pairs:
            trials = {
                t.nct_id: t
                for t in (
                    await session.execute(
                        select(Trial).where(Trial.nct_id.in_([p.nct_id for p in pairs]))
                    )
                )
                .scalars()
                .all()
            }
            judgment_rows = (
                (
                    await session.execute(
                        select(CriterionJudgmentRow)
                        .where(CriterionJudgmentRow.match_run_id == match_id)
                        .order_by(CriterionJudgmentRow.id)
                    )
                )
                .scalars()
                .all()
            )
            for j in judgment_rows:
                judged.setdefault(j.nct_id, []).append(j)

    results = [
        _to_ranked_trial(rank, row, trials.get(row.nct_id), judged.get(row.nct_id, []))
        for rank, row in _rank_rows(pairs)
    ]

    note: NumberedNote | None = None
    profile: PatientProfile | None = None
    if patient is not None:
        if patient.sentences:
            note = NumberedNote.model_validate({"sentences": patient.sentences})
        if patient.profile:
            profile = PatientProfile.model_validate(patient.profile)

    return MatchResult(
        match_id=match_id,
        status=cast("MatchStatus", run.status),
        current_step=cast("PipelineStep | None", run.current_step),
        note=note,
        profile=profile,
        results=results,
        trial_count=len(results),
        total_cost_usd=run.total_cost_usd,
        duration_ms=run.duration_ms,
        error=run.error,
    )


async def _load_run(maker: async_sessionmaker[AsyncSession], match_id: str) -> MatchStatusSnapshot:
    """Snapshot of a run, or 404. Read before any stream is opened.

    The columns are plain ``str`` in SQLAlchemy but a CHECK constraint confines
    them to the ``MatchStatus`` / ``PipelineStep`` vocabularies, so narrowing
    here is a cast rather than a revalidation.
    """
    async with maker() as session:
        run = await session.get(MatchRun, match_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No match run {match_id!r}")
        return MatchStatusSnapshot(
            match_id=match_id,
            status=cast("MatchStatus", run.status),
            current_step=cast("PipelineStep | None", run.current_step),
        )


def _decode_event(data: Any) -> dict[str, Any] | None:
    raw = data.decode("utf-8") if isinstance(data, bytes) else data
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _match_event_stream(
    match_id: str,
    snapshot: MatchStatusSnapshot,
    pubsub: PubSub,
    request: Request,
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> AsyncIterator[str]:
    """Forward ``chan:match:{id}`` to one browser as typed SSE events.

    The channel is why this works behind more than one replica: the worker
    publishes once and whichever gateway holds the connection relays it.
    """
    try:
        yield format_sse("status", snapshot.model_dump(mode="json"))
        if snapshot.status in TERMINAL_STATUSES:
            yield format_sse("done", {"match_id": match_id, "status": snapshot.status})
            return

        while not await request.is_disconnected():
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=heartbeat_seconds
            )
            if message is None:
                yield heartbeat()
                continue

            payload = _decode_event(message.get("data"))
            if payload is None:
                yield format_sse(
                    "error",
                    {
                        "code": "malformed_event",
                        "message": "Dropped an unparseable message from the match channel.",
                        "match_id": match_id,
                    },
                )
                continue

            if (
                payload.get("event_type") == TOPIC_MATCH_COMPLETED
                or payload.get("status") in TERMINAL_STATUSES
            ):
                yield format_sse("completed", payload)
                yield format_sse(
                    "done",
                    {"match_id": match_id, "status": payload.get("status", "completed")},
                )
                return

            yield format_sse("progress", payload)
    finally:
        await pubsub.unsubscribe(chan_match(match_id))
        await pubsub.aclose()


@router.get("/v1/matches/{match_id}/events")
async def match_events(
    match_id: str,
    request: Request,
    maker: SessionmakerDep,
    redis: RedisDep,
) -> StreamingResponse:
    """SSE progress for one match run.

    Subscription happens here, not inside the generator: a Redis failure then
    becomes a real error for the request rather than a 200 whose stream turns
    out to be empty, and no publish can slip through between the status read
    and the subscribe.
    """
    snapshot = await _load_run(maker, match_id)

    pubsub = redis.pubsub()
    await pubsub.subscribe(chan_match(match_id))

    return StreamingResponse(
        _match_event_stream(match_id, snapshot, pubsub, request),
        media_type=SSE_MEDIA_TYPE,
        headers={**SSE_HEADERS, "X-Match-Id": match_id},
    )


# -------------------------------------------------------------------- chat --


@router.post("/v1/matches/{match_id}/chat")
async def match_chat(
    match_id: str,
    body: ChatRequest,
    maker: SessionmakerDep,
    store: StoreDep,
    client: ClientDep,
) -> StreamingResponse:
    """Stream a Claude reply about one match run.

    ``start_reply`` issues the Claude request before returning, so the
    exception chain below runs while a status code can still be set. It is
    specific-first on purpose - see the module docstring.
    """
    await _load_run(maker, match_id)
    conversation_id = body.conversation_id or match_id

    try:
        events = await start_reply(
            conversation_id,
            body.message,
            store=store,
            client=client,
        )
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Claude is rate limited, retry shortly") from exc
    except anthropic.BadRequestError as exc:
        raise HTTPException(400, exc.message) from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Claude API error ({exc.status_code})") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(503, "Could not reach the Claude API") from exc

    return StreamingResponse(
        events,
        media_type=SSE_MEDIA_TYPE,
        headers={**SSE_HEADERS, "X-Conversation-Id": conversation_id},
    )
