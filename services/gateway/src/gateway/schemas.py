"""Wire models for the gateway's own HTTP surface.

These are request/response shapes, not domain contracts: the frozen contracts
live in :mod:`beacon_core.schemas` and are imported, never redefined. They
subclass ``Contract`` only to inherit ``extra="forbid"``, so a client that
misspells a field gets a 422 instead of silently having it ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from beacon_core.schemas import (
    Contract,
    MatchStatus,
    NumberedNote,
    PatientProfile,
    PipelineStep,
    RankedTrial,
)

__all__ = [
    "ChatRequest",
    "DependencyHealth",
    "HealthResponse",
    "MatchAccepted",
    "MatchRequest",
    "MatchResult",
    "MatchStatusSnapshot",
]


class ChatRequest(Contract):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    """Omitted means "this match's own thread": the route defaults it to the
    match id, so a console that only knows the match can still chat."""


class MatchRequest(Contract):
    """Body of ``POST /v1/matches``.

    Either a note (a new synthetic patient) or the id of a patient already
    stored. The gateway only validates, persists and enqueues; profiling,
    retrieval and rule checking belong to the worker.
    """

    patient_note: str | None = Field(default=None, min_length=1)
    patient_id: str | None = Field(default=None, min_length=1, max_length=64)
    config_version: str = Field(default="v1", max_length=32)

    @model_validator(mode="after")
    def _needs_a_patient(self) -> MatchRequest:
        if self.patient_note is None and self.patient_id is None:
            raise ValueError("provide patient_note, patient_id, or both")
        return self


class MatchAccepted(Contract):
    """``202`` body. The work has not started; the outbox row says it will."""

    match_id: str
    patient_id: str
    status: MatchStatus = "queued"
    event_id: str
    """The outbox/Kafka ``event_id``, so a caller can correlate its job with
    the event that triggered it."""


class DependencyHealth(Contract):
    status: Literal["ok", "error"]
    latency_ms: float
    detail: str | None = None
    """Populated only on failure: the exception type and message."""


class HealthResponse(Contract):
    """Per-dependency health. ``degraded`` is reported with a 503 so a load
    balancer takes the replica out rather than reading the body."""

    status: Literal["ok", "degraded"]
    version: str
    dependencies: dict[str, DependencyHealth]


class MatchStatusSnapshot(Contract):
    """First event of an SSE stream, so a client that connects late still knows
    where the run is instead of waiting for the next publish."""

    match_id: str
    status: MatchStatus
    current_step: PipelineStep | None = None


class MatchResult(Contract):
    """Body of ``GET /v1/matches/{match_id}`` - one run's whole answer.

    This route exists because the ``completed`` SSE frame used to be the *only*
    delivery of a finished run's results. A page reload, a late connect or a
    second viewer could never obtain them: ``_match_event_stream`` emits
    ``status`` then ``done`` and returns for an already-terminal run. Worse,
    the 202 from ``POST /v1/matches`` sets ``Location: /v1/matches/{id}``, so
    the header advertised a route that did not exist. Read from Postgres, which
    is the source of truth for all of this anyway.

    ``note`` and ``profile`` are part of the body for a specific reason, not
    for convenience: a ``CriterionJudgment.evidence_sentence_ids`` list is
    meaningless without the ``NumberedNote`` it indexes, so a client that
    renders cited evidence needs both halves in one response. Re-splitting the
    note client-side is not an option - a different split yields different
    sentence IDs and the citation would silently point at the wrong sentence.
    """

    match_id: str
    status: MatchStatus
    current_step: PipelineStep | None = None

    note: NumberedNote | None = None
    """``None`` means the worker has not split the note yet. Deliberately not
    an empty ``NumberedNote``, which would read as "a note with no sentences"
    and make an unresolvable citation look like a data error instead of a
    not-yet."""

    profile: PatientProfile | None = None
    results: list[RankedTrial] = Field(default_factory=list)

    trial_count: int = 0
    """Number of rows in ``results``, so a client can assert it got them all."""

    total_cost_usd: float = 0.0
    duration_ms: float | None = None
    error: str | None = None
    """Set only when ``status == "failed"``."""
