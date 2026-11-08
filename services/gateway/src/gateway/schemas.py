"""Wire models for the gateway's own HTTP surface.

These are request/response shapes, not domain contracts: the frozen contracts
live in :mod:`beacon_core.schemas` and are imported, never redefined. They
subclass ``Contract`` only to inherit ``extra="forbid"``, so a client that
misspells a field gets a 422 instead of silently having it ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from beacon_core.schemas import Contract, MatchStatus, PipelineStep

__all__ = [
    "ChatRequest",
    "DependencyHealth",
    "HealthResponse",
    "MatchAccepted",
    "MatchRequest",
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
