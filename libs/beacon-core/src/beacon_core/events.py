"""Kafka topic names and event payload contracts.

Frozen in Wave 0. Every producer and consumer in the system agrees on these
shapes, so an event written by the gateway today still parses in a worker
deployed next week.

Two properties the envelope exists to provide:

* ``event_id`` makes consumers idempotent. Kafka is at-least-once: the same
  message *will* occasionally arrive twice, and a unique constraint on this ID
  turns the repeat into a no-op rather than a duplicate result.
* ``trace_id`` is propagated through Kafka headers so one OpenTelemetry trace
  spans gateway, broker and worker. Without it you cannot answer "why did that
  patient cost $0.04?".
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from beacon_core.schemas import Contract, MatchStatus, PipelineStep, RankedTrial

__all__ = [
    "ALL_TOPICS",
    "TOPIC_MATCH_COMPLETED",
    "TOPIC_MATCH_DLQ",
    "TOPIC_MATCH_PROGRESS",
    "TOPIC_MATCH_REQUESTED",
    "TOPIC_MATCH_RETRY",
    "TOPIC_TRIALS_UPDATED",
    "EventEnvelope",
    "EventType",
    "MatchCompleted",
    "MatchProgress",
    "MatchRequested",
    "TrialsUpdated",
]

# ------------------------------------------------------------------ topics --

TOPIC_MATCH_REQUESTED = "match.requested"
"""New jobs. Partition key = ``match_id``, so one job's events stay ordered."""

TOPIC_MATCH_PROGRESS = "match.progress"
"""Step-by-step updates, fanned out to the live console over SSE."""

TOPIC_MATCH_COMPLETED = "match.completed"
"""Finished results, for evaluation and analytics."""

TOPIC_MATCH_RETRY = "match.retry"
"""Delayed retry after a transient failure."""

TOPIC_MATCH_DLQ = "match.dlq"
"""Parked after N failures, for inspection. Never silently dropped."""

TOPIC_TRIALS_UPDATED = "trials.updated"
"""Phase 2: daily changed trials, re-indexed incrementally."""

ALL_TOPICS: tuple[str, ...] = (
    TOPIC_MATCH_REQUESTED,
    TOPIC_MATCH_PROGRESS,
    TOPIC_MATCH_COMPLETED,
    TOPIC_MATCH_RETRY,
    TOPIC_MATCH_DLQ,
    TOPIC_TRIALS_UPDATED,
)

EventType = Literal[
    "match.requested",
    "match.progress",
    "match.completed",
    "match.retry",
    "match.dlq",
    "trials.updated",
]


# ------------------------------------------------------------------ payloads --


class EventEnvelope(Contract):
    """Fields common to every Beacon event."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    match_id: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str | None = None
    schema_version: int = 1


class MatchRequested(EventEnvelope):
    event_type: EventType = "match.requested"
    match_id: str
    patient_id: str | None = None
    config_version: str = "v1"


class MatchProgress(EventEnvelope):
    event_type: EventType = "match.progress"
    match_id: str
    step: PipelineStep
    pct: int = Field(ge=0, le=100)
    message: str = ""


class MatchCompleted(EventEnvelope):
    event_type: EventType = "match.completed"
    match_id: str
    status: MatchStatus
    results: list[RankedTrial] = Field(default_factory=list)
    trial_count: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    error: str | None = None


class TrialsUpdated(EventEnvelope):
    """Phase 2 only."""

    event_type: EventType = "trials.updated"
    nct_ids: list[str] = Field(default_factory=list)
