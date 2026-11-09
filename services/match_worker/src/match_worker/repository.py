"""Every write this worker makes, and every one of them idempotent.

Kafka is at-least-once. The same ``match.requested`` message *will* arrive
twice - on a rebalance, on a redelivery after a crash, on a consumer restart
that had not committed its offset - so "write the results" has to mean "make
the results be there", not "insert rows". The three unique constraints that
make that possible already exist in ``beacon_core.models`` and this module is
the only thing that relies on them:

=============================================  =========================================
Constraint                                     What a repeat does instead of duplicating
=============================================  =========================================
``uq_trial_matches_run_nct``                   updates the rank and verdict in place
``uq_criterion_judgments_identity``            updates the label, citations and verdict
``outbox.event_id`` (unique)                   nothing - the event is already queued
``uq_trial_criteria_nct_criterion``            nothing - the rule is already parsed
=============================================  =========================================

Which is why progress ``event_id``s are **derived**, not random: a redelivered
run re-publishes ``match.progress`` for every node it re-runs, and a random ID
would put a second copy of each on the console. See
:func:`match_worker.progress.progress_event`.

The upsert itself is dialect-dispatched rather than hand-rolled as
select-then-insert, because select-then-insert is not atomic and the window it
leaves open is exactly the one a redelivery lands in.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from sqlalchemy import CursorResult, func, select
from sqlalchemy.dialects.postgresql import Insert as PgInsert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from beacon_core.logging import get_logger
from beacon_core.models import (
    CriterionJudgmentRow,
    LlmCall,
    MatchRun,
    Outbox,
    Patient,
    Trial,
    TrialCriterion,
    TrialMatch,
)
from beacon_core.schemas import MatchStatus, PipelineStep
from match_worker.rules import ParsedCriterion

__all__ = ["MatchStore", "PostgresStore", "RunInput", "TrialRules"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunInput:
    """What the worker needs from Postgres to start a run the gateway queued."""

    match_id: str
    patient_id: str | None
    patient_note: str
    config_version: str
    status: MatchStatus


@dataclass(frozen=True, slots=True)
class TrialRules:
    """One trial's parsed rules, as ``check_rules`` wants them."""

    nct_id: str
    title: str
    criteria: list[dict[str, Any]]


class MatchStore(Protocol):
    """The persistence seam the nodes depend on.

    A Protocol so unit tests can run the whole graph against an in-memory store
    that enforces the same unique keys (:mod:`match_worker._testing`), while
    the integration tests - including the crash test - run the real thing
    against the real constraints.
    """

    async def load_run(self, match_id: str) -> RunInput | None: ...
    async def mark_running(self, match_id: str, *, step: PipelineStep | None = None) -> None: ...
    async def set_current_step(self, match_id: str, step: PipelineStep) -> None: ...
    async def finish_run(
        self,
        match_id: str,
        *,
        status: MatchStatus,
        error: str | None,
        total_cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        model: str | None,
        duration_ms: float | None,
    ) -> None: ...
    async def enqueue_event(
        self, *, event_id: str, topic: str, partition_key: str | None, payload: dict[str, Any]
    ) -> bool: ...
    async def save_patient_profile(
        self, patient_id: str | None, *, sentences: list[dict[str, Any]], profile: dict[str, Any]
    ) -> None: ...
    async def load_trial_rules(self, nct_ids: Sequence[str]) -> dict[str, TrialRules]: ...
    async def load_eligibility_text(self, nct_ids: Sequence[str]) -> dict[str, tuple[str, str]]: ...
    async def save_parsed_criteria(
        self, nct_id: str, criteria: Sequence[ParsedCriterion]
    ) -> None: ...
    async def upsert_trial_match(self, match_id: str, row: dict[str, Any]) -> None: ...
    async def save_judgments(
        self, match_id: str, nct_id: str, rows: Sequence[dict[str, Any]]
    ) -> None: ...
    async def record_llm_call(self, row: dict[str, Any]) -> None: ...
    async def count_rows(self, match_id: str) -> dict[str, int]: ...


def _insert_for(session: AsyncSession, table: Any) -> PgInsert:
    """``INSERT ... ON CONFLICT`` for whichever dialect this session speaks.

    Typed as the Postgres construct because that is the one that runs in
    production and the only one whose ``on_conflict_*`` methods mypy knows.
    SQLite's construct carries the same two methods with the same signatures -
    it is accepted here so that a unit test can exercise the real constraint
    behaviour without Docker - and the cast says exactly that and no more.
    """
    name = session.get_bind().dialect.name
    if name == "postgresql":
        return pg_insert(table)
    if name == "sqlite":
        return cast("PgInsert", sqlite_insert(table))
    raise RuntimeError(f"no idempotent insert implemented for dialect {name!r}")


@dataclass
class PostgresStore:
    """The real store. One short transaction per call.

    Short on purpose: the graph checkpoints between nodes, and a transaction
    held open across a node would make the checkpoint and the domain rows
    commit at different times - which is the desynchronisation the outbox
    pattern exists to prevent.
    """

    sessionmaker: async_sessionmaker[AsyncSession]

    async def load_run(self, match_id: str) -> RunInput | None:
        async with self.sessionmaker() as session:
            row = (
                await session.execute(
                    select(
                        MatchRun.id,
                        MatchRun.patient_id,
                        MatchRun.config_version,
                        MatchRun.status,
                        Patient.note_text,
                    )
                    .join(Patient, Patient.id == MatchRun.patient_id, isouter=True)
                    .where(MatchRun.id == match_id)
                )
            ).first()
        if row is None:
            return None
        return RunInput(
            match_id=row.id,
            patient_id=row.patient_id,
            patient_note=row.note_text or "",
            config_version=row.config_version,
            status=cast("MatchStatus", row.status),
        )

    async def mark_running(self, match_id: str, *, step: PipelineStep | None = None) -> None:
        """Flip the run to ``running`` without resetting ``started_at``.

        ``COALESCE`` rather than assignment: a redelivered message must not
        make a run look as though it started over, because ``duration_ms`` is
        reported from this column.
        """
        async with self.sessionmaker.begin() as session:
            run = await session.get(MatchRun, match_id)
            if run is None:
                return
            run.status = "running"
            run.started_at = run.started_at or datetime.now(UTC)
            if step is not None:
                run.current_step = step

    async def set_current_step(self, match_id: str, step: PipelineStep) -> None:
        async with self.sessionmaker.begin() as session:
            run = await session.get(MatchRun, match_id)
            if run is not None:
                run.current_step = step

    async def finish_run(
        self,
        match_id: str,
        *,
        status: MatchStatus,
        error: str | None,
        total_cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        model: str | None,
        duration_ms: float | None,
    ) -> None:
        async with self.sessionmaker.begin() as session:
            run = await session.get(MatchRun, match_id)
            if run is None:
                return
            run.status = status
            run.error = error
            run.total_cost_usd = total_cost_usd
            run.input_tokens = input_tokens
            run.output_tokens = output_tokens
            run.cached_tokens = cached_tokens
            run.model = model
            run.finished_at = datetime.now(UTC)
            run.duration_ms = duration_ms

    async def enqueue_event(
        self, *, event_id: str, topic: str, partition_key: str | None, payload: dict[str, Any]
    ) -> bool:
        """Queue one event in the outbox. ``False`` means it was already there.

        Published *from* the outbox, never produced directly from a node: a
        node that produced straight to Kafka would emit progress for work whose
        checkpoint had not been written, and a crash in between would leave the
        console ahead of the truth.
        """
        async with self.sessionmaker.begin() as session:
            statement = (
                _insert_for(session, Outbox)
                .values(
                    event_id=event_id,
                    topic=topic,
                    partition_key=partition_key,
                    payload=payload,
                )
                .on_conflict_do_nothing(index_elements=["event_id"])
            )
            result = await session.execute(statement)
        # rowcount is 0 when the event was already queued, which is the whole
        # point: the caller learns "already there" without a second query.
        return bool(cast("CursorResult[Any]", result).rowcount)

    async def save_patient_profile(
        self, patient_id: str | None, *, sentences: list[dict[str, Any]], profile: dict[str, Any]
    ) -> None:
        if patient_id is None:
            return
        async with self.sessionmaker.begin() as session:
            patient = await session.get(Patient, patient_id)
            if patient is None:
                return
            patient.sentences = sentences
            patient.profile = profile

    async def load_trial_rules(self, nct_ids: Sequence[str]) -> dict[str, TrialRules]:
        """Already-parsed rules for these trials, keyed by NCT ID.

        Trials with no rows come back absent, not empty: "we have not parsed
        this trial" and "this trial has no rules" are different, and only the
        first one is worth parsing for.
        """
        if not nct_ids:
            return {}
        async with self.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(
                        TrialCriterion.nct_id,
                        TrialCriterion.criterion_id,
                        TrialCriterion.kind,
                        TrialCriterion.text,
                        TrialCriterion.position,
                    )
                    .where(TrialCriterion.nct_id.in_(list(nct_ids)))
                    .order_by(TrialCriterion.nct_id, TrialCriterion.position)
                )
            ).all()
            titles = dict(
                (
                    await session.execute(
                        select(Trial.nct_id, Trial.title).where(Trial.nct_id.in_(list(nct_ids)))
                    )
                ).all()
            )
        grouped: dict[str, TrialRules] = {}
        for row in rows:
            rules = grouped.setdefault(
                row.nct_id, TrialRules(row.nct_id, titles.get(row.nct_id, "") or "", [])
            )
            rules.criteria.append(
                {"criterion_id": row.criterion_id, "kind": row.kind, "text": row.text}
            )
        return grouped

    async def load_eligibility_text(self, nct_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
        """``nct_id -> (title, eligibility_text)`` for trials we hold."""
        if not nct_ids:
            return {}
        async with self.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(Trial.nct_id, Trial.title, Trial.eligibility_text).where(
                        Trial.nct_id.in_(list(nct_ids))
                    )
                )
            ).all()
        return {row.nct_id: (row.title or "", row.eligibility_text or "") for row in rows}

    async def save_parsed_criteria(self, nct_id: str, criteria: Sequence[ParsedCriterion]) -> None:
        if not criteria:
            return
        async with self.sessionmaker.begin() as session:
            statement = (
                _insert_for(session, TrialCriterion)
                .values(
                    [
                        {
                            "nct_id": nct_id,
                            "criterion_id": criterion.criterion_id,
                            "kind": criterion.kind,
                            "position": criterion.position,
                            "text": criterion.text,
                            "parser": "rules",
                        }
                        for criterion in criteria
                    ]
                )
                .on_conflict_do_nothing(index_elements=["nct_id", "criterion_id"])
            )
            await session.execute(statement)

    async def upsert_trial_match(self, match_id: str, row: dict[str, Any]) -> None:
        async with self.sessionmaker.begin() as session:
            values = {"match_run_id": match_id, **row}
            statement = _insert_for(session, TrialMatch).values(**values)
            updatable = {
                key: statement.excluded[key]
                for key in values
                if key not in {"match_run_id", "nct_id"}
            }
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=["match_run_id", "nct_id"], set_=updatable
                )
                if updatable
                else statement.on_conflict_do_nothing(index_elements=["match_run_id", "nct_id"])
            )

    async def save_judgments(
        self, match_id: str, nct_id: str, rows: Sequence[dict[str, Any]]
    ) -> None:
        """Write one trial's judgments. A repeat updates in place.

        Written by the fan-out task that produced them rather than at the end
        of the run, so a worker killed on trial 12 keeps the eleven it already
        judged and only re-judges the one it was holding.
        """
        if not rows:
            return
        async with self.sessionmaker.begin() as session:
            payload = [{"match_run_id": match_id, "nct_id": nct_id, **row} for row in rows]
            statement = _insert_for(session, CriterionJudgmentRow).values(payload)
            updatable = {
                key: statement.excluded[key]
                for key in payload[0]
                if key not in {"match_run_id", "nct_id", "criterion_id"}
            }
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=["match_run_id", "nct_id", "criterion_id"], set_=updatable
                )
            )

    async def record_llm_call(self, row: dict[str, Any]) -> None:
        """Append to the audit log, once per distinct prompt per run.

        **This is the one write in the worker that is deliberately not
        deduplicated**, and the exception is the point rather than an oversight.
        Every cost and latency figure in the README is a query against this
        table, and the invariant that makes those figures defensible is *one row
        per call actually billed*. So:

        * if a resumed run really did re-issue a prompt, we were billed twice
          and the log has to show two rows - collapsing them would under-report
          real spend, which is the one direction of error that matters;
        * if a prompt is logged twice without having been re-issued, that is a
          checkpointing defect, and a unique constraint would hide it behind a
          silent no-op instead of showing it.

        "What did this run cost the user, counting a retried call once?" is a
        real question, but it is a *read-time* one - ``DISTINCT ON
        (prompt_hash)`` in the eval query answers it without having thrown the
        rows away. Collapsing at read time is free; losing rows at write time
        is irreversible.
        """
        async with self.sessionmaker.begin() as session:
            session.add(LlmCall(**row))

    async def count_rows(self, match_id: str) -> dict[str, int]:
        """Row counts for one run - the evidence the crash test prints."""
        async with self.sessionmaker() as session:
            counts: dict[str, int] = {}
            for name, table, column in (
                ("trial_matches", TrialMatch, TrialMatch.match_run_id),
                ("criterion_judgments", CriterionJudgmentRow, CriterionJudgmentRow.match_run_id),
                ("llm_calls", LlmCall, LlmCall.match_run_id),
            ):
                counts[name] = int(
                    (
                        await session.execute(
                            select(func.count()).select_from(table).where(column == match_id)
                        )
                    ).scalar_one()
                )
            counts["outbox"] = int(
                (
                    await session.execute(
                        select(func.count())
                        .select_from(Outbox)
                        .where(Outbox.partition_key == match_id)
                    )
                ).scalar_one()
            )
        return counts

    async def unpublished_events(self, limit: int) -> list[dict[str, Any]]:
        """Oldest unpublished outbox rows, for the relay."""
        async with self.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(
                        Outbox.id,
                        Outbox.event_id,
                        Outbox.topic,
                        Outbox.partition_key,
                        Outbox.payload,
                    )
                    .where(Outbox.published_at.is_(None))
                    .order_by(Outbox.id)
                    .limit(limit)
                )
            ).all()
        return [
            {
                "id": row.id,
                "event_id": row.event_id,
                "topic": row.topic,
                "partition_key": row.partition_key,
                "payload": row.payload,
            }
            for row in rows
        ]

    async def mark_published(self, ids: Iterable[int]) -> None:
        identifiers = list(ids)
        if not identifiers:
            return
        async with self.sessionmaker.begin() as session:
            now = datetime.now(UTC)
            for row_id in identifiers:
                row = await session.get(Outbox, row_id)
                if row is not None:
                    row.published_at = now

    async def mark_publish_failed(self, row_id: int, error: str) -> None:
        async with self.sessionmaker.begin() as session:
            row = await session.get(Outbox, row_id)
            if row is not None:
                row.attempts += 1
                row.last_error = error[:2000]
