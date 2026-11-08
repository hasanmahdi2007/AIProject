"""The Kafka loop: consume, run the graph, *then* commit the offset.

The order in :meth:`MatchWorker.handle_message` is the contract, and it is the
one thing in this file worth reading twice:

1. parse the message;
2. run the graph, which checkpoints to Postgres after every node;
3. write the terminal rows and the ``match.completed`` outbox row;
4. **only then** commit the offset for that one partition.

A worker killed at any point before step 4 has not committed, so the broker
redelivers the message, and step 2 resumes from the last checkpoint instead of
starting over. That is the crash test. Committing first - which
``enable_auto_commit=True`` would do on a timer, which is why
``beacon_core.kafka.build_consumer`` sets it to ``False`` - would turn the same
crash into a job that is silently never finished.

Redelivery is not an error, it is the normal case
-------------------------------------------------
Kafka is at-least-once, so :meth:`handle_message` is written to be run twice:

* a match whose thread is already finished returns immediately - the
  checkpointer, keyed on ``thread_id == match_id``, is what makes "already
  done" observable;
* a match whose thread is half-finished resumes;
* every row written on the way through goes through a unique constraint, so a
  re-run updates rather than duplicates.

Offsets are committed one partition at a time, explicitly.
``consumer.commit()`` with no argument commits the *fetch position* of every
assigned partition, which includes messages that have been fetched into the
buffer and not yet processed - a quiet way to lose a job under load.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import TopicPartition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from beacon_core.events import (
    TOPIC_MATCH_COMPLETED,
    TOPIC_MATCH_DLQ,
    TOPIC_MATCH_REQUESTED,
    MatchCompleted,
    MatchRequested,
)
from beacon_core.kafka import build_consumer, build_producer, trace_from_headers
from beacon_core.logging import get_logger
from beacon_core.settings import Settings, get_settings
from match_worker.backoff import RetryPolicy, is_retryable
from match_worker.budget import BudgetMeter
from match_worker.config import WorkerConfig
from match_worker.context import RunContext, to_config
from match_worker.graph import compile_graph
from match_worker.llm import LlmGateway, SupportsCall
from match_worker.relay import OutboxRelay
from match_worker.repository import MatchStore, PostgresStore
from match_worker.retrieval_client import RetrievalClient
from match_worker.state import PARTIAL_PREFIX, GraphState

__all__ = ["MatchWorker", "RunOutcome", "run_worker"]

log = get_logger(__name__)

_COMPLETION_NAMESPACE = uuid.UUID("9d1f6b22-6c4e-4d8e-9a31-7f2b5c0d4e88")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one message did. Returned so tests can assert on it."""

    match_id: str
    status: str
    resumed: bool
    already_complete: bool
    trials: int
    partial: bool
    error: str | None
    cost_usd: float
    checkpoint_id: str | None = None
    resumed_from_step: str | None = None


@dataclass
class MatchWorker:
    """One consumer, one compiled graph, one store."""

    store: MatchStore
    llm_client: SupportsCall
    graph: CompiledStateGraph[GraphState, None, GraphState, GraphState]
    settings: Settings = field(default_factory=get_settings)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    retrieval: RetrievalClient | None = None
    consumer: AIOKafkaConsumer | None = None
    producer: AIOKafkaProducer | None = None
    relay: OutboxRelay | None = None
    _relay_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ one match --

    def _retrieval_client(self) -> RetrievalClient:
        return self.retrieval or RetrievalClient(
            base_url=self.settings.retrieval.base_url,
            timeout_seconds=self.worker.retrieval_timeout_seconds,
        )

    def _context(
        self, match_id: str, *, patient_id: str | None, trace_id: str | None
    ) -> RunContext:
        budget = BudgetMeter(
            limit_usd=self.worker.max_cost_usd, limit_tokens=self.worker.max_tokens
        )
        return RunContext(
            match_id=match_id,
            store=self.store,
            llm=LlmGateway(
                client=self.llm_client,
                budget=budget,
                match_id=match_id,
                audit=self.store,
                trace_id=trace_id,
                throttle_seconds=self.worker.throttle_seconds,
            ),
            retrieval=self._retrieval_client(),
            budget=budget,
            settings=self.settings,
            worker=self.worker,
            patient_id=patient_id,
            trace_id=trace_id,
        )

    async def run_match(self, event: MatchRequested, *, trace_id: str | None = None) -> RunOutcome:
        """Run or resume one match. Safe to call twice for the same message."""
        run_input = await self.store.load_run(event.match_id)
        if run_input is None:
            raise LookupError(
                f"match {event.match_id} has no match_runs row; the gateway writes the row and "
                f"the outbox event in one transaction, so this message has no job attached"
            )

        ctx = self._context(event.match_id, patient_id=run_input.patient_id, trace_id=trace_id)
        config = to_config(ctx)
        snapshot = await self.graph.aget_state(config)
        checkpointed = dict(snapshot.values) if snapshot.values else {}
        resumed = bool(checkpointed)
        pending = tuple(snapshot.next or ())

        if resumed and not pending:
            # The thread ran to completion already: this is a redelivery of a
            # finished job. Nothing is re-run and nothing is re-written.
            log.info(
                "worker.match.already_complete",
                match_id=event.match_id,
                status=checkpointed.get("status"),
            )
            judgments = checkpointed.get("judgments") or []
            return RunOutcome(
                match_id=event.match_id,
                status=str(checkpointed.get("status") or run_input.status),
                resumed=True,
                already_complete=True,
                trials=len(judgments),
                partial=str(checkpointed.get("error") or "").startswith(PARTIAL_PREFIX),
                error=checkpointed.get("error"),
                cost_usd=float(checkpointed.get("total_cost_usd") or 0.0),
                checkpoint_id=_checkpoint_id(snapshot),
            )

        if resumed:
            # Re-seed the meter from the checkpoint: the budget belongs to the
            # match, not to this attempt, so a crash loop cannot spend it twice.
            ctx.budget.spent_usd = float(checkpointed.get("total_cost_usd") or 0.0)
            log.info(
                "worker.match.resume",
                match_id=event.match_id,
                checkpoint_id=_checkpoint_id(snapshot),
                next_nodes=pending,
                judgments=len(checkpointed.get("judgments") or []),
                spent_usd=ctx.budget.spent_usd,
            )

        started = time.perf_counter()
        # ``None`` resumes the thread from its checkpoint; a value would start a
        # fresh superstep from START and re-run the nodes that already ran.
        graph_input = (
            None
            if resumed
            else GraphState(
                match_id=event.match_id,
                patient_note=run_input.patient_note,
                config_version=event.config_version,
                status="running",
            )
        )
        final = await self.graph.ainvoke(graph_input, config)
        duration_ms = (time.perf_counter() - started) * 1000

        state = GraphState.model_validate(final)
        partial = bool(state.error and state.error.startswith(PARTIAL_PREFIX))
        await self.store.finish_run(
            event.match_id,
            status=state.status,
            error=state.error,
            total_cost_usd=state.total_cost_usd,
            input_tokens=ctx.budget.input_tokens,
            output_tokens=ctx.budget.output_tokens,
            cached_tokens=ctx.budget.cached_tokens,
            model=ctx.model,
            duration_ms=duration_ms,
        )
        completed = MatchCompleted(
            event_id=str(uuid.uuid5(_COMPLETION_NAMESPACE, f"{event.match_id}:completed")),
            match_id=event.match_id,
            status=state.status,
            results=state.ranked,
            trial_count=len(state.ranked),
            total_cost_usd=state.total_cost_usd,
            duration_ms=duration_ms,
            error=state.error,
            trace_id=trace_id,
        )
        await self.store.enqueue_event(
            event_id=completed.event_id,
            topic=TOPIC_MATCH_COMPLETED,
            partition_key=event.match_id,
            payload=completed.model_dump(mode="json"),
        )
        log.info(
            "worker.match.finished",
            match_id=event.match_id,
            status=state.status,
            resumed=resumed,
            trials=len(state.ranked),
            partial=partial,
            cost_usd=round(state.total_cost_usd, 6),
            duration_ms=round(duration_ms, 1),
            llm_calls=ctx.budget.calls,
        )
        return RunOutcome(
            match_id=event.match_id,
            status=state.status,
            resumed=resumed,
            already_complete=False,
            trials=len(state.ranked),
            partial=partial,
            error=state.error,
            cost_usd=state.total_cost_usd,
            checkpoint_id=_checkpoint_id(await self.graph.aget_state(config)),
            resumed_from_step=pending[0] if pending else None,
        )

    # -------------------------------------------------------- one message --

    async def handle_message(
        self, value: dict[str, Any], *, trace_id: str | None = None
    ) -> RunOutcome:
        """Run one message with retries; park it in the DLQ when they run out.

        A retry here is in-process and bounded. The alternative - letting the
        exception escape and relying on redelivery - would hold the partition
        hostage to one bad message, because an uncommitted offset is redelivered
        forever.
        """
        event = MatchRequested.model_validate(value)
        policy = RetryPolicy(
            attempts=self.worker.message_attempts,
            base_seconds=self.worker.message_retry_base_seconds,
            max_seconds=self.worker.message_retry_max_seconds,
        )
        last: BaseException | None = None
        for attempt in range(1, policy.attempts + 1):
            try:
                return await self.run_match(event, trace_id=trace_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
                # A missing job row is a permanent condition: the gateway writes
                # the row and the event in one transaction, so a message with no
                # row will not acquire one by waiting. Everything else gets the
                # attempts it was configured for, retryable or not - a resumed
                # run is cheap, because the checkpoint means it starts where it
                # stopped rather than at the beginning.
                permanent = isinstance(exc, LookupError)
                log.warning(
                    "worker.message.failed",
                    match_id=event.match_id,
                    attempt=attempt,
                    attempts=policy.attempts,
                    permanent=permanent,
                    transport_retryable=is_retryable(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if permanent or attempt == policy.attempts:
                    break
                await asyncio.sleep(policy.delay_for(attempt))

        assert last is not None
        await self._park(event, last, trace_id=trace_id)
        return RunOutcome(
            match_id=event.match_id,
            status="failed",
            resumed=False,
            already_complete=False,
            trials=0,
            partial=False,
            error=f"{type(last).__name__}: {last}",
            cost_usd=0.0,
        )

    async def _park(
        self, event: MatchRequested, exc: BaseException, *, trace_id: str | None
    ) -> None:
        """Mark the run failed and put the message in the DLQ, never drop it."""
        detail = f"{type(exc).__name__}: {exc}"
        await self.store.finish_run(
            event.match_id,
            status="failed",
            error=detail[:2000],
            total_cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            model=None,
            duration_ms=None,
        )
        dlq = MatchCompleted(
            event_id=str(uuid.uuid5(_COMPLETION_NAMESPACE, f"{event.match_id}:dlq")),
            event_type="match.dlq",
            match_id=event.match_id,
            status="failed",
            error=detail[:2000],
            trace_id=trace_id,
        )
        await self.store.enqueue_event(
            event_id=dlq.event_id,
            topic=TOPIC_MATCH_DLQ,
            partition_key=event.match_id,
            payload=dlq.model_dump(mode="json"),
        )
        log.error("worker.message.parked", match_id=event.match_id, error=detail)

    # ----------------------------------------------------------- the loop --

    async def start(self) -> None:
        self.consumer = self.consumer or build_consumer(
            TOPIC_MATCH_REQUESTED, settings=self.settings
        )
        await self.consumer.start()
        if self.worker.relay_enabled:
            self.producer = self.producer or build_producer(self.settings)
            await self.producer.start()
            self.relay = OutboxRelay(
                source=self.store,  # type: ignore[arg-type]
                producer=self.producer,
                poll_seconds=self.worker.relay_poll_seconds,
                batch_size=self.worker.relay_batch_size,
            )
            self._relay_task = asyncio.create_task(self.relay.run())
        log.info(
            "worker.started",
            topics=[TOPIC_MATCH_REQUESTED],
            group=self.settings.kafka.consumer_group,
            relay=self.worker.relay_enabled,
        )

    async def stop(self) -> None:
        if self._relay_task is not None:
            self._relay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._relay_task
            self._relay_task = None
        if self.producer is not None:
            await self.producer.stop()
        if self.consumer is not None:
            await self.consumer.stop()
        log.info("worker.stopped")

    async def consume_forever(self, *, max_messages: int | None = None) -> int:
        """Consume until cancelled, or until ``max_messages`` are handled.

        ``max_messages`` exists so an integration test can run the real loop
        and then return, rather than needing a second mechanism that the demo
        does not use.
        """
        if self.consumer is None:
            raise RuntimeError("call start() first")
        handled = 0
        while max_messages is None or handled < max_messages:
            batches = await self.consumer.getmany(timeout_ms=self.worker.poll_timeout_ms)
            for partition, messages in batches.items():
                for message in messages:
                    trace_id = trace_from_headers(message.headers)
                    await self.handle_message(message.value, trace_id=trace_id)
                    # The checkpoint and the terminal rows are durable by here.
                    # Only now is this message's offset allowed to move, and
                    # only this partition's.
                    await self.consumer.commit(
                        {TopicPartition(partition.topic, partition.partition): message.offset + 1}
                    )
                    handled += 1
                    log.info(
                        "worker.offset.committed",
                        topic=partition.topic,
                        partition=partition.partition,
                        offset=message.offset + 1,
                    )
                    if max_messages is not None and handled >= max_messages:
                        return handled
        return handled


def _checkpoint_id(snapshot: Any) -> str | None:
    configurable = (getattr(snapshot, "config", None) or {}).get("configurable") or {}
    value = configurable.get("checkpoint_id")
    return str(value) if value else None


async def run_worker(
    *,
    settings: Settings | None = None,
    worker: WorkerConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    max_messages: int | None = None,
) -> int:
    """Wire everything up and consume. The entry point ``__main__`` calls."""
    from beacon_core.db import get_sessionmaker
    from beacon_llm import AsyncLlmClient

    cfg = settings or get_settings()
    worker_cfg = worker or WorkerConfig()
    store = PostgresStore(sessionmaker=get_sessionmaker(cfg))

    if checkpointer is not None:
        instance = MatchWorker(
            store=store,
            llm_client=AsyncLlmClient(settings=cfg),
            graph=compile_graph(checkpointer),
            settings=cfg,
            worker=worker_cfg,
        )
        await instance.start()
        try:
            return await instance.consume_forever(max_messages=max_messages)
        finally:
            await instance.stop()

    from match_worker.graph import postgres_checkpointer

    async with postgres_checkpointer(cfg) as saver:
        instance = MatchWorker(
            store=store,
            llm_client=AsyncLlmClient(settings=cfg),
            graph=compile_graph(saver),
            settings=cfg,
            worker=worker_cfg,
        )
        await instance.start()
        try:
            return await instance.consume_forever(max_messages=max_messages)
        finally:
            await instance.stop()
