"""Kafka -> Redis: the hop that makes the console move while a match runs.

The worker never talks to a browser. It writes ``match.progress``,
``match.completed`` and ``match.dlq`` into its outbox, and its relay publishes
them to Kafka. The SSE endpoint, meanwhile, listens on the Redis channel
``chan:match:{id}`` - because the browser's connection can be held by *any*
gateway replica, and Redis pub/sub is what reaches all of them. Before this
module nothing connected the two, so the console showed "Queued" for the whole
run even while the worker was finishing it.

In Spring terms: this is a ``@KafkaListener`` whose handler does
``redisTemplate.convertAndSend(channel, event)``.

Why one *shared* consumer group
-------------------------------
Every replica runs a bridge, all in :data:`BRIDGE_GROUP`, so Kafka hands each
event to exactly **one** of them and that one publishes it to Redis once. Redis
then delivers it to whichever replica holds the browser's connection. Giving
each replica its own group would publish every event once per replica, and each
browser would see every progress step N times.

What it deliberately is not
---------------------------
* **Not durable.** Progress is a live view, not a record. Offsets auto-commit
  and a fresh group starts at ``latest``: replaying yesterday's progress into
  today's channels would be noise. The record is Postgres, and the SSE stream
  re-reads it on every heartbeat (see ``gateway.router._match_event_stream``),
  so an event this bridge misses costs at most one heartbeat of latency, never
  a stuck page.
* **Not fatal.** Kafka being down must not take the API down with it. The loop
  logs, backs off and reconnects; the gateway keeps serving.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol

from aiokafka import AIOKafkaConsumer

from beacon_core.events import TOPIC_MATCH_COMPLETED, TOPIC_MATCH_DLQ, TOPIC_MATCH_PROGRESS
from beacon_core.logging import get_logger
from beacon_core.redis import chan_match
from beacon_core.settings import Settings, get_settings

__all__ = [
    "BRIDGE_GROUP",
    "BRIDGE_TOPICS",
    "ProgressBridge",
    "build_bridge_consumer",
    "decode_value",
]

log = get_logger(__name__)

BRIDGE_TOPICS: Final[tuple[str, ...]] = (
    TOPIC_MATCH_PROGRESS,
    TOPIC_MATCH_COMPLETED,
    TOPIC_MATCH_DLQ,
)
"""``match.dlq`` is here because a parked run is a terminal state too: its
payload carries ``status="failed"``, which the SSE stream turns into the
``completed`` + ``done`` frames that make the console stop waiting."""

BRIDGE_GROUP: Final = "beacon-gateway-progress"
"""Shared by every gateway replica. Distinct from the worker's group, so the
bridge never steals ``match.requested`` work and the worker never sees this."""


class Publisher(Protocol):
    """The one Redis method the bridge needs. ``redis.asyncio.Redis`` fits."""

    def publish(self, channel: str, message: str) -> Awaitable[Any]: ...


class MessageLike(Protocol):
    @property
    def value(self) -> Any: ...


class ConsumerLike(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def __aiter__(self) -> Any: ...


def decode_value(raw: bytes | None) -> dict[str, Any] | None:
    """JSON object or ``None``. Never raises.

    A deserializer that raises inside aiokafka's iterator kills the loop on the
    first junk message - the same failure that once kept the worker from
    starting. Here junk becomes ``None`` and is skipped.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_bridge_consumer(settings: Settings | None = None) -> AIOKafkaConsumer:
    cfg = settings or get_settings()
    return AIOKafkaConsumer(
        *BRIDGE_TOPICS,
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=BRIDGE_GROUP,
        # At-most-once is the right guarantee for a live view; see the module
        # docstring. The worker's group is the one that must not auto-commit.
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=decode_value,
    )


@dataclass
class ProgressBridge:
    """Forwards match events from Kafka to their Redis channel until cancelled."""

    consumer_factory: Callable[[], ConsumerLike]
    publisher: Publisher
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    forwarded: int = 0

    async def forward(self, value: Any) -> bool:
        """Publish one event to ``chan:match:{match_id}``. False if skipped."""
        if not isinstance(value, dict):
            log.warning("gateway.bridge.skipped", reason="not a JSON object")
            return False
        match_id = value.get("match_id")
        if not isinstance(match_id, str) or not match_id:
            log.warning("gateway.bridge.skipped", reason="no match_id", payload=value)
            return False
        await self.publisher.publish(chan_match(match_id), json.dumps(value))
        self.forwarded += 1
        return True

    async def consume(self, consumer: ConsumerLike) -> None:
        """Forward everything one connected consumer yields."""
        async for message in consumer:
            try:
                await self.forward(message.value)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Redis down: drop this event, keep consuming. The SSE stream's
                # heartbeat re-read of Postgres covers the terminal state.
                log.warning("gateway.bridge.publish_failed", error=f"{type(exc).__name__}: {exc}")

    async def run(self) -> None:
        """Connect, consume, and reconnect with backoff. Never raises except
        on cancellation, which is how the gateway's lifespan stops it."""
        delay = self.retry_base_seconds
        while True:
            consumer = self.consumer_factory()
            try:
                await consumer.start()
                log.info("gateway.bridge.started", topics=list(BRIDGE_TOPICS), group=BRIDGE_GROUP)
                delay = self.retry_base_seconds
                await self.consume(consumer)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "gateway.bridge.unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                    retry_in_seconds=delay,
                    detail="live progress paused; SSE falls back to polling Postgres",
                )
            finally:
                with contextlib.suppress(Exception):
                    await consumer.stop()
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.retry_max_seconds)
