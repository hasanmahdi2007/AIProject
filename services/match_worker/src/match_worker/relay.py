"""The outbox relay: the only thing in this service that produces to Kafka.

Nodes write events into ``outbox`` inside the same short transaction as the
rows they describe, and this loop publishes them. That split is the whole
reason the outbox table exists: "commit the result, then publish the event" is
two operations that can half-fail, and a node that produced directly would
announce a step whose checkpoint had not been written - so a crash in the gap
would leave the console claiming work that is about to be re-run.

Ordering and idempotency
------------------------
Rows are published oldest first and the partition key is the ``match_id``, so
one match's events stay in order on one partition. A row is marked published
*after* the broker acknowledges it, which means a crash between the send and
the mark re-publishes that event - at-least-once, in the same direction as
everything else in the system. Consumers deduplicate on ``event_id``, which is
derived rather than random for exactly this reason (see
:mod:`match_worker.progress`).

A caveat worth stating plainly: this relay lives here because no service owned
one when the worker was written, and the worker cannot consume
``match.requested`` unless *something* drains the gateway's outbox. It is not
obviously the match worker's job. The PR proposes moving it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from aiokafka import AIOKafkaProducer

from beacon_core.logging import get_logger

__all__ = ["OutboxRelay", "RelaySource"]

log = get_logger(__name__)


class RelaySource(Protocol):
    async def unpublished_events(self, limit: int) -> list[dict[str, Any]]: ...
    async def mark_published(self, ids: Any) -> None: ...
    async def mark_publish_failed(self, row_id: int, error: str) -> None: ...


@dataclass
class OutboxRelay:
    """Publishes unpublished outbox rows until it is cancelled."""

    source: RelaySource
    producer: AIOKafkaProducer
    poll_seconds: float = 0.5
    batch_size: int = 100
    published: int = 0

    async def drain_once(self) -> int:
        """Publish one batch. Returns how many rows were published."""
        rows = await self.source.unpublished_events(self.batch_size)
        if not rows:
            return 0
        done: list[int] = []
        for row in rows:
            try:
                await self.producer.send_and_wait(
                    row["topic"], value=row["payload"], key=row["partition_key"]
                )
            except Exception as exc:
                log.warning(
                    "worker.relay.publish_failed",
                    event_id=row["event_id"],
                    topic=row["topic"],
                    error=f"{type(exc).__name__}: {exc}",
                )
                await self.source.mark_publish_failed(row["id"], f"{type(exc).__name__}: {exc}")
                continue
            done.append(row["id"])
        await self.source.mark_published(done)
        self.published += len(done)
        log.debug("worker.relay.published", count=len(done))
        return len(done)

    async def run(self) -> None:
        """Poll forever. Cancelled by the worker on shutdown."""
        while True:
            try:
                published = await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("worker.relay.error", error=f"{type(exc).__name__}: {exc}")
                published = 0
            await asyncio.sleep(0 if published == self.batch_size else self.poll_seconds)
