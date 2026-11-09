"""``match.progress`` per node, with derived event IDs, through the outbox.

Two decisions here, both forced by at-least-once delivery:

**Event IDs are derived, not random.** ``EventEnvelope`` defaults ``event_id``
to a fresh UUID4, which is right for a gateway handling a new request and wrong
for a worker that may re-run a node after a crash: a random ID turns one
redelivered run into a second copy of every progress event on the console, and
``outbox.event_id``'s unique constraint - the thing that is supposed to make a
repeat a no-op - never fires. So the ID is a UUID5 over
``(match_id, step, node)``. Re-running ``retrieve`` writes the same ID, the
insert conflicts, and nothing duplicates.

**Nothing is produced to Kafka from a node.** Progress goes into ``outbox`` in
its own short transaction and a relay publishes it. A node that produced
directly would announce work whose checkpoint had not been written yet, so a
crash in the gap would leave the console claiming a step that is about to be
re-run.

A note on the step vocabulary
-----------------------------
``PipelineStep`` carries all seven node names, including ``parse_rules``, as of
the contract change merged on 8 Nov. So every node reports under its own name
and ``MatchRun.current_step`` is exactly the node that is running - there is no
mapping layer here, and the percentages below are the only thing this module
adds to the node names.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from beacon_core.events import TOPIC_MATCH_PROGRESS, MatchProgress
from beacon_core.schemas import PipelineStep

__all__ = [
    "NODE_ORDER",
    "NODE_STEPS",
    "ProgressSink",
    "progress_event",
    "publish_progress",
]

_NAMESPACE = uuid.UUID("5f3a3c1e-0d4b-4f1a-9f2a-1b6c8d9e0a11")
"""Fixed namespace for derived event IDs. Changing it re-duplicates history."""

NODE_ORDER: tuple[str, ...] = (
    "profile_patient",
    "generate_queries",
    "retrieve",
    "parse_rules",
    "check_rules",
    "verify",
    "rank_and_explain",
)

NODE_STEPS: dict[str, tuple[PipelineStep, int]] = {
    "profile_patient": ("profile_patient", 10),
    "generate_queries": ("generate_queries", 20),
    "retrieve": ("retrieve", 35),
    "parse_rules": ("parse_rules", 45),
    "check_rules": ("check_rules", 70),
    "verify": ("verify", 85),
    "rank_and_explain": ("rank_and_explain", 100),
}


class ProgressSink(Protocol):
    async def enqueue_event(
        self, *, event_id: str, topic: str, partition_key: str | None, payload: dict[str, Any]
    ) -> bool: ...


def progress_event(
    *, match_id: str, node: str, message: str = "", trace_id: str | None = None
) -> MatchProgress:
    """One progress event with a derived, stable ``event_id``."""
    step, pct = NODE_STEPS[node]
    return MatchProgress(
        event_id=str(uuid.uuid5(_NAMESPACE, f"{match_id}:{node}:{step}")),
        match_id=match_id,
        step=step,
        pct=pct,
        message=message or node,
        trace_id=trace_id,
    )


async def publish_progress(
    sink: ProgressSink,
    *,
    match_id: str,
    node: str,
    message: str = "",
    trace_id: str | None = None,
) -> MatchProgress:
    """Queue one node's progress event. A repeat is a no-op, by constraint."""
    event = progress_event(match_id=match_id, node=node, message=message, trace_id=trace_id)
    await sink.enqueue_event(
        event_id=event.event_id,
        topic=TOPIC_MATCH_PROGRESS,
        partition_key=match_id,
        payload=event.model_dump(mode="json"),
    )
    return event
