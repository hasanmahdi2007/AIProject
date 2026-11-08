"""Protocols: the seams that let an implementation be swapped without touching
a caller.

Protocols only - no implementations live here. ``ConversationStore`` exists so
the gateway's chat route stops depending on a module-level dict: Wave 1 backs
it with Redis, and Day 5 can move it to Postgres without the route changing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["ConversationStore", "EventPublisher", "LlmCostMeter"]


@runtime_checkable
class ConversationStore(Protocol):
    """Chat history for one conversation, keyed by ``conversation_id``.

    Implementations must be safe across processes: the in-memory dict this
    replaces was lost on restart and wrong behind more than one replica.
    """

    async def get(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return the stored turns, oldest first. Empty list when unknown."""
        ...

    async def append(self, conversation_id: str, *turns: dict[str, Any]) -> None:
        """Append turns atomically.

        Called once, with both the user and the assistant turn, **after** a
        stream completes. Appending the user turn before the stream opens
        leaves a poisoned history behind a failed request.
        """
        ...

    async def clear(self, conversation_id: str) -> None: ...


@runtime_checkable
class EventPublisher(Protocol):
    """Publishes a domain event. Backed by the outbox, not by a direct produce."""

    async def publish(
        self, topic: str, payload: dict[str, Any], *, key: str | None = None
    ) -> None: ...


@runtime_checkable
class LlmCostMeter(Protocol):
    """Records one Claude call for the audit log and the cost figures."""

    async def record(self, call: dict[str, Any]) -> None: ...
