"""Redis-backed chat history.

This is the fix for defect 2. The spike kept history in a module-level
``dict[str, list[dict]]``: lost on every restart, and plain wrong behind more
than one replica, because a follow-up turn would land on a worker that had
never seen the first one.

Keys come from :func:`beacon_core.redis.key_conversation` (``chat:{id}``) so the
format stays a contract rather than a literal repeated in two services, and
every write refreshes a TTL so abandoned conversations expire instead of
accumulating forever.
"""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from beacon_core.redis import key_conversation

__all__ = ["RedisConversationStore"]


class RedisConversationStore:
    """A :class:`beacon_core.protocols.ConversationStore` over a Redis list.

    One list per conversation, one JSON-encoded turn per element, oldest first,
    which makes "read the history" a single ``LRANGE`` and "append the pair" a
    single round trip.
    """

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def get(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return the stored turns, oldest first. Empty list when unknown."""
        raw = await self._redis.lrange(key_conversation(conversation_id), 0, -1)
        turns: list[dict[str, Any]] = []
        for item in raw:
            text = item.decode("utf-8") if isinstance(item, bytes) else item
            turns.append(json.loads(text))
        return turns

    async def append(self, conversation_id: str, *turns: dict[str, Any]) -> None:
        """Append turns in one transaction, refreshing the TTL.

        The user and assistant turns are passed together, after the stream
        completes, so a half-finished exchange is never persisted. MULTI/EXEC
        means a concurrent reader never observes the user turn without its
        reply.
        """
        if not turns:
            return
        key = key_conversation(conversation_id)
        encoded = [json.dumps(turn, separators=(",", ":"), default=str) for turn in turns]
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *encoded)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

    async def clear(self, conversation_id: str) -> None:
        await self._redis.delete(key_conversation(conversation_id))
