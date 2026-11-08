"""Redis client factory and the key namespace.

The key patterns are a contract: the gateway writes ``chan:match:{id}`` and
some other gateway replica reads it, so a typo here is a bug that only shows up
under more than one replica. Build keys with these helpers, never by hand.
"""

from __future__ import annotations

from functools import lru_cache

from redis.asyncio import Redis

from beacon_core.settings import Settings, get_settings

__all__ = [
    "chan_match",
    "get_redis",
    "key_conversation",
    "key_idempotency",
    "key_llm_cache",
    "key_lock",
    "key_rate_limit",
]


@lru_cache(maxsize=1)
def get_redis(settings: Settings | None = None) -> Redis:
    cfg = settings or get_settings()
    return Redis.from_url(cfg.redis.url, decode_responses=True)


def key_rate_limit(api_key: str) -> str:
    """Token-bucket rate limiter, refilled atomically in Lua."""
    return f"rl:{api_key}"


def key_idempotency(idem_key: str) -> str:
    """So a retried POST starts no second job."""
    return f"idem:{idem_key}"


def key_llm_cache(prompt_sha256: str) -> str:
    """Exact-match cache of Claude responses, keyed by prompt hash."""
    return f"llm:{prompt_sha256}"


def key_lock(name: str) -> str:
    """``SET NX`` lock. ``lock:ingest`` keeps ingestion to one runner."""
    return f"lock:{name}"


def key_conversation(conversation_id: str) -> str:
    """Chat history for the ``/v1/matches/{id}/chat`` endpoint."""
    return f"chat:{conversation_id}"


def chan_match(match_id: str) -> str:
    """Pub/sub channel feeding SSE to whichever gateway replica holds the
    browser connection."""
    return f"chan:match:{match_id}"
