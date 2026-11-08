"""FastAPI dependency providers.

Each external thing the routes touch - Postgres, Redis, the conversation store,
the Claude client - arrives as a dependency rather than a module-level global.
That is what lets a test swap in a fake with ``app.dependency_overrides``
instead of monkeypatching a module, and it is why the Claude client is built
lazily: importing :mod:`gateway.main` must not require an API key, or CI cannot
even collect the test suite.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import anthropic
from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from beacon_core.db import get_sessionmaker
from beacon_core.protocols import ConversationStore
from beacon_core.redis import get_redis
from beacon_core.settings import Settings, get_settings
from gateway.conversations import RedisConversationStore

__all__ = [
    "ClientDep",
    "RedisDep",
    "SessionmakerDep",
    "SettingsDep",
    "StoreDep",
    "provide_anthropic_client",
    "provide_conversation_store",
    "provide_redis",
    "provide_sessionmaker",
    "provide_settings",
]


def provide_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(provide_settings)]


def provide_redis() -> Redis:
    return get_redis()


RedisDep = Annotated[Redis, Depends(provide_redis)]


def provide_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return get_sessionmaker()


SessionmakerDep = Annotated[async_sessionmaker[AsyncSession], Depends(provide_sessionmaker)]


def provide_conversation_store(
    redis: RedisDep,
    settings: SettingsDep,
) -> ConversationStore:
    return RedisConversationStore(redis, ttl_seconds=settings.redis.conversation_ttl_seconds)


StoreDep = Annotated[ConversationStore, Depends(provide_conversation_store)]


@lru_cache(maxsize=1)
def _build_anthropic_client() -> anthropic.AsyncAnthropic:
    """One client per process: it owns an HTTP connection pool."""
    key = get_settings().anthropic.api_key.get_secret_value()
    # An empty key means "not configured here" - hand the SDK None and let it
    # read the environment itself, so the failure is the SDK's own clear one.
    return anthropic.AsyncAnthropic(api_key=key or None)


def provide_anthropic_client() -> anthropic.AsyncAnthropic:
    return _build_anthropic_client()


ClientDep = Annotated[anthropic.AsyncAnthropic, Depends(provide_anthropic_client)]
