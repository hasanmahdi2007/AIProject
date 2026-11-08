"""Async SQLAlchemy engine and session factory.

Engine creation is cached per process: an engine owns a connection pool, so
building one per request silently exhausts Postgres connections.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from beacon_core.settings import Settings, get_settings

__all__ = ["dispose_engine", "get_engine", "get_sessionmaker", "session_scope"]


@lru_cache(maxsize=1)
def get_engine(settings: Settings | None = None) -> AsyncEngine:
    cfg = settings or get_settings()
    return create_async_engine(
        cfg.postgres.dsn(),
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=5,
        echo=False,
    )


@lru_cache(maxsize=1)
def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(settings), expire_on_commit=False)


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """One transaction. Commits on success, rolls back on any exception.

    The outbox pattern depends on this: the domain row and its pending event
    must be written inside the same ``session_scope`` block or they can
    desynchronise.
    """
    maker = get_sessionmaker(settings)
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the pool. Call on application shutdown."""
    engine = get_engine()
    await engine.dispose()
