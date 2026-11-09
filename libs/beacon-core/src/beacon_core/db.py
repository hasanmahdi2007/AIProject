"""Async SQLAlchemy engine and session factory.

Engine creation is cached per process: an engine owns a connection pool, so
building one per request silently exhausts Postgres connections.

The cache is keyed on the **DSN**, not on the ``Settings`` object. Both
functions used to be ``@lru_cache``d, which meant the ``settings`` parameter
they advertise could never actually be passed: ``Settings`` is a pydantic model
and therefore unhashable, so ``get_sessionmaker(Settings())`` raised
``TypeError: unhashable type: 'Settings'`` while ``get_sessionmaker()`` worked
fine. A signature that only accepts its own default is a trap, and C1 walked
into it writing the worker's crash test. Keyed on the DSN, injection works, two
callers pointing at the same database still share one pool, and a test pointing
at a different database gets its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from beacon_core.settings import Settings, get_settings

__all__ = ["dispose_engine", "get_engine", "get_sessionmaker", "session_scope"]


_ENGINES: dict[str, AsyncEngine] = {}
"""DSN -> engine. One pool per distinct database per process."""

_SESSIONMAKERS: dict[str, async_sessionmaker[AsyncSession]] = {}


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """The engine for ``settings``, built once per process per DSN.

    Not ``@lru_cache``d: that keys on the argument, and ``Settings`` is an
    unhashable pydantic model, so passing one raised ``TypeError``.
    """
    cfg = settings or get_settings()
    dsn = cfg.postgres.dsn()
    engine = _ENGINES.get(dsn)
    if engine is None:
        engine = create_async_engine(
            dsn,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=5,
            echo=False,
        )
        _ENGINES[dsn] = engine
    return engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """The session factory for ``settings``, sharing that DSN's engine."""
    cfg = settings or get_settings()
    dsn = cfg.postgres.dsn()
    maker = _SESSIONMAKERS.get(dsn)
    if maker is None:
        maker = async_sessionmaker(get_engine(cfg), expire_on_commit=False)
        _SESSIONMAKERS[dsn] = maker
    return maker


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
    """Close every pool this module opened. Call on application shutdown.

    Disposes all cached engines, not just the default one: a test or a tool
    that injected its own ``Settings`` created a second pool, and the old
    ``get_engine()`` call here would silently leave it open - and, worse, build
    a fresh default engine just to dispose it if none had been made yet.
    """
    engines = list(_ENGINES.values())
    _ENGINES.clear()
    _SESSIONMAKERS.clear()
    for engine in engines:
        await engine.dispose()
