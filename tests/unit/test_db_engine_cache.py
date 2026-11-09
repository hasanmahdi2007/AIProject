"""The engine and session-factory cache in :mod:`beacon_core.db`.

No database is touched: ``create_async_engine`` builds a pool lazily and does
not connect, so every assertion here is about caching and identity.

The regression these tests exist for: both functions used to be
``@lru_cache``d, which keys on the argument. ``Settings`` is a pydantic model
and therefore unhashable, so the ``settings`` parameter they advertise could
never be passed - ``get_sessionmaker(Settings())`` raised ``TypeError`` while
``get_sessionmaker()`` worked. A signature that only accepts its own default is
a trap, and it cost an agent real time before being found.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from beacon_core import db
from beacon_core.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_caches() -> Iterator[None]:
    """Leave the module-level caches exactly as they were found.

    These tests deliberately create engines for made-up DSNs; letting them
    leak would hand a later test a pool pointing at a database that is not
    there.
    """
    engines = dict(db._ENGINES)
    makers = dict(db._SESSIONMAKERS)
    db._ENGINES.clear()
    db._SESSIONMAKERS.clear()
    try:
        yield
    finally:
        db._ENGINES.clear()
        db._ENGINES.update(engines)
        db._SESSIONMAKERS.clear()
        db._SESSIONMAKERS.update(makers)


def test_settings_can_actually_be_passed() -> None:
    """The regression. This raised ``TypeError: unhashable type: 'Settings'``."""
    assert db.get_sessionmaker(Settings()) is not None
    assert db.get_engine(Settings()) is not None


def test_one_pool_per_database_per_process() -> None:
    """An engine owns a connection pool, so building one per request silently
    exhausts Postgres connections. Two callers naming the same database must
    share."""
    settings = Settings()

    assert db.get_engine(settings) is db.get_engine(settings)
    assert db.get_sessionmaker(settings) is db.get_sessionmaker(settings)


def test_passing_settings_and_passing_nothing_reach_the_same_pool() -> None:
    """Default ``Settings()`` names the same database as no argument at all, so
    injecting it must not quietly open a second pool against it."""
    assert db.get_engine(Settings()) is db.get_engine()
    assert db.get_sessionmaker(Settings()) is db.get_sessionmaker()


def test_a_different_database_gets_its_own_pool() -> None:
    """The point of accepting ``Settings``: a test pointing somewhere else must
    not be handed the default engine."""
    elsewhere = Settings()
    elsewhere.postgres.port = elsewhere.postgres.port + 1

    assert db.get_engine(elsewhere) is not db.get_engine()
    assert db.get_sessionmaker(elsewhere) is not db.get_sessionmaker()


def test_the_factory_shares_the_engine_for_its_dsn() -> None:
    """``get_sessionmaker`` must not build a second engine behind the scenes."""
    settings = Settings()
    maker = db.get_sessionmaker(settings)

    assert len(db._ENGINES) == 1
    assert maker.kw["bind"] is db.get_engine(settings)


async def test_dispose_closes_every_pool_not_just_the_default() -> None:
    """``dispose_engine`` used to call ``get_engine()`` with no arguments, so an
    injected engine stayed open - and if nothing had built the default engine
    yet, it created one purely in order to dispose it."""
    elsewhere = Settings()
    elsewhere.postgres.port = elsewhere.postgres.port + 1
    db.get_engine()
    db.get_engine(elsewhere)
    assert len(db._ENGINES) == 2

    await db.dispose_engine()

    assert db._ENGINES == {}
    assert db._SESSIONMAKERS == {}


async def test_dispose_on_a_cold_process_builds_nothing() -> None:
    """Shutdown before anything opened a pool must not open one."""
    await db.dispose_engine()

    assert db._ENGINES == {}
