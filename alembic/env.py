"""Alembic environment, async (asyncpg) so there is only one DB driver.

The URL comes from ``beacon_core.settings``, never from ``alembic.ini``: one
connection string for the whole system, and no credentials in a committed file.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from beacon_core.models import Base
from beacon_core.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().postgres.dsn())

target_metadata = Base.metadata


EXTERNALLY_MANAGED_TABLES = frozenset(
    {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
)
"""Tables owned by ``langgraph-checkpoint-postgres``, not by us.

``AsyncPostgresSaver.setup()`` creates these, and it is the library's own
idempotent migration path - the match worker calls it on startup. They are
therefore real tables in the database that are deliberately absent from
``Base.metadata``.

Without this exclusion, autogenerate sees four unexplained tables and proposes
to **drop** them, so ``alembic check`` fails on any machine where the worker
has ever run. It started failing the moment C1's crash test ran against the dev
database. A drift guard that always fails is a drift guard everyone learns to
ignore, and then real drift slips through it - which is the whole reason this
check exists.

Writing our own migration for them instead would be worse: it would duplicate
the library's schema and silently diverge the next time LangGraph changes it.
"""


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Keep externally-managed tables out of autogenerate's comparison.

    Filtering by *name* rather than by object is what makes a reflected table
    alembic has never heard of comparable at all: ``include_object`` is only
    consulted for objects that reflection already produced.
    """
    if type_ == "table":
        return name not in EXTERNALLY_MANAGED_TABLES
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
