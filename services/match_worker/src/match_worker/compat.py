"""Event-loop selection, because the checkpointer cannot use Windows' default.

``langgraph-checkpoint-postgres`` is built on psycopg 3, and psycopg's async
mode cannot run on the ``ProactorEventLoop`` that ``asyncio.run`` picks by
default on Windows::

    psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run
    in async mode.

The project is developed on Windows 11 and CI runs on Ubuntu, so this fails on
exactly one of the two - the one where the demo is recorded. It is not a
warning and there is no fallback: the worker simply cannot open a checkpointer
connection.

So every entry point that runs the worker goes through :func:`run`, which asks
for a selector loop on Windows and leaves every other platform alone.
``loop_factory`` is the supported way to do that as of Python 3.12 (the
alternative, ``WindowsSelectorEventLoopPolicy``, is deprecated).
"""

from __future__ import annotations

import asyncio
import os
import selectors
from collections.abc import Coroutine
from typing import Any

__all__ = ["loop_factory", "run"]


def loop_factory() -> asyncio.AbstractEventLoop:
    """A psycopg-compatible event loop for this platform.

    Branches on ``os.name`` rather than ``sys.platform`` deliberately: mypy
    narrows ``sys.platform`` to the platform it is *running on*, so that form
    reports one of these two branches as unreachable - on Windows locally, and
    on the other branch in Ubuntu CI. ``os.name`` keeps both branches type
    checked everywhere.
    """
    if os.name == "nt":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run``, on a loop the Postgres checkpointer can actually use."""
    return asyncio.run(coro, loop_factory=loop_factory)
