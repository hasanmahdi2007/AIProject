"""ASGI entrypoint: ``uvicorn gateway.main:app``.

The spike did ``from router import router`` - an implicit relative import that
only resolved because the file sat in the working directory. It breaks the
moment the package is installed, which is now. Imports here are absolute.

Nothing at import time touches the network or needs an API key: the engine, the
Redis client and the Claude client are all built on first use, so ``import
gateway.main`` works in CI with no services running.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from beacon_core.db import dispose_engine
from beacon_core.logging import configure_logging
from beacon_core.redis import get_redis
from beacon_core.settings import get_settings
from gateway import __version__
from gateway.router import router as api_router

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Shut the pools down on exit.

    Both are process-cached, so without this a reload leaks Postgres
    connections until the server runs out.
    """
    configure_logging()
    yield
    await get_redis().aclose()
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Beacon gateway",
        version=__version__,
        description=(
            "Clinical-trial matching: submit a patient note, stream the "
            "pipeline's progress, ask questions about the result."
        ),
        lifespan=lifespan,
    )
    # The console is served from another origin (:3000 against the gateway's
    # :8000), so without this the browser rejects every call at the preflight
    # and the user sees only "Failed to fetch". Installed before the router so
    # it wraps every route, including the SSE stream.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Last-Event-ID"],
        expose_headers=["Location", "X-Match-Id"],
    )
    """``Location`` and ``X-Match-Id`` must be exposed explicitly: a browser hides
    every response header from script except a short safelist, so a console
    following the 202's ``Location`` would read ``null`` without this.
    ``Last-Event-ID`` is allowed because that is the header an ``EventSource``
    sends when it reconnects a dropped SSE stream."""

    app.include_router(api_router)
    return app


app = create_app()
