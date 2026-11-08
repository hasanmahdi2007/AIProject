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

from beacon_core.db import dispose_engine
from beacon_core.logging import configure_logging
from beacon_core.redis import get_redis
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
    app.include_router(api_router)
    return app


app = create_app()
