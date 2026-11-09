"""ASGI entrypoint: ``uvicorn retrieval.main:app --port 8001``.

Port 8001 is ``settings.retrieval.base_url``'s port, which is what the match
worker calls.

Nothing at import time touches the network, loads a model or needs an API key.
That is a hard requirement rather than a preference: the unit suite imports this
module to check the request/response contract round-trips, and it has to do so
with no containers running, no ONNX weights on disk, and no Anthropic key - this
service never calls an LLM at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from beacon_core.db import dispose_engine
from beacon_core.logging import configure_logging
from retrieval import __version__
from retrieval.deps import get_qdrant_client
from retrieval.router import router as api_router

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the pools on the way out.

    Both the Qdrant client and the Postgres engine are process-cached, so
    without this a reloading dev server leaks a connection pool per reload until
    Postgres refuses new connections. The ONNX sessions are left to the garbage
    collector; they hold no sockets.
    """
    configure_logging()
    yield
    await get_qdrant_client().close()
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Beacon retrieval",
        version=__version__,
        description=(
            "Hybrid dense + sparse retrieval over the clinical-trial index: "
            "age/sex filters inside the index traversal, Reciprocal Rank Fusion "
            "across both retrievers and all generated queries, then a "
            "cross-encoder rerank. Returns candidates, never recommendations."
        ),
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
