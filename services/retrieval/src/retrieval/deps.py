"""FastAPI dependency providers.

Everything external - the Qdrant connection, the three ONNX models, the
Postgres pool - arrives through a provider rather than a module-level global, so
a test can replace it with ``app.dependency_overrides`` instead of
monkeypatching a module.

Every provider is lazy and cached. Lazy because ``import retrieval.main`` must
work in CI with nothing running and no models on disk: the unit suite imports
the app to check the request/response contract. Cached because each of these
owns something expensive - an HTTP connection pool, an ONNX session with its own
threads, a Postgres pool - and building one per request is the difference
between a fast service and a service that re-reads 400 MB of weights from disk
on every search.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import AsyncQdrantClient

from beacon_core.db import get_sessionmaker
from beacon_core.settings import Settings, get_settings
from retrieval.encoders import get_cross_encoder, get_dense_encoder, get_sparse_encoder
from retrieval.hydrate import PostgresTrialTextStore
from retrieval.search import HybridSearcher
from retrieval.service import RetrievalService

__all__ = [
    "ServiceDep",
    "SettingsDep",
    "get_qdrant_client",
    "provide_service",
    "provide_settings",
]


def provide_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(provide_settings)]


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    """One client, one connection pool, for the process.

    ``prefer_grpc`` is left off: the REST transport is what the brief's "verify
    with curl" workflow talks to, and at ~100 candidates per response the
    serialisation difference is far below the reranker's noise floor. It is one
    line to change if a profile ever says otherwise.
    """
    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant.url)


@lru_cache(maxsize=1)
def provide_service() -> RetrievalService:
    """Build the service once.

    Model loading happens here, on the first request rather than at import, and
    it is the slow part: three ONNX sessions, about 400 MB of weights. A
    production deployment should hit ``/health`` once after start so the first
    real search does not pay for it - which is one of the reasons ``/health``
    touches the collection and the reranker rather than just returning "ok".
    """
    settings = get_settings()
    searcher = HybridSearcher(
        client=get_qdrant_client(),
        settings=settings.qdrant,
        dense=get_dense_encoder(settings),
        sparse=get_sparse_encoder(settings),
    )
    return RetrievalService(
        searcher=searcher,
        text_store=PostgresTrialTextStore(get_sessionmaker(settings)),
        cross_encoder=get_cross_encoder(settings),
        settings=settings,
    )


ServiceDep = Annotated[RetrievalService, Depends(provide_service)]
