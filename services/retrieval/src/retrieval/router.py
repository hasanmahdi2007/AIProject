"""HTTP surface: ``POST /search`` and ``GET /health``.

The route is thin on purpose. It parses the frozen contract, calls the service
and returns the frozen contract; every decision worth defending lives in
:mod:`retrieval.service`, :mod:`retrieval.fusion` and :mod:`retrieval.filters`,
where it can be tested without an HTTP client.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status

from beacon_core.schemas import SearchRequest, SearchResponse
from retrieval import __version__
from retrieval.collection import CollectionVerificationError
from retrieval.deps import ServiceDep, SettingsDep

__all__ = ["router"]

log = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse, summary="Hybrid trial retrieval")
async def search(request: SearchRequest, service: ServiceDep) -> SearchResponse:
    """Narrow the trial corpus to ``rerank_top_k`` candidates for one patient.

    Hybrid dense + sparse search per query, with the age/sex filter evaluated
    inside the index traversal, fused with RRF twice (dense+sparse, then across
    queries), then reranked by a cross-encoder.

    The returned trials are **candidates, not recommendations**: no eligibility
    decision is made here. ``timings_ms`` carries the per-stage cost.
    """
    try:
        return await service.search(request)
    except CollectionVerificationError as exc:
        # The index is not there or not usable. 503 rather than 500: the service
        # is fine, its dependency is not, and the message says which.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/health", summary="Dependency-by-dependency health")
async def health(service: ServiceDep, settings: SettingsDep) -> dict[str, Any]:
    """Report what the Qdrant collection actually declares.

    A probe that only answers ``{"status": "ok"}`` is close to useless: it is
    the per-dependency detail that tells you whether to rebuild the index or
    look at Postgres. This one reports the things that otherwise fail silently -
    the sparse field's IDF modifier, the dense dimension, the payload indexes
    behind the filters - and returns 503 when any of them is wrong, so a
    container orchestrator refuses to route traffic to a process that would
    return quietly degraded results.
    """
    body: dict[str, Any] = {
        "service": "retrieval",
        "version": __version__,
        "qdrant_url": settings.qdrant.url,
    }
    try:
        report = await service.verify()
    except CollectionVerificationError as exc:
        body["status"] = "unhealthy"
        body["problems"] = [str(exc)]
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, body) from exc
    except Exception as exc:  # pragma: no cover - transport failures
        body["status"] = "unhealthy"
        body["problems"] = [f"cannot reach Qdrant: {exc!r}"]
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, body) from exc

    body.update(
        {
            "status": "ok" if report.ok else "degraded",
            "collection": report.name,
            "points": report.points,
            "dense_dim": report.dense_dim,
            "sparse_modifier": report.sparse_modifier,
            "indexed_payload_fields": list(report.indexed_payload_fields),
            "problems": list(report.problems),
            "warnings": list(report.warnings),
        }
    )
    if not report.ok:
        log.error("retrieval_collection_unusable", problems=report.problems)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, body)
    if report.warnings:
        log.warning("retrieval_collection_warnings", warnings=report.warnings)
    return body
