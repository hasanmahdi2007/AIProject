"""HTTP client for the retrieval service. **No search logic lives here.**

``services/retrieval/**`` belongs to another agent and is unwritten at the time
of this commit, but its HTTP contract is frozen and merged:
``SearchRequest`` and ``SearchResponse`` in ``beacon_core.schemas``. So this
module is a transport and nothing more - it posts the frozen request to
``settings.retrieval.base_url`` and validates the frozen response.

Nothing in this worker fuses rankings, applies an age or sex filter, or
reranks. Not even temporarily: a "temporary" post-filter here would be the
overfiltering bug, implemented in the one place nobody would think to look for
it, and it would make the retrieval service's own tests pass while the system
returned four candidates instead of twenty. The filters are *sent* - inside the
request, so they are evaluated inside the index traversal - and that is the
whole of this file's involvement.

The tests stub this at the HTTP layer with ``respx``, against the same frozen
models, so they verify the request this worker actually sends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from beacon_core.logging import get_logger
from beacon_core.schemas import (
    AgeSexFilter,
    PatientProfile,
    RankedQuery,
    SearchRequest,
    SearchResponse,
)
from beacon_core.settings import RetrievalSettings

__all__ = ["RetrievalClient", "RetrievalError", "build_search_request"]

log = get_logger(__name__)


class RetrievalError(Exception):
    """Retrieval was unreachable or answered with something unusable."""


def build_search_request(
    *, queries: list[RankedQuery], profile: PatientProfile | None, settings: RetrievalSettings
) -> SearchRequest:
    """Assemble the frozen request from this run's queries and profile.

    ``include_unknown`` stays ``True``: a trial whose age bounds are absent or
    unparseable must survive the filter and let the rule checker decide. An
    unknown bound is "possibly eligible", never "exclude", and a wrongly
    excluded trial never appears anywhere for anyone to notice.

    ``apply_age_sex_filters=False`` sends an *empty* filter rather than a
    different request shape, because that configuration is an ablation row and
    has to differ from the full system in exactly one respect.
    """
    filters = (
        AgeSexFilter(
            age=profile.age if profile else None,
            sex=profile.sex if profile else None,
            include_unknown=True,
        )
        if settings.apply_age_sex_filters
        else AgeSexFilter(include_unknown=True)
    )
    return SearchRequest(
        queries=queries,
        filters=filters,
        fuse_top_k=settings.fuse_top_k,
        rerank_top_k=settings.rerank_top_k,
        rrf_k=settings.rrf_k,
        use_rerank=settings.use_rerank,
    )


@dataclass
class RetrievalClient:
    """``POST {base_url}/search``, typed both ways."""

    base_url: str
    timeout_seconds: float = 30.0
    client: httpx.AsyncClient | None = None

    @property
    def search_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/search"

    async def search(self, request: SearchRequest) -> SearchResponse:
        payload: dict[str, Any] = request.model_dump(mode="json")
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await client.post(self.search_url, json=payload)
            response.raise_for_status()
            parsed = SearchResponse.model_validate(response.json())
        except httpx.HTTPError as exc:
            raise RetrievalError(f"retrieval at {self.search_url} failed: {exc}") from exc
        except ValueError as exc:
            # A 200 whose body is not a SearchResponse is a contract violation,
            # not a transport blip: surfacing it as RetrievalError keeps the
            # worker's retry policy from hammering a mismatched service.
            raise RetrievalError(
                f"retrieval at {self.search_url} returned an unparseable body: {exc}"
            ) from exc
        finally:
            if own_client:
                await client.aclose()

        log.info(
            "worker.retrieval.search",
            url=self.search_url,
            queries=len(request.queries),
            candidates=len(parsed.candidates),
            total_before_rerank=parsed.total_before_rerank,
            timings_ms=parsed.timings_ms,
        )
        return parsed
