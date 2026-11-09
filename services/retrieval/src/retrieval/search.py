"""The Qdrant half: two searches per query, with the filter inside each one.

This module is where "the filter is a constraint on the index traversal" stops
being a sentence and becomes a wire format. The filter built by
:mod:`retrieval.filters` is attached to every :class:`QueryRequest` as
``filter=``, which Qdrant evaluates *during* the HNSW graph walk (and, for the
sparse field, during the inverted-index scan), so the walk only ever visits and
only ever counts points that satisfy it. Ask for 20 under a 10%-selective filter
and 20 come back.

The two designs this deliberately is not:

* **Filter first, then search the survivors.** Materialising the matching subset
  of 375,580 trials and searching that is a full scan: the whole point of an
  approximate index is that it is never asked to look at everything, and a
  pre-filter asks it to do exactly that before it starts.
* **Search, then filter the results.** The overfiltering bug. The graph walk
  visits ``hnsw_ef`` points chosen purely by vector proximity; dropping the
  non-matching ones afterwards leaves roughly ``hnsw_ef x selectivity``
  survivors, so with ``hnsw_ef=40`` and a 10% filter a request for 20
  candidates returns about 4 - with no error, no warning, and a response that
  looks like a legitimately short result set. ``test_filtered_top_k_is_not_top_4``
  in the integration suite runs both paths against the same live collection and
  asserts the difference.

**One network round trip.** All ``2N`` searches (dense and sparse for each of
the N queries) go out in a single ``query_batch_points`` call. Qdrant executes
them server-side and returns ``2N`` result lists, so latency is one round trip
plus the slowest search rather than ``2N`` sequential round trips.

**Why not Qdrant's own server-side fusion?** ``prefetch`` with
``FusionQuery(fusion=Fusion.RRF)`` would fuse dense and sparse inside Qdrant and
is the obvious thing to reach for. It is not usable here, for two reasons that
are both about this project's requirements rather than about Qdrant:

1. ``FusionQuery`` exposes no ``k``. Qdrant's RRF uses its own fixed constant,
   and ``settings.retrieval.rrf_k`` is an ablation row - it has to be a number
   this code can change from config.
2. Server-side fusion can only fuse the prefetches of *one* request. The
   cross-query level - fusing N queries with weights derived from
   ``RankedQuery.rank`` - has no server-side expression at all.

So the raw ranked lists come back and :mod:`retrieval.fusion` fuses them here.
The cost is shipping ``2N x limit`` IDs and payloads over the wire instead of
``limit``; the payloads are six small scalar fields, which is why that is
affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client import models

from beacon_core.schemas import RankedQuery
from beacon_core.settings import QdrantSettings
from retrieval.encoders import DenseQueryEncoder, SparseQueryEncoder
from retrieval.protocols import QdrantReader

__all__ = ["EncodedQuery", "Hit", "HybridSearcher", "QueryHits"]

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Hit:
    """One point returned by one retriever, in rank order within its branch."""

    nct_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EncodedQuery:
    """One generated query, already embedded by both bi-encoders.

    Encoding is a separate step from searching so that the two can be timed
    separately - ``timings_ms`` has to show whether a slow request was the
    embedder or Qdrant, or the whole "the cheap stage is cheap" story is an
    assertion rather than a measurement - and so that :meth:`HybridSearcher.search`
    can be tested against a fake client with hand-written vectors and no models
    loaded.
    """

    rank: int
    text: str
    dense: list[float]
    sparse: models.SparseVector


@dataclass(frozen=True, slots=True)
class QueryHits:
    """Both branches' hits for one generated query.

    ``query_rank`` is carried through from ``RankedQuery.rank`` rather than
    being the list index, because the cross-query fusion weight is a function of
    the declared importance of the query, not of where it happens to sit in a
    list.
    """

    query_rank: int
    query_text: str
    dense: list[Hit]
    sparse: list[Hit]

    def dense_ids(self) -> list[str]:
        return [hit.nct_id for hit in self.dense]

    def sparse_ids(self) -> list[str]:
        return [hit.nct_id for hit in self.sparse]


class HybridSearcher:
    """Runs the ``2N`` filtered searches and returns their raw ranked lists.

    It deliberately does no fusion, no reranking and no scoring of its own: this
    is the I/O boundary, and keeping it free of ranking logic is what lets the
    ranking logic be tested without a container.
    """

    def __init__(
        self,
        client: QdrantReader,
        settings: QdrantSettings,
        dense: DenseQueryEncoder,
        sparse: SparseQueryEncoder,
    ) -> None:
        self._client = client
        self._settings = settings
        self._dense = dense
        self._sparse = sparse

    @property
    def client(self) -> QdrantReader:
        """Exposed so the health check can inspect the same connection the
        searches use, rather than opening a second one that might point
        somewhere else."""
        return self._client

    def encode(self, queries: list[RankedQuery]) -> list[EncodedQuery]:
        """Embed every query with both bi-encoders.

        Both encoders are called per query rather than in one batch: N is 3 by
        default and at most 10, so batching would save a few milliseconds of
        ONNX overhead and cost the per-query association that the cross-query
        fusion weight depends on.
        """
        return [
            EncodedQuery(
                rank=query.rank,
                text=query.text,
                dense=self._dense.encode_query(query.text),
                sparse=self._sparse.encode_query(query.text),
            )
            for query in queries
        ]

    def _requests(
        self,
        queries: list[EncodedQuery],
        *,
        search_filter: models.Filter | None,
        limit: int,
        hnsw_ef: int | None,
    ) -> list[models.QueryRequest]:
        """Build ``2N`` query requests: dense then sparse, per query, in order.

        The ordering is load-bearing - :meth:`search` pairs the responses back
        up by position - so the response count is checked rather than trusted.
        """
        requests: list[models.QueryRequest] = []
        for query in queries:
            requests.append(
                models.QueryRequest(
                    query=query.dense,
                    using=self._settings.dense_vector_name,
                    # The filter. Inside the request, evaluated during the walk.
                    filter=search_filter,
                    limit=limit,
                    with_payload=True,
                    # hnsw_ef is left to the collection's own config unless a
                    # caller (the integration test, reproducing the brief's
                    # ef_search=40 scenario) pins it. No literal default here:
                    # Qdrant already raises ef to at least `limit`, and
                    # second-guessing that from this side would be a hidden
                    # tuning knob that no ablation row covers.
                    params=None if hnsw_ef is None else models.SearchParams(hnsw_ef=hnsw_ef),
                )
            )
            requests.append(
                models.QueryRequest(
                    query=query.sparse,
                    using=self._settings.sparse_vector_name,
                    # The same filter, inside the sparse search too. A filter
                    # applied to only one branch is worse than none: the fused
                    # list then contains ineligible trials from the unfiltered
                    # branch, and they are indistinguishable from eligible ones.
                    filter=search_filter,
                    limit=limit,
                    with_payload=True,
                    # No search params: the sparse field is an inverted index,
                    # so hnsw_ef is meaningless for it.
                )
            )
        return requests

    async def search(
        self,
        queries: list[EncodedQuery],
        *,
        search_filter: models.Filter | None,
        limit: int,
        hnsw_ef: int | None = None,
    ) -> list[QueryHits]:
        """Run every branch of every query in one batched call.

        Args:
            queries: the already-encoded queries, from :meth:`encode`.
            search_filter: the hard filter, or ``None`` for unconstrained.
            limit: per-branch depth. Set to ``fuse_top_k`` by the caller: each
                branch retrieves as deep as the fused set will be, so fusion has
                material to disagree over. Shallower branches would make RRF
                mostly a no-op, since documents absent from a short list
                contribute nothing.
            hnsw_ef: pin the HNSW beam width, or ``None`` to use the
                collection's configured value.

        Returns:
            One :class:`QueryHits` per query, branches in rank order.
        """
        requests = self._requests(
            queries, search_filter=search_filter, limit=limit, hnsw_ef=hnsw_ef
        )
        responses = await self._client.query_batch_points(
            collection_name=self._settings.collection,
            requests=requests,
        )
        if len(responses) != len(requests):
            raise RuntimeError(
                f"Qdrant returned {len(responses)} result sets for {len(requests)} "
                "requests; cannot pair them back to their queries"
            )

        results: list[QueryHits] = []
        for index, query in enumerate(queries):
            dense_response = responses[index * 2]
            sparse_response = responses[index * 2 + 1]
            results.append(
                QueryHits(
                    query_rank=query.rank,
                    query_text=query.text,
                    dense=self._to_hits(dense_response.points, branch="dense"),
                    sparse=self._to_hits(sparse_response.points, branch="sparse"),
                )
            )
        return results

    def _to_hits(self, points: list[models.ScoredPoint], *, branch: str) -> list[Hit]:
        """Convert scored points to hits, dropping any without an NCT ID.

        A point whose payload has no ``nct_id`` cannot be used: every later
        stage - the Postgres hydration, the rule checker, the final answer -
        keys on it, and the Qdrant point ID is a uuid5 of the NCT ID and so
        cannot be reversed. Such a point is a corrupt write rather than a
        normal case, so it is dropped with a warning instead of being allowed to
        propagate as a candidate with no identity.
        """
        hits: list[Hit] = []
        for point in points:
            payload = point.payload or {}
            nct_id = payload.get("nct_id")
            if not isinstance(nct_id, str) or not nct_id:
                log.warning(
                    "qdrant_point_without_nct_id",
                    point_id=str(point.id),
                    branch=branch,
                    collection=self._settings.collection,
                )
                continue
            hits.append(Hit(nct_id=nct_id, score=point.score, payload=dict(payload)))
        return hits
