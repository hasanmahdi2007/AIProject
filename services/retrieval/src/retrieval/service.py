"""The funnel: 375,580 -> ``fuse_top_k`` -> ``rerank_top_k``.

One request walks five stages, each timed into ``SearchResponse.timings_ms`` so
the cost story is a measurement:

===========  =========================================================
``embed``    N query strings -> N dense vectors + N sparse vectors.
``qdrant``   one batched call, ``2N`` filtered searches inside the index.
``fuse``     RRF twice, in pure Python: dense+sparse, then across queries.
``hydrate``  one Postgres statement for the fused top ``fuse_top_k``.
``rerank``   cross-encoder over ``fuse_top_k`` pairs -> ``rerank_top_k``.
===========  =========================================================

plus ``total``. The expected shape is ``rerank`` dominating everything else by
an order of magnitude, which is the whole argument for the funnel narrowing
before it gets there; if it ever does not, the funnel's proportions are wrong
and the numbers say so.

**What this returns and what it does not.** A ``TrialCandidate`` is a candidate,
never a suggestion. Retrieval answers one question - "is this trial about the
right disease?" - and nothing here decides eligibility. In particular,
embeddings cannot compare numbers: a 24-year-old's note will happily retrieve an
"age 25 or older" trial with a high cosine, because "25" and "24" are
near-identical tokens in a 384-dimensional space trained on meaning. That is
*fine and expected*. Numeric correctness comes from the typed payload filter in
:mod:`retrieval.filters` and then from the LLM rule checker reading the
eligibility text - never from a similarity score, and never from the reranker,
which is a better relevance model and still not a reasoner about integers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import structlog

from beacon_core.schemas import RankedQuery, SearchRequest, SearchResponse, Sex, TrialCandidate
from beacon_core.settings import Settings
from retrieval.collection import CollectionReport, verify_collection
from retrieval.encoders import CrossEncoder
from retrieval.filters import build_filter
from retrieval.fusion import fuse_across_queries, fuse_hybrid, rank_ids
from retrieval.hydrate import TrialText, TrialTextStore, rerank_document
from retrieval.search import HybridSearcher, QueryHits

__all__ = ["ResolvedParams", "RetrievalService", "resolve_params"]

log = structlog.get_logger(__name__)

_VALID_SEX: frozenset[str] = frozenset({"male", "female", "all"})


@dataclass(frozen=True, slots=True)
class ResolvedParams:
    """The knobs for one request, after config and request have been reconciled."""

    fuse_top_k: int
    rerank_top_k: int
    rrf_k: int
    use_rerank: bool
    apply_filters: bool


def resolve_params(request: SearchRequest, settings: Settings) -> ResolvedParams:
    """Settings are authoritative; the request may override, explicitly.

    This exists because of a subtlety that would otherwise make half the
    ablation matrix unreachable. ``SearchRequest`` declares non-``None``
    defaults - ``rrf_k=60``, ``fuse_top_k=100`` and so on - so by the time a
    body has been parsed there is no way to tell "the caller asked for 60" from
    "the caller said nothing and pydantic filled in 60". Reading the parsed
    value unconditionally would make ``settings.retrieval.rrf_k`` dead config:
    setting ``BEACON_RETRIEVAL__RRF_K=30`` would change nothing at all, with no
    error, and the sweep row would quietly measure k=60 four times.

    Pydantic's ``model_fields_set`` records which fields the incoming JSON
    actually carried, so a field the caller omitted falls through to settings
    and a field the caller sent wins. An ablation is then a config change, and a
    caller who really does want a different ``k`` for one request can still say
    so.

    ``apply_age_sex_filters`` has no request-side equivalent at all - the
    contract is frozen and this is not a field worth adding to it, since no
    caller should be choosing per request whether hard filters apply - so it is
    read from settings only.
    """
    sent = request.model_fields_set
    retrieval = settings.retrieval
    return ResolvedParams(
        fuse_top_k=request.fuse_top_k if "fuse_top_k" in sent else retrieval.fuse_top_k,
        rerank_top_k=request.rerank_top_k if "rerank_top_k" in sent else retrieval.rerank_top_k,
        rrf_k=request.rrf_k if "rrf_k" in sent else retrieval.rrf_k,
        use_rerank=request.use_rerank if "use_rerank" in sent else retrieval.use_rerank,
        apply_filters=retrieval.apply_age_sex_filters,
    )


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    """Record a stage's wall-clock milliseconds.

    ``perf_counter`` rather than ``time()``: the latter is subject to NTP steps,
    and a stage that reports -3 ms is worse than no number.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = (time.perf_counter() - start) * 1000.0


def _as_sex(value: object) -> Sex | None:
    """Coerce a payload value to the contract's ``Sex``, or ``None``.

    A payload holding something unexpected must not raise a 500 in the middle of
    a search, and must not be guessed at either: an unrecognised value becomes
    ``None``, which the contract defines as "unknown", which means the rule
    checker decides. The same direction of caution as everywhere else here.
    """
    if isinstance(value, str) and value in _VALID_SEX:
        # mypy cannot narrow a `str in frozenset` to a Literal, so this is the
        # one place the Literal is asserted by hand rather than inferred.
        return value  # type: ignore[return-value]
    return None


def _as_float(value: object) -> float | None:
    """Payload number -> float, with anything non-numeric read as unknown."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


class RetrievalService:
    """Everything one ``POST /search`` does, with its dependencies injected."""

    def __init__(
        self,
        *,
        searcher: HybridSearcher,
        text_store: TrialTextStore,
        cross_encoder: CrossEncoder | None,
        settings: Settings,
        hnsw_ef: int | None = None,
    ) -> None:
        """
        Args:
            cross_encoder: ``None`` is allowed and means "this process cannot
                rerank" - used by the unit suite, which must run with no models
                on disk. It is not the same thing as ``use_rerank=False``: that
                is a deliberate ablation, this is a missing capability, and the
                two are logged differently so an accidentally un-reranked eval
                run is visible rather than indistinguishable from an intended
                one.
            hnsw_ef: pinned beam width, for tests that need to reproduce a
                specific ``ef_search``. Production leaves it ``None``.
        """
        self._searcher = searcher
        self._text_store = text_store
        self._cross_encoder = cross_encoder
        self._settings = settings
        self._hnsw_ef = hnsw_ef

    async def verify(self) -> CollectionReport:
        """Check the collection's declarations. See :mod:`retrieval.collection`."""
        return await verify_collection(self._searcher.client, self._settings.qdrant)

    async def search(self, request: SearchRequest) -> SearchResponse:
        params = resolve_params(request, self._settings)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        search_filter = build_filter(request.filters, enabled=params.apply_filters)

        with _timed(timings, "embed"):
            encoded = self._searcher.encode(list(request.queries))

        with _timed(timings, "qdrant"):
            per_query = await self._searcher.search(
                encoded,
                search_filter=search_filter,
                # Each branch goes as deep as the fused set will be. Shallower
                # branches would make RRF close to a no-op.
                limit=params.fuse_top_k,
                hnsw_ef=self._hnsw_ef,
            )

        with _timed(timings, "fuse"):
            fused_ids, rrf_scores = self._fuse(per_query, rrf_k=params.rrf_k)
            kept_ids = fused_ids[: params.fuse_top_k]

        with _timed(timings, "hydrate"):
            texts = await self._text_store.fetch(kept_ids)

        payloads = _collect_payloads(per_query)
        branch_scores = _collect_branch_scores(per_query)

        with _timed(timings, "rerank"):
            ordered_ids, rerank_scores = self._rerank(
                kept_ids,
                texts=texts,
                # The rank-1 query, not all N. See :meth:`_rerank`.
                query=min(request.queries, key=lambda q: q.rank),
                use_rerank=params.use_rerank,
            )

        candidates = [
            _build_candidate(
                nct_id=nct_id,
                final_rank=position + 1,
                payload=payloads.get(nct_id, {}),
                text=texts.get(nct_id),
                dense_score=branch_scores[0].get(nct_id),
                sparse_score=branch_scores[1].get(nct_id),
                rrf_score=rrf_scores.get(nct_id),
                rerank_score=rerank_scores.get(nct_id),
            )
            for position, nct_id in enumerate(ordered_ids[: params.rerank_top_k])
        ]

        timings["total"] = (time.perf_counter() - total_start) * 1000.0
        log.info(
            "retrieval_search",
            queries=len(request.queries),
            filtered=search_filter is not None,
            fused=len(fused_ids),
            returned=len(candidates),
            reranked=params.use_rerank and self._cross_encoder is not None,
            **{f"ms_{stage}": round(value, 1) for stage, value in timings.items()},
        )
        return SearchResponse(
            candidates=candidates,
            # The size of the fused set *before* truncation to rerank_top_k, so
            # a caller can see whether the funnel actually had 100 candidates to
            # choose 20 from or whether the filter left it only 25.
            total_before_rerank=len(kept_ids),
            timings_ms={stage: round(value, 3) for stage, value in timings.items()},
        )

    # ------------------------------------------------------------- stages --

    def _fuse(
        self,
        per_query: Sequence[QueryHits],
        *,
        rrf_k: int,
    ) -> tuple[list[str], dict[str, float]]:
        """Both levels of RRF. See :mod:`retrieval.fusion` for the arithmetic.

        Level 1 fuses each query's dense and sparse lists into a per-query
        ranking. Level 2 fuses those rankings across queries, weighted by
        ``1/RankedQuery.rank``.

        Returns the final ID order and the cross-query scores, which go onto the
        candidates as ``rrf_score``.
        """
        level_one: list[tuple[int, list[str]]] = []
        for hits in per_query:
            within = fuse_hybrid(hits.dense_ids(), hits.sparse_ids(), k=rrf_k)
            level_one.append((hits.query_rank, rank_ids(within)))
        across = fuse_across_queries(level_one, k=rrf_k)
        return rank_ids(across), across

    def _rerank(
        self,
        nct_ids: Sequence[str],
        *,
        texts: Mapping[str, TrialText],
        query: RankedQuery,
        use_rerank: bool,
    ) -> tuple[list[str], dict[str, float]]:
        """Reorder the fused set with the cross-encoder.

        **Against the rank-1 query only, not all N.** The reranker is the
        dominant cost of the request at roughly 100 pairs per second, so scoring
        every candidate against every query would multiply the most expensive
        stage by N - about 300 pairs and three seconds at the default N=3, and
        ten seconds at the top of the Day 3 sweep - to re-ask a question the
        first stage already answered. The N queries exist to widen *recall*:
        each one pulls in trials the others miss, and RRF has already folded all
        of that into which 100 candidates are here. Precision at the top is one
        question - "given the patient's primary clinical situation, is this the
        right trial?" - and the rank-1 query is by construction the one that
        states it. (The alternative, ``max`` over per-query cross-encoder
        scores, is the obvious next ablation; it is an N-times-cost experiment,
        not a default.)

        ``use_rerank=False`` returns the RRF order unchanged and no scores, so
        the no-rerank ablation row differs from the headline run in exactly one
        respect: the ordering of the same candidate set, compared at the same k.
        """
        if not use_rerank or self._cross_encoder is None:
            if use_rerank and self._cross_encoder is None:
                log.warning("rerank_requested_but_no_cross_encoder_loaded")
            return list(nct_ids), {}
        if not nct_ids:
            return [], {}

        documents = [rerank_document(texts.get(nct_id)) for nct_id in nct_ids]
        scores = self._cross_encoder.score(query.text, documents)
        by_id = dict(zip(nct_ids, scores, strict=True))
        # Descending score; NCT ID breaks exact ties so the order is
        # reproducible across runs, as in rank_ids.
        ordered = sorted(by_id, key=lambda nct_id: (-by_id[nct_id], nct_id))
        return ordered, by_id


def _collect_payloads(per_query: Sequence[QueryHits]) -> dict[str, dict[str, object]]:
    """One payload per NCT ID, from whichever branch saw it first.

    The payloads are identical whichever branch returned the point - they come
    from the same stored point - so first-wins needs no tie-break.
    """
    payloads: dict[str, dict[str, object]] = {}
    for hits in per_query:
        for hit in (*hits.dense, *hits.sparse):
            payloads.setdefault(hit.nct_id, dict(hit.payload))
    return payloads


def _collect_branch_scores(
    per_query: Sequence[QueryHits],
) -> tuple[dict[str, float], dict[str, float]]:
    """Best raw dense and sparse score per NCT ID, across all queries.

    **Diagnostics only.** These are what ``TrialCandidate.dense_score`` and
    ``sparse_score`` carry, and they exist so a human debugging a surprising
    result can see "this came from the sparse side only". Nothing in this
    service ever adds, averages, normalises or compares them - the moment two of
    these numbers are combined arithmetically, the reason RRF exists has been
    forgotten.
    """
    dense: dict[str, float] = {}
    sparse: dict[str, float] = {}
    for hits in per_query:
        for hit in hits.dense:
            dense[hit.nct_id] = max(dense.get(hit.nct_id, hit.score), hit.score)
        for hit in hits.sparse:
            sparse[hit.nct_id] = max(sparse.get(hit.nct_id, hit.score), hit.score)
    return dense, sparse


def _build_candidate(
    *,
    nct_id: str,
    final_rank: int,
    payload: Mapping[str, object],
    text: TrialText | None,
    dense_score: float | None,
    sparse_score: float | None,
    rrf_score: float | None,
    rerank_score: float | None,
) -> TrialCandidate:
    """Assemble one row of the answer from the payload and the hydrated text.

    The typed age and sex bounds come from the Qdrant payload - the same values
    the filter was evaluated against - so what the caller sees is what the
    filter saw. They are copied onto the candidate because the rule checker
    needs them to re-check numerically against the patient's actual age: the
    filter proved the trial is *possibly* eligible (unknown bounds survive), and
    only the checker can say more.
    """
    return TrialCandidate(
        nct_id=nct_id,
        title="" if text is None else text.title,
        conditions=[] if text is None else list(text.conditions),
        interventions=[] if text is None else list(text.interventions),
        phase=_as_str(payload.get("phase")),
        status=_as_str(payload.get("status")),
        min_age_years=_as_float(payload.get("min_age_years")),
        max_age_years=_as_float(payload.get("max_age_years")),
        sex=_as_sex(payload.get("sex")),
        dense_score=dense_score,
        sparse_score=sparse_score,
        rrf_score=rrf_score,
        rerank_score=rerank_score,
        retrieval_rank=final_rank,
    )
