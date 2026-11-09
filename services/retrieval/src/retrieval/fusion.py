"""Reciprocal Rank Fusion. Pure, synchronous, no I/O - so it is unit-testable.

**RRF is rank-based, never score-based.** ``score(doc) = sum_i w_i / (k + r_i)``
where ``r_i`` is the doc's 1-based rank in ranking ``i`` and docs absent from a
ranking simply contribute nothing.

The whole reason this function exists is that the two retrievers produce
numbers that are not the same kind of quantity. A cosine similarity of 0.82 and
a BM25 score of 14.3 do not live on a common scale: BM25 is unbounded above and
grows with document frequency and length normalisation, cosine is bounded in
[-1, 1] and its spread depends on the embedding model's anisotropy. Min-max
normalising each list and adding them looks reasonable and is not - the
normalisation constants come from whatever happened to be in *this* result list,
so the same document gets a different "normalised score" depending on which
other documents came back with it. Ranks discard the magnitudes on purpose, and
that is the feature.

Fusion happens at two levels here:

1. **Within one query**: the dense ranking and the sparse ranking for that
   query are fused with equal weight. Neither retriever is a priori better -
   dense wins on paraphrase, sparse wins on rare exact tokens (an NCT ID, a
   gene name, "pembrolizumab") - and the point of running both is to not have
   to choose.
2. **Across the N queries**: each query's fused ranking is fused again, with
   query of ``rank`` r weighted ``1/r``.

Why ``1/r`` for the cross-query weights (:func:`query_weight`): the generator
emits queries already ordered by importance, rank 1 being the patient's primary
diagnosis facet and later ranks secondary facets (biomarker, prior therapy).
Uniform weights throw that ordering away; an exponential decay (0.5^r) makes
query 4 onward contribute almost nothing, which at the N=10 end of the Day 3
sweep would make the extra queries dead weight and the sweep uninformative.
``1/r`` is the same functional form as RRF itself - one idea in the system
rather than two - it is scale-free, and it decays slowly enough that a trial
found *only* by the rank-3 query still enters the fused top 100 (it needs to:
that is exactly the biomarker-only trial multi-query retrieval exists to find)
while a trial found by the rank-1 query outranks it on ties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "fuse_across_queries",
    "fuse_hybrid",
    "query_weight",
    "rank_ids",
    "rrf_scores",
]


def rrf_scores(
    rankings: Sequence[Sequence[str]],
    *,
    k: int,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fuse ranked ID lists into ``{doc_id: rrf_score}``.

    Args:
        rankings: one ranked list of document IDs per retriever, best first. A
            document missing from a list contributes nothing for that list,
            which is the correct behaviour and not a special case: it is what
            makes RRF tolerate retrievers that return different depths.
        k: the RRF constant. Larger ``k`` flattens the weight curve, so the
            difference between rank 1 and rank 10 matters less and agreement
            across retrievers matters more. ``k=60`` is the value from Cormack
            et al. 2009 and the default in ``settings.retrieval.rrf_k``; it is
            read from config and never written as a literal here, because it is
            an ablation row.
        weights: per-ranking multipliers, defaulting to 1.0 each. Used for the
            cross-query level; the dense/sparse level leaves it unset.

    Returns:
        Every ID seen in any ranking, mapped to its summed score. Unsorted -
        :func:`rank_ids` does the ordering, so callers that only need a lookup
        do not pay for a sort.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(f"weights has length {len(weights)}, expected {len(rankings)}")
    if k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {k}")

    scores: dict[str, float] = {}
    for i, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[i]
        seen: set[str] = set()
        for position, doc_id in enumerate(ranking):
            # A retriever that returns the same ID twice (Qdrant will not, but a
            # fake or a future retriever might) must not be able to inflate a
            # document by double-counting it. Keep the best rank only.
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank = position + 1  # RRF ranks are 1-based: the top hit is 1/(k+1).
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


def rank_ids(scores: Mapping[str, float]) -> list[str]:
    """Order IDs by score, descending, breaking ties by ID.

    The tie-break is deterministic on purpose. Floating-point sums of
    ``1/(k+r)`` collide exactly when two documents occupy mirrored ranks, and an
    arbitrary order there would make the same request return different results
    on different runs, which would make an eval score impossible to reproduce.
    """
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def fuse_hybrid(
    dense_ranking: Sequence[str],
    sparse_ranking: Sequence[str],
    *,
    k: int,
) -> dict[str, float]:
    """Level 1: fuse one query's dense and sparse rankings, equally weighted."""
    return rrf_scores([dense_ranking, sparse_ranking], k=k)


def query_weight(query_rank: int) -> float:
    """Cross-query weight for a query of the given ``RankedQuery.rank``.

    ``1/rank``: rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.333. See the module
    docstring for why this shape and not uniform or exponential.
    """
    if query_rank < 1:
        raise ValueError(f"RankedQuery.rank is 1-based, got {query_rank}")
    return 1.0 / query_rank


def fuse_across_queries(
    per_query: Sequence[tuple[int, Sequence[str]]],
    *,
    k: int,
) -> dict[str, float]:
    """Level 2: fuse each query's already-fused ranking, weighted by query rank.

    Args:
        per_query: ``(query_rank, ranked_ids)`` pairs, one per generated query.
            ``query_rank`` is ``RankedQuery.rank``, not the index in the list -
            the caller may have filtered queries out, and the weight must track
            the declared importance rather than list position.
        k: the RRF constant, the same one used at level 1.

    Note this fuses *ranks of the fused rankings*, not the level-1 RRF scores.
    Summing the level-1 scores would reintroduce the problem RRF exists to
    avoid: those sums are not comparable across queries either, because a query
    whose dense and sparse lists agree produces systematically larger numbers
    than one whose lists disagree, which would silently weight queries by how
    *easy* they were rather than how *important* they are.
    """
    rankings = [ranked for _, ranked in per_query]
    weights = [query_weight(query_rank) for query_rank, _ in per_query]
    return rrf_scores(rankings, k=k, weights=weights)
