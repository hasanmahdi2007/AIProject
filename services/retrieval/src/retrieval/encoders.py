"""The three local models, loaded lazily and shared process-wide.

Nothing in this module calls a paid API. ``bge-small-en-v1.5``,
``Qdrant/bm25`` and ``bge-reranker-base`` all run on CPU through FastEmbed's
ONNX runtime, at no cost and with no network access once cached. Query
*generation* is an LLM job and belongs to the worker's ``generate_queries``
node; this service receives queries already written, in
``SearchRequest.queries``.

Why these are lazy and cached rather than built in the module body:

* ``import retrieval.main`` must not download 400 MB of ONNX weights. The unit
  suite imports the app to check the request/response contract and runs with no
  models and no containers.
* Each model holds an ONNX session with its own thread pool. Building one per
  request would re-read the weights from disk every time and is the difference
  between a 40 ms rerank and a 4 s one.

**The bi-encoder / cross-encoder asymmetry is the shape of the whole funnel**,
so it is worth being precise about which of these does what:

:class:`DenseEncoder` and the sparse encoder are *bi-encoders*: they map a
query and a trial into the same space **separately**. That independence is the
only reason 375,580 trial vectors could be computed once by B1 at ingest time
and searched in milliseconds now - the trial side of the computation does not
depend on the query. The price is that the model never sees the pair together,
so it scores "are these two texts about similar things" and cannot reason over
the pair.

:class:`CrossEncoder` is the opposite trade. It concatenates query and trial
into one input and runs a full transformer forward pass over the pair with
cross-attention between them, which is far more accurate and **impossible to
precompute**: there is no trial-side vector to cache, because the representation
depends on the query. Measured on this machine it manages roughly 100 pairs per
second on CPU. 375,580 pairs would therefore be about an hour per search.

That asymmetry is the entire justification for the funnel being
375,580 -> 100 -> 20 and not some other shape: the cheap model is the only one
that can look at everything, the accurate model is the only one worth trusting
near the top, and ``fuse_top_k=100`` is the number of pairs the accurate model
can afford to read in about a second.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from beacon_core.settings import Settings, get_settings

__all__ = [
    "CrossEncoder",
    "DenseEncoder",
    "SparseEncoder",
    "get_cross_encoder",
    "get_dense_encoder",
    "get_sparse_encoder",
]


@dataclass(frozen=True, slots=True)
class DenseEncoder:
    """``bge-small-en-v1.5`` -> 384 floats.

    The model name comes from ``settings.qdrant.dense_model``, which is the
    same setting B1's ingest reads. That is not tidiness: query vectors and
    document vectors must come from the *same* model or the cosine between them
    is meaningless, and a mismatch raises nothing - it just returns bad results
    with confident scores.
    """

    model: TextEmbedding
    dim: int

    def encode_query(self, text: str) -> list[float]:
        """Embed one query.

        ``query_embed`` rather than ``embed`` to say which side of the
        asymmetric objective this text is. Checked rather than assumed, because
        it matters: for this model in FastEmbed 0.9
        ``TextEmbeddingBase.query_embed`` is a pass-through to ``embed`` and
        adds **no** instruction prefix, so it produces byte-identical vectors to
        the ``embed`` call B1's ingest uses for the trial overviews. The two
        halves of the space therefore line up today, and the naming keeps them
        lined up if a future model in config does add a prefix. (A mismatch here
        raises nothing - it returns plausible-looking vectors whose cosines are
        meaningless.)
        """
        vector = [float(value) for value in next(iter(self.model.query_embed(text)))]
        if len(vector) != self.dim:
            # A silent dimension change (someone swaps the model in config and
            # forgets dense_dim) would make Qdrant reject or, worse, a
            # differently-sized collection accept nonsense. Fail here.
            raise ValueError(
                f"dense model returned {len(vector)} dims, collection expects {self.dim}"
            )
        return vector


@dataclass(frozen=True, slots=True)
class SparseEncoder:
    """``Qdrant/bm25`` -> a sparse vector of token hashes to weights.

    Only the term *frequency* half of BM25 is computed here. The inverse
    document frequency half is corpus-level knowledge that a single query does
    not have, and Qdrant applies it server-side - but **only** if the sparse
    field was declared with ``modifier: IDF``. See
    :func:`retrieval.collection.verify_collection`, which asserts that, because
    omitting it raises no error and merely degrades rare-term matching, which is
    the entire reason BM25 is in this project.
    """

    model: SparseTextEmbedding

    def encode_query(self, text: str) -> models.SparseVector:
        embedding = next(iter(self.model.query_embed(text)))
        return models.SparseVector(
            indices=[int(i) for i in embedding.indices],
            values=[float(v) for v in embedding.values],
        )


@dataclass(frozen=True, slots=True)
class CrossEncoder:
    """``bge-reranker-base``: reads query and trial together, one pass per pair.

    See the module docstring for why this cannot be precomputed and why that
    dictates the funnel's shape.
    """

    model: TextCrossEncoder
    batch_size: int = 16

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score every ``(query, document)`` pair. Higher is more relevant.

        Returned values are raw logits, not probabilities, and are comparable
        only within a single call - which is all that is needed, since they are
        used to sort one candidate set. They are deliberately *not* fed back
        into RRF: by this point ranks have done their job and the cross-encoder
        is the final authority on ordering.
        """
        if not documents:
            return []
        return [
            float(score)
            for score in self.model.rerank(query, documents, batch_size=self.batch_size)
        ]


@lru_cache(maxsize=1)
def get_dense_encoder(settings: Settings | None = None) -> DenseEncoder:
    cfg = settings or get_settings()
    return DenseEncoder(
        model=TextEmbedding(model_name=cfg.qdrant.dense_model),
        dim=cfg.qdrant.dense_dim,
    )


@lru_cache(maxsize=1)
def get_sparse_encoder(settings: Settings | None = None) -> SparseEncoder:
    cfg = settings or get_settings()
    return SparseEncoder(model=SparseTextEmbedding(model_name=cfg.qdrant.sparse_model))


@lru_cache(maxsize=1)
def get_cross_encoder(settings: Settings | None = None) -> CrossEncoder:
    cfg = settings or get_settings()
    return CrossEncoder(model=TextCrossEncoder(model_name=cfg.qdrant.reranker_model))
