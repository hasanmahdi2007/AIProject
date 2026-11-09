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

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from beacon_core.settings import Settings, get_settings

__all__ = [
    "CrossEncoder",
    "DenseEncoder",
    "DenseQueryEncoder",
    "PairScorer",
    "RerankModel",
    "SparseEncoder",
    "SparseQueryEncoder",
    "get_cross_encoder",
    "get_dense_encoder",
    "get_sparse_encoder",
]


class DenseQueryEncoder(Protocol):
    """What :class:`~retrieval.search.HybridSearcher` needs from the embedder.

    The rest of the service depends on these three protocols rather than on the
    concrete classes below, for one concrete reason: the unit suite must pass
    with **no ONNX weights on disk**. CI's fast job would otherwise have to
    download 400 MB of models to check that RRF adds fractions correctly. A stub
    returning a hand-written vector satisfies this, and the real models are
    confined to the integration tests, which skip when the cache is cold.
    """

    def encode_query(self, text: str) -> list[float]: ...


class SparseQueryEncoder(Protocol):
    def encode_query(self, text: str) -> models.SparseVector: ...


class PairScorer(Protocol):
    """A cross-encoder: scores ``(query, document)`` pairs. Higher is better."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class RerankModel(Protocol):
    """The slice of FastEmbed's ``TextCrossEncoder`` that :class:`CrossEncoder`
    uses. A protocol so a test can inject a fake and check the *batching*
    without 280 MB of weights - the length-sorting below is pure index
    bookkeeping and a bug in it would silently return the right scores against
    the wrong trials."""

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = ...
    ) -> Iterable[float]: ...


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

    model: RerankModel
    batch_size: int = 8
    """8, measured, not guessed - and smaller is faster here, which is the
    opposite of the usual batching intuition. See :meth:`score`."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score every ``(query, document)`` pair. Higher is more relevant.

        Returned values are raw logits, not probabilities, and are comparable
        only within a single call - which is all that is needed, since they are
        used to sort one candidate set. They are deliberately *not* fed back
        into RRF: by this point ranks have done their job and the cross-encoder
        is the final authority on ordering.

        **Documents are batched in length order.** The transformer pads every
        sequence in a batch to the longest member, so a single long trial makes
        every other pair in its batch cost full length. Trial overviews are very
        uneven - mean 729 characters, p95 1,922, max 3,052 over 134,120 real
        rows - so in arrival order most of the compute is spent on padding.

        Measured here, 100 real overviews against one query on 8 CPU cores:

        ===========  ==============  ====================  =======
        batch_size   arrival order   length-sorted order   gain
        ===========  ==============  ====================  =======
        8            57.2 s          33.6 s                1.7x
        16           83.4 s          50.5 s                1.7x
        32           115.9 s         46.4 s                2.5x
        ===========  ==============  ====================  =======

        Two things fall out of that table. Sorting is worth 1.7-2.5x for about
        ten lines, and in arrival order a *bigger* batch is strictly worse
        (57 s -> 116 s), because a wider batch is more likely to contain an
        outlier that everything else gets padded up to. Hence
        ``batch_size = 8``: sorted-and-narrow was the fastest combination
        measured.

        This is purely a cost optimisation and changes no result. Each pair is
        scored independently, with attention masked over the padding, so batch
        composition cannot affect a score - confirmed rather than assumed: the
        benchmark compared the two orderings pair by pair and the maximum
        difference was exactly ``0.0``.
        """
        if not documents:
            return []
        # Indices of `documents`, shortest first. Sorting the indices rather
        # than the strings is what makes the result restorable: `order[j]` is
        # where the j-th scored document belongs in the caller's list.
        order = sorted(range(len(documents)), key=lambda index: len(documents[index]))
        scores = self.model.rerank(
            query, [documents[index] for index in order], batch_size=self.batch_size
        )
        restored = [0.0] * len(documents)
        for index, score in zip(order, scores, strict=True):
            restored[index] = float(score)
        return restored


# The caches are keyed on the model name - a string - and not on ``Settings``.
# ``Settings`` is a pydantic model with no ``__hash__``, so ``@lru_cache`` on a
# function taking one raises ``TypeError: unhashable type: 'Settings'`` the
# moment anybody passes a non-default settings object. (The same bug was found
# and fixed in ``beacon_core.db.get_engine`` the same day.) Keying on the model
# name is also the more correct cache key: the name is the only thing that
# decides which weights get loaded, and two callers with different ``Settings``
# but the same model should share one ONNX session rather than loading 280 MB
# twice.


@lru_cache(maxsize=4)
def _dense_encoder(model_name: str, dim: int) -> DenseEncoder:
    return DenseEncoder(model=TextEmbedding(model_name=model_name), dim=dim)


@lru_cache(maxsize=4)
def _sparse_encoder(model_name: str) -> SparseEncoder:
    return SparseEncoder(model=SparseTextEmbedding(model_name=model_name))


@lru_cache(maxsize=4)
def _cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model=TextCrossEncoder(model_name=model_name))


def get_dense_encoder(settings: Settings | None = None) -> DenseEncoder:
    cfg = (settings or get_settings()).qdrant
    return _dense_encoder(cfg.dense_model, cfg.dense_dim)


def get_sparse_encoder(settings: Settings | None = None) -> SparseEncoder:
    cfg = (settings or get_settings()).qdrant
    return _sparse_encoder(cfg.sparse_model)


def get_cross_encoder(settings: Settings | None = None) -> CrossEncoder:
    cfg = (settings or get_settings()).qdrant
    return _cross_encoder(cfg.reranker_model)
