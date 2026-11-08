"""The two encoders that produce the two named vectors on each Qdrant point.

Dense and sparse answer different questions and neither replaces the other:

* ``BAAI/bge-small-en-v1.5`` (384-d, cosine) puts *"advanced lung tumour"* near
  *"metastatic NSCLC"*, which is what a free-text patient note needs.
* ``Qdrant/bm25`` matches the literal token. ``EGFR``, ``HbA1c``, ``NCT0123``
  and ``pembrolizumab`` are precisely the terms a 384-d general-purpose encoder
  smears together, and they are usually the terms that decide a trial.

Both models are loaded once per process. FastEmbed downloads them to its own
cache on first use (about 130 MB for bge-small) and then runs them under
onnxruntime on the CPU; model construction takes a couple of minutes the first
time and under two seconds afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from beacon_core.settings import QdrantSettings

__all__ = ["Encoder", "SparseValues"]


@dataclass(frozen=True, slots=True)
class SparseValues:
    """One BM25 vector: parallel term-id and weight arrays.

    Deliberately not a ``qdrant_client`` model, so the encoder stays testable
    and the Qdrant types live only in :mod:`ingestion.store`.
    """

    indices: list[int]
    values: list[float]


class Encoder:
    """Dense + sparse encoding of trial overviews, batched.

    Instantiating this constructs both FastEmbed models, so build one per
    process and reuse it. :meth:`encode` preserves input order, which the dual
    write relies on to pair a vector with the right ``nct_id``.
    """

    def __init__(self, settings: QdrantSettings, *, threads: int | None = None) -> None:
        # Imported lazily: loading FastEmbed pulls in onnxruntime, which costs
        # a second and some hundreds of MB of RSS. The parser and splitter
        # tests import this package and must not pay for it.
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._settings = settings
        self.dense = TextEmbedding(settings.dense_model, threads=threads)
        self.sparse = SparseTextEmbedding(settings.sparse_model, threads=threads)

    @property
    def dense_dim(self) -> int:
        return self._settings.dense_dim

    def encode(
        self, texts: Sequence[str], *, batch_size: int = 64
    ) -> tuple[list[list[float]], list[SparseValues]]:
        """Encode a batch, returning dense vectors and sparse vectors in order.

        Raises:
            ValueError: if the dense model returns a width other than the
                configured ``dense_dim``. A silently different width would be
                rejected by Qdrant later with a far less obvious error, and a
                *matching* width from the wrong model would not be rejected at
                all.
        """
        if not texts:
            return [], []
        dense = [vector.tolist() for vector in self.dense.embed(texts, batch_size=batch_size)]
        for vector in dense:
            if len(vector) != self._settings.dense_dim:
                raise ValueError(
                    f"{self._settings.dense_model} produced {len(vector)} dims, "
                    f"expected {self._settings.dense_dim}"
                )
        sparse = [
            SparseValues(indices=[int(i) for i in sv.indices], values=[float(v) for v in sv.values])
            for sv in self.sparse.embed(texts, batch_size=batch_size)
        ]
        return dense, sparse
