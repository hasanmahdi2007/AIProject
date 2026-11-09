"""Assertions about the live Qdrant collection, checked rather than assumed.

Everything verified here is something B1's ingest is responsible for setting and
this service is dependent on - and every one of them, if wrong, degrades results
without raising. A service that silently returns worse answers is harder to
debug than one that refuses to start, so these are checked once and reported on
``GET /health``.

The four checks:

1. **The collection exists.** Obvious, but worth a distinct message: the
   alternative is a 404 from deep inside the client on the first search.
2. **The dense vector field exists under the configured name and has the
   configured dimension.** A 384-dim query against a 768-dim field is rejected
   by Qdrant, which is the benign case; the malignant case is a collection
   rebuilt with a different model at the same dimension, which is why the
   dimension check is necessary but not sufficient and the model name is logged.
3. **The sparse vector field is declared with ``modifier: IDF``.** This is the
   one that fails most quietly. BM25 needs corpus-level inverse document
   frequency to know that "carcinoma" carries almost no information in a corpus
   of oncology trials while "ERBB2" carries a great deal. Qdrant computes IDF
   server-side, and **only** when the field declares the modifier - the sparse
   encoder on the query side cannot supply it, because a single query has no
   view of the corpus. Omit the modifier and every search still succeeds, still
   returns 20 trials, and rare-term matching is gone - which is the entire
   reason BM25 is in this project alongside the dense index.
4. **Payload indexes exist on the three filter fields.** Without them Qdrant
   can still answer a filtered query, by scanning the payload of every
   candidate point, so again nothing raises - it just gets slow, and the
   "filters inside the traversal" design collapses into a full scan with extra
   steps. This one is reported as a warning rather than a hard failure because
   the results stay *correct*; it is a latency defect, not a correctness one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient, models

from beacon_core.settings import QdrantSettings
from retrieval.filters import MAX_AGE_FIELD, MIN_AGE_FIELD, SEX_FIELD

__all__ = ["CollectionReport", "CollectionVerificationError", "verify_collection"]

FILTERED_PAYLOAD_FIELDS = (MIN_AGE_FIELD, MAX_AGE_FIELD, SEX_FIELD)


class CollectionVerificationError(RuntimeError):
    """The collection cannot serve correct hybrid search. Raised loudly."""


@dataclass(frozen=True, slots=True)
class CollectionReport:
    """What the collection actually declares, for ``/health`` and for logs."""

    name: str
    points: int
    dense_dim: int | None
    sparse_modifier: str | None
    indexed_payload_fields: tuple[str, ...]
    problems: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.problems


async def verify_collection(
    client: AsyncQdrantClient,
    settings: QdrantSettings,
) -> CollectionReport:
    """Inspect the collection and report every discrepancy found.

    Every problem is collected rather than raised at the first one: told "the
    sparse modifier is missing" you fix that, restart, and are then told the
    payload indexes are missing too. One report, one rebuild.

    Raises:
        CollectionVerificationError: only if the collection does not exist at
            all, since nothing further can be inspected.
    """
    if not await client.collection_exists(settings.collection):
        raise CollectionVerificationError(
            f"Qdrant collection {settings.collection!r} does not exist at {settings.url}. "
            "Run the B1 ingest (`uv run python -m ingestion.run`) before searching."
        )

    info = await client.get_collection(settings.collection)
    params = info.config.params
    problems: list[str] = []
    warnings: list[str] = []

    # --- dense vector field ---
    dense_dim: int | None = None
    vectors = params.vectors
    if isinstance(vectors, dict):
        dense_params = vectors.get(settings.dense_vector_name)
        if dense_params is None:
            problems.append(
                f"no dense vector field named {settings.dense_vector_name!r}; "
                f"found {sorted(vectors)}"
            )
        else:
            dense_dim = dense_params.size
            if dense_dim != settings.dense_dim:
                problems.append(
                    f"dense field {settings.dense_vector_name!r} has size {dense_dim}, "
                    f"but settings.qdrant.dense_dim is {settings.dense_dim} "
                    f"({settings.dense_model})"
                )
    else:
        # An unnamed single-vector collection. The hybrid design needs two named
        # vectors on one point so that a dense hit and a sparse hit refer to the
        # same trial and can be fused by ID.
        problems.append(
            "collection uses a single unnamed vector; hybrid search needs named "
            f"vectors {settings.dense_vector_name!r} and {settings.sparse_vector_name!r}"
        )

    # --- sparse vector field and its IDF modifier ---
    sparse_config = params.sparse_vectors or {}
    sparse_modifier: str | None = None
    sparse_params = sparse_config.get(settings.sparse_vector_name)
    if sparse_params is None:
        problems.append(
            f"no sparse vector field named {settings.sparse_vector_name!r}; "
            f"found {sorted(sparse_config)}"
        )
    else:
        modifier = sparse_params.modifier
        sparse_modifier = None if modifier is None else str(modifier)
        wants_idf = settings.sparse_idf_modifier
        has_idf = modifier == models.Modifier.IDF
        if wants_idf and not has_idf:
            problems.append(
                f"sparse field {settings.sparse_vector_name!r} declares modifier "
                f"{sparse_modifier!r}, not 'idf'. BM25 without corpus IDF loses "
                "document-frequency weighting and degrades rare-term matching "
                "silently - recreate the collection with modifier=IDF."
            )
        elif has_idf and not wants_idf:
            warnings.append(
                "collection declares modifier 'idf' but settings.qdrant."
                "sparse_idf_modifier is False; the setting is being ignored"
            )

    # --- payload indexes on the filter fields ---
    indexed = tuple(sorted(info.payload_schema or {}))
    missing = [name for name in FILTERED_PAYLOAD_FIELDS if name not in indexed]
    if missing:
        warnings.append(
            f"no payload index on {missing}: filtered search still returns correct "
            "results but falls back to scanning payloads instead of constraining "
            "the index traversal, so latency degrades with corpus size"
        )

    return CollectionReport(
        name=settings.collection,
        points=info.points_count or 0,
        dense_dim=dense_dim,
        sparse_modifier=sparse_modifier,
        indexed_payload_fields=indexed,
        problems=tuple(problems),
        warnings=tuple(warnings),
    )
