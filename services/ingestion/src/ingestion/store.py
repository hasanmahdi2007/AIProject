"""The dual write: Postgres first, then Qdrant, idempotent by ``nct_id``.

The ordering, and why there is a commit marker
----------------------------------------------
Postgres is the source of truth and Qdrant is a derived index. Two stores
cannot be written atomically, so the question is not *whether* they can diverge
but which direction of divergence is recoverable. Postgres first:

* **Postgres written, Qdrant not** -- the trial exists but is unsearchable.
  Detectable (``indexed_at IS NULL``) and fixable by re-running.
* **Qdrant written, Postgres not** -- a point whose ``nct_id`` resolves to
  nothing. Retrieval returns a candidate that cannot be rendered, and nothing
  in the data says it is wrong.

``trials.indexed_at`` is set in a *third* step, after the Qdrant upsert
acknowledges. So its meaning is precise: **both stores hold this trial.** A row
with ``indexed_at IS NULL`` is an interrupted trial, and re-running picks
exactly those up again. That is the whole resume mechanism -- no offsets, no
checkpoint file, nothing that can disagree with the data.

Idempotency
-----------
* The ``trials`` row is an ``INSERT ... ON CONFLICT (nct_id) DO UPDATE``, so a
  re-run rewrites the same values.
* Each Qdrant point id is ``uuid5(BEACON_NAMESPACE, nct_id)`` -- derived, not
  random -- so a re-run overwrites the same point instead of adding a second
  copy of the trial. This is the single most important line in the module: a
  random id here would double the collection on every run and no error would
  be raised.
* Criteria are upserted on ``(nct_id, criterion_id)`` and any criterion that
  the splitter no longer produces is deleted, so the rule set converges rather
  than accumulating.

Hence: re-running the full ingest changes nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from beacon_core.models import Trial, TrialCriterion
from beacon_core.settings import QdrantSettings
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ingestion.criteria import Criterion
from ingestion.embedding import SparseValues

__all__ = [
    "AGE_PAYLOAD_FIELDS",
    "BEACON_NAMESPACE",
    "PAYLOAD_INDEXES",
    "QdrantIndex",
    "SparseModifierMismatch",
    "TrialPoint",
    "count_criteria",
    "count_trials",
    "indexed_nct_ids",
    "mark_indexed",
    "point_id",
    "upsert_criteria",
    "upsert_trials",
]

BEACON_NAMESPACE = uuid.UUID("5f3e1b7a-0c2d-4e8f-9a6b-1d2c3e4f5a6b")
"""Fixed UUID namespace for deriving Qdrant point ids from NCT IDs.

A constant literal, not ``uuid.uuid4()`` at import time: the whole point is
that the same ``nct_id`` maps to the same point id on every machine and in
every process, for ever. Changing this value orphans the entire collection.
"""

AGE_PAYLOAD_FIELDS = ("min_age_years", "max_age_years")

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "nct_id": models.PayloadSchemaType.KEYWORD,
    "min_age_years": models.PayloadSchemaType.FLOAT,
    "max_age_years": models.PayloadSchemaType.FLOAT,
    "sex": models.PayloadSchemaType.KEYWORD,
}
"""Payload fields that get an index.

B2 must filter *inside* the index traversal: an unindexed filter field makes
Qdrant fall back to a full scan of the payload for every query, which on
375,580 points turns a 20 ms search into seconds. The age fields and ``sex``
are the filters the brief specifies; ``nct_id`` is indexed because looking a
trial up by its identifier is how every hand-check and every debugging session
starts. ``phase`` and ``status`` are left unindexed on purpose -- they are
returned, not filtered on, in v1, and each index costs build time and memory.
"""


class SparseModifierMismatch(RuntimeError):
    """Raised when an existing collection's sparse field lacks ``modifier: IDF``.

    BM25 without corpus IDF is not an error state Qdrant reports; it simply
    stops weighting by document frequency, so ``EGFR`` scores like ``patient``
    and rare-term matching -- the entire reason BM25 is in this system --
    degrades in silence. Refusing to write into such a collection is the only
    way to make the failure loud.
    """


@dataclass(frozen=True, slots=True)
class TrialPoint:
    """One Qdrant point: two named vectors and a filterable payload."""

    nct_id: str
    dense: list[float]
    sparse: SparseValues
    payload: dict[str, Any]


def point_id(nct_id: str) -> str:
    """Deterministic point id for a trial. See :data:`BEACON_NAMESPACE`."""
    return str(uuid.uuid5(BEACON_NAMESPACE, nct_id))


# ----------------------------------------------------------------- Postgres --


async def upsert_trials(session: AsyncSession, rows: Sequence[dict[str, Any]]) -> int:
    """Insert or update ``trials`` rows by ``nct_id``; returns rows written.

    ``indexed_at`` is not touched here. On an update it is explicitly reset to
    ``NULL``, because the row's content has just changed and the Qdrant point
    has not been rewritten yet -- leaving a stale ``indexed_at`` in place would
    claim agreement between the stores that does not hold.
    """
    if not rows:
        return 0
    statement = pg_insert(Trial).values(list(rows))
    updatable = {
        column.name: statement.excluded[column.name]
        for column in Trial.__table__.columns
        if column.name not in {"nct_id", "created_at", "indexed_at"}
    }
    updatable["indexed_at"] = None
    await session.execute(
        statement.on_conflict_do_update(index_elements=[Trial.nct_id], set_=updatable)
    )
    return len(rows)


async def upsert_criteria(session: AsyncSession, criteria: Sequence[Criterion]) -> int:
    """Replace the criteria of every trial mentioned in ``criteria``.

    Upserts on ``(nct_id, criterion_id)`` so existing row ids stay stable, then
    deletes any criterion of those trials that the splitter no longer produces.
    Both halves are needed: without the delete, improving the splitter would
    leave the old rules behind and every trial would accumulate duplicates.
    """
    if not criteria:
        return 0
    rows = [
        {
            "nct_id": c.nct_id,
            "criterion_id": c.criterion_id,
            "kind": c.kind,
            "position": c.position,
            "text": c.text,
            "parser": c.parser,
        }
        for c in criteria
    ]
    statement = pg_insert(TrialCriterion).values(rows)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[TrialCriterion.nct_id, TrialCriterion.criterion_id],
            set_={
                "kind": statement.excluded.kind,
                "position": statement.excluded.position,
                "text": statement.excluded.text,
                "parser": statement.excluded.parser,
            },
        )
    )
    keep: dict[str, set[str]] = {}
    for c in criteria:
        keep.setdefault(c.nct_id, set()).add(c.criterion_id)
    for nct_id, ids in keep.items():
        await session.execute(
            delete(TrialCriterion).where(
                TrialCriterion.nct_id == nct_id,
                TrialCriterion.criterion_id.notin_(ids),
            )
        )
    return len(rows)


async def clear_criteria(session: AsyncSession, nct_ids: Sequence[str]) -> None:
    """Drop every criterion of these trials (used when a trial yields none)."""
    if not nct_ids:
        return
    await session.execute(delete(TrialCriterion).where(TrialCriterion.nct_id.in_(nct_ids)))


async def mark_indexed(
    session: AsyncSession, nct_ids: Sequence[str], *, when: datetime | None = None
) -> None:
    """Record that both stores now hold these trials.

    Called only after Qdrant has acknowledged the upsert; see the module
    docstring. ``indexed_at`` is the resume marker, so setting it early would
    make an interrupted run look complete.
    """
    if not nct_ids:
        return
    stamp = when or datetime.now(UTC)
    await session.execute(
        Trial.__table__.update()
        .where(Trial.nct_id.in_(nct_ids))
        .values(indexed_at=stamp)
    )


async def indexed_nct_ids(session: AsyncSession) -> set[str]:
    """Every ``nct_id`` already present in *both* stores.

    375,580 identifiers are about 4 MB in memory, which is cheaper than a
    per-batch round trip and makes the skip decision a set lookup.
    """
    result = await session.execute(select(Trial.nct_id).where(Trial.indexed_at.isnot(None)))
    return set(result.scalars().all())


async def count_trials(session: AsyncSession, *, indexed_only: bool = False) -> int:
    statement = select(func.count()).select_from(Trial)
    if indexed_only:
        statement = statement.where(Trial.indexed_at.isnot(None))
    return int((await session.execute(statement)).scalar() or 0)


async def count_criteria(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count()).select_from(TrialCriterion))).scalar() or 0)


# ------------------------------------------------------------------- Qdrant --


class QdrantIndex:
    """The ``trials`` collection: two named vectors, one point per trial."""

    def __init__(self, settings: QdrantSettings, client: AsyncQdrantClient | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncQdrantClient(url=settings.url)

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._settings.collection

    async def close(self) -> None:
        await self._client.close()

    def _sparse_params(self) -> models.SparseVectorParams:
        modifier = (
            models.Modifier.IDF if self._settings.sparse_idf_modifier else models.Modifier.NONE
        )
        return models.SparseVectorParams(modifier=modifier)

    async def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection if absent, and verify it if present.

        The verification is not decoration. ``modifier: IDF`` can only be set
        when the sparse field is declared; a collection created without it
        accepts every upsert and every query and merely returns worse results.
        So an existing collection whose modifier disagrees with
        ``settings.qdrant.sparse_idf_modifier`` raises
        :class:`SparseModifierMismatch` rather than being written into.
        """
        exists = await self._client.collection_exists(self.collection)
        if exists and recreate:
            await self._client.delete_collection(self.collection)
            exists = False
        if not exists:
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self._settings.dense_vector_name: models.VectorParams(
                        size=self._settings.dense_dim,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={self._settings.sparse_vector_name: self._sparse_params()},
            )
        else:
            await self._verify_sparse_modifier()
        await self._ensure_payload_indexes()

    async def _verify_sparse_modifier(self) -> None:
        info = await self._client.get_collection(self.collection)
        sparse = (info.config.params.sparse_vectors or {}).get(
            self._settings.sparse_vector_name
        )
        if sparse is None:
            raise SparseModifierMismatch(
                f"collection {self.collection!r} has no sparse vector field "
                f"{self._settings.sparse_vector_name!r}; recreate it with --recreate-collection"
            )
        expected = self._sparse_params().modifier
        actual = sparse.modifier or models.Modifier.NONE
        if actual != expected:
            raise SparseModifierMismatch(
                f"collection {self.collection!r} declares sparse modifier {actual!r} but "
                f"settings.qdrant.sparse_idf_modifier implies {expected!r}. BM25 without IDF "
                "degrades silently; recreate the collection with --recreate-collection."
            )

    async def _ensure_payload_indexes(self) -> None:
        info = await self._client.get_collection(self.collection)
        existing = set(info.payload_schema or {})
        for field_name, schema in PAYLOAD_INDEXES.items():
            if field_name in existing:
                continue
            await self._client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

    async def upsert(self, points: Iterable[TrialPoint], *, wait: bool = True) -> int:
        """Upsert trial points. Returns the number sent.

        ``wait=True`` is the default because the caller sets ``indexed_at``
        straight afterwards, and that flag must not be able to run ahead of the
        write it attests to.
        """
        batch = [
            models.PointStruct(
                id=point_id(p.nct_id),
                vector={
                    self._settings.dense_vector_name: p.dense,
                    self._settings.sparse_vector_name: models.SparseVector(
                        indices=p.sparse.indices, values=p.sparse.values
                    ),
                },
                payload=p.payload,
            )
            for p in points
        ]
        if not batch:
            return 0
        await self._client.upsert(collection_name=self.collection, points=batch, wait=wait)
        return len(batch)

    async def count(self) -> int:
        result = await self._client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    async def info(self) -> models.CollectionInfo:
        return await self._client.get_collection(self.collection)

    async def retrieve(self, nct_ids: Sequence[str]) -> list[models.Record]:
        """Fetch points by NCT ID, payload and both vectors included.

        Used by the hand-check report, which compares each payload against the
        Postgres row it was derived from.
        """
        return await self._client.retrieve(
            collection_name=self.collection,
            ids=[point_id(nct_id) for nct_id in nct_ids],
            with_payload=True,
            with_vectors=True,
        )
