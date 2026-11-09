"""End-to-end ingest against the running compose stack.

Every test here is marked ``integration`` and so is excluded from the fast
unit run and from CI. They need the TREC archives cached under
``IR_DATASETS_HOME`` and the stack up on the shifted ports (Postgres 15432,
Qdrant 6333) that ``beacon_core.settings`` already defaults to.

Run them with::

    uv run pytest services/ingestion -m integration

They write real corpus rows into the real ``trials`` table and the real
collection, idempotently, so they are safe to repeat -- but do not run them
while a full ingest is in flight: both would be writing the same rows and the
counts these assert on would move underneath them.
"""

from __future__ import annotations

import pytest
from qdrant_client import models
from sqlalchemy import select

from beacon_core.db import session_scope
from beacon_core.models import Trial
from beacon_core.settings import get_settings
from ingestion import store
from ingestion.run import RunConfig, ingest
from ingestion.store import QdrantIndex, point_id

pytestmark = pytest.mark.integration

SAMPLE = 24
"""Small enough to embed in a few seconds on a CPU, large enough to span
several real records with and without age bounds."""


async def _counts() -> tuple[int, int, int]:
    index = QdrantIndex(get_settings().qdrant)
    try:
        points = await index.count()
    finally:
        await index.close()
    async with session_scope() as session:
        rows = await store.count_trials(session)
        indexed = await store.count_trials(session, indexed_only=True)
    return points, rows, indexed


async def test_ingest_is_idempotent_by_nct_id() -> None:
    """Running the same slice twice must change nothing.

    This is the property the whole dual write is built around: Qdrant point
    ids are ``uuid5(namespace, nct_id)`` rather than random, and the
    ``trials`` write is an upsert on ``nct_id``. If either were otherwise, the
    second pass would double the collection and no error would be raised.

    The second pass is run with ``force=True`` on purpose. Without it the
    resume marker would skip every record and the test would prove only that
    skipping works; with it, every row and point is genuinely rewritten, and
    the counts still have to be identical.
    """
    config = RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=SAMPLE)
    await ingest(config)
    before = await _counts()

    await ingest(RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=SAMPLE, force=True))
    after = await _counts()

    assert before == after, f"re-ingest changed the stores: {before} -> {after}"
    points, rows, indexed = after
    assert rows >= SAMPLE
    assert indexed == rows, "every row must be marked as present in both stores"
    assert points == rows


async def test_every_point_carries_both_named_vectors() -> None:
    await ingest(RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=0))
    cfg = get_settings()
    async with session_scope() as session:
        nct_ids = list(
            (
                await session.execute(select(Trial.nct_id).order_by(Trial.nct_id).limit(SAMPLE))
            ).scalars()
        )
    index = QdrantIndex(cfg.qdrant)
    try:
        records = await index.retrieve(nct_ids)
    finally:
        await index.close()

    assert len(records) == len(nct_ids)
    for record in records:
        vectors = record.vector
        assert isinstance(vectors, dict)
        assert set(vectors) == {cfg.qdrant.dense_vector_name, cfg.qdrant.sparse_vector_name}
        dense = vectors[cfg.qdrant.dense_vector_name]
        sparse = vectors[cfg.qdrant.sparse_vector_name]
        assert isinstance(dense, list)
        assert len(dense) == cfg.qdrant.dense_dim
        assert isinstance(sparse, models.SparseVector)
        assert len(sparse.indices) == len(sparse.values) > 0


async def test_payload_agrees_with_the_postgres_row() -> None:
    """The hand-check, as an assertion rather than a printout."""
    await ingest(RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=0))
    cfg = get_settings()
    async with session_scope() as session:
        rows = list(
            (await session.execute(select(Trial).order_by(Trial.nct_id).limit(SAMPLE))).scalars()
        )
    index = QdrantIndex(cfg.qdrant)
    try:
        records = {r.id: r for r in await index.retrieve([row.nct_id for row in rows])}
    finally:
        await index.close()

    for row in rows:
        record = records[point_id(row.nct_id)]
        payload = record.payload or {}
        assert payload["nct_id"] == row.nct_id
        assert payload["min_age_years"] == row.min_age_years
        assert payload["max_age_years"] == row.max_age_years
        assert payload["sex"] == row.sex
        assert payload["phase"] == row.phase
        assert payload["status"] == row.status
        # Eligibility text is Postgres-only; the retrieval unit is a trial.
        assert "eligibility_text" not in payload
        assert "overview_text" not in payload


async def test_null_ages_survive_the_round_trip_as_null() -> None:
    """Real trials with no stated age must read back as NULL, not 0 or 999.

    ``NCT00000104`` is one: both bounds are ``N/A`` in the source XML.
    """
    await ingest(RunConfig(limit=200, batch_size=32, criteria_limit=0))
    cfg = get_settings()
    async with session_scope() as session:
        row = (
            await session.execute(select(Trial).where(Trial.nct_id == "NCT00000104"))
        ).scalar_one()
    assert row.raw_min_age == "N/A"
    assert row.raw_max_age == "N/A"
    assert row.min_age_years is None
    assert row.max_age_years is None

    index = QdrantIndex(cfg.qdrant)
    try:
        record = (await index.retrieve(["NCT00000104"]))[0]
    finally:
        await index.close()
    payload = record.payload or {}
    assert payload["min_age_years"] is None
    assert payload["max_age_years"] is None


async def test_the_live_sparse_field_declares_idf() -> None:
    """Asserted against a real Qdrant, because the stub cannot prove the server
    accepted the modifier -- only that the client sent it."""
    cfg = get_settings()
    index = QdrantIndex(cfg.qdrant)
    try:
        await index.ensure_collection()
        info = await index.info()
    finally:
        await index.close()
    sparse = (info.config.params.sparse_vectors or {})[cfg.qdrant.sparse_vector_name]
    assert sparse.modifier == models.Modifier.IDF
    assert {"min_age_years", "max_age_years", "sex"} <= set(info.payload_schema or {})
