"""End-to-end ingest against a real Postgres and a real Qdrant.

Marked ``integration``, which in this repo means **CI does run them**:
``.github/workflows/ci.yml`` has a second job that brings up the compose stack
and runs ``pytest -m integration``. An earlier version of this file claimed the
opposite, and that false belief is what produced the bug -- the tests raised
``CorpusNotCachedError`` on a runner with no TREC corpus and failed the PR. A
test that cannot run in this environment must **skip**, never fail, so every
precondition here is a fixture that calls :func:`pytest.skip` with a hint, the
pattern ``services/gateway/tests/test_integration_api.py`` established.

Three preconditions, each its own fixture so the diagnosis is specific:

* the TREC archives cached under ``IR_DATASETS_HOME`` (:func:`corpus`);
* Postgres reachable with the schema applied (:func:`live_sessions`);
* Qdrant reachable (:func:`live_index`).

Locally::

    uv run pytest services/ingestion -m integration

These write real corpus rows into the real ``trials`` table and the real
collection, idempotently, so they are safe to repeat -- but do not run them
while a full ingest is in flight: both would be writing the same rows and the
counts they assert on would move underneath them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from beacon_core.models import Trial
from beacon_core.settings import Settings
from ingestion import store
from ingestion.corpus import CorpusNotCachedError, corpus_is_cached, iter_raw_trials
from ingestion.run import RunConfig, ingest
from ingestion.store import QdrantIndex, point_id

pytestmark = pytest.mark.integration

INTEGRATION_HINT = (
    "Bring the stack up with "
    "`docker compose -f deploy/compose.yaml up -d --wait` and apply the schema "
    "with `uv run alembic upgrade head`. The host ports are shifted: Postgres "
    "15432, Redis 16379."
)

CORPUS_HINT = (
    "The TREC collection archives are not cached. Fetch them once with "
    '`uv run python -c "from evals.trec import iter_trials; next(iter_trials(limit=1))"` '
    "(~1.8 GB compressed). CI has no corpus, so these tests skip there."
)

SAMPLE = 24
"""Small enough to embed in a few seconds on a CPU, large enough to span
several real records with and without age bounds."""


@pytest.fixture
def corpus() -> None:
    """Skip unless the collection archives are on disk.

    Ordered before the embedder is ever constructed, so a runner without the
    corpus never pays for FastEmbed's 130 MB model download either.
    """
    if not corpus_is_cached():
        pytest.skip(CORPUS_HINT)
    try:
        next(iter_raw_trials(limit=1))
    except CorpusNotCachedError as exc:  # pragma: no cover - belt and braces
        pytest.skip(f"{exc} {CORPUS_HINT}")


@pytest.fixture
async def live_engine() -> AsyncIterator[AsyncEngine]:
    """A fresh engine per test, not ``beacon_core.db``'s cached one.

    The cache is per process while every test gets a fresh event loop
    (``asyncio_default_fixture_loop_scope = "function"``), and an asyncpg
    connection belongs to the loop that opened it. Sharing the cached pool
    across two tests fails with *"got Future attached to a different loop"* --
    a real bug, not a CI artifact: it bites the second of any two of these
    tests run in sequence, locally included.
    """
    engine = create_async_engine(Settings().postgres.dsn())
    try:
        async with engine.connect():
            pass
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Postgres unreachable: {exc}. {INTEGRATION_HINT}")
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def live_sessions(live_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(live_engine, expire_on_commit=False)


@pytest.fixture
async def live_index() -> AsyncIterator[QdrantIndex]:
    index = QdrantIndex(Settings().qdrant)
    try:
        await index.ensure_collection()
    except Exception as exc:
        await index.close()
        pytest.skip(f"Qdrant unreachable: {exc}. {INTEGRATION_HINT}")
    try:
        yield index
    finally:
        await index.close()


async def _counts(
    sessions: async_sessionmaker[AsyncSession], index: QdrantIndex
) -> tuple[int, int, int]:
    points = await index.count()
    async with sessions.begin() as session:
        rows = await store.count_trials(session)
        indexed = await store.count_trials(session, indexed_only=True)
    return points, rows, indexed


async def test_ingest_is_idempotent_by_nct_id(
    corpus: None,
    live_sessions: async_sessionmaker[AsyncSession],
    live_index: QdrantIndex,
) -> None:
    """Running the same slice twice must change nothing.

    This is the property the whole dual write is built around: Qdrant point
    ids are ``uuid5(namespace, nct_id)`` rather than random, and the ``trials``
    write is an upsert on ``nct_id``. If either were otherwise the second pass
    would double the collection and no error would be raised.

    The second pass uses ``force=True`` on purpose. Without it the resume
    marker would skip every record and the test would prove only that skipping
    works; with it, every row and point is genuinely rewritten, and the counts
    still have to be identical.
    """
    config = RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=SAMPLE)
    await ingest(config, sessions=live_sessions)
    before = await _counts(live_sessions, live_index)

    await ingest(
        RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=SAMPLE, force=True),
        sessions=live_sessions,
    )
    after = await _counts(live_sessions, live_index)

    assert before == after, f"re-ingest changed the stores: {before} -> {after}"
    points, rows, indexed = after
    assert rows >= SAMPLE
    assert points == indexed, "every indexed row must have exactly one point"


async def test_every_point_carries_both_named_vectors(
    corpus: None,
    live_sessions: async_sessionmaker[AsyncSession],
    live_index: QdrantIndex,
) -> None:
    await ingest(RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=0), sessions=live_sessions)
    cfg = Settings()
    async with live_sessions.begin() as session:
        nct_ids = list(
            (
                await session.execute(select(Trial.nct_id).order_by(Trial.nct_id).limit(SAMPLE))
            ).scalars()
        )
    records = await live_index.retrieve(nct_ids)

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


async def test_payload_agrees_with_the_postgres_row(
    corpus: None,
    live_sessions: async_sessionmaker[AsyncSession],
    live_index: QdrantIndex,
) -> None:
    """The hand-check, as an assertion rather than a printout."""
    await ingest(RunConfig(limit=SAMPLE, batch_size=8, criteria_limit=0), sessions=live_sessions)
    async with live_sessions.begin() as session:
        rows = list(
            (await session.execute(select(Trial).order_by(Trial.nct_id).limit(SAMPLE))).scalars()
        )
    records = {r.id: r for r in await live_index.retrieve([row.nct_id for row in rows])}

    for row in rows:
        payload = records[point_id(row.nct_id)].payload or {}
        assert payload["nct_id"] == row.nct_id
        assert payload["min_age_years"] == row.min_age_years
        assert payload["max_age_years"] == row.max_age_years
        assert payload["sex"] == row.sex
        assert payload["phase"] == row.phase
        assert payload["status"] == row.status
        # Eligibility text is Postgres-only; the retrieval unit is a trial.
        assert "eligibility_text" not in payload
        assert "overview_text" not in payload


async def test_null_ages_survive_the_round_trip_as_null(
    corpus: None,
    live_sessions: async_sessionmaker[AsyncSession],
    live_index: QdrantIndex,
) -> None:
    """A real trial with no stated age must read back NULL, not 0 or 999.

    ``NCT00000104`` is one: both bounds are ``N/A`` in the source XML. It is
    the second record of the collection, so a 200-record slice covers it.
    """
    await ingest(RunConfig(limit=200, batch_size=32, criteria_limit=0), sessions=live_sessions)
    async with live_sessions.begin() as session:
        row = (
            await session.execute(select(Trial).where(Trial.nct_id == "NCT00000104"))
        ).scalar_one()
    assert row.raw_min_age == "N/A"
    assert row.raw_max_age == "N/A"
    assert row.min_age_years is None
    assert row.max_age_years is None

    payload = (await live_index.retrieve(["NCT00000104"]))[0].payload or {}
    assert payload["min_age_years"] is None
    assert payload["max_age_years"] is None


async def test_unparseable_age_is_stored_as_null(
    live_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Round-trip proof for the shape the brief calls out, straight to SQL.

    Needs no corpus -- the record is hand-built, because the April-2021
    snapshot happens to contain no prose age bounds. A later snapshot may, and
    when it does the trial must stay findable rather than acquire a 65-year
    upper bound nobody wrote.
    """
    from sqlalchemy import delete

    from ingestion._testing import WordTokenizer
    from ingestion.corpus import RawTrial
    from ingestion.overview import OverviewBuilder
    from ingestion.pipeline import derive_trial

    nct_id = "NCT09999991"
    raw = RawTrial(
        nct_id=nct_id,
        title="Null Age Round Trip",
        conditions=("Hypertension",),
        interventions=(),
        phase=None,
        status="Recruiting",
        brief_summary=None,
        detailed_description=None,
        eligibility_text=None,
        raw_gender="Nonbinary",
        raw_min_age="65 Years and older",
        raw_max_age="N/A",
    )
    derived = derive_trial(raw, OverviewBuilder(WordTokenizer(), max_tokens=512))
    try:
        async with live_sessions.begin() as session:
            await store.upsert_trials(session, [derived.trial_row()])
        async with live_sessions.begin() as session:
            row = (
                await session.execute(
                    select(
                        Trial.min_age_years,
                        Trial.max_age_years,
                        Trial.sex,
                        Trial.raw_min_age,
                    ).where(Trial.nct_id == nct_id)
                )
            ).one()
        assert row.min_age_years is None
        assert row.max_age_years is None
        assert row.sex is None
        assert row.raw_min_age == "65 Years and older"
    finally:
        async with live_sessions.begin() as session:
            await session.execute(delete(Trial).where(Trial.nct_id == nct_id))


async def test_the_live_sparse_field_declares_idf(live_index: QdrantIndex) -> None:
    """Asserted against a real Qdrant, because the unit-test stub can only
    prove the client *sent* ``modifier: IDF``, not that the server kept it."""
    cfg = Settings()
    info = await live_index.info()
    sparse = (info.config.params.sparse_vectors or {})[cfg.qdrant.sparse_vector_name]
    assert sparse.modifier == models.Modifier.IDF
    assert {"min_age_years", "max_age_years", "sex"} <= set(info.payload_schema or {})


async def test_cross_check_against_ir_datasets(corpus: None) -> None:
    """The XML reading and ``evals.trec`` must agree on what they share.

    They must also disagree exactly where ``ir_datasets`` drops data, which is
    the measurement that justifies ``ingestion.corpus`` existing at all.
    """
    from ingestion.corpus import cross_check_against_ir_datasets

    counts = cross_check_against_ir_datasets(sample=200)
    assert counts["checked"] == 200
    assert counts["id_mismatch"] == 0
    assert counts["title_mismatch"] == 0
    assert counts["eligibility_mismatch"] == 0
    assert counts["irds_condition_empty"] > 0


def test_reads_real_records_from_the_cached_archives(corpus: None) -> None:
    trials = list(iter_raw_trials(limit=5))
    assert len(trials) == 5
    assert trials[0].nct_id == "NCT00000102"
    assert all(t.nct_id.startswith("NCT") for t in trials)
    assert all(t.title for t in trials)


def test_skip_is_reproducible(corpus: None) -> None:
    """``skip`` must land on the same record every time, on every machine."""
    first_three = [t.nct_id for t in iter_raw_trials(limit=3)]
    skipped = [t.nct_id for t in iter_raw_trials(skip=1, limit=2)]
    assert skipped == first_three[1:]
