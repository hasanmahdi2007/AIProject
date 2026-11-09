"""The real models: bge-small, BM25 and bge-reranker-base, on CPU.

**These skip rather than download.** ``bge-reranker-base`` is ~280 MB and
``bge-small-en-v1.5`` ~130 MB, and CI runs ``pytest -m integration`` on a fresh
runner with a cold FastEmbed cache. :func:`retrieval.testing.missing_models`
inspects the cache directory instead of trying to load the model, because
attempting the load *is* the download. Everything that can be tested without
weights lives in ``test_integration_qdrant.py``, which does run in CI - the
overfiltering and NULL-safety tests among them.

Nothing here calls a paid API. All three models are local and free; this service
never talks to an LLM at all.

Nothing here writes to Postgres or to the shared ``trials`` collection. The
semantic fixture is created and dropped per test under its own name, and the
two tests that touch real data are read-only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from beacon_core.schemas import RankedQuery, SearchRequest
from beacon_core.settings import QdrantSettings, Settings
from retrieval.encoders import get_cross_encoder, get_dense_encoder, get_sparse_encoder
from retrieval.hydrate import (
    MappingTrialTextStore,
    PostgresTrialTextStore,
    TrialText,
    rerank_document,
)
from retrieval.search import HybridSearcher
from retrieval.service import RetrievalService
from retrieval.testing import (
    INTEGRATION_HINT,
    SEMANTIC_TRIALS,
    create_semantic_collection,
    missing_models,
    semantic_text_store,
)

pytestmark = pytest.mark.integration

COLLECTION = "trials_test_b2_models"

LUNG_QUERY = RankedQuery(
    rank=1,
    text="52-year-old with metastatic EGFR-mutant non-small cell lung cancer",
    facet="diagnosis",
)


def require_models() -> Settings:
    settings = Settings()
    absent = missing_models(settings.qdrant)
    if absent:
        pytest.skip(
            f"model weights not cached: {', '.join(absent)}. These are ~400 MB in "
            'total; cache them with `uv run python -c "from fastembed import '
            "TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')\"` (and the "
            "same for Qdrant/bm25 and BAAI/bge-reranker-base) rather than letting "
            "a test download them."
        )
    return settings


async def live_client(settings: Settings) -> AsyncQdrantClient:
    client = AsyncQdrantClient(url=settings.qdrant.url)
    try:
        await client.get_collections()
    except Exception as exc:
        await client.close()
        pytest.skip(f"Qdrant unreachable at {settings.qdrant.url}: {exc}. {INTEGRATION_HINT}")
    return client


@pytest.fixture
async def service() -> AsyncIterator[RetrievalService]:
    """The real pipeline over the semantic fixture, embedded for real.

    One client and one set of encoders per test. The encoders are
    process-cached by ``functools.lru_cache`` and hold no sockets, so they are
    safe across event loops; the Qdrant client is not, and is rebuilt and
    closed per test.
    """
    settings = require_models()
    client = await live_client(settings)

    fixture_settings: QdrantSettings = Settings().qdrant
    fixture_settings.collection = COLLECTION

    dense = get_dense_encoder(settings)
    sparse = get_sparse_encoder(settings)
    texts = semantic_text_store()
    await create_semantic_collection(
        client,
        COLLECTION,
        fixture_settings,
        dense_vectors=[dense.encode_query(t.text) for t in SEMANTIC_TRIALS],
        sparse_vectors=[sparse.encode_query(t.text) for t in SEMANTIC_TRIALS],
    )
    try:
        yield RetrievalService(
            searcher=HybridSearcher(
                client=client, settings=fixture_settings, dense=dense, sparse=sparse
            ),
            text_store=MappingTrialTextStore(texts),
            cross_encoder=get_cross_encoder(settings),
            settings=settings,
        )
    finally:
        await client.delete_collection(COLLECTION)
        await client.close()


class TestRealRetrieval:
    async def test_a_lung_cancer_query_retrieves_lung_cancer_trials(
        self, service: RetrievalService
    ) -> None:
        """The baseline sanity check: the embeddings are doing something.

        Every test in the container-free suite uses stub vectors, so nothing
        there can catch "the query is being embedded with the wrong model" or
        "the dense and sparse branches are transposed". This can.
        """
        response = await service.search(
            SearchRequest.model_validate(
                {"queries": [LUNG_QUERY.model_dump()], "rerank_top_k": 5, "use_rerank": False}
            )
        )

        top = [c.nct_id for c in response.candidates]
        assert any(nct_id.startswith("NCTFIX000") for nct_id in top[:3]), (
            f"expected the lung-cancer trials (NCTFIX0001-0003) near the top, got {top}"
        )
        assert "NCTFIX0032" not in top[:3], "the heart-failure trial is not about lung cancer"

    async def test_the_cross_encoder_changes_the_order(self, service: RetrievalService) -> None:
        """Measured rather than assumed: if the reranker never changed anything
        it would be 17 seconds of CPU for nothing, and the ablation row would be
        a flat line."""
        body = {"queries": [LUNG_QUERY.model_dump()], "fuse_top_k": 12, "rerank_top_k": 8}
        without = await service.search(SearchRequest.model_validate({**body, "use_rerank": False}))
        with_rerank = await service.search(
            SearchRequest.model_validate({**body, "use_rerank": True})
        )

        assert [c.nct_id for c in without.candidates] != [
            c.nct_id for c in with_rerank.candidates
        ], "the cross-encoder returned exactly the RRF order, which is suspicious"
        assert all(c.rerank_score is not None for c in with_rerank.candidates)
        assert all(c.rerank_score is None for c in without.candidates)

    async def test_the_hard_filter_still_applies_with_real_vectors(
        self, service: RetrievalService
    ) -> None:
        """The filter and the embeddings are independent, but a regression that
        dropped the filter would only show up with real retrieval."""
        response = await service.search(
            SearchRequest.model_validate(
                {
                    "queries": [{"rank": 1, "text": "prostate cancer with lung metastases"}],
                    "filters": {"age": 45, "sex": "female"},
                    "rerank_top_k": 20,
                    "use_rerank": False,
                }
            )
        )

        returned = {c.nct_id for c in response.candidates}
        assert "NCTFIX0020" not in returned, "a male-only trial, for a female patient"
        assert returned, "the filter should not have emptied the result set"

    async def test_embeddings_cannot_compare_numbers_and_that_is_fine(
        self, service: RetrievalService
    ) -> None:
        """The documented limitation, asserted as behaviour.

        A 45-year-old's query retrieves the 65-and-over trial happily - "65" and
        "45" are near-identical tokens to a model trained on meaning. With no
        filter it comes back; with the typed payload filter it does not. This is
        the whole reason numeric eligibility is a filter and a rule checker
        rather than a similarity score.
        """
        body = {
            "queries": [{"rank": 1, "text": "immunotherapy for non-small cell lung carcinoma"}],
            "rerank_top_k": 20,
            "use_rerank": False,
        }
        unfiltered = await service.search(SearchRequest.model_validate(body))
        filtered = await service.search(
            SearchRequest.model_validate({**body, "filters": {"age": 45}})
        )

        assert "NCTFIX0010" in {c.nct_id for c in unfiltered.candidates}, (
            "the 65-and-over trial is semantically relevant, and retrieval says so"
        )
        assert "NCTFIX0010" not in {c.nct_id for c in filtered.candidates}, (
            "the typed filter, not the similarity score, is what knows 45 < 65"
        )

    async def test_timings_show_the_reranker_dominating(self, service: RetrievalService) -> None:
        """The cost story, measured on this hardware rather than asserted.

        The funnel's shape is justified by the cross-encoder being orders of
        magnitude more expensive per document than the bi-encoders. If that ever
        stops being true the funnel's proportions are wrong, and this is where
        it would show.
        """
        response = await service.search(
            SearchRequest.model_validate(
                {"queries": [LUNG_QUERY.model_dump()], "fuse_top_k": 12, "use_rerank": True}
            )
        )

        timings = response.timings_ms
        assert timings["rerank"] > timings["embed"]
        assert timings["rerank"] > timings["qdrant"]
        assert timings["fuse"] < 10.0, "pure-Python RRF over a few hundred IDs is free"


class TestCrossEncoderDiscriminates:
    def test_it_scores_a_relevant_trial_above_an_irrelevant_one(self) -> None:
        """Eight pairs, not a hundred: enough to prove the model is reading the
        pair, cheap enough to belong in a test. On this machine
        ``bge-reranker-base`` manages about 6 pairs per second over
        trial-length documents, so a 100-pair rerank is ~17 s - which is the
        measurement behind the funnel narrowing to 100 before it gets here.
        """
        settings = require_models()
        encoder = get_cross_encoder(settings)
        lung = next(t for t in SEMANTIC_TRIALS if t.nct_id == "NCTFIX0001")
        heart = next(t for t in SEMANTIC_TRIALS if t.nct_id == "NCTFIX0032")

        scores = encoder.score(LUNG_QUERY.text, [lung.text, heart.text])

        assert scores[0] > scores[1]

    def test_the_document_is_the_text_that_was_embedded(self) -> None:
        """``overview_text`` byte-for-byte, so an improvement from reranking is
        attributable to the model and not to it having been handed better
        input."""
        text = TrialText(nct_id="NCTFIX0001", title="t", overview_text="the embedded text")

        assert rerank_document(text) == "the embedded text"


class TestPostgresHydration:
    """The text half, against the real ``trials`` table. Read-only.

    No fixture rows are written: the test reads NCT IDs out of the live Qdrant
    collection and hydrates those, which is exactly what the service does on a
    real request, and leaves the shared database untouched.

    The engine is built and disposed inside the test. ``pytest`` gives every
    test a fresh event loop, and ``beacon_core.db`` caches its engine
    process-wide, so a pool created in one test's loop and reused in the next
    raises ``got Future attached to a different loop``.
    """

    async def test_it_hydrates_titles_for_real_nct_ids(self) -> None:
        settings = Settings()
        client = await live_client(settings)
        try:
            if not await client.collection_exists(settings.qdrant.collection):
                pytest.skip("no live trials collection; run the B1 ingest")
            points, _ = await client.scroll(
                settings.qdrant.collection, limit=5, with_payload=True, with_vectors=False
            )
        finally:
            await client.close()
        nct_ids = [
            str((p.payload or {})["nct_id"]) for p in points if (p.payload or {}).get("nct_id")
        ]
        if not nct_ids:
            pytest.skip("the live collection is empty; run the B1 ingest")

        engine = create_async_engine(settings.postgres.dsn())
        try:
            async with engine.connect():
                pass
        except Exception as exc:
            await engine.dispose()
            pytest.skip(f"Postgres unreachable: {exc}. {INTEGRATION_HINT}")
        try:
            store = PostgresTrialTextStore(async_sessionmaker(engine, expire_on_commit=False))
            found = await store.fetch(nct_ids)
        finally:
            await engine.dispose()

        assert found, (
            f"none of {nct_ids} is in the trials table: Qdrant and Postgres have "
            "diverged, which is the known cost of the dual write"
        )
        for nct_id, text in found.items():
            assert text.nct_id == nct_id
            assert text.title, f"{nct_id} has no title to show or to rerank on"
            assert text.overview_text, f"{nct_id} has no embedded text to rerank on"

    async def test_an_unknown_nct_id_is_absent_rather_than_an_error(self) -> None:
        """A candidate missing from Postgres must not raise: the service keeps
        it as a candidate with no text rather than silently deleting a trial
        because of an infrastructure inconsistency."""
        settings = Settings()
        engine = create_async_engine(settings.postgres.dsn())
        try:
            async with engine.connect():
                pass
        except Exception as exc:
            await engine.dispose()
            pytest.skip(f"Postgres unreachable: {exc}. {INTEGRATION_HINT}")
        try:
            store = PostgresTrialTextStore(async_sessionmaker(engine, expire_on_commit=False))
            found = await store.fetch(["NCT_DOES_NOT_EXIST_B2"])
        finally:
            await engine.dispose()

        assert found == {}

    async def test_an_empty_id_list_does_not_hit_the_database(self) -> None:
        """``WHERE nct_id IN ()`` is a syntax error in some dialects and a
        guaranteed-empty scan in others. Short-circuited either way."""
        # A DSN that could not possibly connect, to prove it is never used.
        engine = create_async_engine("postgresql+asyncpg://nowhere:5432/none")
        store = PostgresTrialTextStore(async_sessionmaker(engine))

        assert await store.fetch([]) == {}


class TestFilterAblationOnRealVectors:
    async def test_apply_age_sex_filters_false_returns_the_excluded_trial(
        self, service: RetrievalService
    ) -> None:
        """The ablation row is reachable by config alone, end to end with real
        embeddings: the same request that excludes a male-only trial includes it
        when the setting is off."""
        settings = Settings()
        settings.retrieval.apply_age_sex_filters = False
        unfiltered_service = RetrievalService(
            # The same searcher, so the fixture collection is not re-upserted.
            searcher=service.searcher,
            text_store=MappingTrialTextStore(semantic_text_store()),
            cross_encoder=None,
            settings=settings,
        )

        response = await unfiltered_service.search(
            SearchRequest.model_validate(
                {
                    "queries": [{"rank": 1, "text": "prostate cancer with lung metastases"}],
                    "filters": {"age": 45, "sex": "female"},
                    "rerank_top_k": 20,
                    "use_rerank": False,
                }
            )
        )

        assert "NCTFIX0020" in {c.nct_id for c in response.candidates}
