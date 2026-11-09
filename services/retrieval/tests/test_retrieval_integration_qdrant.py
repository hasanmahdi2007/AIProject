"""Against a live Qdrant. No models loaded, so this runs in CI.

CI genuinely runs ``pytest -m integration`` against the compose stack, so
everything here skips with a message when a prerequisite is absent rather than
failing. What it does *not* need is any ONNX weights: the fixtures use
deterministic pseudo-random vectors and the queries use
:class:`StubDenseEncoder`, because none of these tests is about relevance. They
are about which points a filter admits and how many come back - and that makes
the single most important test in this service runnable on a cold runner.

Two fixture collections are created and dropped per test, both prefixed
``trials_test_b2_``. Nothing here reads or writes the shared ``trials``
collection except :class:`TestLiveCollection`, which is strictly read-only.

Every test builds its own ``AsyncQdrantClient`` and closes it. ``pytest``
is configured with ``asyncio_default_fixture_loop_scope = "function"``, so each
test gets a fresh event loop; a client (and its connection pool) created in one
loop and reused in another raises ``got Future attached to a different loop``
and then cascades into ``Event loop is closed``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from beacon_core.schemas import AgeSexFilter, RankedQuery
from beacon_core.settings import QdrantSettings, Settings
from retrieval.collection import verify_collection
from retrieval.filters import MAX_AGE_FIELD, MIN_AGE_FIELD, SEX_FIELD, build_filter
from retrieval.search import HybridSearcher
from retrieval.testing import (
    INTEGRATION_HINT,
    SCALE_FILTER_AGE,
    SEMANTIC_TRIALS,
    StubDenseEncoder,
    StubSparseEncoder,
    create_scale_collection,
    create_semantic_collection,
    deterministic_vectors,
)

pytestmark = pytest.mark.integration

SCALE_COLLECTION = "trials_test_b2_scale"
SEMANTIC_COLLECTION = "trials_test_b2_semantic"

QUERY = [RankedQuery(rank=1, text="metastatic non-small cell lung cancer")]

# The brief's scenario, and the reason this file exists: a beam of 40 visited
# points under a filter that matches 10% of the corpus leaves about four
# survivors if the filter is applied afterwards.
HNSW_EF = 40
BEAM = 40
TOP_K = 20


async def live_client() -> AsyncQdrantClient:
    """A client of our own, or a skip. Never shared between tests."""
    settings = Settings()
    client = AsyncQdrantClient(url=settings.qdrant.url)
    try:
        await client.get_collections()
    except Exception as exc:
        await client.close()
        pytest.skip(f"Qdrant unreachable at {settings.qdrant.url}: {exc}. {INTEGRATION_HINT}")
    return client


def fixture_settings(collection: str) -> QdrantSettings:
    """Production Qdrant settings, pointed at a fixture collection."""
    settings = Settings().qdrant
    settings.collection = collection
    return settings


def searcher_for(client: AsyncQdrantClient, settings: QdrantSettings) -> HybridSearcher:
    return HybridSearcher(
        client=client,
        settings=settings,
        dense=StubDenseEncoder(dim=settings.dense_dim),
        sparse=StubSparseEncoder(),
    )


@pytest.fixture
async def scale() -> AsyncIterator[tuple[AsyncQdrantClient, QdrantSettings, int]]:
    """1,000 random points, exactly 100 of which pass an age-45 filter."""
    client = await live_client()
    settings = fixture_settings(SCALE_COLLECTION)
    eligible = await create_scale_collection(client, SCALE_COLLECTION, settings, size=1000)
    try:
        yield client, settings, eligible
    finally:
        await client.delete_collection(SCALE_COLLECTION)
        await client.close()


@pytest.fixture
async def semantic() -> AsyncIterator[tuple[AsyncQdrantClient, QdrantSettings]]:
    """The ~24 hand-written trials with deliberately awkward age/sex payloads."""
    client = await live_client()
    settings = fixture_settings(SEMANTIC_COLLECTION)
    dense, sparse = deterministic_vectors(len(SEMANTIC_TRIALS), settings)
    await create_semantic_collection(
        client,
        SEMANTIC_COLLECTION,
        settings,
        dense_vectors=dense,
        sparse_vectors=sparse,
    )
    try:
        yield client, settings
    finally:
        await client.delete_collection(SEMANTIC_COLLECTION)
        await client.close()


class TestOverfiltering:
    """**The test this whole service is organised around.**

    A filter that matches 10% of the collection, a beam of 40, and a request for
    20 candidates. Done right - the filter inside the request, so the graph walk
    only visits points that satisfy it - 20 come back. Done the tempting way -
    search, then drop the rows that fail - about 4 come back, with no error and
    no warning, and a response that is indistinguishable from "only four trials
    matched".

    Both paths are run here against the same collection in the same test, so
    the comparison is not an argument about how HNSW behaves; it is two numbers.
    """

    async def test_filtered_top_20_returns_20_not_4(
        self, scale: tuple[AsyncQdrantClient, QdrantSettings, int]
    ) -> None:
        client, settings, eligible = scale
        searcher = searcher_for(client, settings)
        encoded = searcher.encode(QUERY)
        search_filter = build_filter(AgeSexFilter(age=SCALE_FILTER_AGE))
        assert search_filter is not None

        # --- the correct path: the filter is part of the query ---
        inside = await searcher.search(
            encoded, search_filter=search_filter, limit=TOP_K, hnsw_ef=HNSW_EF
        )

        # --- the bug: the same search, unfiltered, filtered afterwards ---
        unfiltered = await searcher.search(encoded, search_filter=None, limit=BEAM, hnsw_ef=HNSW_EF)
        post_filtered_dense = [
            hit for hit in unfiltered[0].dense if hit.payload[MIN_AGE_FIELD] <= SCALE_FILTER_AGE
        ]

        assert len(inside[0].dense) == TOP_K, (
            "a filtered top-20 must return 20 candidates; fewer means the filter "
            "is not constraining the index traversal"
        )
        assert len(inside[0].sparse) == TOP_K
        assert len(post_filtered_dense) < TOP_K // 2, (
            f"post-filtering a beam of {BEAM} left {len(post_filtered_dense)} of "
            f"{TOP_K} requested - this is the overfiltering bug, and it is what "
            "the service would silently do if the filter were applied to the "
            "result list instead of being passed into the query"
        )
        # The expected value is ~4 (10% of 40). Asserted as a range rather than
        # a number because it depends on which 40 points the beam happened to
        # visit, and a flaky test here would get deleted rather than fixed.
        assert len(post_filtered_dense) <= 10
        assert eligible == 100, "the fixture's selectivity is 10%, by construction"

    async def test_every_returned_candidate_satisfies_the_filter(
        self, scale: tuple[AsyncQdrantClient, QdrantSettings, int]
    ) -> None:
        """Returning 20 would be worthless if they were the wrong 20."""
        client, settings, _ = scale
        searcher = searcher_for(client, settings)
        search_filter = build_filter(AgeSexFilter(age=SCALE_FILTER_AGE))

        hits = await searcher.search(
            searcher.encode(QUERY), search_filter=search_filter, limit=TOP_K, hnsw_ef=HNSW_EF
        )

        for branch in (hits[0].dense, hits[0].sparse):
            assert branch
            for hit in branch:
                assert hit.payload[MIN_AGE_FIELD] <= SCALE_FILTER_AGE
                assert hit.payload[MAX_AGE_FIELD] >= SCALE_FILTER_AGE

    async def test_the_filter_holds_at_a_depth_above_the_eligible_count(
        self, scale: tuple[AsyncQdrantClient, QdrantSettings, int]
    ) -> None:
        """The honest upper bound of the claim, and it is not "exactly all".

        Asking for 150 when 100 points match returns *nearly* 100 - measured at
        95 on one run and 100 on another. The search is still approximate: a
        filtered HNSW walk is constrained to matching points, which is what
        fixes the overfiltering bug, but it does not thereby become exhaustive,
        and recall at a depth approaching the size of the matching set is below
        1.0.

        This is why the claim this file makes is "a filtered top-20 returns 20",
        not "a filtered search returns every match". At k=20 against 100
        matching points the depth requested is a fifth of what is available and
        the beam is never the binding constraint; at k=150 it is. Asserting
        equality here would have been a flaky test, which is worse than no test
        because it gets deleted rather than fixed.
        """
        client, settings, eligible = scale
        searcher = searcher_for(client, settings)
        search_filter = build_filter(AgeSexFilter(age=SCALE_FILTER_AGE))

        hits = await searcher.search(
            searcher.encode(QUERY), search_filter=search_filter, limit=150, hnsw_ef=HNSW_EF
        )

        assert len(hits[0].dense) <= eligible, "it cannot invent matching points"
        assert len(hits[0].dense) >= eligible * 0.8, (
            f"returned {len(hits[0].dense)} of {eligible} matching points; well "
            "below approximate-recall territory and into post-filtering territory"
        )


class TestNullSafety:
    """A trial with an unknown bound must survive. Both directions, every field.

    This is the more dangerous half of the filter's job: a wrongly *included*
    trial gets rejected a few seconds later by the rule checker, while a wrongly
    *excluded* one never appears anywhere for anyone to notice. On the real
    corpus ``max_age_years`` is NULL for about half the trials, so getting this
    wrong discards half the index.

    The assertions are on set membership at a depth greater than the fixture, so
    the filter is the only thing deciding what comes back.
    """

    DEPTH = 50
    FORTY_FIVE = AgeSexFilter(age=45)

    async def returned(
        self,
        semantic: tuple[AsyncQdrantClient, QdrantSettings],
        filters: AgeSexFilter,
    ) -> set[str]:
        client, settings = semantic
        searcher = searcher_for(client, settings)
        hits = await searcher.search(
            searcher.encode(QUERY),
            search_filter=build_filter(filters),
            limit=self.DEPTH,
        )
        return {hit.nct_id for hit in hits[0].dense}

    async def test_a_null_min_age_survives_an_age_filter(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """NCTFIX0001 has no parsed age bounds at all - "N/A" on the source
        record, or ``"65 Years and older"``, which B1 deliberately parses to
        ``None`` rather than risk reading as a 65-year *maximum*."""
        survivors = await self.returned(semantic, self.FORTY_FIVE)

        assert "NCTFIX0001" in survivors

    async def test_a_null_max_age_survives_when_the_min_is_known(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """NCTFIX0002 is ``min=18, max=None``. The known bound admits a
        45-year-old and the unknown one must not veto it - which is why the two
        bounds are separate clauses."""
        survivors = await self.returned(semantic, self.FORTY_FIVE)

        assert "NCTFIX0002" in survivors

    async def test_a_null_min_age_survives_when_the_max_is_known(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        survivors = await self.returned(semantic, self.FORTY_FIVE)

        assert "NCTFIX0003" in survivors

    async def test_the_other_direction_strict_mode_drops_them(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """The same filter with ``include_unknown=False``.

        This is what makes the previous three tests meaningful: it proves they
        pass because of the ``is_null``/``is_empty`` disjuncts and not because
        Qdrant happens to admit NULLs through a bare range condition. If range
        conditions passed NULLs on their own, this test would fail.
        """
        survivors = await self.returned(semantic, AgeSexFilter(age=45, include_unknown=False))

        assert "NCTFIX0001" not in survivors
        assert "NCTFIX0002" not in survivors, "min=18 is known but max=None is not"
        assert "NCTFIX0003" not in survivors

    async def test_a_null_sex_survives_a_sex_filter(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        survivors = await self.returned(semantic, AgeSexFilter(sex="female"))

        assert "NCTFIX0001" in survivors, "sex is None: unknown, so possibly eligible"

    async def test_strict_mode_drops_a_null_sex(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        survivors = await self.returned(semantic, AgeSexFilter(sex="female", include_unknown=False))

        assert "NCTFIX0001" not in survivors

    async def test_positive_evidence_still_excludes(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """NULL-tolerance must not degenerate into no filter at all.

        NCTFIX0010 is 65-and-over and NCTFIX0011 caps at 17; both have a
        *parsed* bound that excludes a 45-year-old, so both must go.
        """
        survivors = await self.returned(semantic, self.FORTY_FIVE)

        assert "NCTFIX0010" not in survivors, "min_age_years=65 excludes a 45-year-old"
        assert "NCTFIX0011" not in survivors, "max_age_years=17 excludes a 45-year-old"

    async def test_a_boundary_age_is_included(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """A 65-year-old is eligible for a 65-and-over trial, and for one
        capped at 65. Strict bounds would delete exactly the boundary
        patients."""
        survivors = await self.returned(semantic, AgeSexFilter(age=65))

        assert "NCTFIX0010" in survivors  # min 65
        assert "NCTFIX0030" in survivors  # 18-75
        assert "NCTFIX0033" not in survivors  # capped at 64


class TestSexFilter:
    DEPTH = 50

    async def returned(
        self,
        semantic: tuple[AsyncQdrantClient, QdrantSettings],
        filters: AgeSexFilter,
    ) -> set[str]:
        client, settings = semantic
        searcher = searcher_for(client, settings)
        hits = await searcher.search(
            searcher.encode(QUERY),
            search_filter=build_filter(filters),
            limit=self.DEPTH,
        )
        return {hit.nct_id for hit in hits[0].dense}

    async def test_a_male_only_trial_is_excluded_for_a_female_patient(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        survivors = await self.returned(semantic, AgeSexFilter(age=45, sex="female"))

        assert "NCTFIX0020" not in survivors, "a prostate cancer trial enrols males only"

    async def test_a_female_only_trial_is_kept_for_a_female_patient(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        survivors = await self.returned(semantic, AgeSexFilter(age=45, sex="female"))

        assert "NCTFIX0021" in survivors

    async def test_unrestricted_trials_are_kept_for_either_sex(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """``sex == "all"`` is most trials. Forgetting to match it would delete
        almost the whole index for every patient."""
        female = await self.returned(semantic, AgeSexFilter(age=45, sex="female"))
        male = await self.returned(semantic, AgeSexFilter(age=45, sex="male"))

        assert "NCTFIX0030" in female and "NCTFIX0030" in male
        assert "NCTFIX0031" in female and "NCTFIX0031" in male

    async def test_the_mirror_case(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        survivors = await self.returned(semantic, AgeSexFilter(age=45, sex="male"))

        assert "NCTFIX0020" in survivors
        assert "NCTFIX0021" not in survivors
        assert "NCTFIX0022" not in survivors


class TestFixtureCollectionDeclarations:
    async def test_the_fixture_declares_idf_like_production(
        self, semantic: tuple[AsyncQdrantClient, QdrantSettings]
    ) -> None:
        """If the fixture's sparse field behaved differently from production's,
        the fixture would be testing something else."""
        client, settings = semantic

        report = await verify_collection(client, settings)

        assert report.ok, report.problems
        assert report.sparse_modifier == "idf"
        assert set(report.indexed_payload_fields) >= {MIN_AGE_FIELD, MAX_AGE_FIELD, SEX_FIELD}


class TestLiveCollection:
    """Read-only assertions about the shared ``trials`` collection.

    Nothing here writes, creates or deletes. The point count is deliberately
    never asserted: the full ingest is a multi-hour job that may be a third
    done, finished, or resuming, and a test that encodes today's number would
    fail for a reason that is not a defect.
    """

    @staticmethod
    async def live_trials() -> tuple[AsyncQdrantClient, QdrantSettings]:
        client = await live_client()
        settings = Settings().qdrant
        if not await client.collection_exists(settings.collection):
            await client.close()
            pytest.skip(
                f"collection {settings.collection!r} does not exist; run the B1 ingest first"
            )
        info = await client.get_collection(settings.collection)
        if not info.points_count:
            await client.close()
            pytest.skip(f"collection {settings.collection!r} is empty; run the B1 ingest")
        return client, settings

    async def test_the_sparse_field_declares_modifier_idf(self) -> None:
        """The silent one. BM25 without corpus IDF still returns 20 results and
        has lost the rare-term weighting that is the only reason it is here
        next to a dense index."""
        client, settings = await self.live_trials()
        try:
            report = await verify_collection(client, settings)
        finally:
            await client.close()

        assert report.sparse_modifier == "idf", (
            "recreate the collection with modifier=IDF; nothing raises without it"
        )
        assert report.problems == (), report.problems

    async def test_the_filter_fields_are_payload_indexed(self) -> None:
        """Without these, a filtered query is still *correct* but scans payloads
        instead of constraining the traversal - the design collapses into a full
        scan with extra steps."""
        client, settings = await self.live_trials()
        try:
            report = await verify_collection(client, settings)
        finally:
            await client.close()

        assert set(report.indexed_payload_fields) >= {MIN_AGE_FIELD, MAX_AGE_FIELD, SEX_FIELD}
        assert report.warnings == (), report.warnings

    async def test_null_tolerance_keeps_a_large_share_of_the_real_corpus(self) -> None:
        """The NULL-safety argument, measured on real data instead of asserted.

        B1's census puts ``max_age_years`` at NULL for ~49.8% of trials and
        ``min_age_years`` at ~9.6%. So the strict filter is expected to discard
        roughly half the index relative to the NULL-safe one. The assertion is a
        loose inequality rather than a percentage, because the ingest is
        incomplete and resuming, and the exact share moves - but a 20% gap
        cannot be produced by anything other than NULL handling.
        """
        client, settings = await self.live_trials()
        try:
            total = (await client.get_collection(settings.collection)).points_count or 0
            null_safe = await client.count(
                settings.collection,
                count_filter=build_filter(AgeSexFilter(age=45, include_unknown=True)),
                exact=True,
            )
            strict = await client.count(
                settings.collection,
                count_filter=build_filter(AgeSexFilter(age=45, include_unknown=False)),
                exact=True,
            )
        finally:
            await client.close()

        assert strict.count < null_safe.count, (
            "if these are equal, the is_null/is_empty disjuncts are not doing "
            "anything and NULL-safety is not actually implemented"
        )
        discarded = (null_safe.count - strict.count) / max(null_safe.count, 1)
        assert discarded > 0.20, (
            f"strict filtering discards only {discarded:.1%} of the "
            f"{null_safe.count} trials a 45-year-old is possibly eligible for; "
            "B1's census says ~50% of trials have a NULL max_age_years, so a "
            "share this small suggests the age fields are not being parsed"
        )
        assert null_safe.count <= total
