"""The collection verifier: does it notice the things that fail silently?

Every case here is a collection that answers searches perfectly happily while
being wrong. The missing IDF modifier is the headline: BM25 without
corpus-level inverse document frequency still returns a ranking, still returns
the right *number* of results, and has lost the rare-term weighting that is the
only reason a sparse index is in this system next to a dense one.

These run against a stub, which is why they can cover the failure cases at all -
a live Qdrant would have to be deliberately misconfigured four different ways.
``test_integration_qdrant.py`` separately asserts the real collection passes.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from beacon_core.settings import Settings
from retrieval.collection import CollectionVerificationError, verify_collection
from retrieval.testing import StubQdrantClient, stub_collection_info


def qdrant_settings() -> Settings:
    return Settings()


class TestHealthyCollection:
    async def test_a_correct_collection_has_no_problems(self) -> None:
        settings = qdrant_settings()
        client = StubQdrantClient(info=stub_collection_info())

        report = await verify_collection(client, settings.qdrant)

        assert report.ok
        assert report.problems == ()
        assert report.warnings == ()
        assert report.sparse_modifier == "idf"
        assert report.dense_dim == settings.qdrant.dense_dim
        assert report.points == 375_580


class TestTheIdfModifier:
    @pytest.mark.parametrize("modifier", [None, models.Modifier.NONE])
    async def test_a_missing_idf_modifier_is_a_hard_problem(
        self, modifier: models.Modifier | None
    ) -> None:
        """The silent one. No error, correct result counts, degraded ranking."""
        client = StubQdrantClient(info=stub_collection_info(sparse_modifier=modifier))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert not report.ok
        assert any("idf" in problem.lower() for problem in report.problems)

    async def test_the_message_says_what_to_do_about_it(self) -> None:
        """A problem report that does not name the fix costs an hour of reading."""
        client = StubQdrantClient(info=stub_collection_info(sparse_modifier=None))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert any("recreate the collection" in problem for problem in report.problems)

    async def test_declaring_idf_when_the_setting_is_off_is_only_a_warning(self) -> None:
        """Results are correct and better than configured, so it does not block
        traffic - but the setting is being ignored and somebody should know."""
        settings = qdrant_settings()
        settings.qdrant.sparse_idf_modifier = False
        client = StubQdrantClient(info=stub_collection_info())

        report = await verify_collection(client, settings.qdrant)

        assert report.ok
        assert any("ignored" in warning for warning in report.warnings)


class TestVectorFields:
    async def test_a_missing_sparse_field_is_a_problem(self) -> None:
        client = StubQdrantClient(info=stub_collection_info(sparse_name=None))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert not report.ok
        assert any("sparse vector field" in problem for problem in report.problems)

    async def test_a_wrong_dense_dimension_is_a_problem(self) -> None:
        """768 against a 384-dim query is the benign case - Qdrant rejects it.
        The check exists for the malignant neighbour: a collection rebuilt with
        a *different* model at the same dimension, which is why the message
        names the configured model as well as the number."""
        client = StubQdrantClient(info=stub_collection_info(dense_dim=768))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert not report.ok
        assert any("768" in problem for problem in report.problems)
        assert any("bge-small" in problem for problem in report.problems)

    async def test_an_unnamed_single_vector_collection_is_a_problem(self) -> None:
        """Hybrid search needs both vectors on one point, or a dense hit and a
        sparse hit cannot be known to be the same trial and RRF has nothing to
        fuse by."""
        client = StubQdrantClient(info=stub_collection_info(unnamed_vector=True))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert not report.ok
        assert any("single unnamed vector" in problem for problem in report.problems)

    async def test_a_missing_dense_field_names_what_was_found_instead(self) -> None:
        client = StubQdrantClient(info=stub_collection_info(dense_name="embedding"))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert not report.ok
        assert any("embedding" in problem for problem in report.problems)


class TestPayloadIndexes:
    async def test_missing_payload_indexes_are_a_warning_not_a_problem(self) -> None:
        """A latency defect, not a correctness one: Qdrant still answers a
        filtered query correctly by scanning payloads. It is a warning because
        blocking traffic over it would be worse than the slowdown - but it does
        mean the "filter inside the traversal" design has collapsed into a full
        scan with extra steps, so it is reported loudly."""
        client = StubQdrantClient(info=stub_collection_info(payload_fields=("nct_id",)))

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert report.ok
        assert len(report.warnings) == 1
        assert "min_age_years" in report.warnings[0]
        assert "sex" in report.warnings[0]

    async def test_a_fully_indexed_collection_warns_about_nothing(self) -> None:
        client = StubQdrantClient(info=stub_collection_info())

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert report.indexed_payload_fields == ("max_age_years", "min_age_years", "nct_id", "sex")
        assert report.warnings == ()


class TestMissingCollection:
    async def test_raises_rather_than_reporting(self) -> None:
        """Nothing further can be inspected, and the caller needs a 503 with a
        message naming the ingest command rather than a report full of
        consequential problems."""
        client = StubQdrantClient(exists=False)

        with pytest.raises(CollectionVerificationError, match="does not exist"):
            await verify_collection(client, qdrant_settings().qdrant)

    async def test_the_message_names_the_ingest_command(self) -> None:
        client = StubQdrantClient(exists=False)

        with pytest.raises(CollectionVerificationError, match="ingestion.run"):
            await verify_collection(client, qdrant_settings().qdrant)


class TestEveryProblemIsCollected:
    async def test_all_discrepancies_are_reported_at_once(self) -> None:
        """Told only the first one, you fix it, restart, and are told the next.
        One report, one rebuild."""
        client = StubQdrantClient(
            info=stub_collection_info(
                dense_dim=768,
                sparse_modifier=None,
                payload_fields=("nct_id",),
            )
        )

        report = await verify_collection(client, qdrant_settings().qdrant)

        assert len(report.problems) == 2
        assert len(report.warnings) == 1
