"""The dual write's two load-bearing details: IDF, and deterministic point ids.

Both failures these tests exist for are silent. A collection built without
``modifier: IDF`` serves every query without complaint and merely ranks badly;
a random point id doubles the collection on every run and raises nothing. So
they are checked here against a recording stub, and again against the live
stack in test_ingest_live.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from beacon_core.settings import QdrantSettings
from ingestion.embedding import SparseValues
from ingestion.store import (
    PAYLOAD_INDEXES,
    QdrantIndex,
    SparseModifierMismatch,
    TrialPoint,
    point_id,
)


class RecordingClient:
    """Enough of ``AsyncQdrantClient`` to observe how the collection is built."""

    def __init__(self, *, exists: bool = False, modifier: models.Modifier | None = None) -> None:
        self._exists = exists
        self._modifier = modifier
        self.created: dict[str, Any] | None = None
        self.indexes: list[str] = []
        self.deleted = False
        self.upserted: list[models.PointStruct] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return self._exists

    async def delete_collection(self, collection_name: str) -> None:
        self.deleted = True
        self._exists = False

    async def create_collection(self, **kwargs: Any) -> None:
        self.created = kwargs
        self._exists = True
        self._modifier = kwargs["sparse_vectors_config"]["sparse"].modifier

    async def get_collection(self, collection_name: str) -> Any:
        # Duck-typed rather than a real ``models.CollectionInfo``: that model
        # requires a dozen unrelated sub-configs, and pinning their fields here
        # would make this stub break on every qdrant-client release for
        # reasons that have nothing to do with ingestion.
        sparse = (
            {"sparse": models.SparseVectorParams(modifier=self._modifier)}
            if self._modifier is not None
            else {}
        )
        params = SimpleNamespace(
            sparse_vectors=sparse,
            vectors={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        )
        return SimpleNamespace(
            config=SimpleNamespace(params=params),
            payload_schema=dict.fromkeys(self.indexes),
        )

    async def create_payload_index(self, *, field_name: str, **kwargs: Any) -> None:
        self.indexes.append(field_name)

    async def upsert(self, *, collection_name: str, points: Any, wait: bool = True) -> None:
        self.upserted.extend(points)

    async def close(self) -> None:
        return None


def _index(client: RecordingClient, **overrides: Any) -> QdrantIndex:
    return QdrantIndex(QdrantSettings(**overrides), client=client)  # type: ignore[arg-type]


# --- the IDF modifier -------------------------------------------------------


async def test_collection_is_created_with_the_idf_modifier() -> None:
    client = RecordingClient()
    await _index(client).ensure_collection()
    assert client.created is not None
    sparse = client.created["sparse_vectors_config"]["sparse"]
    assert sparse.modifier == models.Modifier.IDF, (
        "BM25 without corpus IDF scores a rare term like a common one and reports no error at all"
    )


async def test_dense_vector_is_384_dim_cosine_and_named_dense() -> None:
    client = RecordingClient()
    await _index(client).ensure_collection()
    assert client.created is not None
    dense = client.created["vectors_config"]["dense"]
    assert dense.size == 384
    assert dense.distance == models.Distance.COSINE


async def test_payload_indexes_cover_the_age_and_sex_filters() -> None:
    client = RecordingClient()
    await _index(client).ensure_collection()
    assert set(client.indexes) == set(PAYLOAD_INDEXES)
    assert {"min_age_years", "max_age_years", "sex"} <= set(client.indexes)
    assert PAYLOAD_INDEXES["min_age_years"] == models.PayloadSchemaType.FLOAT
    assert PAYLOAD_INDEXES["sex"] == models.PayloadSchemaType.KEYWORD


async def test_an_existing_collection_without_idf_is_refused() -> None:
    """The silent-degradation guard: refuse rather than write into it."""
    client = RecordingClient(exists=True, modifier=models.Modifier.NONE)
    with pytest.raises(SparseModifierMismatch, match=r"IDF|idf"):
        await _index(client).ensure_collection()
    assert client.upserted == []


async def test_an_existing_collection_with_idf_is_accepted() -> None:
    client = RecordingClient(exists=True, modifier=models.Modifier.IDF)
    await _index(client).ensure_collection()
    assert client.created is None, "an existing, correct collection is left alone"


async def test_an_existing_collection_without_the_sparse_field_is_refused() -> None:
    client = RecordingClient(exists=True, modifier=None)
    with pytest.raises(SparseModifierMismatch, match="sparse"):
        await _index(client).ensure_collection()


async def test_recreate_drops_the_collection_first() -> None:
    client = RecordingClient(exists=True, modifier=models.Modifier.NONE)
    await _index(client).ensure_collection(recreate=True)
    assert client.deleted is True
    assert client.created is not None


async def test_idf_can_be_turned_off_only_by_configuration() -> None:
    """``settings.qdrant.sparse_idf_modifier`` is honoured, not hardcoded."""
    client = RecordingClient()
    await _index(client, sparse_idf_modifier=False).ensure_collection()
    assert client.created is not None
    assert client.created["sparse_vectors_config"]["sparse"].modifier == models.Modifier.NONE


# --- idempotency ------------------------------------------------------------


def test_point_id_is_derived_from_the_nct_id() -> None:
    assert point_id("NCT00000105") == point_id("NCT00000105")
    assert point_id("NCT00000105") != point_id("NCT00000102")
    # Pinned: changing the namespace orphans every point already indexed.
    assert point_id("NCT00000105") == "ea97db9a-9d94-5ccd-bd38-6d14f48d2bec"


async def test_upserting_the_same_trial_twice_targets_one_point() -> None:
    client = RecordingClient()
    index = _index(client)
    await index.ensure_collection()
    point = TrialPoint(
        nct_id="NCT00000105",
        dense=[0.0] * 384,
        sparse=SparseValues(indices=[1, 2], values=[0.5, 0.25]),
        payload={"nct_id": "NCT00000105"},
    )
    await index.upsert([point])
    await index.upsert([point])
    assert len({p.id for p in client.upserted}) == 1, "a re-run must overwrite, not append"


async def test_upsert_carries_both_named_vectors() -> None:
    client = RecordingClient()
    index = _index(client)
    await index.ensure_collection()
    await index.upsert(
        [
            TrialPoint(
                nct_id="NCT1",
                dense=[0.1] * 384,
                sparse=SparseValues(indices=[7], values=[1.5]),
                payload={"nct_id": "NCT1"},
            )
        ]
    )
    vector = client.upserted[0].vector
    assert isinstance(vector, dict)
    assert set(vector) == {"dense", "sparse"}
    dense = vector["dense"]
    sparse = vector["sparse"]
    assert isinstance(dense, list)
    assert len(dense) == 384
    assert isinstance(sparse, models.SparseVector)
    assert sparse.indices == [7]


async def test_upserting_nothing_is_a_no_op() -> None:
    client = RecordingClient()
    index = _index(client)
    await index.ensure_collection()
    assert await index.upsert([]) == 0


# The live-stack counterparts of these checks are in test_ingest_live.py,
# where the skip fixtures live: a stub can only prove the client sent
# modifier: IDF, not that the server kept it.
