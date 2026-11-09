"""``POST /search``: the frozen contract, over real HTTP.

The service is exercised through ``TestClient``, with the Qdrant client, the
encoders, the document store and the reranker replaced by stubs. What is under
test is the contract boundary: that a ``SearchRequest`` body is accepted exactly
as ``beacon_core.schemas`` defines it, that the response deserialises back into
a ``SearchResponse``, and that a malformed body is rejected rather than
half-interpreted.

``extra="forbid"`` on ``Contract`` is doing real work here. A caller who sends
``{"top_k": 20}`` meaning ``rerank_top_k`` gets a 422, instead of a silent 20 ->
the-default-anyway that would have been indistinguishable from success.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from beacon_core.schemas import SearchResponse
from beacon_core.settings import Settings
from retrieval.deps import provide_service, provide_settings
from retrieval.hydrate import MappingTrialTextStore, TrialText
from retrieval.main import create_app
from retrieval.search import HybridSearcher
from retrieval.service import RetrievalService
from retrieval.testing import (
    StubCrossEncoder,
    StubDenseEncoder,
    StubQdrantClient,
    StubSparseEncoder,
    stub_collection_info,
)

BRANCHES = [["NCT00000001", "NCT00000002"], ["NCT00000002", "NCT00000003"]]

PAYLOADS = {
    nct_id: {
        "nct_id": nct_id,
        "min_age_years": None,
        "max_age_years": None,
        "sex": "all",
        "phase": "Phase 2",
        "status": "Recruiting",
    }
    for nct_id in ("NCT00000001", "NCT00000002", "NCT00000003")
}

TEXTS = {
    nct_id: TrialText(
        nct_id=nct_id,
        title=f"A study of something in {nct_id}",
        conditions=["Non-Small Cell Lung Carcinoma"],
        interventions=["osimertinib"],
        overview_text=f"{nct_id}: non-small cell lung carcinoma, osimertinib.",
    )
    for nct_id in PAYLOADS
}

BODY: dict[str, Any] = {
    "queries": [
        {"rank": 1, "text": "metastatic EGFR-mutant non-small cell lung cancer", "facet": "diagnosis"},
        {"rank": 2, "text": "EGFR exon 19 deletion", "facet": "biomarker"},
    ],
    "filters": {"age": 52, "sex": "female", "include_unknown": True},
    "fuse_top_k": 100,
    "rerank_top_k": 20,
    "rrf_k": 60,
    "use_rerank": True,
}


def build_service(
    *,
    branches: list[list[str]] | None = None,
    info_kwargs: dict[str, Any] | None = None,
    exists: bool = True,
) -> RetrievalService:
    settings = Settings()
    client = StubQdrantClient(
        # The stub is strict about the request count, so the number of
        # canned branches has to match 2 x the queries in the body.
        BRANCHES * 2 if branches is None else branches,
        PAYLOADS,
        info=stub_collection_info(**(info_kwargs or {})),
        exists=exists,
    )
    return RetrievalService(
        searcher=HybridSearcher(
            client=client,
            settings=settings.qdrant,
            dense=StubDenseEncoder(dim=settings.qdrant.dense_dim),
            sparse=StubSparseEncoder(),
        ),
        text_store=MappingTrialTextStore(TEXTS),
        cross_encoder=StubCrossEncoder(scores={"osimertinib": 1.0}),
        settings=settings,
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    # A zero-argument lambda, not ``build_service`` itself: FastAPI inspects
    # an override's signature as a dependency and would try to populate
    # ``info_kwargs`` from the request body.
    app.dependency_overrides[provide_service] = lambda: build_service()
    app.dependency_overrides[provide_settings] = lambda: Settings()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestRoundTrip:
    def test_the_contract_body_is_accepted_as_written(self, client: TestClient) -> None:
        response = client.post("/search", json=BODY)

        assert response.status_code == 200, response.text

    def test_the_response_deserialises_back_into_the_contract(self, client: TestClient) -> None:
        """The actual round trip: whatever FastAPI serialised must validate as
        the same model the worker will parse it into."""
        payload = client.post("/search", json=BODY).json()

        parsed = SearchResponse.model_validate(payload)

        assert len(parsed.candidates) == 3
        assert parsed.total_before_rerank == 3
        assert parsed.timings_ms["total"] > 0

    def test_the_response_carries_no_fields_outside_the_contract(
        self, client: TestClient
    ) -> None:
        payload = client.post("/search", json=BODY).json()

        assert set(payload) == {"candidates", "total_before_rerank", "timings_ms"}
        assert set(payload["candidates"][0]) == {
            "nct_id",
            "title",
            "conditions",
            "interventions",
            "phase",
            "status",
            "min_age_years",
            "max_age_years",
            "sex",
            "dense_score",
            "sparse_score",
            "rrf_score",
            "rerank_score",
            "retrieval_rank",
        }

    def test_a_minimal_body_is_enough(self) -> None:
        """Only ``queries`` is required; everything else falls through to
        settings. This is what the worker actually sends."""
        app = create_app()
        app.dependency_overrides[provide_service] = lambda: build_service(branches=BRANCHES)
        app.dependency_overrides[provide_settings] = lambda: Settings()
        with TestClient(app) as test_client:
            response = test_client.post(
                "/search", json={"queries": [{"rank": 1, "text": "metastatic melanoma"}]}
            )

        assert response.status_code == 200, response.text
        assert SearchResponse.model_validate(response.json()).candidates

    def test_every_timing_stage_appears(self, client: TestClient) -> None:
        timings = client.post("/search", json=BODY).json()["timings_ms"]

        assert set(timings) == {"embed", "qdrant", "fuse", "hydrate", "rerank", "total"}


class TestRejectedBodies:
    def test_an_unknown_field_is_a_422_not_a_silent_default(self) -> None:
        """``extra="forbid"``. A caller who writes ``top_k`` meaning
        ``rerank_top_k`` must be told, not quietly given the default."""
        app = create_app()
        app.dependency_overrides[provide_service] = lambda: build_service()
        with TestClient(app) as test_client:
            response = test_client.post(
                "/search",
                json={"queries": [{"rank": 1, "text": "melanoma"}], "top_k": 20},
            )

        assert response.status_code == 422

    @pytest.mark.parametrize(
        ("body", "why"),
        [
            ({}, "queries is required"),
            ({"queries": []}, "min_length=1: a search with no query is a bug"),
            ({"queries": [{"rank": 0, "text": "m"}]}, "rank is 1-based"),
            ({"queries": [{"rank": 1, "text": ""}]}, "an empty query string"),
            (
                {"queries": [{"rank": 1, "text": "m"}], "filters": {"age": 200}},
                "age is bounded at 120",
            ),
            (
                {"queries": [{"rank": 1, "text": "m"}], "filters": {"sex": "other"}},
                "Sex is a closed vocabulary of male/female/all",
            ),
            (
                {"queries": [{"rank": 1, "text": "m"}], "rrf_k": 0},
                "rrf_k >= 1, or rank 60 divides by zero",
            ),
            (
                {"queries": [{"rank": 1, "text": "m"}], "rerank_top_k": 0},
                "asking for zero candidates is a bug",
            ),
        ],
    )
    def test_malformed_bodies_are_rejected(
        self, client: TestClient, body: dict[str, Any], why: str
    ) -> None:
        response = client.post("/search", json=body)

        assert response.status_code == 422, f"{why}: {response.text}"


class TestHealth:
    def test_a_healthy_collection_reports_its_declarations(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ok"
        assert body["sparse_modifier"] == "idf"
        assert body["dense_dim"] == 384
        assert "min_age_years" in body["indexed_payload_fields"]

    def test_a_collection_without_idf_is_unhealthy(self) -> None:
        """503, not a 200 with a warning field nobody reads: a process that
        would return quietly degraded results should not be routed traffic."""
        app = create_app()
        app.dependency_overrides[provide_service] = lambda: build_service(
            info_kwargs={"sparse_modifier": None}
        )
        app.dependency_overrides[provide_settings] = lambda: Settings()
        with TestClient(app) as test_client:
            response = test_client.get("/health")

        assert response.status_code == 503
        assert "idf" in response.text.lower()

    def test_a_missing_collection_is_unhealthy(self) -> None:
        app = create_app()
        app.dependency_overrides[provide_service] = lambda: build_service(exists=False)
        app.dependency_overrides[provide_settings] = lambda: Settings()
        with TestClient(app) as test_client:
            response = test_client.get("/health")

        assert response.status_code == 503
        assert "does not exist" in response.text


class TestImportSafety:
    def test_importing_the_app_needs_no_models_containers_or_api_key(self) -> None:
        """Asserted rather than assumed, because it is what keeps the fast unit
        job fast: ``create_app`` must not load 400 MB of ONNX weights, open a
        socket, or read an Anthropic key. This service never calls an LLM.
        """
        app = create_app()

        assert app.title == "Beacon retrieval"
        assert set(app.openapi()["paths"]) == {"/search", "/health"}
