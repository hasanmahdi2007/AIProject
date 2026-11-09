"""The whole pipeline, with every external thing stubbed.

No containers, no ONNX weights, no network. What is under test here is the
*wiring*: that the filter reaches both branches of every query, that the two
levels of RRF are composed in the right order, that the reranker is given the
fused set and its output reorders the answer, and that the config knobs are
actually read. Relevance is not under test - the stub client returns canned
rankings - which is deliberate, because relevance is not a property this code
can be unit-tested for.
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client import models

from beacon_core.schemas import AgeSexFilter, RankedQuery, SearchRequest
from beacon_core.settings import Settings
from retrieval.hydrate import MappingTrialTextStore, TrialText
from retrieval.search import HybridSearcher
from retrieval.service import RetrievalService, resolve_params
from retrieval.testing import StubCrossEncoder, StubDenseEncoder, StubQdrantClient, StubSparseEncoder

# The hand-computed case from test_fusion, now driven through the service:
# one query, dense = [A, B, C], sparse = [C, A, D] -> A, C, B, D, with the
# letters spelled NCTA..NCTD because the contract requires >= 3 characters.
ONE_QUERY_BRANCHES = [["NCTA", "NCTB", "NCTC"], ["NCTC", "NCTA", "NCTD"]]

PAYLOADS = {
    "NCTA": {"nct_id": "NCTA", "min_age_years": 18.0, "max_age_years": None, "sex": "all",
          "phase": "Phase 3", "status": "Recruiting"},
    "NCTB": {"nct_id": "NCTB", "min_age_years": None, "max_age_years": None, "sex": None,
          "phase": None, "status": None},
    "NCTC": {"nct_id": "NCTC", "min_age_years": 65.0, "max_age_years": 80.0, "sex": "male",
          "phase": "Phase 1", "status": "Completed"},
    "NCTD": {"nct_id": "NCTD", "min_age_years": 0.0, "max_age_years": 17.0, "sex": "female",
          "phase": "Phase 2", "status": "Recruiting"},
}

TEXTS = {
    nct_id: TrialText(
        nct_id=nct_id,
        title=f"Trial {nct_id}",
        conditions=[f"Condition {nct_id}"],
        interventions=[f"Drug {nct_id}"],
        overview_text=f"Trial {nct_id} studies melanoma." if nct_id == "NCTD" else f"Trial {nct_id}.",
    )
    for nct_id in PAYLOADS
}


def build_service(
    branches: list[list[str]],
    *,
    settings: Settings | None = None,
    cross_encoder: StubCrossEncoder | None = None,
    texts: dict[str, TrialText] | None = None,
) -> tuple[RetrievalService, StubQdrantClient]:
    cfg = settings or Settings()
    client = StubQdrantClient(branches, PAYLOADS)
    searcher = HybridSearcher(
        client=client,
        settings=cfg.qdrant,
        dense=StubDenseEncoder(dim=cfg.qdrant.dense_dim),
        sparse=StubSparseEncoder(),
    )
    service = RetrievalService(
        searcher=searcher,
        text_store=MappingTrialTextStore(TEXTS if texts is None else texts),
        cross_encoder=cross_encoder,
        settings=cfg,
    )
    return service, client


def one_query(
    *,
    filters: AgeSexFilter | None = None,
    fuse_top_k: int | None = None,
    rerank_top_k: int | None = None,
    use_rerank: bool | None = None,
) -> SearchRequest:
    """Build a request through ``model_validate``, so that only the arguments
    given here land in ``model_fields_set`` - which is what
    :func:`resolve_params` reads to decide whether settings or the caller wins.
    Constructing the model directly would mark every passed field as explicit
    and make the settings fall-through untestable.
    """
    body: dict[str, Any] = {"queries": [{"rank": 1, "text": "metastatic melanoma"}]}
    if filters is not None:
        body["filters"] = filters.model_dump()
    if fuse_top_k is not None:
        body["fuse_top_k"] = fuse_top_k
    if rerank_top_k is not None:
        body["rerank_top_k"] = rerank_top_k
    if use_rerank is not None:
        body["use_rerank"] = use_rerank
    return SearchRequest.model_validate(body)


class TestFilterReachesEverySearch:
    """The correctness item: the filter is on the request, not applied after."""

    async def test_both_branches_of_the_query_carry_the_filter(self) -> None:
        service, client = build_service(ONE_QUERY_BRANCHES)

        await service.search(one_query(filters=AgeSexFilter(age=45, sex="female")))

        assert len(client.requests) == 2, "one dense and one sparse request"
        for request in client.requests:
            assert request.filter is not None, (
                "a branch searched without the filter contributes ineligible "
                "trials that are indistinguishable from eligible ones"
            )
            assert request.filter.must is not None
            assert len(request.filter.must) == 3

    async def test_every_branch_of_every_query_carries_it(self) -> None:
        service, client = build_service(
            [["NCTA"], ["NCTB"], ["NCTC"], ["NCTD"], ["NCTA"], ["NCTB"]],
        )
        request = SearchRequest(
            queries=[
                RankedQuery(rank=1, text="melanoma"),
                RankedQuery(rank=2, text="BRAF V600E"),
                RankedQuery(rank=3, text="prior ipilimumab"),
            ],
            filters=AgeSexFilter(age=45),
        )

        await service.search(request)

        assert len(client.requests) == 6, "2 branches x 3 queries, one round trip"
        assert all(r.filter is not None for r in client.requests)

    async def test_the_two_branches_use_the_two_named_vectors(self) -> None:
        """A transposed ``using`` would search the sparse field with a dense
        vector, which Qdrant rejects - but searching dense twice would not
        raise, and would silently delete the BM25 half of the system."""
        settings = Settings()
        service, client = build_service(ONE_QUERY_BRANCHES, settings=settings)

        await service.search(one_query())

        assert client.requests[0].using == settings.qdrant.dense_vector_name
        assert client.requests[1].using == settings.qdrant.sparse_vector_name
        assert isinstance(client.requests[0].query, list)
        assert isinstance(client.requests[1].query, models.SparseVector)

    async def test_no_filter_is_sent_when_the_request_constrains_nothing(self) -> None:
        service, client = build_service(ONE_QUERY_BRANCHES)

        await service.search(one_query())

        assert all(r.filter is None for r in client.requests)

    async def test_each_branch_is_retrieved_to_the_fused_depth(self) -> None:
        """Shallower branches would make RRF close to a no-op, since a document
        absent from a short list contributes nothing to its score."""
        service, client = build_service(ONE_QUERY_BRANCHES)

        await service.search(one_query(fuse_top_k=37))

        assert [r.limit for r in client.requests] == [37, 37]


class TestFusionThroughTheService:
    async def test_the_hand_computed_order_survives_the_pipeline(self) -> None:
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))

        assert [c.nct_id for c in response.candidates] == ["NCTA", "NCTC", "NCTB", "NCTD"]

    async def test_rrf_score_is_attached_and_descending(self) -> None:
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))

        scores = [c.rrf_score for c in response.candidates]
        assert all(s is not None for s in scores)
        present = [s for s in scores if s is not None]
        assert present == sorted(present, reverse=True)

    async def test_raw_branch_scores_are_reported_but_not_fused(self) -> None:
        """``dense_score``/``sparse_score`` are diagnostics. The stub gives the
        two branches different scales on purpose: a trial seen only by the
        sparse branch has no dense score, and the ordering still comes from the
        ranks."""
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))
        by_id = {c.nct_id: c for c in response.candidates}

        assert by_id["NCTD"].dense_score is None, "D was never returned by the dense branch"
        assert by_id["NCTD"].sparse_score is not None
        assert by_id["NCTB"].sparse_score is None
        assert by_id["NCTA"].dense_score is not None and by_id["NCTA"].sparse_score is not None

    async def test_cross_query_weighting_decides_a_symmetric_case(self) -> None:
        """The hand-computed cross-query case, end to end.

        Query 1 (rank 1) ranks A then B; query 2 (rank 2) ranks B then A. Each
        query's two branches agree, so level-1 fusion leaves [A, B] and [B, A].
        Level 2 weights them 1.0 and 0.5, giving A = 1/61 + 0.5/62 and
        B = 1/62 + 0.5/61, so A wins. Unweighted, this would be a tie broken by
        ID - which would also put A first, so the test checks the *scores* and
        not only the order.
        """
        service, _ = build_service([["NCTA", "NCTB"], ["NCTA", "NCTB"], ["NCTB", "NCTA"], ["NCTB", "NCTA"]])
        request = SearchRequest(
            queries=[
                RankedQuery(rank=1, text="primary diagnosis"),
                RankedQuery(rank=2, text="secondary facet"),
            ],
            use_rerank=False,
        )

        response = await service.search(request)
        by_id = {c.nct_id: c.rrf_score for c in response.candidates}

        assert by_id["NCTA"] == pytest.approx(1.0 / 61 + 0.5 / 62)
        assert by_id["NCTB"] == pytest.approx(1.0 / 62 + 0.5 / 61)
        assert [c.nct_id for c in response.candidates] == ["NCTA", "NCTB"]


class TestRerank:
    async def test_the_cross_encoder_reorders_the_fused_set(self) -> None:
        """D is last after RRF (one third-place sparse hit) and first after
        rerank, because it is the only trial whose text is about melanoma. This
        is the funnel working: the cheap stage got it into the candidate set at
        all, the accurate stage moved it to the top."""
        reranker = StubCrossEncoder(scores={"melanoma": 9.0})
        service, _ = build_service(ONE_QUERY_BRANCHES, cross_encoder=reranker)

        response = await service.search(one_query(use_rerank=True))

        assert [c.nct_id for c in response.candidates][0] == "NCTD"
        assert response.candidates[0].rerank_score == 9.0

    async def test_it_sees_the_whole_fused_set_not_just_the_answer(self) -> None:
        """``fuse_top_k`` pairs in, ``rerank_top_k`` out. Reranking only the
        top 20 would make the stage pointless: it could reorder them but never
        promote candidate 21, which is the entire value it adds."""
        reranker = StubCrossEncoder(scores={})
        service, _ = build_service(ONE_QUERY_BRANCHES, cross_encoder=reranker)

        await service.search(one_query(rerank_top_k=2))

        assert reranker.calls == [("metastatic melanoma", 4)]

    async def test_it_is_scored_against_the_rank_one_query_only(self) -> None:
        """N-times the cost for a question the first stage already answered.
        Documented in ``RetrievalService._rerank``; asserted here so a change of
        mind has to be deliberate."""
        reranker = StubCrossEncoder(scores={})
        service, _ = build_service(
            [["NCTA", "NCTB"], ["NCTA", "NCTB"], ["NCTB", "NCTA"], ["NCTB", "NCTA"]], cross_encoder=reranker
        )
        request = SearchRequest(
            queries=[
                RankedQuery(rank=2, text="secondary facet"),
                RankedQuery(rank=1, text="primary diagnosis"),
            ],
        )

        await service.search(request)

        assert [query for query, _ in reranker.calls] == ["primary diagnosis"]

    async def test_use_rerank_false_keeps_the_rrf_order_and_scores_nothing(self) -> None:
        """The ablation row: same candidate set, same k, ordering only."""
        reranker = StubCrossEncoder(scores={"melanoma": 9.0})
        service, _ = build_service(ONE_QUERY_BRANCHES, cross_encoder=reranker)

        response = await service.search(one_query(use_rerank=False))

        assert [c.nct_id for c in response.candidates] == ["NCTA", "NCTC", "NCTB", "NCTD"]
        assert reranker.calls == []
        assert all(c.rerank_score is None for c in response.candidates)

    async def test_a_missing_cross_encoder_degrades_to_the_rrf_order(self) -> None:
        """Not the same as the ablation - it is a missing capability - but it
        must not 500, because a retrieval answer without the rerank is still a
        usable answer."""
        service, _ = build_service(ONE_QUERY_BRANCHES, cross_encoder=None)

        response = await service.search(one_query(use_rerank=True))

        assert [c.nct_id for c in response.candidates] == ["NCTA", "NCTC", "NCTB", "NCTD"]


class TestResponseShape:
    async def test_candidates_are_truncated_to_rerank_top_k(self) -> None:
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(rerank_top_k=2, use_rerank=False))

        assert len(response.candidates) == 2
        assert response.total_before_rerank == 4, "the fused set before truncation"

    async def test_retrieval_rank_is_one_based_and_dense(self) -> None:
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))

        assert [c.retrieval_rank for c in response.candidates] == [1, 2, 3, 4]

    async def test_every_stage_is_timed(self) -> None:
        """``timings_ms`` is the cost story. A stage missing from it is a stage
        nobody can argue about."""
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query())

        assert set(response.timings_ms) == {
            "embed",
            "qdrant",
            "fuse",
            "hydrate",
            "rerank",
            "total",
        }
        assert all(value >= 0.0 for value in response.timings_ms.values())

    async def test_typed_bounds_come_from_the_payload_the_filter_saw(self) -> None:
        """The rule checker re-checks age numerically, so the candidate has to
        carry the same bounds the filter was evaluated against."""
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))
        by_id = {c.nct_id: c for c in response.candidates}

        assert by_id["NCTC"].min_age_years == 65.0
        assert by_id["NCTC"].max_age_years == 80.0
        assert by_id["NCTC"].sex == "male"
        assert by_id["NCTC"].phase == "Phase 1"

    async def test_an_unknown_bound_stays_unknown_on_the_candidate(self) -> None:
        """``None`` must not be coerced to 0 or 999 anywhere on the way out."""
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))
        by_id = {c.nct_id: c for c in response.candidates}

        assert by_id["NCTB"].min_age_years is None
        assert by_id["NCTB"].max_age_years is None
        assert by_id["NCTB"].sex is None

    async def test_text_fields_are_hydrated_from_the_document_store(self) -> None:
        service, _ = build_service(ONE_QUERY_BRANCHES)

        response = await service.search(one_query(use_rerank=False))

        assert response.candidates[0].title == "Trial NCTA"
        assert response.candidates[0].conditions == ["Condition NCTA"]
        assert response.candidates[0].interventions == ["Drug NCTA"]

    async def test_a_candidate_missing_from_postgres_survives_with_no_text(self) -> None:
        """A dual-write divergence must not silently delete a trial. It loses
        its title and has nothing for the reranker to read; it stays a
        candidate."""
        service, _ = build_service(ONE_QUERY_BRANCHES, texts={})

        response = await service.search(one_query(use_rerank=False))

        assert [c.nct_id for c in response.candidates] == ["NCTA", "NCTC", "NCTB", "NCTD"]
        assert response.candidates[0].title == ""

    async def test_a_point_with_no_nct_id_is_dropped_rather_than_anonymous(self) -> None:
        """Every later stage keys on the NCT ID and the point ID is a one-way
        uuid5 of it, so a payload without one is unusable rather than partial."""
        client = StubQdrantClient([["NCTA", "NCTB"], ["NCTA"]], {"NCTB": {}})
        cfg = Settings()
        service = RetrievalService(
            searcher=HybridSearcher(
                client=client,
                settings=cfg.qdrant,
                dense=StubDenseEncoder(dim=cfg.qdrant.dense_dim),
                sparse=StubSparseEncoder(),
            ),
            text_store=MappingTrialTextStore(TEXTS),
            cross_encoder=None,
            settings=cfg,
        )

        response = await service.search(one_query())

        assert [c.nct_id for c in response.candidates] == ["NCTA"]


class TestResolveParams:
    """Settings must be authoritative, or half the ablation matrix is unreachable."""

    def test_an_omitted_field_falls_through_to_settings(self) -> None:
        settings = Settings()
        settings.retrieval.rrf_k = 30
        settings.retrieval.fuse_top_k = 50
        request = SearchRequest.model_validate(
            {"queries": [{"rank": 1, "text": "melanoma"}]},
        )

        params = resolve_params(request, settings)

        assert params.rrf_k == 30
        assert params.fuse_top_k == 50

    def test_an_explicit_field_wins_even_when_it_equals_the_schema_default(self) -> None:
        """The case that makes ``model_fields_set`` necessary. The contract's
        default is 60; a caller who *sends* 60 while config says 30 means 60."""
        settings = Settings()
        settings.retrieval.rrf_k = 30
        request = SearchRequest.model_validate(
            {"queries": [{"rank": 1, "text": "melanoma"}], "rrf_k": 60},
        )

        assert resolve_params(request, settings).rrf_k == 60

    def test_apply_age_sex_filters_comes_from_settings_only(self) -> None:
        settings = Settings()
        settings.retrieval.apply_age_sex_filters = False
        request = SearchRequest.model_validate({"queries": [{"rank": 1, "text": "m"}]})

        assert resolve_params(request, settings).apply_filters is False

    def test_use_rerank_follows_settings_when_unspecified(self) -> None:
        settings = Settings()
        settings.retrieval.use_rerank = False
        request = SearchRequest.model_validate({"queries": [{"rank": 1, "text": "m"}]})

        assert resolve_params(request, settings).use_rerank is False

    async def test_the_rrf_k_setting_actually_changes_the_scores(self) -> None:
        """End-to-end proof that the k ablation row is wired up, not just
        resolved: the same rankings fused at k=30 give the k=30 arithmetic."""
        settings = Settings()
        settings.retrieval.rrf_k = 30
        service, _ = build_service(ONE_QUERY_BRANCHES, settings=settings)
        request = SearchRequest.model_validate(
            {"queries": [{"rank": 1, "text": "melanoma"}], "use_rerank": False}
        )

        response = await service.search(request)
        by_id = {c.nct_id: c.rrf_score for c in response.candidates}

        assert by_id["NCTA"] == pytest.approx(1 / 31)
        assert by_id["NCTC"] == pytest.approx(1 / 32)

    async def test_the_filter_ablation_actually_removes_the_filter(self) -> None:
        settings = Settings()
        settings.retrieval.apply_age_sex_filters = False
        service, client = build_service(ONE_QUERY_BRANCHES, settings=settings)

        await service.search(one_query(filters=AgeSexFilter(age=45, sex="female")))

        assert all(r.filter is None for r in client.requests)
