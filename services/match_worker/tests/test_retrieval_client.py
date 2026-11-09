"""The retrieval seam, stubbed at HTTP.

``services/retrieval/**`` belongs to another agent and is not written yet, so
these tests assert against the frozen contract instead of against an
implementation: the request this worker sends, and its handling of what comes
back. Nothing here searches, fuses or filters - that is the retrieval
service's, and a "temporary" filter in this worker would be the overfiltering
bug hidden where nobody looks for it.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from beacon_core.schemas import PatientProfile, RankedQuery, SearchRequest, SearchResponse
from beacon_core.settings import RetrievalSettings
from match_worker._testing import search_response
from match_worker.retrieval_client import RetrievalClient, RetrievalError, build_search_request

QUERIES = [RankedQuery(rank=1, text="metastatic breast cancer", facet="diagnosis")]
PROFILE = PatientProfile(age=58, sex="female", conditions=["breast cancer"])


def test_request_carries_the_filters_and_keeps_unknowns_eligible() -> None:
    request = build_search_request(queries=QUERIES, profile=PROFILE, settings=RetrievalSettings())
    assert request.filters.age == 58
    assert request.filters.sex == "female"
    # A trial whose age bounds are absent must survive the filter: unknown
    # means "possibly eligible", and a wrongly excluded trial is invisible.
    assert request.filters.include_unknown is True
    assert request.rerank_top_k == 20
    assert request.rrf_k == 60


def test_filters_can_be_ablated_without_changing_the_request_shape() -> None:
    request = build_search_request(
        queries=QUERIES,
        profile=PROFILE,
        settings=RetrievalSettings(apply_age_sex_filters=False),
    )
    assert request.filters.age is None
    assert request.filters.sex is None
    assert request.filters.include_unknown is True


def test_a_profile_with_no_age_sends_no_age_filter() -> None:
    request = build_search_request(
        queries=QUERIES, profile=PatientProfile(), settings=RetrievalSettings()
    )
    assert request.filters.age is None


@respx.mock
async def test_posts_the_frozen_contract_and_parses_the_response() -> None:
    route = respx.post("http://retrieval.invalid/search").mock(
        return_value=httpx.Response(200, json=search_response().model_dump(mode="json"))
    )
    client = RetrievalClient(base_url="http://retrieval.invalid")

    request = build_search_request(queries=QUERIES, profile=PROFILE, settings=RetrievalSettings())
    response = await client.search(request)

    assert route.called
    sent = SearchRequest.model_validate_json(route.calls.last.request.content)
    assert sent == request
    assert isinstance(response, SearchResponse)
    assert [c.nct_id for c in response.candidates] == [
        "NCT04000001",
        "NCT04000002",
        "NCT04000003",
    ]


@respx.mock
async def test_a_trailing_slash_in_the_base_url_does_not_double_up() -> None:
    route = respx.post("http://retrieval.invalid/search").mock(
        return_value=httpx.Response(200, json=search_response().model_dump(mode="json"))
    )
    client = RetrievalClient(base_url="http://retrieval.invalid/")
    await client.search(
        build_search_request(queries=QUERIES, profile=PROFILE, settings=RetrievalSettings())
    )
    assert route.called


@respx.mock
async def test_a_server_error_becomes_a_retrieval_error() -> None:
    respx.post("http://retrieval.invalid/search").mock(return_value=httpx.Response(503))
    client = RetrievalClient(base_url="http://retrieval.invalid")
    with pytest.raises(RetrievalError, match="failed"):
        await client.search(
            build_search_request(queries=QUERIES, profile=PROFILE, settings=RetrievalSettings())
        )


@respx.mock
async def test_a_200_that_is_not_a_search_response_is_a_contract_violation() -> None:
    respx.post("http://retrieval.invalid/search").mock(
        return_value=httpx.Response(200, json={"trials": ["NCT01"]})
    )
    client = RetrievalClient(base_url="http://retrieval.invalid")
    with pytest.raises(RetrievalError, match="unparseable"):
        await client.search(
            build_search_request(queries=QUERIES, profile=PROFILE, settings=RetrievalSettings())
        )
