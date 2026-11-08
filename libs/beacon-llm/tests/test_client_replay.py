"""The client, end to end, against the recorded fixtures - with the network off.

Every test in this module runs with the socket layer blocked and with no API
key in the environment. That is a stronger proof than mocking the HTTP client:
it catches *any* transport, including a code path that constructed its own
client, and it is the state CI runs in.

A note on why this is not done with ``respx``, which the brief suggested:
anthropic 2.6.0 is built on ``httpx2``, not ``httpx``. ``respx`` 0.23 patches
``httpx``, so it would install cleanly, intercept nothing, and report that no
request was made - passing for the wrong reason, which is the exact failure
mode this file exists to rule out. Blocking the socket is transport-agnostic
and cannot pass vacuously: :func:`test_the_network_block_itself_works` proves
the block bites.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from anthropic.types import Message, Usage
from pydantic import SecretStr

from beacon_core.models import LlmCall
from beacon_core.schemas import CriterionJudgment
from beacon_core.settings import Settings
from beacon_llm import (
    AsyncLlmClient,
    FixtureMissingError,
    FixtureMode,
    FixtureStore,
    LlmClient,
    PromptTooLargeError,
    RefusalError,
    StructuredOutputError,
    build_cached_prompt,
)
from beacon_llm._testing import (
    COMMITTED_FIXTURES,
    RULE_CHECK_INSTRUCTIONS,
    SAMPLE_NOTE,
    SAMPLE_TRIAL_RULES,
    SECOND_TRIAL_RULES,
)


class NetworkAccessDuringReplay(AssertionError):
    """Raised if anything in this module tries to open a connection."""


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
"""Addresses the block lets through.

Loopback has to stay open: on Windows, ``asyncio``'s proactor event loop builds
its own self-pipe out of a 127.0.0.1 socket pair, so blocking loopback stops the
async tests from starting rather than stopping them from calling out. Nothing
reachable only over loopback is the Anthropic API, so the proof is unaffected.
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound sockets for the duration of each test.

    ``getaddrinfo`` as well as ``connect``: a DNS lookup is already a packet on
    the wire, and the SDK resolves before it connects.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def check(host: object) -> None:
        if str(host) not in LOOPBACK_HOSTS:
            raise NetworkAccessDuringReplay(
                f"replay mode attempted to reach {host!r}; it must read fixtures only"
            )

    def host_of(address: object) -> object:
        return address[0] if isinstance(address, tuple) and address else address

    def connect(self: socket.socket, address: object) -> None:
        check(host_of(address))
        real_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        check(host_of(address))
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    def create_connection(address: object, *args: object, **kwargs: object) -> socket.socket:
        check(host_of(address))
        return real_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        check(host)
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def client() -> LlmClient:
    """A replay client with no credentials, pointed at the committed fixtures."""
    settings = Settings(llm_fixtures_dir=COMMITTED_FIXTURES)
    settings.anthropic.api_key = SecretStr("")
    return LlmClient(settings=settings, mode=FixtureMode.REPLAY)


async def test_the_async_client_replays_identically() -> None:
    """The async path is the same logic with an ``await`` in front.

    The worker (agent C1) is a LangGraph graph and the gateway is FastAPI, so
    both will use :class:`AsyncLlmClient`. It shares the plan, guard,
    validation, pricing and fixture code with the sync client; this test is
    what keeps "shares" true, by asserting the two produce the same result from
    the same fixture.
    """
    settings = Settings(llm_fixtures_dir=COMMITTED_FIXTURES)
    settings.anthropic.api_key = SecretStr("")
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )

    sync_result = LlmClient(settings=settings, mode=FixtureMode.REPLAY).call(
        prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5", effort="low"
    )
    async_result = await AsyncLlmClient(settings=settings, mode=FixtureMode.REPLAY).call(
        prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5", effort="low"
    )

    assert async_result.parsed == sync_result.parsed
    assert async_result.cost == sync_result.cost
    assert async_result.prompt_hash == sync_result.prompt_hash
    assert async_result.served_from_fixture


async def test_the_async_client_also_enforces_the_token_guard() -> None:
    """The guard is not something only the sync path bothers with."""
    settings = Settings(llm_fixtures_dir=COMMITTED_FIXTURES)
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable="Criterion text. " * 30_000,
    )
    with pytest.raises(PromptTooLargeError):
        await AsyncLlmClient(settings=settings, mode=FixtureMode.REPLAY).call(
            prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5"
        )


def test_the_network_block_itself_works() -> None:
    """Guards against this module passing vacuously.

    If the block ever stopped biting, every other test here would still pass
    while silently making real calls. This one fails first.
    """
    with pytest.raises(NetworkAccessDuringReplay):
        socket.getaddrinfo("api.anthropic.com", 443)
    with pytest.raises(NetworkAccessDuringReplay):
        socket.create_connection(("api.anthropic.com", 443))


def test_replay_returns_a_validated_criterion_judgment(client: LlmClient) -> None:
    """The recorded answer comes back as a ``CriterionJudgment``, not as text.

    The fixture was recorded from a real Haiku 5.5 call made with
    ``output_config.format``, so this also demonstrates that the structured
    output path produced schema-valid JSON.
    """
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    result = client.call(
        prompt=prompt,
        output_model=CriterionJudgment,
        model="claude-haiku-5-5",
        effort="low",
    )

    assert isinstance(result.parsed, CriterionJudgment)
    assert result.parsed.criterion_id == "INC-3"
    assert result.parsed.kind == "inclusion"
    assert result.served_from_fixture
    assert result.stop_reason == "end_turn"
    # Evidence IDs point at real sentences in the note - the invariant the
    # verifier enforces later, demonstrated here on a real recorded answer.
    assert result.parsed.evidence_sentence_ids
    assert set(result.parsed.evidence_sentence_ids) <= SAMPLE_NOTE.valid_ids


def test_the_cost_meter_reads_the_recorded_cache_read_tokens(client: LlmClient) -> None:
    """The second recorded call is a cache hit, and is priced as one.

    Recorded live: 86 fresh input tokens, 2032 read from cache, 117 output.
    0.00000860 + 0.00002032 + 0.00005850 = 0.00008742 USD at the Haiku base
    tier. Same arithmetic as ``test_cost``, but here it runs through the client
    on a real ``usage`` object rather than on numbers typed into a test.
    """
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SECOND_TRIAL_RULES,
    )
    result = client.call(
        prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5", effort="low"
    )

    assert result.cost.cache_read_tokens == 2032
    assert result.cost.cache_hit
    assert result.cost.input_tokens == 86
    assert result.cost.output_tokens == 117
    assert result.cost.billed_tier == "haiku_base"
    assert result.cost.cost_usd == pytest.approx(0.00008742)


def test_the_first_recorded_call_paid_to_write_the_cache(client: LlmClient) -> None:
    """Call 1 writes 2032 tokens; call 2 reads them back for 12.5x less.

    Both halves asserted from recorded data, which is as close as an offline
    test can get to proving the cache ordering pays for itself.
    """
    first = client.call(
        prompt=build_cached_prompt(
            instructions=RULE_CHECK_INSTRUCTIONS,
            note=SAMPLE_NOTE,
            variable=SAMPLE_TRIAL_RULES,
        ),
        output_model=CriterionJudgment,
        model="claude-haiku-5-5",
        effort="low",
    )
    second = client.call(
        prompt=build_cached_prompt(
            instructions=RULE_CHECK_INSTRUCTIONS,
            note=SAMPLE_NOTE,
            variable=SECOND_TRIAL_RULES,
        ),
        output_model=CriterionJudgment,
        model="claude-haiku-5-5",
        effort="low",
    )

    assert first.cost.cache_write_tokens == 2032
    assert first.cost.cache_read_tokens == 0
    assert second.cost.cache_read_tokens == 2032
    assert second.cost.cache_write_cost_usd == 0
    assert first.cost.cache_write_cost_usd == second.cost.cache_read_cost_usd * Decimal("12.5")


def test_the_audit_row_populates_every_llm_call_column(client: LlmClient) -> None:
    """``LlmCall(**result.audit_row(...))`` must construct.

    ``audit_row`` returns a plain dict so this package needs no SQLAlchemy
    session on the hot path; this test is what keeps the dict in step with the
    frozen table definition that agent W0 owns.
    """
    result = client.call(
        prompt=build_cached_prompt(
            instructions=RULE_CHECK_INSTRUCTIONS,
            note=SAMPLE_NOTE,
            variable=SAMPLE_TRIAL_RULES,
        ),
        output_model=CriterionJudgment,
        model="claude-haiku-5-5",
        effort="low",
    )
    row = result.audit_row(match_run_id="run-1", step="check_rules", trace_id="trace-1")

    columns = set(LlmCall.__table__.columns.keys())
    assert set(row) <= columns, f"not columns of llm_calls: {set(row) - columns}"

    # Everything the table wants from a call, minus what only the DB supplies.
    supplied_elsewhere = {"id", "created_at"}
    assert columns - supplied_elsewhere - set(row) == set()

    orm_row = LlmCall(**row)
    assert orm_row.model == "claude-haiku-5-5"
    assert orm_row.effort == "low"
    assert orm_row.billed_tier == "haiku_base"
    assert orm_row.cache_read_tokens == 0
    assert orm_row.cache_write_tokens == 2032
    assert orm_row.cost_usd == pytest.approx(0.000337)
    assert orm_row.stop_reason == "end_turn"
    assert len(orm_row.prompt_hash) == 64
    assert orm_row.step == "check_rules"


def test_an_unrecorded_prompt_raises_instead_of_calling_out(client: LlmClient) -> None:
    """The guarantee, stated as a test: a miss is an error, never a request."""
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable="a trial nobody recorded",
    )
    with pytest.raises(FixtureMissingError):
        client.call(prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5")


def test_a_recorded_answer_that_does_not_fit_the_model_raises(tmp_path: Path) -> None:
    """A schema change invalidates fixtures, and must say so.

    ``CriterionJudgment`` forbids extra fields, so adding a field to it - or
    replaying a response recorded before one was removed - fails here rather
    than producing a half-populated judgment.
    """
    settings = Settings(llm_fixtures_dir=tmp_path / "llm")
    replay_client = LlmClient(settings=settings, mode=FixtureMode.REPLAY)
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )

    FixtureStore(settings.llm_fixtures_dir).save(
        prompt.fingerprint(),
        message=Message(
            id="msg_stale",
            model="claude-haiku-5-5",
            role="assistant",
            type="message",
            stop_reason="end_turn",
            content=[{"type": "text", "text": '{"criterion_id": "INC-3"}'}],
            usage=Usage(input_tokens=1, output_tokens=1),
        ),
        request={},
        latency_ms=0.0,
        request_id=None,
    )

    with pytest.raises(StructuredOutputError) as excinfo:
        replay_client.call(prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5")
    assert excinfo.value.model_name == "CriterionJudgment"
    assert excinfo.value.raw == '{"criterion_id": "INC-3"}'


def test_a_truncated_response_is_not_mistaken_for_a_bad_schema(tmp_path: Path) -> None:
    """``stop_reason == "max_tokens"`` means raise max_tokens, not fix the model."""
    settings = Settings(llm_fixtures_dir=tmp_path / "llm")
    replay_client = LlmClient(settings=settings, mode=FixtureMode.REPLAY)
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    FixtureStore(settings.llm_fixtures_dir).save(
        prompt.fingerprint(),
        message=Message(
            id="msg_cut",
            model="claude-haiku-5-5",
            role="assistant",
            type="message",
            stop_reason="max_tokens",
            content=[{"type": "text", "text": '{"criterion_id": "INC-'}],
            usage=Usage(input_tokens=1, output_tokens=1),
        ),
        request={},
        latency_ms=0.0,
        request_id=None,
    )

    with pytest.raises(StructuredOutputError, match="truncated"):
        replay_client.call(prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5")


def test_a_refusal_is_its_own_error(tmp_path: Path) -> None:
    """A refusal will refuse again, so retry logic must not swallow it."""
    settings = Settings(llm_fixtures_dir=tmp_path / "llm")
    replay_client = LlmClient(settings=settings, mode=FixtureMode.REPLAY)
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    FixtureStore(settings.llm_fixtures_dir).save(
        prompt.fingerprint(),
        message=Message(
            id="msg_refused",
            model="claude-haiku-5-5",
            role="assistant",
            type="message",
            stop_reason="refusal",
            content=[],
            usage=Usage(input_tokens=1, output_tokens=1),
        ),
        request={},
        latency_ms=0.0,
        request_id=None,
    )

    with pytest.raises(RefusalError):
        replay_client.call(prompt=prompt, output_model=CriterionJudgment, model="claude-haiku-5-5")
