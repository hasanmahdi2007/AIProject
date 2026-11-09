"""The budget meter and the retry policy - the two things that decide whether a
failure costs money.
"""

from __future__ import annotations

import random

import anthropic
import httpx
import pytest

from beacon_core.schemas import TrialJudgment
from beacon_core.settings import KafkaSettings
from beacon_llm import (
    CostBreakdown,
    PromptTooLargeError,
    RefusalError,
    StructuredOutputError,
    price_tokens,
)
from match_worker.backoff import RetryPolicy, is_retryable, retry_async
from match_worker.budget import BudgetExhausted, BudgetMeter
from match_worker.config import WorkerConfig
from match_worker.state import COULD_NOT_EVALUATE, could_not_evaluate, is_partial, partial_reasons


def _cost(usd_target_tokens: int = 1_000_000) -> CostBreakdown:
    return price_tokens(
        "claude-haiku-5-5", input_tokens=usd_target_tokens, output_tokens=0, cache_read_tokens=0
    )


# ------------------------------------------------------------------- budget --


def test_meter_accumulates_from_real_usage_and_returns_the_delta() -> None:
    meter = BudgetMeter(limit_usd=1.0, limit_tokens=10_000_000)
    cost = price_tokens(
        "claude-haiku-5-5", input_tokens=1000, output_tokens=200, cache_read_tokens=5000
    )
    delta = meter.record(cost, step="check_rules")

    assert delta == pytest.approx(cost.cost_usd)
    assert meter.spent_usd == pytest.approx(cost.cost_usd)
    # The cliff is judged on prompt tokens, which include cache reads.
    assert meter.tokens == 1000 + 5000 + 200
    assert meter.cached_tokens == 5000
    assert meter.calls == 1


def test_check_raises_once_the_dollar_ceiling_is_reached() -> None:
    meter = BudgetMeter(limit_usd=0.05, limit_tokens=10_000_000)
    meter.check(step="check_rules")  # fine while there is budget left

    meter.record(_cost(1_000_000), step="check_rules")  # $0.10 of Haiku input
    assert meter.exhausted
    with pytest.raises(BudgetExhausted) as raised:
        meter.check(step="check_rules")
    assert "dollar ceiling" in str(raised.value)
    assert raised.value.spent_usd > raised.value.limit_usd


def test_check_raises_on_the_token_ceiling_too() -> None:
    meter = BudgetMeter(limit_usd=100.0, limit_tokens=500)
    meter.record(_cost(1000), step="profile_patient")
    with pytest.raises(BudgetExhausted, match="token ceiling"):
        meter.check(step="retrieve")


def test_snapshot_reports_both_limits() -> None:
    meter = BudgetMeter(limit_usd=0.5, limit_tokens=1000)
    snapshot = meter.snapshot()
    assert snapshot.exhausted is False
    assert snapshot.limit_usd == 0.5
    assert snapshot.limit_tokens == 1000


# -------------------------------------------------------- partial markers --


def test_could_not_evaluate_is_marked_and_detectable() -> None:
    judgment = could_not_evaluate("NCT01", "per-match budget exhausted")
    assert judgment.category == "insufficient_info"
    assert judgment.summary is not None
    assert judgment.summary.startswith(COULD_NOT_EVALUATE)
    assert judgment.judgments == []

    everything = [judgment, TrialJudgment(nct_id="NCT02", category="eligible")]
    assert is_partial(everything) is True
    assert partial_reasons(everything) == ["per-match budget exhausted"]


def test_a_complete_run_is_not_partial() -> None:
    assert is_partial([TrialJudgment(nct_id="NCT02", category="eligible")]) is False


# -------------------------------------------------------------- retryable --


def test_outcomes_are_not_retried_but_transport_is() -> None:
    not_retryable = [
        BudgetExhausted(reason="x", spent_usd=1.0, limit_usd=1.0, tokens=1),
        RefusalError(model="claude-haiku-5-5", request_id=None),
        PromptTooLargeError(model="claude-haiku-5-5", prompt_tokens=200_000, ceiling=100_000),
        ValueError("a bug in our own code"),
    ]
    for exc in not_retryable:
        assert is_retryable(exc) is False, exc

    assert is_retryable(StructuredOutputError(model_name="X", detail="bad json", raw="{")) is True
    assert is_retryable(httpx.ConnectError("boom")) is True
    assert is_retryable(TimeoutError()) is True

    response = httpx.Response(503, request=httpx.Request("POST", "https://example.invalid"))
    status_error = anthropic.APIStatusError("overloaded", response=response, body=None)  # type: ignore[arg-type]
    assert is_retryable(status_error) is True

    bad_request = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
    assert (
        is_retryable(
            anthropic.APIStatusError("bad request", response=bad_request, body=None)  # type: ignore[arg-type]
        )
        is False
    )


async def test_retry_async_retries_then_succeeds_and_sleeps_with_jitter() -> None:
    slept: list[float] = []
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("blip")
        return "ok"

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    policy = RetryPolicy(attempts=3, base_seconds=1.0, max_seconds=8.0)
    result = await retry_async(
        flaky, policy=policy, what="unit", sleep=fake_sleep, rng=random.Random(7)
    )

    assert result == "ok"
    assert calls["n"] == 3
    # Full jitter: each delay is a uniform draw inside its own window, so the
    # twenty parallel trials of a fan-out do not retry in lockstep.
    assert len(slept) == 2
    assert 0 <= slept[0] <= 1.0
    assert 0 <= slept[1] <= 2.0
    assert slept[0] != slept[1]


async def test_retry_async_gives_up_after_the_configured_attempts() -> None:
    calls = {"n": 0}

    async def always_fails() -> str:
        calls["n"] += 1
        raise httpx.ConnectError("blip")

    async def no_sleep(_seconds: float) -> None:
        return None

    with pytest.raises(httpx.ConnectError):
        await retry_async(
            always_fails,
            policy=RetryPolicy(attempts=2, base_seconds=0.0, max_seconds=0.0),
            what="unit",
            sleep=no_sleep,
        )
    assert calls["n"] == 2


async def test_retry_async_does_not_retry_an_outcome() -> None:
    calls = {"n": 0}

    async def refuses() -> str:
        calls["n"] += 1
        raise RefusalError(model="claude-haiku-5-5", request_id="req_1")

    with pytest.raises(RefusalError):
        await retry_async(
            refuses, policy=RetryPolicy(attempts=5, base_seconds=0.0, max_seconds=0.0), what="unit"
        )
    assert calls["n"] == 1


def test_delay_window_is_capped() -> None:
    policy = RetryPolicy(attempts=10, base_seconds=1.0, max_seconds=4.0)
    assert policy.delay_for(1, rng=random.Random(1)) <= 1.0
    assert policy.delay_for(9, rng=random.Random(1)) <= 4.0


# ------------------------------------------------------------------ config --


def test_message_attempts_matches_the_systems_kafka_retry_policy() -> None:
    """The DLQ threshold is the one the frozen settings describe.

    ``settings.kafka.max_retries`` is the system-wide statement of "after this
    many failures the message goes to match.dlq"; the worker keeps its own knob
    so one deployment can be made more patient, but the default must agree or
    the documented behaviour and the real behaviour differ.
    """
    assert WorkerConfig().message_attempts == KafkaSettings().max_retries == 3


def test_the_default_budget_is_a_runaway_guard_not_a_normal_limit() -> None:
    config = WorkerConfig()
    # Twenty Haiku rule checks on a cached prefix land near a cent, so the
    # default must be far above a normal run and far below a surprise.
    assert 0.1 <= config.max_cost_usd <= 5.0
    assert config.trial_attempts == 2
    assert config.throttle_seconds == 0.0
