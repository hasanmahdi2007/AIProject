"""The prompt-token guard.

The regression this guards against has a name: somebody reads "twenty Claude
calls per patient", thinks "that's twenty round trips, batch them", and ships
one call with twenty trials' rules in it. On Haiku 5.5 that single call lands
past 100,000 prompt tokens and every token in it reprices at 5x. No exception,
no warning, nothing in the logs - just a bill.

So the guard runs before the request is sent, on every path including replay,
and these tests are what make the refactor above fail in CI instead of on the
invoice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beacon_core.schemas import CriterionJudgment
from beacon_core.settings import Settings
from beacon_llm import (
    FixtureMissingError,
    HeuristicTokenCounter,
    LlmClient,
    PromptTooLargeError,
    build_cached_prompt,
    enforce_prompt_budget,
)
from beacon_llm._testing import RULE_CHECK_INSTRUCTIONS, SAMPLE_NOTE, SAMPLE_TRIAL_RULES
from beacon_llm.fixtures import FixtureMode

CEILING = 100_000


def test_at_the_ceiling_is_allowed_and_one_over_is_not() -> None:
    """The refusal is strictly above the ceiling, matching the price tier."""
    enforce_prompt_budget(model="claude-haiku-5-5", prompt_tokens=CEILING, ceiling=CEILING)
    with pytest.raises(PromptTooLargeError) as excinfo:
        enforce_prompt_budget(model="claude-haiku-5-5", prompt_tokens=CEILING + 1, ceiling=CEILING)

    error = excinfo.value
    assert error.model == "claude-haiku-5-5"
    assert error.prompt_tokens == CEILING + 1
    assert error.ceiling == CEILING
    # The message has to explain the trap, because whoever triggers it is
    # mid-refactor and believes batching is an optimisation.
    assert "cache reads" in str(error)
    assert "5x" in str(error)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_models_without_a_cliff_are_not_refused(model: str) -> None:
    """There is no point refusing a call whose price would not change."""
    enforce_prompt_budget(model=model, prompt_tokens=900_000, ceiling=CEILING)


def test_the_heuristic_counter_counts_system_messages_and_schema() -> None:
    """Anything it skips is tokens the guard cannot see.

    The output schema is part of the priced prompt, so it is counted too.
    """
    counter = HeuristicTokenCounter(chars_per_token=4)
    system = [{"type": "text", "text": "a" * 400}]
    messages = [{"role": "user", "content": [{"type": "text", "text": "b" * 400}]}]

    without_schema = counter.count(
        model="claude-haiku-5-5",
        system=system,  # type: ignore[arg-type]
        messages=messages,  # type: ignore[arg-type]
        output_config={},
    )
    with_schema = counter.count(
        model="claude-haiku-5-5",
        system=system,  # type: ignore[arg-type]
        messages=messages,  # type: ignore[arg-type]
        output_config={"format": {"type": "json_schema", "schema": {"x": "y" * 400}}},
    )
    assert without_schema == 200
    assert with_schema > without_schema


def test_the_heuristic_counter_handles_a_plain_string_content() -> None:
    """``content`` may be a bare string rather than a block list."""
    counter = HeuristicTokenCounter(chars_per_token=4)
    assert (
        counter.count(
            model="claude-haiku-5-5",
            system=[],
            messages=[{"role": "user", "content": "c" * 800}],
            output_config={},
        )
        == 200
    )


def test_batching_twenty_trials_into_one_call_is_refused_offline(tmp_path: Path) -> None:
    """The regression test proper, and it needs no network.

    ``LlmClient`` in replay mode still counts and still refuses, so this fails
    in CI on a machine with no API key - which is the only place the refactor
    would ever be reviewed.
    """
    settings = Settings(llm_fixtures_dir=tmp_path / "llm")
    client = LlmClient(settings=settings, mode=FixtureMode.REPLAY)

    twenty_trials = "\n\n".join(
        f"Trial NCT0400{i:04d} eligibility rules:\n" + ("Criterion text. " * 1600)
        for i in range(20)
    )
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS, note=SAMPLE_NOTE, variable=twenty_trials
    )

    with pytest.raises(PromptTooLargeError) as excinfo:
        client.call(
            prompt=prompt,
            output_model=CriterionJudgment,
            model=settings.anthropic.cheap_model,
            effort="low",
        )
    assert excinfo.value.prompt_tokens > settings.anthropic.haiku_prompt_token_ceiling


def test_a_single_trial_passes_the_guard_and_then_looks_for_a_fixture(tmp_path: Path) -> None:
    """Proof the previous test failed on size and not on everything failing.

    One trial clears the guard; the call then stops for the ordinary reason -
    replay has no recording of it.
    """
    settings = Settings(llm_fixtures_dir=tmp_path / "llm")
    client = LlmClient(settings=settings, mode=FixtureMode.REPLAY)
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )

    with pytest.raises(FixtureMissingError):
        client.call(prompt=prompt, output_model=CriterionJudgment, effort="low")
