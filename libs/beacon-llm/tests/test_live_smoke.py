"""Live smoke tests. Marked ``integration``; excluded from the default run.

Two things cannot be proved offline, so they are proved here, once, with the
cheapest model:

1. **That prompt caching actually engages.** A ``cache_control`` breakpoint on a
   prefix shorter than 512 tokens is accepted, ignored, and reports no error.
   The only way to know the cache worked is to read
   ``usage.cache_read_input_tokens`` from a real second call and find it
   non-zero. Asserting it is the whole point of this file.
2. **That ``output_config.format`` really returns schema-valid JSON** on the
   current models, without assistant prefill and without a forced
   ``tool_choice`` - neither of which these models accept.

Run with::

    BEACON_LLM_MODE=record uv run pytest libs/beacon-llm -m integration -s

``record`` rather than ``live`` on purpose: the first run pays for two Haiku
calls and writes the responses into ``tests/fixtures/llm/``, and every run after
that re-checks the same assertions against the recorded ``usage`` for free. The
recorded pair is also what the replay tests use, so the committed fixtures carry
real cache-read numbers instead of invented ones.
"""

from __future__ import annotations

import pytest

from beacon_core.schemas import CriterionJudgment
from beacon_core.settings import Settings
from beacon_llm import FixtureMode, LlmClient, build_cached_prompt, resolve_mode
from beacon_llm._testing import (
    COMMITTED_FIXTURES,
    RULE_CHECK_INSTRUCTIONS,
    SAMPLE_NOTE,
    SAMPLE_TRIAL_RULES,
    SECOND_TRIAL_RULES,
)

pytestmark = pytest.mark.integration


def test_structured_output_and_prompt_caching() -> None:
    """Two calls sharing one cached prefix; the second must read from cache.

    This is the shape of the real workload in miniature: one patient, two
    trials, one prefix. Twenty trials is the same thing nineteen more times.
    """
    mode = resolve_mode()
    if mode is FixtureMode.REPLAY:
        pytest.skip("set BEACON_LLM_MODE=record (or live) to run the live smoke test")

    settings = Settings(llm_fixtures_dir=COMMITTED_FIXTURES)
    client = LlmClient(settings=settings, mode=mode)
    model = settings.anthropic.cheap_model

    first_prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    second_prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SECOND_TRIAL_RULES,
    )

    # The thing the whole builder exists to guarantee.
    assert first_prompt.prefix_hash == second_prompt.prefix_hash
    assert first_prompt.estimated_prefix_tokens > 512

    first = client.call(
        prompt=first_prompt,
        output_model=CriterionJudgment,
        model=model,
        effort="low",
        max_tokens=1024,
    )
    second = client.call(
        prompt=second_prompt,
        output_model=CriterionJudgment,
        model=model,
        effort="low",
        max_tokens=1024,
    )

    for label, result in (("call 1", first), ("call 2", second)):
        usage = result.message.usage
        print(
            f"\n{label}: model={result.model} tier={result.cost.billed_tier} "
            f"input={usage.input_tokens} "
            f"cache_write={usage.cache_creation_input_tokens} "
            f"cache_read={usage.cache_read_input_tokens} "
            f"output={usage.output_tokens} "
            f"prompt_total={result.cost.prompt_tokens} "
            f"cost=${result.cost.cost_usd:.8f} stop={result.stop_reason} "
            f"request_id={result.request_id}"
        )
        print(f"{label} parsed: {result.parsed.model_dump_json()}")

    # Structured outputs: validated instances, not free text that happens to
    # look like JSON.
    assert isinstance(first.parsed, CriterionJudgment)
    assert isinstance(second.parsed, CriterionJudgment)
    assert first.parsed.criterion_id
    assert set(first.parsed.evidence_sentence_ids) <= SAMPLE_NOTE.valid_ids
    assert set(second.parsed.evidence_sentence_ids) <= SAMPLE_NOTE.valid_ids

    # Call 1 writes the prefix into the cache; call 2 reads it back.
    assert (first.message.usage.cache_creation_input_tokens or 0) > 0, (
        "call 1 wrote nothing to the cache: the prefix is probably under the "
        "512-token minimum, which fails silently"
    )
    assert (second.message.usage.cache_read_input_tokens or 0) > 0, (
        "call 2 read nothing from the cache: the prefix was not byte-identical, "
        "or more than the TTL elapsed between the calls"
    )
    assert second.cost.cache_hit
    assert second.message.usage.cache_read_input_tokens == (
        first.message.usage.cache_creation_input_tokens
    ), "the cached region differed between the two calls"

    # Haiku, small prompt: both calls must be billed at the base tier.
    assert first.cost.billed_tier == "haiku_base"
    assert second.cost.billed_tier == "haiku_base"

    total = first.cost.total_cost_usd + second.cost.total_cost_usd
    print(f"\ntwo-call total: ${total:.8f}")
