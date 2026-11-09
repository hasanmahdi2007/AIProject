"""The prompt-cache contract for the rule check.

These are the tests that have to survive the owner rewriting
``rule_check_prompt.py``. They assert *structure*, never wording: the cached
prefix is byte-identical across every trial of one patient, it clears the
512-token floor, the trial's rules are outside it, and the breakpoint sits on
the note block.

Each of those failing silently costs five to twenty times more per call and
raises no error, which is why they are asserted rather than documented.
"""

from __future__ import annotations

from beacon_llm import MIN_CACHEABLE_PREFIX_TOKENS, build_cached_prompt
from match_worker._testing import TRIALS, numbered_note
from match_worker.outputs import TrialRuleCheck
from match_worker.rule_check_prompt import render_trial_rules, rule_check_instructions


def _prompt_for(trial_index: int) -> object:
    trial = TRIALS[trial_index]
    criteria = [
        {"criterion_id": c.criterion_id, "kind": c.kind, "text": c.text} for c in trial.criteria
    ]
    return build_cached_prompt(
        instructions=rule_check_instructions(),
        note=numbered_note(),
        variable=render_trial_rules(nct_id=trial.nct_id, title=trial.title, criteria=criteria),
    )


def test_the_cacheable_prefix_is_byte_identical_across_trials() -> None:
    """The entire source of the cache saving: 19 of 20 calls read this prefix."""
    prompts = [_prompt_for(i) for i in range(len(TRIALS))]
    prefixes = {p.prefix_hash for p in prompts}  # type: ignore[attr-defined]
    assert len(prefixes) == 1, "the prefix differs per trial, so nothing will cache"

    # ...and the full prompts still differ, or the fixtures would collide and
    # every trial would be judged against the first trial's rules.
    assert len({p.fingerprint() for p in prompts}) == len(TRIALS)  # type: ignore[attr-defined]


def test_the_prefix_clears_the_minimum_cacheable_size() -> None:
    prompt = _prompt_for(0)
    assert prompt.estimated_prefix_tokens > MIN_CACHEABLE_PREFIX_TOKENS  # type: ignore[attr-defined]
    assert prompt.maybe_too_short_to_cache is False  # type: ignore[attr-defined]


def test_block_order_is_instructions_then_note_then_this_trials_rules() -> None:
    prompt = _prompt_for(0)
    system = prompt.system_blocks()  # type: ignore[attr-defined]
    messages = prompt.messages()  # type: ignore[attr-defined]

    assert len(system) == 2
    assert system[0]["text"] == rule_check_instructions()
    assert "cache_control" not in system[0]
    # The breakpoint is on the note, so the cached region is instructions+note
    # and stops there.
    assert system[1]["cache_control"]["type"] == "ephemeral"
    assert system[1]["text"].startswith("1. A 58-year-old woman")

    variable = messages[0]["content"][0]["text"]
    assert TRIALS[0].nct_id in variable
    # The per-trial rules must be outside the cached prefix.
    assert TRIALS[0].nct_id not in prompt.cacheable_prefix  # type: ignore[attr-defined]
    assert "NCT04000001:inc:01" in variable


def test_every_parsed_criterion_reaches_the_prompt_with_its_kind() -> None:
    trial = TRIALS[0]
    criteria = [
        {"criterion_id": c.criterion_id, "kind": c.kind, "text": c.text} for c in trial.criteria
    ]
    rendered = render_trial_rules(nct_id=trial.nct_id, title=trial.title, criteria=criteria)
    for criterion in criteria:
        assert criterion["criterion_id"] in rendered
        assert criterion["text"] in rendered
    assert rendered.count("kind: inclusion") == 3
    assert rendered.count("kind: exclusion") == 2
    assert f"Return exactly {len(criteria)} judgments" in rendered


def test_the_output_schema_is_the_frozen_criterion_judgment() -> None:
    """A sixth field here would change the schema sent to Claude, bust the
    cached prefix and invalidate every fixture."""
    schema = TrialRuleCheck.model_json_schema()
    judgment = schema["$defs"]["CriterionJudgment"]
    assert set(judgment["properties"]) == {
        "criterion_id",
        "kind",
        "label",
        "evidence_sentence_ids",
        "explanation",
    }
