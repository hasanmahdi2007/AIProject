"""The prompt builder: is the ordering really structural?

The property under test is not "the blocks come out in the right order" but
"there is no way to get the wrong order". So these tests check the shape of the
rendered request, and check that twenty prompts for twenty different trials
produce one byte-identical prefix - which is the condition the API's cache
actually keys on.
"""

from __future__ import annotations

import json

import pytest

from beacon_core.schemas import NoteSentence, NumberedNote
from beacon_llm import MIN_CACHEABLE_PREFIX_TOKENS, build_cached_prompt, render_numbered_note

# Importing a fixture function into a test module is enough to register it
# there; see beacon_llm._testing for why these do not live in a conftest.
from beacon_llm._testing import RULE_CHECK_INSTRUCTIONS, SAMPLE_NOTE, SAMPLE_TRIAL_RULES


def test_numbered_note_renders_the_ids_the_model_will_cite() -> None:
    note = NumberedNote(
        sentences=[
            NoteSentence(sentence_id=1, text="Patient is 58."),
            NoteSentence(sentence_id=2, text="ECOG 1."),
        ]
    )
    assert render_numbered_note(note) == "1. Patient is 58.\n2. ECOG 1."


def test_system_holds_two_blocks_with_the_breakpoint_on_the_note() -> None:
    """Instructions, then note, then the breakpoint. Nothing else in system."""
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    blocks = prompt.system_blocks()

    assert len(blocks) == 2
    assert blocks[0]["text"] == RULE_CHECK_INSTRUCTIONS
    assert "cache_control" not in blocks[0]

    assert blocks[1]["text"] == render_numbered_note(SAMPLE_NOTE)
    assert blocks[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_the_variable_part_is_the_user_turn_and_nothing_else() -> None:
    """The per-trial text is in ``messages``, not in ``system``.

    Different fields of the request, not merely a different position in a list:
    appending to the conversation cannot land inside the cached region.
    """
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    messages = prompt.messages()

    assert messages == [{"role": "user", "content": [{"type": "text", "text": SAMPLE_TRIAL_RULES}]}]
    assert SAMPLE_TRIAL_RULES not in json.dumps(prompt.system_blocks())


def test_no_assistant_turn_is_ever_produced() -> None:
    """Assistant prefill returns 400 on the current models.

    There is no field for it and no code path to it, so the mistake is not
    available rather than merely discouraged.
    """
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    assert [m["role"] for m in prompt.messages()] == ["user"]


def test_twenty_trials_share_one_byte_identical_prefix() -> None:
    """The real workload: one patient, twenty trials, one cached prefix.

    If this fails, every one of the twenty calls pays full price for the prefix
    and no error is raised anywhere. It is the single most expensive thing in
    this library that can break without a symptom.
    """
    prompts = [
        build_cached_prompt(
            instructions=RULE_CHECK_INSTRUCTIONS,
            note=SAMPLE_NOTE,
            variable=f"Trial NCT0400{i:04d}, criterion INC-1: some rule text.",
        )
        for i in range(20)
    ]

    prefixes = {p.prefix_hash for p in prompts}
    assert len(prefixes) == 1

    rendered_prefixes = {json.dumps(p.system_blocks(), sort_keys=True) for p in prompts}
    assert len(rendered_prefixes) == 1

    # ...while the full fingerprints differ, so each call gets its own fixture.
    assert len({p.fingerprint() for p in prompts}) == 20


def test_fingerprint_is_stable_and_sensitive() -> None:
    """Same prompt, same hash; any change, a different hash.

    Stability is what makes the fixture key work across runs and machines;
    sensitivity is what stops a changed prompt quietly replaying a stale answer.
    """
    first = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    same = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    assert first.fingerprint() == same.fingerprint()
    assert len(first.fingerprint()) == 64

    changed_ttl = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
        cache_ttl="1h",
    )
    changed_instructions = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS + " Also be concise.",
        note=SAMPLE_NOTE,
        variable=SAMPLE_TRIAL_RULES,
    )
    assert first.fingerprint() != changed_ttl.fingerprint()
    assert first.fingerprint() != changed_instructions.fingerprint()
    assert first.prefix_hash != changed_instructions.prefix_hash


def test_a_short_prefix_is_flagged_as_probably_uncacheable() -> None:
    """Under 512 tokens the breakpoint is accepted and ignored, with no error.

    There is nothing to catch at runtime, so the builder flags it and the
    client logs a warning. The live smoke test is what actually proves the
    cache engaged.
    """
    short = build_cached_prompt(
        instructions="Judge the criterion.",
        note="1. Patient is 58.",
        variable="INC-1: age over 18",
    )
    assert short.estimated_prefix_tokens < MIN_CACHEABLE_PREFIX_TOKENS
    assert short.maybe_too_short_to_cache


def test_a_realistic_prefix_is_not_flagged() -> None:
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note=SAMPLE_NOTE,
        variable="INC-1: age over 18",
    )
    assert prompt.estimated_prefix_tokens > MIN_CACHEABLE_PREFIX_TOKENS
    assert not prompt.maybe_too_short_to_cache


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"instructions": "  ", "note": "1. x", "variable": "y"},
            "instructions",
        ),
        ({"instructions": "a", "note": "   ", "variable": "y"}, "note"),
        ({"instructions": "a", "note": "1. x", "variable": ""}, "variable"),
    ],
)
def test_empty_slots_are_rejected(kwargs: dict[str, str], match: str) -> None:
    """An empty slot means the three-part structure has collapsed."""
    with pytest.raises(ValueError, match=match):
        build_cached_prompt(**kwargs)  # type: ignore[arg-type]


def test_a_prerendered_note_string_is_accepted() -> None:
    """Not every prompt is note-shaped; query generation is not."""
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,
        note="free text context",
        variable="generate queries",
    )
    assert prompt.note_text == "free text context"
