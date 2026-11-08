"""The one seam every Claude call in Beacon passes through.

Typical use, from a pipeline step:

.. code-block:: python

    from beacon_core.schemas import CriterionJudgment
    from beacon_llm import LlmClient, build_cached_prompt

    client = LlmClient()                       # replay unless BEACON_LLM_MODE says otherwise
    prompt = build_cached_prompt(
        instructions=RULE_CHECK_INSTRUCTIONS,   # stable across the whole patient
        note=numbered_note,                     # cache breakpoint lands here
        variable=trial_rules_text,              # the only part that changes per trial
    )
    result = client.call(
        prompt=prompt,
        output_model=CriterionJudgment,
        model="claude-haiku-5-5",
        effort="high",
    )
    session.add(LlmCall(**result.audit_row(match_run_id=run_id, step="check_rules")))

Four properties this package exists to hold, each of which fails silently if it
is left to call sites:

* **Cache ordering is structural.** ``build_cached_prompt`` takes three strings
  and renders them in one fixed order, with the breakpoint after the note.
  There is no argument that reorders them, because a reordered prefix costs 5x
  to 20x more and raises no error.
* **Nothing is billed at a guessed rate.** :mod:`beacon_llm.pricing` holds the
  verified table as data, including the fact that the cache-read discount is
  0.1x on Haiku 5.5 but 0.05x on Opus 5.5 and Sonnet 5.5. An unpriced model
  raises rather than costing zero.
* **The Haiku cliff is guarded, not documented.** Prompts are counted before
  the request leaves the process and refused above
  ``settings.anthropic.haiku_prompt_token_ceiling``, on the replay path too, so
  a batching regression is a test failure.
* **Replay never reaches the network.** The default mode is replay and a
  missing fixture raises; there is no fall-through to the API.
"""

from __future__ import annotations

from beacon_llm.capabilities import Effort, assert_supports, model_info
from beacon_llm.client import ADAPTIVE_THINKING, AsyncLlmClient, CallResult, LlmClient
from beacon_llm.cost import CostBreakdown, price_tokens, price_usage
from beacon_llm.errors import (
    BeaconLlmError,
    CapabilityError,
    FixtureMissingError,
    PromptTooLargeError,
    RefusalError,
    StructuredOutputError,
    UnknownModelPriceError,
)
from beacon_llm.fixtures import FixtureMode, FixtureStore, resolve_mode
from beacon_llm.guard import (
    ApiTokenCounter,
    HeuristicTokenCounter,
    TokenCounter,
    enforce_prompt_budget,
)
from beacon_llm.pricing import (
    CACHE_WRITE_MULTIPLIERS,
    HAIKU_PROMPT_TOKEN_CLIFF,
    MIN_CACHEABLE_PREFIX_TOKENS,
    PRICE_TABLE,
    BilledTier,
    CacheTtl,
    ModelPricing,
    TierPrices,
    known_models,
    pricing_for,
)
from beacon_llm.prompt import CachedPrompt, build_cached_prompt, render_numbered_note

__version__ = "0.1.0"

__all__ = [
    "ADAPTIVE_THINKING",
    "CACHE_WRITE_MULTIPLIERS",
    "HAIKU_PROMPT_TOKEN_CLIFF",
    "MIN_CACHEABLE_PREFIX_TOKENS",
    "PRICE_TABLE",
    "ApiTokenCounter",
    "AsyncLlmClient",
    "BeaconLlmError",
    "BilledTier",
    "CacheTtl",
    "CachedPrompt",
    "CallResult",
    "CapabilityError",
    "CostBreakdown",
    "Effort",
    "FixtureMissingError",
    "FixtureMode",
    "FixtureStore",
    "HeuristicTokenCounter",
    "LlmClient",
    "ModelPricing",
    "PromptTooLargeError",
    "RefusalError",
    "StructuredOutputError",
    "TierPrices",
    "TokenCounter",
    "UnknownModelPriceError",
    "__version__",
    "assert_supports",
    "build_cached_prompt",
    "enforce_prompt_budget",
    "known_models",
    "model_info",
    "price_tokens",
    "price_usage",
    "pricing_for",
    "render_numbered_note",
    "resolve_mode",
]
