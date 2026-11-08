# ADR 0005 — Model choice and the cost controls

- **Status:** accepted; model roles locked 2026-11-08
- **Date:** 2026-11-08
- **Affects:** `beacon_core.settings.AnthropicSettings`, `libs/beacon-llm/**`, `beacon_core.models.LlmCall`

## Context

Beacon's token spend is almost entirely one stage. For each patient, ~20
candidate trials are checked in parallel, one Claude call per trial, each
receiving the standing instructions plus the numbered patient note plus that
trial's rules. `Beacon_System_Design.pdf` §2 puts roughly **95% of all token
spend in step 7** (`check_rules`). Everything else — profiling, query
generation, the verifier, the summaries — is noise next to it.

The project's total API budget is about **$30–80**. So the model decision is not
"which model is best" but "which model where, and what stops the bill from
surprising us".

### Verified prices

Taken from `platform.claude.com` and recorded in `docs/AGENT_BRIEFS.md` §A3 as
verified **8 November 2026**; encoded as data in
`beacon_llm.pricing.PRICE_TABLE`. These model IDs are current and valid and
must not be "corrected" from memory.

| Model | Input $/MTok | Output $/MTok | Cache read | Cache-read multiplier |
|---|---|---|---|---|
| `claude-haiku-5-5` (≤100,000 prompt tokens) | 0.10 | 0.50 | 0.01 | **0.1×** |
| `claude-haiku-5-5` (>100,000 prompt tokens) | 0.50 | 2.50 | 0.05 | 0.1× |
| `claude-sonnet-5-5` | 2 | 10 | 0.10 | **0.05×** |
| `claude-opus-5-5` | 4 | 20 | 0.20 | **0.05×** |

Two facts in that table are money leaks if missed:

1. **Haiku 5.5 has a 5× pricing cliff at 100,000 prompt tokens**, and the
   threshold counts *all* input — fresh input, cache reads and cache writes
   together. Each request is priced on its own, so twenty small calls and one
   batched call are not the same money even at identical total tokens.
2. **The cache-read discount is not flat.** 0.1× on Haiku, 0.05× on Sonnet and
   Opus. Writing "× 0.1" once in the cost function would overcharge both models
   the project actually reports on.

Cache-*write* multipliers (1.25× for 5m, 2× for 1h) are the standard published
figures but are **not** part of the verified table, so they live in
`CACHE_WRITE_MULTIPLIERS` and the cache-write cost is reported as its own
component of `CostBreakdown` — the uncertainty is never buried in a total.

## Options considered

| Option | Verdict |
|---|---|
| One model everywhere (Haiku) | Rejected. The headline number would be reported on the cheapest model, which undersells the system and invites "so how good is it really?". |
| One model everywhere (Sonnet or Opus) | Rejected on cost: development means hundreds of iterations, and tuning on a $2/$10 or $4/$20 model burns the entire budget before the harness works. |
| Haiku for dev, **Opus** for the headline run | Rejected. See below. |
| **Haiku for dev and bulk; Sonnet 5.5 for the single headline run; Opus 5.5 as the judge only** | **Chosen.** |
| A non-Claude model as the judge | Out of scope for this project, and it would add a second provider's SDK, pricing table and failure modes to maintain. |

### Why Sonnet and not Opus for the headline

The guide offers Opus 5.5 as "the most capable option if the budget allows".
Declined, deliberately, on 8 November 2026:

- The headline claim is about the **system** — hybrid retrieval, rule-level
  judging, citation verification — not about the frontier model inside it. A
  number that only exists on the most expensive model is a weaker claim about
  the architecture.
- `check_rules` is a constrained structured-output task against a frozen
  five-field schema. The marginal benefit of Opus over Sonnet on that shape of
  task is **unmeasured here**, while the cost is exactly 2×.
- If a later ablation shows a real gap, that is a **row in the table**, not a
  change of headline.

### Why Opus as the judge

A judge must differ from the model it grades, or it is grading itself. The
headline run is Sonnet 5.5, so the judge is Opus 5.5 — stated in
`AnthropicSettings.judge_model`'s docstring for exactly this reason. The judge
runs only on a small hand-labelled sample (on the order of **$1**), and its
agreement with human labels is reported as Cohen's κ = **{judge_kappa}**.
Without that calibration an LLM-judge number is an opinion, not a measurement.

## Decision

| Role | Model | Setting |
|---|---|---|
| Development, tuning, every bulk/ablation run | `claude-haiku-5-5` | `AnthropicSettings.cheap_model` |
| The single headline run on held-out TREC 2022 | `claude-sonnet-5-5` | `AnthropicSettings.headline_model` |
| LLM judge, small sample only | `claude-opus-5-5` | `AnthropicSettings.judge_model` |

Embeddings (`bge-small-en-v1.5`), BM25 and the cross-encoder
(`bge-reranker-base`) all run locally through FastEmbed/ONNX at **$0**. No part
of retrieval costs money.

### The controls, each pinned to code

| Control | Where | What it prevents |
|---|---|---|
| Prompt-token ceiling, counted **before** the request leaves the process | `beacon_llm.guard`, `AnthropicSettings.haiku_prompt_token_ceiling` = 100,000 | the "batch all 20 trials into one call" refactor crossing the Haiku cliff silently |
| Cache-ordered prompt with three fixed slots | `beacon_llm.prompt.CachedPrompt` | a caller putting variable text in the cached prefix, which costs 5–20× and raises no error |
| Minimum cacheable prefix = 512 tokens | `MIN_CACHEABLE_PREFIX_TOKENS`, mirrored in settings | a `cache_control` breakpoint being accepted, ignored, and paying full price forever |
| `billed_tier` column on every call | `beacon_core.models.LlmCall` | the Haiku cliff being invisible after the fact — two rows with identical token counts can differ 5× and nothing else would say why |
| Record/replay, **default `replay`**, no path from a miss to the network | `beacon_llm.fixtures` | CI spending money; also makes eval diffs mean code changes, not model variance |
| `Decimal` arithmetic end to end | `beacon_llm.cost` | float drift across thousands of ~1e-5 USD calls summed into a README figure |
| Unknown model raises | `pricing_for` → `UnknownModelPriceError` | a model missing from the table appearing in `llm_calls` as free |
| Batches API for full benchmark runs | planned, eval harness | paying list price for a run whose latency does not matter |
| Exact-match response cache | `llm:{sha256}` in Redis | re-paying for an identical prompt |
| Account-level monthly spend limit | Anthropic Console | everything above failing at once |

## Consequences

- **Cost per patient is a measurement, not an estimate.** The figures are
  **{cost_per_patient_haiku}** and **{cost_per_patient_sonnet}**, and the full
  held-out run is **{cost_full_run_sonnet}** — all pending the audit log. The
  guide's ~$0.03/patient and ~$60/full-Sonnet-run numbers are *its* estimates
  and are never reported as results.
- **The tuned configuration is tuned on a different model than the headline.**
  This is the sharp edge of the decision: prompts and thresholds swept on Haiku
  may not be optimal for Sonnet. Mitigation: the chosen configuration is run on
  **both** models and the results table carries both rows, so the gap is visible
  rather than assumed away. If it turns out to be large, the honest fix is a
  second, smaller sweep on Sonnet — which is a budget decision made with a
  number in hand.
- **`thinking` has exactly one shape.** `beacon_llm.capabilities` asked the
  Models API (2026-11-08): `thinking.types.enabled.supported` is false on all
  three models, `adaptive` is true on all three, `disabled` is false on Sonnet
  and Opus. So every call uses `thinking={"type": "adaptive"}` and
  `output_config.effort` is the only dial — low for query generation, higher for
  rule checks.
- **A model deprecation is a one-file change.** Prices and IDs are data in
  `pricing.py` and `AnthropicSettings`; nothing downstream multiplies a
  hardcoded rate.
- Running ~20 parallel calls per patient against a shared tokens-per-minute
  limit means the gateway's rate limiter is not the only limiter that matters; a
  shared LLM TPM budget across workers is part of the design and not yet built.
