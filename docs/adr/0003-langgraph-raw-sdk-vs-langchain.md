# ADR 0003 — LangGraph for orchestration, the raw Anthropic SDK inside every node

- **Status:** accepted
- **Date:** 2026-11-08
- **Affects:** `libs/beacon-llm/**`, `services/match_worker`, `beacon_core.schemas.MatchState`

## Context

The match pipeline is a small state machine. `beacon_core.schemas.PipelineStep`
fixes its vocabulary at six steps:

```
profile_patient → generate_queries → retrieve → check_rules → verify → rank_and_explain
```

It needs four things from an orchestration layer: typed state, a checkpoint
after every step so a crashed worker resumes mid-pipeline, fan-out to ~20
parallel per-trial calls, and retries with backoff.

It needs four *different* things from the LLM layer, and these are the ones that
decided the ADR, because each fails **silently** when a framework gets it wrong:

1. **Byte-exact prompt-cache ordering.** One patient means ~20 rule-check calls
   that share a large prefix (standing instructions + the numbered note) and
   differ only in one trial's rules. Put the shared part first with a cache
   breakpoint after it and 19 of 20 calls read it back at 0.1× (Haiku) or 0.05×
   (Sonnet/Opus) of input price. Put the variable part first and the prefix is
   no longer byte-identical: no error, no warning, a bill 5–20× larger.
2. **The raw `usage` object.** `cache_read_input_tokens`,
   `cache_creation_input_tokens` and the exact input total are needed to price a
   call at all — and Haiku 5.5's 5× cliff at 100,000 prompt tokens counts cache
   reads toward the threshold (ADR 0005). A wrapper that normalises `usage` into
   `{prompt_tokens, completion_tokens}` destroys the only data that makes the
   cost meter correct.
3. **Structured outputs against a frozen schema.** `CriterionJudgment` is the
   JSON schema sent to `client.messages.parse()`. Adding a field changes the
   schema, busts the cached prefix and invalidates every recorded fixture.
4. **Record/replay with no fallthrough to the network.** CI must be free and
   deterministic, and a store that silently fell through to a live call would
   turn that from a guarantee into a hope.

## Options considered

| Option | Verdict |
|---|---|
| **Raw SDK + hand-rolled orchestration** (no graph library) | Rejected. Checkpoint-per-node, resumable state and parallel fan-out are the parts worth not writing, and writing them badly is the likeliest source of a lost job. |
| **LangChain LCEL chains + `langchain_anthropic`** | Rejected. The wrapper owns message assembly and normalises `usage`, which breaks requirements 1 and 2 above invisibly. Also the single answer an interviewer cannot be given is "the framework did that". |
| **LlamaIndex** | Rejected. Its strength is the retrieval/indexing layer, which is exactly the part Beacon writes itself in order to be able to defend it. |
| **LangGraph + LangChain's LLM wrappers** | Rejected — half the problem. LangGraph is the part worth having; the wrapper is the part that hurts. |
| **LangGraph for orchestration + the raw Anthropic SDK inside each node** | **Chosen.** |

## Decision

**LangGraph owns the graph. Nothing between Beacon and the Anthropic SDK.**

LangGraph provides:

- typed state — `MatchState`, a Pydantic model holding the profile, the numbered
  note, queries, candidates, judgments, the ranked answer, status, current step
  and `total_cost_usd`;
- a checkpoint written to Postgres after every node, which is what makes a
  worker killed during step 5 resume at step 5 and not step 1;
- `Send`-based fan-out for the 20 parallel rule checks;
- per-node retry configuration.

`libs/beacon-llm` is the whole LLM layer, written deliberately, one module per
hazard:

| Module | What it owns | Why it cannot be delegated |
|---|---|---|
| `prompt.py` | `CachedPrompt` — three fixed string slots (`instructions`, `note`, `variable`) rendered in one order | There is no constructor taking a list of blocks and no parameter that reorders them, so no caller *can* put variable text in the cached prefix |
| `pricing.py` | the price table as data, the 100K cliff, per-model cache multipliers | Writing "× 0.1" once would overcharge the two models the project reports on |
| `cost.py` | `Decimal` arithmetic over the real `Usage` | Per-call costs are ~1e-5 USD summed into a README figure |
| `guard.py` | counts prompt tokens *before* the request leaves the process, raises above the ceiling | Makes "batch all 20 trials into one call" a loud failure instead of a 5× invoice |
| `fixtures.py` | record/replay keyed by prompt hash, **default `replay`**, no path from a miss to the network | "Zero network calls in CI" has to be a guarantee |
| `capabilities.py` | asks `GET /v1/models/{id}` instead of hardcoding a capability table | A hardcoded table is wrong the moment a model ships |
| `errors.py`, `client.py` | typed failures, `stop_reason` handling including refusals | A refusal read as an answer is a wrong judgment with a confident explanation |

## Consequences

- **Four or five modules exist that a framework would have supplied.** That is
  roughly the cost of this decision, and it is real work that is not pipeline
  logic.
- **The payoff is that every number is derivable by hand.** The cost per patient
  comes from a price table we can point at, applied to a `usage` object we
  logged, with the billed tier recorded on the row.
- **LangGraph's checkpointer couples the worker to its state-serialisation
  format.** A LangGraph major version bump is a migration of in-flight
  checkpoints. Accepted: in-flight jobs are seconds long and the queue can be
  drained before an upgrade.
- **No LangSmith.** Observability is OpenTelemetry into Arize Phoenix, which the
  compose stack already runs, plus the `llm_calls` table for dollars.
- `capabilities.py` records a finding that shaped the client: on all three of
  Haiku 5.5, Sonnet 5.5 and Opus 5.5 (checked 2026-11-08),
  `thinking.types.enabled.supported` is **false** and `adaptive` is **true**, and
  `disabled` is false on Sonnet and Opus. So there is exactly one thinking code
  path — `thinking={"type": "adaptive"}` on every call, with
  `output_config.effort` as the only dial. A per-model branch would have nothing
  to branch on. Hardcoding the older `{"type": "enabled", "budget_tokens": N}`
  shape would simply have failed.
- Prototyping is slower than chaining prewritten components. This is the trade
  the project is making on purpose: the pipeline is the thing being examined in
  an interview, so it is the thing that has to be ours.
