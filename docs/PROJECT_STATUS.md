# Beacon: project status and code review (2026-11-14)

Short version: **the engineering is strong, and the evidence isn't there yet.** The code
reads like a senior engineer's: hybrid RAG, a checkpointed agent graph, an outbox, a DLQ and a
real eval harness. CI is green with about 760 unit tests. But no benchmark numbers are
published yet, and a live match has never been shown end to end in the console. For an AI
engineering portfolio, **the numbers are the headline**, so everything below is ordered to
get them.

---

## 1. What works today (verified 2026-11-14)

| Area | State | Quality |
|---|---|---|
| CI: ruff, format, mypy (146 files), pytest (about 760 unit tests) | all green | strong |
| Docker stack (Postgres, Redis, Qdrant, Redpanda, Phoenix) | 5/5 healthy | done |
| Gateway: `POST /v1/matches` with job + outbox in one transaction, `GET` result, SSE endpoint, chat | works | strong; **live progress missing, see F1** |
| Ingestion: TREC → Postgres + Qdrant (dense + BM25), age/sex parser, rule splitter | **239,336 / 375,580 trials indexed (64%)** | strong; resumable |
| Retrieval: hybrid search, two-level RRF k=60, NULL-safe filters inside HNSW, cross-encoder rerank | works, about 34 s per request on CPU | **best part of the repo** |
| Match worker: 7-node LangGraph, `Send` fan-out, Postgres checkpoints, offset after checkpoint, DLQ, budget cap | works (3 runs completed on 9 Nov) | strong; **citation bug, see F2** |
| LLM layer: structured outputs, cache-ordered prompts, cost meter, Sonnet only for `check_rules` | works | strong |
| Evals: TREC metrics, bootstrap CIs, ablation table, Opus judge + Cohen's κ, provenance guard | harness done, **0 numbers produced** | strong code, no output |
| Web console (Next.js): evidence highlighting, fixture replay, lookup by ID | works | good |
| Observability (OTel → Phoenix) | `telemetry.py` is a no-op stub | not started |

**Agents:** none are running. All 10 agent branches are merged into master. One leftover holds
unfinished work (see F4). The other worktrees are clean and can be deleted.

---

## 2. Code review findings (most severe first)

**F1. The console never updates live. It sits on "Queued" even after the run finishes.**
*High; this is the other half of the Queued bug.* **FIXED 14 Nov:** `gateway/bridge.py` (shared consumer group `beacon-gateway-progress` → `PUBLISH chan:match:{id}`), plus the SSE stream re-reads Postgres on every heartbeat so a lost event can't leave the page stuck. Tests: `services/gateway/tests/test_progress_bridge.py`.
The SSE endpoint (`services/gateway/src/gateway/router.py:417`) only forwards messages from
the Redis channel `chan:match:{id}`. **No production code publishes to that channel.** The
worker writes progress and completion to the outbox, and from there to Kafka (`match.progress`,
`match.completed`), and nothing consumes those topics. `ARCHITECTURE.md` draws this bridge
(`K -.-> G`, then `G <-> Redis`), but it was never built.
- Workaround today: copy the match ID, wait about 2 minutes, then paste it into the console's
  "look up" box.
- Fix (free, no LLM): a background task in the gateway that consumes `match.progress` and
  `match.completed` and `PUBLISH`es each event to `chan_match(match_id)`. It is a Kafka
  `@KafkaListener` that fans out to a Redis topic. It can be tested with a fake event, so
  nothing gets billed.

**F2. The citation gate is bypassed. A hallucinated citation can still exclude a trial.**
*High; correctness, and it breaks a README metric.*
`services/match_worker/src/match_worker/nodes.py:550-552` **drops** invalid sentence IDs
*before* the verdict is derived. After that, `citations_valid()` (`verdict.py:65`) is always
true, so:
1. A `met` on an exclusion rule that cites a nonexistent sentence becomes `met` with no
   citations. That counts as valid (`all([]) == True`) and **excludes the trial**, which is
   exactly what `verdict.py`'s docstring says must never happen.
2. `invalid_citation_rate`, a README headline metric, always reads 0.

Fix: compute validity on the raw IDs before filtering, and keep the raw IDs in the row. Also
decide whether a `met`/`not_met` with **zero** citations should be demoted to
`insufficient_info` (recommended). Add a node-level test. Unit tests are free.

**F3. Seven stuck "queued" runs, and nothing cleans them up.** *Medium; data hygiene.* **FIXED 14 Nov:** all 10 unfinished runs deleted (with checkpoints, outbox rows and orphan patients; backup in the session scratchpad), and the worker's Kafka group was moved to the end (lag 0). The 3 completed runs are kept as examples. There is still no automatic reaper.
The runs from 8–9 Nov have outbox rows marked published (by hand, all in the same batch at
21:29:30) but were never put on Kafka. They will show "Queued" forever. Mark them `failed`
with a note. Optionally add a reaper that fails runs queued for more than N minutes.

**F4. Unfinished eval work in a leftover worktree.** *Medium; blocks task 3.* **FIXED 14 Nov:** brought into master with the missing tests. New trap noted in its docstring: every row keeps `rerank_top_k=20`, so the run-file generator must ask for 100 candidates or "Recall@100" silently becomes recall at 20.
`.claude/worktrees/agent-ab1fb1e8db908d203` has 60 uncommitted lines in
`evals/src/evals/ablations.py`: `FREE_ABLATION_TABLE` (5 rows, $0) and
`RunConfig.requires_llm`. Its docstring says "a test asserts every row here is free", but that
test does not exist. Bring the change into master and write the test.

**F5. The docs have drifted.** *Low.* `ARCHITECTURE.md:98` still says "Steps 3–7 are not yet
implemented", and the diagram says "6 nodes" where the worker has 7. A reviewer who reads the
docs first will notice.

**F6. Housekeeping.** *Low.* Ten stale worktrees and their `agent/*` branches. After
F4, delete them with `git worktree remove` + `git branch -d`.

Checked and fine: the outbox relay marks a row published only after `send_and_wait`. The
offset is committed per partition, after the checkpoint. The budget is re-seeded from the
checkpoint on resume. The filters handle missing values correctly and run inside the
search. RRF settings resolve through `model_fields_set`.

---

## 3. What is left, in order (apps close 2026-11-15)

| # | Task | Who | Cost |
|---|---|---|---|
| 1 | Fix F1 (progress bridge) + F2 (citation gate) + F3 + F4 | Claude | free |
| 2 | **Start the ingest now**: about 136k trials, several hours of CPU. Every number needs it. | Claude / Hasan | free |
| 3 | Live match in the console, end to end | **Hasan** | about $0.10–0.50 per match, capped at $0.50 |
| 4 | Free eval rows: BM25 / dense / hybrid / rerank / filters → Recall@100, nDCG@10 | Claude | free |
| 5 | Rewrite the `TODO(owner)` prompt in `rule_check_prompt.py`; about 150 hand labels for κ | **Hasan** | free |
| 6 | Paid rows: N=3 query generation, `full_system` headline | **Hasan** | about $3 + $1 + $12.50 |
| 7 | README: replace the `{placeholders}` (13 metrics) with real numbers; add a 30 s demo GIF | Claude | free |
| 8 | Fix the docs drift (F5), clean up worktrees (F6) | Claude | free |
| 9 | OTel → Phoenix traces (C3). Nice to have, so cut it if time runs out. | Claude | free |

**If there's only time for one thing:** get the ingest to 100%, then run the free retrieval
ablation table. "Hybrid + rerank beats BM25 by X on Recall@100 on TREC CT 2021, with 95%
CIs" is the single sentence that makes this an AI engineering project and not just a backend
project.

---

## 4. What Hasan should review personally (interviewers will ask)

Read these until you can explain them without notes, in this order:

1. `services/retrieval/src/retrieval/filters.py`: why missing values must pass, and why the
   filter goes *inside* the HNSW walk. This is the best interview story in the repo.
2. `services/retrieval/src/retrieval/fusion.py` + `service.py`: RRF and the funnel
   (375k → 100 → 20), and why the reranker sees only the rank-1 query.
3. `services/match_worker/src/match_worker/verdict.py`: the verdict is computed in code, not
   by the LLM. Know F2 and its fix.
4. `services/match_worker/src/match_worker/nodes.py` (`check_rules`) + `state.py`: the `Send`
   fan-out, the reducers, and the order that makes the cached prompt prefix reusable.
5. `services/match_worker/src/match_worker/runner.py` + `relay.py`: at-least-once delivery,
   committing after the checkpoint, the outbox. In Spring terms, a transactional outbox plus a
   manual-ack `@KafkaListener`.
6. `evals/src/evals/metrics.py`, `bootstrap.py`, `judge.py`: what Recall@100 / nDCG@10 mean,
   why bootstrap CIs, and what Cohen's κ proves about the judge.
7. `services/match_worker/src/match_worker/rule_check_prompt.py`: **yours to write.** Its
   wording is the part of the system you can most credibly claim as your own.
