# Next session: start here

Brief handoff, written 2026-11-14. For the details (ports, traps, measured facts) read
[docs/RESUME_HERE.md](docs/RESUME_HERE.md). This file covers only what you need to start.
**Read [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) next**: the code review findings (F1–F6)
and the remaining-work order supersede the task list below.
**After 15 Nov:** build the weekly live ClinicalTrials.gov sync, designed in
[docs/LIVE_SYNC_PLAN.md](docs/LIVE_SYNC_PLAN.md). Never update the frozen TREC index.

## What Beacon is

**Beacon matches a patient to clinical trials using RAG.** You paste a free-text patient note
and get back ranked ClinicalTrials.gov trials, each with clickable evidence for every
eligibility criterion. It is evaluated on TREC Clinical Trials 2021/2022 and compared with
TrialGPT. It is Hasan's flagship portfolio project, and FAANG internship applications close on
**2026-11-15**.

**How a request flows:** web console (Next.js, :3000) → gateway (FastAPI, :8000) writes the job
and an outbox row in one transaction → the worker's relay publishes the outbox to Kafka
(Redpanda) → the match worker (a 7-node LangGraph with Postgres checkpoints and a DLQ) calls the
retrieval service (:8001: hybrid BM25 + dense in Qdrant, RRF k=60, NULL-safe age/sex filters,
cross-encoder rerank) → Claude checks the rules (Sonnet for the eligibility decision, Haiku for
everything else) → progress streams back to the console over SSE.

Stack: Python/uv workspace (`libs/`, `services/`, `evals/`), Postgres 15432, Redis 16379,
Qdrant 6333, Redpanda 9092, Phoenix 6006, all in `deploy/compose.yaml`. Every command goes
through `.\tasks.ps1` (the Makefile for this repo).

## Ground rules (Hasan's)

- **Never spend API credits yourself.** Running the worker, a live match, or an LLM eval row
  bills Hasan's key. Only Hasan runs those. Free work (unit tests, ingest, retrieval-only evals)
  is fine.
- Explain things in Java/Spring Boot terms first (outbox ≈ transactional outbox with
  `@Transactional`; alembic ≈ Flyway).
- Never run a bare `git add -A`. Stage files by name.

## How to run it (4 terminals, Docker Desktop must be up)

```powershell
.\tasks.ps1 up; .\tasks.ps1 ps     # all five containers must read "healthy"
.\tasks.ps1 gateway                # :8000
.\tasks.ps1 retrieval              # :8001
.\tasks.ps1 worker                 # SPENDS CREDITS - Hasan only
cd web; npm run dev                # :3000
```

## Just fixed (2026-11-14), not committed yet

**The bug:** "Match trials" showed *Queued — accepted, not started* and never moved.
- **Root cause:** nothing started the match worker. The gateway only writes an outbox row. The
  worker's relay is what publishes it to Kafka, and the worker is what consumes it. `tasks.ps1`
  had no `worker` task and RESUME_HERE never said to start one, so every job sat in the outbox.
  **Fix:** a new `.\tasks.ps1 worker` task.
- Two more bugs were already fixed in the working tree, and they would have hit right after:
  1. `runner.py`: a leftover junk Kafka message (`{"probe": "a1-infra"}`) crashed the worker on
     every start. Undeserialisable messages are now logged and dropped.
  2. `config.py` / `retrieval_client.py`: the retrieval timeout went from 30 s to 240 s. Reranking
     100 candidates on CPU takes about 34 s, so every match had timed out into the DLQ.
- Match-worker unit tests pass (offline). The whole flow has **not** been run end to end, because
  that spends credits. **Hasan tests it.**
- **Warning:** when the worker first starts, it also processes every job already queued from
  earlier clicks, and each one bills. To clear old jobs first, ask the next session how.

**To do:** commit the uncommitted work. Stage files by name (see `git status`): the worker
fixes, the progress bridge (`gateway/bridge.py` + tests), the free ablation table, and these docs.
The design map is published at https://claude.ai/artifact/D2MDzDZay8Ni5tFjwEdaTU

## Remaining tasks, in priority order

1. **Hasan tests a live match** in the console, then commit the fix above.
2. **Finish the ingest (free, about 11 h of CPU).** On 9 Nov it was at 152,296 / 375,580 trials.
   Run `uv run python -m ingestion.run`; it resumes where it stopped. Every published number
   needs the full index.
3. **Free evals:** the BM25 / dense / hybrid / rerank / filters / N=1 ablation rows. This needs a
   small generator that writes TREC run files, then `uv run python -m evals.report`.
   Recall@100 costs $0.
4. **Hasan only:** rewrite the `TODO(owner)` prompt in `match_worker/rule_check_prompt.py` (keep
   it above 512 tokens), and write about 150 hand labels in
   `evals/calibration/hand_labels.template.jsonl` for Cohen's κ.
5. **Paid eval rows (Hasan runs them):** N=3 query generation and `full_system`. Estimated at
   about $3 on Haiku, $1 for the Opus judge and $12.50 for the Sonnet headline run.
6. **C3 observability:** OpenTelemetry → Phoenix, with trace context carried in Kafka headers.
   Fill in the stub at `libs/beacon-core/src/beacon_core/telemetry.py`. Do it on its own, since
   it touches every package.
7. Replace the `{placeholders}` in README with the real numbers. Never publish a number measured
   against a partial index; the harness blocks that anyway.
