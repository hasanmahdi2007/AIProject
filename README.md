# Beacon

Clinical-trial matching: a patient note in, a ranked list of trials out, with
every eligibility decision traced to the sentence it came from.

> **Decision support, not medical advice.** Beacon is a research and portfolio
> system evaluated on the public TREC Clinical Trials benchmark with
> **synthetic** patient notes. It has no clinical validity, it must not be used
> for care decisions, and no real patient record should ever be entered into it.

**Status:** Wave 0 (foundation and contracts). Not yet runnable end to end.
See `docs/AGENT_BRIEFS.md` for what is built, by whom, and in what order.

---

## What it does

375,000 trials cannot go into a prompt, so the work splits in two, and that
split is what "RAG" means here:

| Stage | Cost | Does |
|---|---|---|
| **Retrieve** | milliseconds, no LLM | 375,000 → ~100 → 20 candidates |
| **Read** | seconds, 20 parallel Claude calls | judges every eligibility rule of those 20 |

Retrieval decides what Claude is *allowed to see*, so a good trial dropped
there can never be recovered. That is why **Recall@100** is the metric that
governs the whole system.

The pipeline: profile the patient → generate 3 facet-diverse queries → hybrid
dense + BM25 search with age/sex filters applied **inside** the index traversal
→ fuse with RRF (k=60) → cross-encoder rerank to 20 → judge each rule with a
cited sentence ID → verify the citations → rank and explain.

Three independent layers decide different things, and only one of them decides
eligibility:

| Layer | Answers | Numerically reliable |
|---|---|---|
| Embeddings + BM25 + reranker | is this trial *about* the right disease? | no, and it does not need to be |
| Hard payload filters | exact range comparison on typed fields | yes |
| LLM rule checker | does this patient meet *this rule*? | yes, with a citation |

Embeddings cannot compare numbers — a 24-year-old's note will happily retrieve
a trial requiring age 25+. That is not a bug, because retrieval never decides
eligibility.

## Design documents

| File | What it is for |
|---|---|
| `Beacon_Project_Guide.pdf` | the spec: what to build, day by day |
| `Beacon_Design_Decisions.pdf` | **why** — ADRs, the traps, the cost model |
| `Beacon_System_Design.pdf` | **what** — flowcharts, toolkit, contracts (keep this one open) |
| `docs/AGENT_BRIEFS.md` | how the work is split across agents, with the file-ownership matrix |

## Getting started

Developed natively on Windows against Docker Desktop. CI runs Ubuntu, so the
code stays path-agnostic.

```powershell
# 1. Install uv (once), then restart the shell
irm https://astral.sh/uv/install.ps1 | iex

# 2. Install the workspace from the lockfile
uv sync --locked

# 3. Configure
Copy-Item .env.example .env     # then set ANTHROPIC_API_KEY

# 4. Check it
.\tasks.ps1 imports             # every workspace package imports
.\tasks.ps1 check               # ruff + mypy + pytest, the same as CI
```

`tasks.ps1` is a thin wrapper, not a build system — every command it runs is
written out below, so a Linux cloner needs no PowerShell:

| Task | Command it runs |
|---|---|
| `sync` | `uv sync --locked` |
| `imports` | `uv run python -c "import beacon_core, beacon_llm, gateway, retrieval, match_worker, ingestion, evals"` |
| `up` / `down` / `ps` | `docker compose -f deploy/compose.yaml up -d` / `down` / `ps` |
| `lint` / `fmt` / `types` | `uv run ruff check .` / `ruff format .` / `mypy .` |
| `test` | `uv run pytest -m "not integration"` |
| `check` | all four of lint, `ruff format --check`, types, test |
| `migrate` | `uv run alembic upgrade head` |
| `gateway` | `uv run uvicorn gateway.main:app --reload --port 8000` |

Docker Desktop must be running before `up` or the integration tests;
`tasks.ps1` checks `docker info` first and says so plainly if it is not.

## Layout

A `uv` workspace. Every package is importable from day one, even where the body
is a stub — that is what makes "CI green" mean something.

```
libs/beacon-core/        schemas, settings, models, db, logging, redis, kafka, telemetry
                         FROZEN CONTRACTS - read-only to every agent after Wave 0
libs/beacon-llm/         Claude wrapper: structured outputs, caching, cost meter, record/replay
services/gateway/        FastAPI: auth, rate limits, /v1/matches, SSE
services/retrieval/      FastAPI: hybrid search + RRF + rerank
services/match_worker/   LangGraph pipeline, Kafka consumer
services/ingestion/      TREC corpus to Postgres + Qdrant
evals/                   TREC loaders, ranx metrics, ablations, reports
alembic/                 migrations (orchestrator-owned)
deploy/                  compose.yaml and service configs
web/                     Next.js console
```

## Data sources

Official sources only, no scraping of third-party aggregators:

- **TREC Clinical Trials 2021/2022** via `ir_datasets` (NIST) — the 375K-trial
  corpus, 125 synthetic patient topics, and expert graded relevance judgments.
- **ClinicalTrials.gov API v2** — the official JSON REST API, phase 2 only.

Claude's web search stays **out** of the matching loop entirely: every citation
must resolve into the indexed corpus, or the faithfulness metric measures
nothing. Only the ingestion job touches the network.

Tuning happens on the 2021 split. Headline numbers come from held-out 2022, and
no number is published that was not measured there.
