# Beacon — Agent Briefs

**Purpose.** One self-contained brief per agent, written to be **copy-pasted directly into an agent prompt**.
Each agent starts cold with no memory of the design conversation, so every brief restates what it needs.

**How to use this file.** Pick a role, read its *Preconditions*, copy the fenced **Paste-ready prompt**
into a new agent, and review the resulting PR against its *Definition of Done*. Nothing else.

**Version.** v1.0 — 8 Nov 2026. Companion documents: `Beacon_Project_Guide.pdf` (what to build),
`Beacon_Design_Decisions.pdf` (why), `Beacon_System_Design.pdf` (flowcharts + toolkit).

---

## 1. Operating protocol

| Rule | Detail |
|---|---|
| **Isolation** | Every agent works in its **own git worktree** on its own branch `agent/<id>-<slug>`. Parallel agents in one tree collide even with disjoint paths. |
| **Integration** | Agent opens a PR into `master`. The **orchestrator** merges, in wave order. Agents never merge their own work. |
| **Approval gate** | **Hasan reviews and accepts each wave before the next wave launches.** No exceptions — this is the control point. |
| **Contracts are frozen** | `libs/beacon-core/` schemas and settings are written once in Wave 0 and are **read-only** to every later agent. |
| **Shared files are orchestrator-owned** | `uv.lock`, root `pyproject.toml`, `alembic/versions/**`, `.github/workflows/ci.yml`. An agent that needs a dependency or a migration **requests it in its PR description**; it does not edit those files. |
| **No scope creep** | An agent that believes it needs to touch another agent's path stops and reports instead of doing it. |
| **Every PR must be green** | `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass before review. |
| **No `conftest.py` inside a package's test directory** | The repo has exactly one, at `tests/conftest.py`. A second one anywhere (e.g. `services/gateway/tests/conftest.py`) makes `mypy .` fail with *"Duplicate module named conftest"*, because both resolve to the bare module name `conftest`. Adding `__init__.py` trades that for a pytest failure (*"Plugin already registered under a different name"*), and the real fixes — `explicit_package_bases` for mypy, `consider_namespace_packages` for pytest — cannot work through a hyphenated directory like `libs/beacon-llm/`. **So: put shared test material in a uniquely-named module inside your own package** (A3's `beacon_llm/_testing.py` is the reference) and import it from your tests. Found by A3, 8 Nov. |

### Dependency graph and critical path

```
W0  Foundation / Contracts        (serial — blocks everything)
     |
     +--> A1 Infra & Compose ----+
     +--> A2 Gateway             |
     +--> A3 LLM library --------+--> B1 Ingestion --> B2 Retrieval --> B3 Evals
     +--> A4 TREC data ----------+                          |              |
                                                            +--> C1 Worker +
                                        C3 Observability  (SOLO — no other agent active)
                                        C4 Web console    (independent after A2)
                                        D  Docs & ADRs    (independent, any time)
```

**Critical path: `W0 → A4 → B1 → B2 → B3`.** B1 contains the 30–90 minute embedding run — the only
unavoidable wall-clock wait in the project. Unblock B1 as early as possible and run it overnight.

### File ownership matrix

| Path | Exclusive owner | Everyone else |
|---|---|---|
| `libs/beacon-core/**` | W0 / orchestrator | read-only, permanently |
| `uv.lock`, root `pyproject.toml` | orchestrator | request, do not edit |
| `alembic/versions/**` | orchestrator | request, do not edit |
| `.github/workflows/ci.yml` | orchestrator | do not edit |
| `deploy/**` | A1 | do not touch |
| `services/gateway/**` | A2 | do not touch |
| `libs/beacon-llm/**` | A3 | do not touch |
| `evals/src/evals/trec.py` | A4 | do not touch |
| `services/ingestion/**` | B1 | do not touch |
| `services/retrieval/**` | B2 | do not touch |
| `evals/src/evals/{metrics,ablations,report}.py` | B3 | do not touch |
| `services/match_worker/**` | C1 | do not touch |
| `web/**` | C4 | do not touch |
| `docs/**` | D | do not touch |

### Role index

| ID | Role | Wave | Parallel with | Rough effort |
|---|---|---|---|---|
| **W0** | Foundation & Contracts | 0 | — (serial) | 3–4 h |
| **A1** | Infrastructure & Compose | 1 | A2, A3, A4 | 2 h |
| **A2** | Gateway service | 1 | A1, A3, A4 | 3 h |
| **A3** | Claude LLM library | 1 | A1, A2, A4 | 4 h |
| **A4** | TREC data access | 1 | A1, A2, A3 | 2 h |
| **B1** | Ingestion pipeline | 2 | B2*, B3* | 5 h + overnight |
| **B2** | Retrieval service | 2 | B1, B3 | 5 h |
| **B3** | Evaluation harness | 2 | B1, B2 | 4 h |
| **C1** | Match worker (LangGraph + Kafka) | 3 | — | 6 h |
| **C3** | Observability | 3 | **none — runs alone** | 3 h |
| **C4** | Web console | 3 | anything | 4 h |
| **D** | Docs, README, ADRs | any | anything | 3 h |

\* B2 and B3 can start against a 10,000-trial subset or a mocked contract before B1 finishes the full index.

### What you should write yourself

The guide is explicit: *"You must be able to explain every important line."* Project deep-dive
interviews target three things — **retrieval fusion, the rule checker with citations, and the
evaluation metrics**. Delegate scaffolding; write those three yourself.

| Safe to delegate | Write yourself (pair, don't delegate) |
|---|---|
| W0 scaffold, A1 Compose, A4 loaders, C4 console, D docs | **B2** — RRF fusion, filter construction, rerank |
| A2 gateway migration, C3 OTel wiring | **C1 / A3** — the rule-check prompt and `CriterionJudgment` handling |
| B1 plumbing (the parsers still need your review) | **B3** — nDCG/recall wiring, kappa calibration, bootstrap CIs |

---

## 2. W0 — Foundation & Contracts

> **Serial. Nothing else starts until this is merged.** Every other agent imports what this produces.

**Mission.** Create the `uv` workspace, the package skeletons, and the frozen contracts that let
four agents work in parallel without disagreeing about data shapes.

**Exclusive write paths.** Root `pyproject.toml`, `uv.lock`, all nine package `pyproject.toml`,
`libs/beacon-core/**`, `alembic/**`, `.env.example`, `tasks.ps1`, `.github/workflows/ci.yml`, `.gitignore`.

**Read-only context.** `Beacon_Design_Decisions.pdf` §5 (contracts), `Beacon_System_Design.pdf` p5,
existing `main.py` / `router.py` / `service.py` / `requirements.txt`.

**Preconditions.** None.

**Deliverables.**
- `uv` workspace: `[tool.uv.workspace] members = ["libs/*", "services/*", "evals"]`, `requires-python = ">=3.12"`
- Nine packages, each with `pyproject.toml` and an importable `src/<pkg>/__init__.py`
- `beacon_core.schemas` — `PatientProfile`, `NumberedNote`, `TrialCandidate`, `CriterionJudgment`, `MatchState`
- `beacon_core.settings` — one `Settings` (pydantic-settings), nested: `postgres`, `redis`, `qdrant`, `kafka`, `anthropic`, `ir_datasets_home`
- `beacon_core.logging` — structlog JSON renderer, trace-ID processor stubbed
- `beacon_core.db` — async SQLAlchemy 2 engine + `async_sessionmaker`
- `beacon_core.{kafka,redis,telemetry}` — typed stubs that import cleanly
- `beacon_core.protocols.ConversationStore` — Protocol only, no implementation
- Alembic initial migration creating the ten tables from `Beacon_System_Design.pdf` p5
- Retrieval service request/response models (in `beacon_core.schemas`), **no implementation**
- Kafka event payload models
- `.env.example` with every key and no values; `tasks.ps1` (`up`/`down`/`lint`/`types`/`test`), with a `docker info` precondition check
- CI on `ubuntu-latest`: `astral-sh/setup-uv`, `uv sync --locked`, ruff, mypy, pytest
- Delete `requirements.txt`; `git mv` the three root `.py` files into `services/gateway/src/gateway/` **unmodified** (A2 fixes them)

**Definition of Done.**
```bash
uv sync --locked
uv run python -c "import beacon_core, beacon_llm, gateway, retrieval, match_worker, ingestion, evals"
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest
uv run alembic upgrade head        # against a scratch Postgres
```
CI green on the PR.

**Must not do.** No business logic. No Docker Compose (that is A1). No fixing the gateway bugs
(that is A2). No guessing at `ir_datasets` dataset IDs (that is A4).

**Known traps.**
- Code must be path-agnostic (`pathlib`, no backslash literals) — developed on Windows, CI runs Ubuntu.
- `anthropic >= 2.3.0` is already in the current venv; carry that floor forward.
- Schema names are frozen the moment this merges. Get `CriterionJudgment` exactly right.

**Paste-ready prompt.**
```
You are the Foundation agent for Beacon, a clinical-trial matching engine at
C:\Users\hasan\Desktop\AIProject (Windows, Docker Desktop, Python 3.12.10).

Read first: docs/AGENT_BRIEFS.md section "W0", plus Beacon_System_Design.pdf page 5
(the contracts page) and Beacon_Design_Decisions.pdf section 5.

Your job is the uv workspace, the nine package skeletons, and the frozen contracts in
libs/beacon-core. Everything in the W0 Deliverables list in that brief, nothing beyond it.

Hard constraints:
- You own ONLY the paths listed under "Exclusive write paths" in the W0 brief.
- Write NO business logic. Stubs and contracts only.
- Use pathlib everywhere; CI runs on ubuntu-latest although development is on Windows.
- git mv main.py, router.py, service.py into services/gateway/src/gateway/ WITHOUT
  changing their contents. A later agent fixes them.
- Delete requirements.txt once uv.lock exists.

Done when all four commands in the W0 "Definition of Done" pass. Work on branch
agent/w0-foundation and open a PR. Report anything you could not do rather than
working around it.
```

---

## 3. A1 — Infrastructure & Compose

**Mission.** A five-service Docker Compose stack where `docker compose ps` reports **healthy**, not merely *running*.

**Exclusive write paths.** `deploy/**`.

**Read-only context.** `Beacon_System_Design.pdf` p1 and p4 (ports), `beacon_core.settings`.

**Preconditions.** W0 merged.

**Deliverables.** `deploy/compose.yaml` with healthchecks and named volumes; `deploy/init.sql` (creates the database only — schema comes from Alembic).

| Service | Image | Port | Healthcheck |
|---|---|---|---|
| postgres | `postgres:17` | **15432** on the host (5432 inside) | `pg_isready` + `SELECT 1` |
| redis | `redis:8` | **16379** on the host (6379 inside) | `redis-cli ping` |
| qdrant | `qdrant/qdrant` | 6333 | `/readyz` |
| redpanda | `redpandadata/redpanda` | 9092 | `rpk cluster health` |
| phoenix | `arizephoenix/phoenix` | 6006 | HTTP root |

**Definition of Done.**
```bash
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml ps     # all five report (healthy)
docker compose -f deploy/compose.yaml down && docker compose -f deploy/compose.yaml up -d
# volumes persisted: postgres still has its database
```

**Must not do.** No Python. No schema DDL in `init.sql` beyond `CREATE DATABASE` — Alembic is the single source of truth for schema.

**Known traps.** `depends_on: condition: service_healthy`, not bare `depends_on`. Redpanda needs explicit advertised listeners to be reachable from the host *and* from other containers. Pin image tags.

**Paste-ready prompt.**
```
You are the Infrastructure agent for Beacon at C:\Users\hasan\Desktop\AIProject
(Windows + Docker Desktop).

Read first: docs/AGENT_BRIEFS.md section "A1".

Build deploy/compose.yaml with exactly five services - postgres:17, redis:8,
qdrant/qdrant, redpandadata/redpanda, arizephoenix/phoenix - each with a real
healthcheck and a named volume where it stores data. Add deploy/init.sql that
creates the database and nothing else (Alembic owns the schema).

Hard constraints:
- You own ONLY deploy/**. Touch nothing else, especially no Python and no CI config.
- Pin every image tag. Use depends_on with condition: service_healthy.
- Redpanda must be reachable both from the host and from sibling containers -
  get the advertised listeners right.

Done when `docker compose -f deploy/compose.yaml ps` shows all five as (healthy),
and a down/up cycle preserves the Postgres volume. Branch agent/a1-infra, open a PR.
```

---

## 4. A2 — Gateway service

**Mission.** Turn the three migrated spike files into a real gateway, fixing the three defects the guide names in §14.

**Exclusive write paths.** `services/gateway/**`.

**Read-only context.** `Beacon_Design_Decisions.pdf` §7, `beacon_core.schemas`, `beacon_core.protocols.ConversationStore`.

**Preconditions.** W0 merged.

**Deliverables.**
- `GET /health` — checks Postgres *and* Redis, returns per-dependency status
- `POST /v1/matches` — validates, writes request + outbox row in **one transaction**, returns `202` + `match_id`
- `GET /v1/matches/{id}/events` — SSE stream
- `POST /v1/matches/{id}/chat` — the migrated chat endpoint
- Redis-backed `ConversationStore` implementing the W0 Protocol
- Unit tests, including a regression test for defect 1 below

**The three defects to fix (they are the point of this task):**
1. `service.py` appends the user turn to history **before** the Claude stream opens, so a failed request poisons the history. Build the outgoing message list locally; persist user **and** assistant turns together only after the stream completes.
2. `conversations: dict[str, list[dict]]` is process memory — lost on restart, wrong across replicas. Move to Redis, keyed `chat:{conversation_id}` with a TTL, behind the `ConversationStore` Protocol.
3. Errors are written into the response **text**. Emit typed SSE events (`token` / `error` / `done`) and set `media_type="text/event-stream"`.

**Keep what is already correct:** opening the Claude stream *before* returning the response, so auth/rate-limit/validation failures become real HTTP codes; the specific-first exception chain; the `refusal` stop-reason check.

**Definition of Done.**
```bash
uv run pytest services/gateway
uv run uvicorn gateway.main:app --port 8000
curl localhost:8000/health                       # 200, postgres+redis both ok
# SSE stream emits typed events; a forced mid-stream failure emits an `error` event,
# never an error string inside a `token` payload
```

**Must not do.** Do not change the model ID — `claude-haiku-5-5` is correct. Do not implement the match pipeline (that is C1). Do not edit `beacon_core`.

**Known traps.** `MODEL = "claude-haiku-5-5"` is a **valid, current** model — leave it. The outbox row and the request row must be in the same transaction or the whole pattern is pointless.

**Paste-ready prompt.**
```
You are the Gateway agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "A2", and the three files already at
services/gateway/src/gateway/ (main.py, router.py, service.py) which were moved
there unmodified from the repo root.

Your job: turn that chat spike into a real gateway. Fix exactly the three defects
listed in the A2 brief (history poisoning on failure, in-process conversation dict,
errors written into response text). Add GET /health, POST /v1/matches returning 202
with an outbox row written in the SAME transaction, and an SSE events endpoint.

Hard constraints:
- You own ONLY services/gateway/**.
- libs/beacon-core is READ-ONLY. Import its schemas and the ConversationStore
  Protocol; do not modify them.
- Do NOT change the model ID. "claude-haiku-5-5" is a current, valid model.
- Keep the existing good behaviour: open the Claude stream before returning the
  response so errors become real HTTP codes; keep the specific-first exception
  chain; keep the refusal stop_reason check.
- Write a regression test proving a failed request does not poison conversation history.

Done when the A2 Definition of Done commands pass. Branch agent/a2-gateway, open a PR.
```

---

## 5. A3 — Claude LLM library

**Mission.** The single seam through which every Claude call passes — structured outputs, cache-ordered prompts, cost metering, budget guards, and record/replay so CI costs $0.

**Exclusive write paths.** `libs/beacon-llm/**`.

**Read-only context.** `Beacon_Design_Decisions.pdf` §10 (verified prices + the five API facts), `beacon_core.schemas`.

**Preconditions.** W0 merged.

**Deliverables.**
- `call()` wrapping `client.messages.parse()` with a Pydantic model — structured outputs via `output_config.format` (**not** the deprecated `output_format`)
- Cache-ordered prompt builder: stable instructions → patient note (**cache breakpoint**) → per-trial rules
- Cost meter reading `usage`, with the price table **and per-model cache-read multipliers as data**
- Haiku 100K prompt-token guard (see traps)
- Single adaptive-thinking path with `output_config.effort` passed per call site
- Record/replay layer: JSON fixtures on disk keyed by prompt hash
- Tests: replay runs with zero network; cost meter matches a hand-computed figure; the guard raises

**Verified price table (do not re-derive):**

| Model | ID | Input/MTok | Output/MTok | Cache read |
|---|---|---|---|---|
| Opus 5.5 | `claude-opus-5-5` | $4 | $20 | $0.20 (0.05×) |
| Sonnet 5.5 | `claude-sonnet-5-5` | $2 | $10 | $0.10 (0.05×) |
| Haiku 5.5 (≤100K prompt) | `claude-haiku-5-5` | $0.10 | $0.50 | $0.01 (0.1×) |
| Haiku 5.5 (>100K prompt) | `claude-haiku-5-5` | $0.50 | $2.50 | $0.05 |

**Definition of Done.**
```bash
uv run pytest libs/beacon-llm            # replay tests, zero network calls
# a live two-call smoke test shows usage.cache_read_input_tokens > 0 on the second call
# the cost meter's output for a known usage object matches a hand calculation
# the token guard raises above the configured ceiling
```

**Must not do.** Do not use assistant prefill (400 on current models). Do not use forced `tool_choice` on Opus 5.5 / Sonnet 5.5 (unsupported) — structured outputs instead. Do not send `budget_tokens` (rejected on all three models).

**Known traps.**
- **Haiku 5.5 has a pricing cliff at 100,000 prompt tokens** — above it rates jump 5×. The threshold counts *all* input tokens including cache reads. Count tokens before the call, refuse above a configured ceiling, and record the billed tier in the `llm_calls` row.
- Cache-read multiplier is **0.1× on Haiku but 0.05× on Opus/Sonnet**. Not a flat rate.
- Minimum cacheable prefix is **512 tokens**; below it caching silently does nothing and raises no error.

**Paste-ready prompt.**
```
You are the LLM-library agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "A3" (it contains the verified price table -
use it verbatim, do not look prices up or recall them from memory).

Build libs/beacon-llm: a single wrapper every Claude call in the project goes through.
It must provide structured outputs via client.messages.parse() with Pydantic models,
a cache-ordered prompt builder (stable instructions -> patient note -> per-trial rules,
with the cache breakpoint after the note), a cost meter with per-model cache-read
multipliers as data, a prompt-token guard, and a record/replay fixture layer so CI
makes zero network calls.

Hard constraints:
- You own ONLY libs/beacon-llm/**. libs/beacon-core is read-only.
- Use output_config.format, NOT the deprecated output_format parameter.
- Do NOT use assistant prefill, forced tool_choice, or budget_tokens - all are
  rejected or unsupported on the current models.
- Haiku 5.5 costs 5x more above 100,000 prompt tokens, counting cache reads toward
  the threshold. Guard against it and record the billed tier per call.

Done when the A3 Definition of Done passes. Branch agent/a3-llm, open a PR.
```

---

## 6. A4 — TREC data access

**Mission.** Make the benchmark loadable, with dataset IDs **discovered and documented**, never guessed.

**Exclusive write paths.** `evals/src/evals/trec.py`, `evals/tests/**`, `evals/src/evals/__main__.py`.

**Preconditions.** W0 merged.

**Deliverables.** `load_topics(year)`, `load_qrels(year)`, `iter_trials()`; dataset IDs as named constants with a comment recording how they were found; a CLI that prints one topic and its qrels count.

**Definition of Done.**
```bash
uv run python -m evals.trec --topic 2021 --first    # prints a patient note + qrels count
uv run pytest evals
```

**Must not do.** **Do not guess dataset IDs.** Enumerate `ir_datasets.registry` for entries under `clinicaltrials`, confirm the 2021 and 2022 track IDs, and record them. Do not download the full corpus inside a test.

**Known traps.** `IR_DATASETS_HOME` must point somewhere with ≥25 GB free, outside the repo, and must come from `beacon_core.settings`. The full corpus download is slow — make it resumable and keep it out of CI.

**Paste-ready prompt.**
```
You are the TREC data agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "A4".

Build evals/src/evals/trec.py exposing load_topics(year), load_qrels(year) and
iter_trials(), backed by the ir_datasets library, for the TREC Clinical Trials
2021 and 2022 tracks. Add a small CLI that prints one 2021 topic and its qrels count.

Hard constraints:
- You own ONLY evals/src/evals/trec.py, evals/src/evals/__main__.py and evals/tests/**.
- DO NOT GUESS THE DATASET IDS. Enumerate ir_datasets.registry, find the entries under
  "clinicaltrials", verify which correspond to the 2021 and 2022 tracks, and record them
  as named constants with a comment saying how you confirmed them.
- Read IR_DATASETS_HOME from beacon_core.settings; never hardcode a path.
- Tests must not download the full corpus.

Done when `uv run python -m evals.trec --topic 2021 --first` prints a patient note and
a qrels count, and pytest passes. Branch agent/a4-trec, open a PR.
```

---

## 7. B1 — Ingestion pipeline

**Mission.** Turn 375,000 TREC trials into a searchable Qdrant index plus structured Postgres rows — resumably and idempotently.

**Exclusive write paths.** `services/ingestion/**`.

**Preconditions.** W0, A1, A4 merged.

**Deliverables.**
- TREC → Postgres: structured fields (NCT ID, title, conditions, interventions, phase, status, min/max age, sex, raw eligibility text)
- Overview builder: `title + conditions + interventions + brief_summary`, **most discriminative fields first**, logging how many trials exceed the 512-token cap
- Qdrant upsert: named vectors `dense` (bge-small-en-v1.5) + `sparse` (**`modifier: IDF`**), filterable payload, payload indexes on `min_age`/`max_age`/`sex`
- Rule splitter with unit tests on real messy records
- Age/sex parser with unit tests on `"18 Years"`, `"6 Months"`, `"N/A"`, missing
- Idempotent and resumable by `nct_id`

**Definition of Done.**
```bash
uv run python -m ingestion.run --limit 1000     # fast smoke path
uv run python -m ingestion.run                  # full run, ~30-90 min
# Qdrant reports 375K points with BOTH named vectors
# re-running the full ingest changes nothing (idempotent)
# truncation rate is logged as a number
uv run pytest services/ingestion                # parser + splitter tests
```

**Must not do.** Do not fixed-size chunk the eligibility text — splitting a rule mid-sentence makes its judgment meaningless. Do not index eligibility text into Qdrant (Postgres only). Do not edit the Alembic migration — request changes.

**Known traps.**
- **`modifier: IDF` on the sparse vector field.** Omit it and BM25 silently degrades with no error.
- **Dual write.** Postgres first, then Qdrant. These can diverge — that is the known cost of choosing Qdrant, so resumability is not optional.
- Age fields are frequently absent or unparseable. **Unknown must mean "possibly eligible", never "exclude"** — store `NULL`, do not default to 0 or 999.

**Paste-ready prompt.**
```
You are the Ingestion agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "B1", and page 3 of Beacon_System_Design.pdf
(the ingestion flowchart).

Build services/ingestion: read the TREC Clinical Trials corpus via evals.trec,
write structured rows to Postgres, and upsert one Qdrant point per trial carrying
TWO named vectors (dense from bge-small-en-v1.5 via FastEmbed, sparse BM25) plus a
filterable payload. Include a rule splitter and an age/sex parser, both unit tested
against real messy records.

Hard constraints:
- You own ONLY services/ingestion/**. beacon-core and the Alembic migrations are read-only.
- The Qdrant sparse vector field MUST be configured with modifier: IDF. Without it BM25
  loses document-frequency weighting and degrades silently with no error.
- Build the overview text with the most discriminative fields FIRST (title, conditions,
  interventions, then brief summary) because bge-small truncates at 512 tokens silently.
  Log how many trials exceed the cap.
- NEVER fixed-size-chunk the eligibility criteria. Split it into individual rules on
  structure (section headers, bullets, numbering). Eligibility text goes to Postgres only,
  never into Qdrant.
- Unparseable or missing age MUST be stored as NULL meaning "possibly eligible".
  Never default it to 0 or 999 - that silently deletes good trials.
- The whole job must be resumable and idempotent by NCT ID.

Done when the B1 Definition of Done passes. Branch agent/b1-ingestion, open a PR.
```

---

## 8. B2 — Retrieval service

> **Recommended: write this one yourself.** It is the most-probed code in an interview.

**Mission.** Hybrid search that narrows 375,000 trials to 20, with filters applied *inside* the index traversal.

**Exclusive write paths.** `services/retrieval/**`.

**Preconditions.** W0 merged; B1 merged (or a ≥10,000-trial subset indexed).

**Deliverables.**
- FastAPI service implementing the W0 request/response contract
- Qdrant hybrid query using `prefetch` + server-side RRF fusion
- `rrf(rankings, k=60)` as a pure, unit-tested function; `k` configurable
- NULL-safe age/sex filters passed as Qdrant filters (**not** applied after the search)
- Cross-encoder rerank of the top 100 → top 20 (`bge-reranker-base`)
- Configurable query count `N` (default 3)

**Definition of Done.**
```bash
uv run pytest services/retrieval
# filter test: a query with an age filter returns 20 candidates, not 4
# RRF unit test against a hand-computed example
# p95 latency under 500 ms over the full 375K index
```

**Must not do.** **Never post-filter.** Taking the top k and then filtering produces the overfiltering bug — with `ef_search=40` and a 10%-selective filter you average 4 survivors from 40. Never exclude a trial because a field is `NULL`.

**Known traps.** Qdrant payload fields used in filters need payload **indexes** or filtering falls back to a slow scan. The reranker is the CPU bottleneck — batch it. RRF uses **ranks only**, never raw scores: a cosine of 0.82 and a BM25 of 14.3 are not comparable.

**Paste-ready prompt.**
```
You are the Retrieval agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "B2", and Beacon_Design_Decisions.pdf
section 4 ("Where the hard filters go") - that section is the specification for the
single most important behaviour in this task.

Build services/retrieval: a FastAPI service implementing the request/response models
already defined in beacon_core.schemas. For each query it runs a Qdrant hybrid search
(dense + sparse) fused with Reciprocal Rank Fusion at k=60, applies age/sex filters,
reranks the top 100 with BAAI/bge-reranker-base, and returns the top 20.

Hard constraints:
- You own ONLY services/retrieval/**. beacon-core is read-only.
- Filters MUST be passed into the Qdrant query so they are evaluated inside the index
  traversal. NEVER retrieve top-k and then filter the results: with ef_search=40 and a
  filter matching 10% of rows you average 4 survivors out of 40. Write a test proving
  a filtered query returns 20 candidates, not 4.
- Filters must be NULL-safe: a trial whose min_age or max_age is NULL must SURVIVE the
  filter. NULL means "possibly eligible", never "exclude".
- Implement rrf() as a pure function with a unit test against a hand-computed example.
  RRF uses ranks only, never raw scores.
- Make k, the rerank depth, and the final top-N configurable, not hardcoded.

Done when the B2 Definition of Done passes, including p95 under 500 ms on the full
index. Branch agent/b2-retrieval, open a PR.
```

---

## 9. B3 — Evaluation harness

> **Recommended: write this one yourself.** Every number on your resume comes from it.

**Mission.** Turn retrieval output into defensible numbers, with a clean train/test split.

**Exclusive write paths.** `evals/src/evals/{metrics,ablations,report,judge}.py`, `evals/tests/**`.

**Preconditions.** W0, A4 merged; B2 merged or mocked behind the HTTP contract.

**Deliverables.**
- ranx wiring: nDCG@10, P@10, Recall@100, MRR against graded TREC qrels
- Ablation runner producing the table: BM25 only · dense only · hybrid · +query generation · +rerank · +filters · full system
- **N sweep: `N ∈ {1, 3, 5, 10}` on TREC 2021**
- Bootstrap confidence intervals and paired significance tests
- LLM judge with Cohen's κ calibration against hand labels
- One-command report generation

**Definition of Done.**
```bash
uv run python -m evals.ablations --tune 2021 --test 2022
uv run python -m evals.report        # regenerates the full report from scratch
uv run pytest evals
```

**Must not do.** **Never evaluate on 2022 while tuning.** Never write an estimated number into a report. The judge must be `claude-opus-5-5` — a different, stronger model than both Haiku 5.5 and the Sonnet 5.5 headline model.

**Known traps.** TREC qrels are **graded** (eligible 2 / excluded 1 / not relevant 0) — use graded nDCG, not binary. With 50 test patients a 2-point difference is probably noise; report CIs. The judge costs about $1 because it runs on a sample, not the full benchmark.

**Paste-ready prompt.**
```
You are the Evaluation agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "B3", and Beacon_Design_Decisions.pdf
sections 6 and 10.

Build the evaluation harness in evals/: ranx-based metrics (graded nDCG@10, P@10,
Recall@100, MRR) against TREC qrels, an ablation runner, a sweep of the generated-query
count N over {1, 3, 5, 10}, bootstrap confidence intervals, paired significance tests,
an LLM judge calibrated with Cohen's kappa, and a one-command report generator.

Hard constraints:
- You own ONLY evals/src/evals/{metrics,ablations,report,judge}.py and evals/tests/**.
  Do NOT touch evals/src/evals/trec.py - another agent owns it.
- TUNE ON TREC 2021 ONLY. Never read the 2022 labels during tuning. 2022 is the
  held-out test set and is scored once.
- TREC qrels are graded (2 = eligible, 1 = excluded, 0 = not relevant). Use graded
  nDCG, not a binary collapse.
- The LLM judge must use claude-opus-5-5 - deliberately a different and stronger model
  than both the Haiku 5.5 development model and the Sonnet 5.5 headline model, so it is
  never grading its own family's output.
- Never emit an estimated number. Every figure in a report must come from a run.

Done when the B3 Definition of Done passes. Branch agent/b3-evals, open a PR.
```

---

## 10. C1 — Match worker (LangGraph + Kafka)

**Mission.** The durable pipeline: a seven-node LangGraph state machine consuming Kafka, checkpointing to Postgres, surviving a kill.

**Exclusive write paths.** `services/match_worker/**`.

**Preconditions.** W0, A3, B2 merged.

**Deliverables.**
- `StateGraph` over `MatchState`: `profile_patient → generate_queries → retrieve → parse_rules → check_rules → verify → rank_and_explain`
- `Send` fan-out: one parallel `check_rules` task per candidate trial
- Postgres checkpointer after every node
- Kafka consumer with offsets committed **after** checkpointing; retries with exponential backoff + jitter; DLQ after N failures
- Idempotent writes (unique constraints)
- Per-match token and dollar budget; on breach, return partial results **clearly marked**
- Progress events published per node

**Definition of Done.**
```bash
uv run pytest services/match_worker
# crash test: start a match, kill the worker container mid-run, restart it;
# the job completes, resumes from the last checkpoint, and produces NO duplicate rows
```

**Must not do.** Do not let a framework own the prompts — call the Anthropic SDK through `beacon_llm` directly inside each node. Do not commit Kafka offsets before the checkpoint is durable. Do not let one trial's failure abort the other 19.

**Known traps.** Kafka is at-least-once: the same message *will* arrive twice. The crash test is a deliverable, not a nice-to-have — it is the demo clip. One trial failing twice gets marked "could not evaluate" and the run continues.

**Paste-ready prompt.**
```
You are the Match Worker agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "C1", and page 1 of Beacon_System_Design.pdf.

Build services/match_worker: a LangGraph StateGraph over the MatchState schema with
seven nodes (profile_patient, generate_queries, retrieve, parse_rules, check_rules,
verify, rank_and_explain), consuming match.requested from Kafka and checkpointing to
Postgres after every node. Use LangGraph's Send primitive to fan out one parallel
check_rules task per candidate trial.

Hard constraints:
- You own ONLY services/match_worker/**. beacon-core, beacon-llm and the retrieval
  service are read-only dependencies.
- Use LangGraph for ORCHESTRATION ONLY. Every Claude call goes through the beacon_llm
  wrapper so prompts, caching and cost accounting stay under our control.
- Commit Kafka offsets only AFTER the checkpoint is durable. Kafka is at-least-once:
  the same message will sometimes arrive twice, so all writes must be idempotent.
- If one trial's rule check fails twice, mark that trial "could not evaluate" and let
  the other 19 continue. Never abort the whole run for one trial.
- Enforce a per-match token and dollar budget. On breach, return partial results
  clearly marked as partial - never a silent truncation.

Done when the crash test passes: start a match, kill the worker mid-run, restart it,
and the job finishes from its last checkpoint with no duplicate rows.
Branch agent/c1-worker, open a PR.
```

---

## 11. C3 — Observability

> **Runs alone.** It edits every package, so no other agent may be active.

**Mission.** One trace per request, spanning gateway → Kafka → worker → every LLM and retrieval call.

**Exclusive write paths.** Small additions across all packages — **sole agent active**.

**Preconditions.** C1 merged.

**Deliverables.** OTel setup in `beacon_core.telemetry`; instrumentation in every service; **trace context propagated through Kafka message headers**; structlog JSON logs carrying the trace ID; Phoenix receiving LLM spans with tokens, cost and latency; a `/system` summary endpoint.

**Definition of Done.**
```bash
# one POST /v1/matches produces ONE trace in Phoenix spanning gateway, Kafka, worker,
# all 7 nodes and all 20 LLM calls, with tokens, cost and latency on each LLM span
# every log line carries the trace ID
```

**Must not do.** Do not refactor anything while instrumenting. Additive changes only.

**Known traps.** Trace context does **not** cross Kafka automatically — inject and extract it in message headers manually. That link is the whole point.

**Paste-ready prompt.**
```
You are the Observability agent for Beacon at C:\Users\hasan\Desktop\AIProject.
You are the ONLY agent running right now - you may touch every package.

Read first: docs/AGENT_BRIEFS.md section "C3".

Add OpenTelemetry end to end: configure it in beacon_core.telemetry, instrument the
gateway, retrieval service and match worker, and export LLM spans to Arize Phoenix with
tokens, cost and latency attached. Add the trace ID to every structlog line.

Hard constraints:
- ADDITIVE CHANGES ONLY. Do not refactor, rename or restructure anything while
  instrumenting. If something looks wrong, report it instead of fixing it.
- Trace context does NOT cross Kafka by itself. Inject it into message headers on
  publish and extract it on consume. That single link is the main deliverable.

Done when one POST /v1/matches yields ONE trace covering gateway -> Kafka -> worker ->
all 7 nodes -> all 20 LLM calls. Branch agent/c3-observability, open a PR.
```

---

## 12. C4 — Web console

**Mission.** A thin demo UI. Deliberately minimal.

**Exclusive write paths.** `web/**`.

**Preconditions.** A2 merged (gateway contract exists).

**Deliverables.** One or two Next.js pages: paste a note → live progress via `EventSource` → ranked trials → rule table with the cited evidence sentence highlighted on click → plain-language patient summary → missing-facts list. Visible "not medical advice" notice.

**Definition of Done.**
```bash
npm run build && npm run start
# paste a TREC note, watch live progress, open the top trial, click a rule and see its
# evidence sentence highlight in the note
```

**Must not do.** No business logic in the frontend. No auth system. Keep it to two pages.

**Paste-ready prompt.**
```
You are the Web Console agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "C4", and the gateway's OpenAPI schema at
http://localhost:8000/docs.

Build web/: a deliberately thin Next.js console. One main page: paste a patient note,
POST it to /v1/matches, subscribe to the SSE events endpoint with EventSource and show
live progress, then render the ranked trials. Opening a trial shows its rule table;
clicking a rule highlights the exact patient-note sentence cited as evidence. Also show
the plain-language patient summary and the missing-facts list.

Hard constraints:
- You own ONLY web/**.
- No business logic in the frontend - it renders what the API returns, nothing more.
- No authentication system. Two pages maximum. Tailwind for styling.
- A visible "decision support, not medical advice" notice on every page.

Done when a pasted TREC note produces live progress and a clickable evidence highlight.
Branch agent/c4-console, open a PR.
```

---

## 13. D — Docs, README & ADRs

**Mission.** Make the repo understandable to a stranger in ten minutes.

**Exclusive write paths.** `docs/**`, `README.md`.

**Preconditions.** None, but the results tables need B3.

**Deliverables.** `README.md` per the guide's §18.2 checklist (one-sentence pitch, demo GIF, architecture diagram, results + ablation tables with CIs, one-command run instructions tested from a clean clone, design decisions linked to ADRs, known limitations, medical disclaimer, data sources); `docs/ARCHITECTURE.md` seeded from `Beacon_Design_Decisions.pdf`; six ADRs.

**The six ADRs** (context / options / decision / consequences, one page each):
`0001` Qdrant vs pgvector+pg_search · `0002` Kafka vs Redis Streams · `0003` LangGraph + raw SDK vs LangChain · `0004` trial-level vs rule-level indexing · `0005` model choice and cost · `0006` the 2021/2022 tuning split

**Must not do.** **Never write a performance number that was not measured.** Leave placeholders until B3 produces real figures. Do not claim clinical validity.

**Known traps.** ADR 0001 must **not** argue that Qdrant was chosen for scale — 375K vectors is small for any engine, and an interviewer will catch it. The real reason is BM25: Postgres `tsvector`/`ts_rank` is not BM25.

**Paste-ready prompt.**
```
You are the Documentation agent for Beacon at C:\Users\hasan\Desktop\AIProject.

Read first: docs/AGENT_BRIEFS.md section "D", plus all three PDFs in the repo root.

Write README.md following the checklist in section 18.2 of Beacon_Project_Guide.pdf,
seed docs/ARCHITECTURE.md from Beacon_Design_Decisions.pdf, and write six ADRs
(0001-0006) in docs/adr/, each with context, options considered, the decision, and
its consequences.

Hard constraints:
- You own ONLY docs/** and README.md.
- NEVER write a performance number that has not been measured. Use explicit
  placeholders like {nDCG@10} until the evaluation harness produces real figures.
- ADR 0001 must NOT say Qdrant was chosen for scale. 375,000 vectors is small for any
  engine and that claim is false. The real reason is BM25: Postgres tsvector/ts_rank is
  a different ranking function, so a pgvector build would lose the standard keyword
  baseline that TREC and TrialGPT results are compared against. The cost of the choice
  is a Postgres-to-Qdrant dual write.
- Do not claim clinical validity. Claim benchmark performance with confidence intervals.

Branch agent/d-docs, open a PR.
```
