# Resume here

Written 2026-11-09, when the weekly model limit ran low with the engine built and
the measurements not yet taken. This is the shortest path back in.

---

## 1. How to boot the whole thing

Everything runs through `tasks.ps1` (this project's Makefile — there is no `make`
on Windows). `.\tasks.ps1 help` lists every task.

```powershell
# 0. Docker Desktop must be running first. It is NOT a service - it is an app,
#    and nothing below works until its engine is up. Check with:  docker info
#    If it is closed, launch it from:
#    C:\Users\hasan\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe

.\tasks.ps1 sync      # install the workspace from uv.lock
.\tasks.ps1 up        # start postgres, redis, qdrant, redpanda, phoenix
.\tasks.ps1 ps        # must read "healthy", not merely "running"
.\tasks.ps1 migrate   # alembic upgrade head (first time only)
.\tasks.ps1 check     # lint + format + types + unit tests = what CI runs
```

**Host ports are deliberately shifted.** Postgres is **15432**, not 5432, because
this machine runs its own PostgreSQL 18 service on 5432; Redis is **16379**, not
6379. Qdrant 6333, Redpanda 9092, Phoenix 6006 are normal. `beacon_core.settings`
already defaults to the shifted ones, so a plain local run needs no env override.

Then, to see it move:

```powershell
.\tasks.ps1 gateway     # http://localhost:8000  - /health, POST /v1/matches
.\tasks.ps1 retrieval   # http://localhost:8001  - POST /search
cd web; npm install; npm run dev    # http://localhost:3000 - the console
```

The console runs against the gateway named by `NEXT_PUBLIC_GATEWAY_URL`.
`cd web; npm run verify` runs the offline checks; `npm run verify:live` adds the
gateway ones and **skips cleanly** if nothing is listening.

### Is the project finished?

**The engine is built and verified. The measurements are not taken.** Concretely:

| | State |
|---|---|
| Contracts, settings, migrations, CI | done, merged |
| Compose stack, 5 services healthy | done, merged |
| Gateway: `/health`, `POST /v1/matches`, SSE, `GET /v1/matches/{id}`, chat | done, merged |
| `beacon_llm`: structured outputs, cache ordering, cost meter, 100K guard, replay | done, merged |
| Ingestion: TREC → Postgres + Qdrant, IDF, rule splitter, age parser | done, merged |
| Retrieval: hybrid, RRF k=60, NULL-safe filters inside traversal, reranker | done, merged |
| Match worker: 7-node LangGraph on Kafka, Postgres checkpoints, DLQ, resume | done, merged |
| Eval harness: Recall@100, graded nDCG, P@10, bootstrap CIs, Cohen's κ | done, merged |
| Web console: paste note → progress → ranked trials → clickable evidence | done, merged |
| Docs: README, ARCHITECTURE, 6 ADRs | done, merged |
| **The full 375,580-trial index** | **incomplete — see §2** |
| **Every performance number** | **unmeasured — all `{placeholders}`** |
| **Observability (C3)** | **not started — see §3** |
| **The rule-check prompt** | **reserved for the owner — see §4** |
| **~150 hand labels for κ** | **not done — see §4** |

Nothing fabricated: `README.md` has an explicit placeholder index, and the eval
harness **refuses to mark a report publishable** unless the corpus is complete.

---

## 2. Finish the ingest — free, and the prerequisite for every number

It costs **no API credits and no model tokens**. It is local CPU: the embedders
(`bge-small-en-v1.5`, BM25) and the reranker all run offline.

```powershell
.\tasks.ps1 up
uv run python -m ingestion.run        # resumes; skips what is already indexed
```

It is **idempotent by `nct_id`** — point ids are `uuid5(namespace, nct_id)`,
`trials` upserts, and `indexed_at` is set **last, only after Qdrant acknowledges**,
so a NULL marks an interrupted trial and re-running is a no-op. It has already
survived two abrupt kills with Postgres and Qdrant exactly equal.

**Measured reality, not the guide's estimate:** about **9.5 trials/second** on this
hardware (4c/8t, CPU-only ONNX, no GPU), so the full corpus is roughly **11 hours**,
not the "30–90 minutes" the project guide claims. Progress as of writing:
**152,296 of 375,580 (40.6%)**.

Check progress any time:

```powershell
docker compose -f deploy/compose.yaml exec -T postgres psql -U beacon -d beacon `
  -c "SELECT count(*) FROM trials WHERE indexed_at IS NOT NULL;"
curl http://localhost:6333/collections/trials
```

Those two numbers must stay equal. They are the dual-write invariant.

---

## 3. C3 — Observability. DEFERRED, not cancelled. Resume Saturday.

Deferred on 2026-11-09 only because the weekly model limit ran low. It is the last
unstarted agent task and it is still wanted.

**It must run SOLO — no other agent active** — because OpenTelemetry touches every
package, which is exactly why it was sequenced last.

`libs/beacon-core/src/beacon_core/telemetry.py` is a deliberate **42-line no-op
stub** with the signatures already frozen (`setup_tracing`, `span`,
`current_traceparent`), so every service already calls into it and nothing breaks
while it does nothing. C3 fills it in without touching a single call site.

Scope when it resumes:

- real tracer provider and OTLP exporter to Phoenix (`settings.otel_endpoint`,
  already `http://localhost:6006/v1/traces`)
- spans through the gateway → Kafka → worker → retrieval → Claude path
- **trace context propagated through Kafka headers** — this is the hard part and
  the interesting one: without it the trace breaks at the queue boundary and you
  get two unrelated traces instead of one journey
- token, cost and latency attributes on every LLM span, so "why did that patient
  cost $0.04?" is answerable from one trace
- one trace showing all 7 nodes and all 20 rule-check calls for a single patient

Note that `.\tasks.ps1 up` already starts Phoenix on 6006, so the backend is there
waiting; only the instrumentation is missing.

---

## 4. The two things no agent can do

**The rule-check prompt.** `services/match_worker/src/match_worker/rule_check_prompt.py`
has a `TODO(owner)` block. Rewrite `rule_check_instructions()` (the stable cached
prefix) and `render_trial_rules()` (the per-trial variable block). Everything around
it is complete and reads none of the prompt text. Two constraints the tests enforce:
the prefix must stay above **512 tokens** or prompt caching silently does nothing,
and the output schema stays the frozen five-field `CriterionJudgment`. Fixtures are
authored from those two functions at run time, so rewriting them does not invalidate
the tests.

**~150 hand labels.** `evals/calibration/hand_labels.template.jsonl`. This is the
one genuinely undelegatable task: Cohen's κ measures how well the Opus judge agrees
with **your** labels. If a model writes them, κ measures two models agreeing with
each other, which is not calibration — and publishing it as such would be false.
Budget 2–3 hours. Target κ ≥ 0.6.

---

## 5. Measurement: five of the eight ablation rows cost nothing

The eval harness scores **TREC run files**, not a live service — deliberately, so it
can score anything and stays re-runnable. Something must produce those files; that
generator is the last piece of plumbing.

```powershell
uv run python -m evals.report --runs <dir-of-run-files> --indexed-trials <count>
# once tuning is frozen, add:  --i-have-finished-tuning "<reason, >=20 chars>"
```

| Ablation row | Needs Claude? |
|---|---|
| BM25 only | **no — free** |
| Dense only | **no — free** |
| Hybrid RRF k=60 | **no — free** |
| + cross-encoder rerank | **no — free** |
| + age/sex filters | **no — free** |
| + LLM query generation, N=1 (raw note baseline) | **no — free** |
| + LLM query generation, N=3 and the N sweep | yes |
| + LLM rule-level matching (`full_system`) | yes |

So **Recall@100 — the metric that governs the whole system — is measurable for
$0.00** once the index is complete. Only the last two rows need credits
(~$3 Haiku full run, ~$1 Opus judge, ~$12.50 Sonnet headline).

**Do not report any number measured against a partial index.** The harness already
enforces this: `UNKNOWN_COVERAGE` is the default and is treated as harshly as
known-partial, and a partial report is stamped *"NOT PUBLISHABLE … NOT comparable to
published TREC or TrialGPT figures"* above the numbers.

---

## 6. Measured facts worth not re-deriving

All confirmed directly against Postgres and Qdrant, not taken from a report.

| Fact | Value | Why it matters |
|---|---|---|
| 512-token truncation rate | **2.350%** (3,056 / 130,024) | the overview field order is load-bearing; this is the cost of it |
| `max_age_years` is NULL | **49.8%** | a filter excluding unknowns discards **half the index** — this is why `include_unknown=True` |
| `min_age_years` is NULL | 9.6% | same |
| NULL-safe vs strict filter at age 45 | admits 109,372 vs 46,524 | strict discards **57.5%** |
| Criteria from the regex fallback | **12.9%** (21,308 / 164,602) | read a weak rule-matching row against this before blaming the prompt |
| Cross-encoder throughput | **~3 pairs/sec**, not ~100 | the guide is wrong by two orders of magnitude; 100 candidates took 161 s |
| Length-sorted reranker batching | **1.7–2.5× faster**, results identical (max diff 0.0) | batch padding was pure waste |
| Search latency (retrieve+fuse+hydrate) | 377 ms median / 612 ms max at 133K | meets the 500 ms p95 target; rerank is the separate slow stage |
| Overfiltering, same collection both ways | filter-inside **20**, post-filter **2** | the whole reason filters go inside the traversal |

---

## 7. Traps that have already cost time — do not rediscover them

- **A test module basename must be unique across the workspace.** Two packages each
  with `tests/test_pipeline.py` breaks `mypy .` ("Duplicate module named"). pytest
  survives it via `--import-mode=importlib`; mypy does not, and `__init__.py` does
  not fix it in a src layout. Documented in `pyproject.toml`.
- **Integration tests must `pytest.skip`, never fail**, when a prerequisite is
  missing. CI genuinely runs `pytest -m integration` in a second job.
- **`asyncio_default_fixture_loop_scope = "function"`** gives every test a fresh
  event loop while `beacon_core.db` caches its engine per process — reusing a pool
  across loops raises `got Future attached to a different loop`. Build a per-test
  engine.
- **`respx` cannot intercept the Anthropic SDK** (`anthropic` 2.6 is built on
  `httpx2`), so such a test passes while asserting nothing. Use replay fixtures.
  respx *is* fine for the retrieval service, which uses plain `httpx`.
- **`ruff format` formats Python inside Markdown fences**, so hand-aligned comments
  in a `.md` file fail `ruff format --check`.
- **Qdrant's `full_scan_threshold` / `indexing_threshold` are kilobytes, not points.**
- **`NCT04000001/2/3`** in the shared `trials` table are the match worker's test
  fixtures, not corpus records. They are the only `indexed_at IS NULL` rows, which is
  the same signal the ingest uses to detect a broken dual write — worth cleaning up.
- Never run a bare `git add -A` in this repo: it has already committed an unrelated
  file and deleted a tracked PDF.
