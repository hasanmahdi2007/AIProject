# ADR 0001 — Qdrant alongside Postgres, not pgvector

- **Status:** accepted
- **Date:** 2026-11-08
- **Affects:** `deploy/compose.yaml`, `beacon_core.settings.QdrantSettings`, `services/ingestion`, `services/retrieval`

## Context

Beacon retrieves over the TREC Clinical Trials corpus: **375,580 trials**
(verified by enumerating `ir_datasets`; see `evals/src/evals/trec.py`). Each
trial gets one 384-dimension dense vector from `BAAI/bge-small-en-v1.5` and one
sparse BM25 vector, and retrieval must apply hard age/sex filters *inside* the
search rather than after it.

PostgreSQL is already in the stack as the source of truth, so `pgvector` is the
option that adds nothing. Choosing anything else means running a second
datastore and paying for it. That cost has to be justified by something real.

### The reason it is *not* about scale

375,580 × 384 dims × 4 bytes ≈ **576 MB of dense vectors**. That is a small
index by any standard. The project's design-decision notes (carried in the
agent brief; `Beacon_Design_Decisions.pdf` itself is not in this repository)
record published third-party figures of roughly **p50 4.8 ms (pgvector) vs
3.1 ms (Qdrant) at 500K vectors**, with pgvector comfortable to around 5M
vectors — single-digit milliseconds, next to a cross-encoder rerank of ~1 s and
an LLM read stage of ~60 s. **These are not Beacon's own measurements** and are
quoted only to size the argument.

"We chose Qdrant to scale" is therefore the wrong answer, and an interviewer who
knows the numbers will say so. `Beacon_System_Design.pdf` §6 ranks a single
Qdrant node as bottleneck **#3**, biting at around 10M+ vectors — behind the
CPU cross-encoder and LLM cost.

### The reason it *is* about BM25

The hybrid-search ablation needs a **standard BM25 keyword baseline**, because
that is the baseline every TREC Clinical Trials result and the TrialGPT paper
are compared against. Postgres full-text search does not provide one:
`tsvector` + `ts_rank`/`ts_rank_cd` is a *different ranking function* — no Okapi
length normalisation, no `k1`/`b` term-frequency saturation, and weights that
are per-field rather than corpus-IDF-derived. Reporting a `ts_rank` row and
calling it "BM25" would make the headline comparison meaningless.

A second detail: BM25 needs corpus-level IDF. Qdrant computes it only when the
sparse field is declared with `modifier=IDF`; omitting that raises no error and
silently degrades rare-term matching, which is the entire reason BM25 is in the
system (`EGFR`, `HbA1c`, `BRCA1`). That flag is pinned in
`QdrantSettings.sparse_idf_modifier`.

## Options considered

| Option | Verdict |
|---|---|
| **pgvector only**, keyword side on `tsvector`/`ts_rank` | Rejected. Loses the comparable BM25 baseline, which is the one row the whole ablation table rests on. |
| **pgvector + ParadeDB `pg_search`** (real BM25 inside Postgres) | Viable, rejected. It gives true BM25, but it means a non-stock Postgres image, a second index type with its own build and refresh path, and hand-written RRF in SQL. The "one datastore" saving mostly evaporates, and the extension becomes a hard dependency of the schema. |
| **Qdrant alongside Postgres** | **Chosen.** |
| Hosted vector DB (Pinecone and similar) | Rejected. Cannot run locally, cannot be part of `docker compose up`, and hides the internals the project exists to demonstrate. |
| Elasticsearch / OpenSearch | Rejected. Real BM25 and real vectors, but a far heavier container than Qdrant for a 576 MB index, and dense+sparse+RRF in one request is more awkward. |

## Decision

Run **Qdrant** as a separate service, with **one point per trial** carrying two
named vectors (`dense`, `sparse`) plus a filterable payload
(`min_age_years`, `max_age_years`, `sex`, `phase`, `status`, `nct_id`).
Postgres remains the source of truth.

What this buys, concretely:

1. **A real BM25 baseline**, with corpus IDF handled natively by the engine.
2. **Dense + sparse + RRF fusion expressed server-side in one query**, so the
   two halves are scored against the same point and fused without a round trip
   per modality.
3. **Filters evaluated inside HNSW traversal.** This is not a convenience —
   post-filtering the top-k is the overfiltering bug documented on
   `beacon_core.schemas.AgeSexFilter`: with `ef_search=40` and a filter matching
   10% of rows you average four survivors out of forty, so "top 20" silently
   returns 4.

## Consequences

**The admitted cost is the dual write.** Ingestion writes Postgres first, then
Qdrant. Those are two systems that can half-fail, and they are the only place in
Beacon where two stores can drift apart. The price is paid as a hard
requirement on the ingest job: it must be **resumable and idempotent by
`nct_id`**, so re-running it after any failure converges rather than duplicating
or skipping. `trials.nct_id` is the primary key and `indexed_at` records whether
a row made it into Qdrant.

Other consequences, stated honestly:

- One more container, one more health check, one more backup story, one more
  query language in the codebase. Five backing services instead of four.
- Qdrant has no transactions. There is no way to make the Postgres commit and
  the Qdrant upsert atomic; the recovery story is re-running ingestion, not
  rolling back.
- Payload fields are duplicated between `trials` and the Qdrant payload. The
  schema change that updates one and forgets the other is a silent filter bug,
  so both are written from the same ingestion code path.
- **The reversal is cheap if it is ever wanted.** The retrieval service is
  reached only through `SearchRequest`/`SearchResponse` in
  `beacon_core.schemas`, which name no engine. A pgvector + `pg_search` build
  behind the same contract is the realistic fallback if the five-container stack
  ever has to shrink.
- Nothing here is a scale claim. If the corpus grew past roughly 10M vectors the
  fix is sharding and replicas, and that is a *different* decision from this one.
