# ADR 0004 — Index one point per trial, over the overview only

- **Status:** accepted (phase 1); rule-level indexing is a phase-2 ablation, gated on measurement
- **Date:** 2026-11-08
- **Affects:** `services/ingestion`, `beacon_core.models.Trial`, `beacon_core.models.TrialCriterion`, `QdrantSettings`

## Context

A ClinicalTrials.gov record has two very different kinds of text:

- an **overview** — title, conditions, interventions, brief summary — which
  answers "what is this trial about?";
- an **eligibility block** — typically 10–50 inclusion and exclusion rules,
  heavy with negation, numeric thresholds and boilerplate — which answers "who
  may enrol?".

The retrieval unit has to be chosen, and the choice is constrained by two facts:

1. **The answer unit is a trial**, and TREC qrels are graded per
   *(topic, NCT ID)* pair. Anything indexed below trial level needs an
   aggregation step before it can be scored at all.
2. **`bge-small-en-v1.5` truncates silently at 512 tokens.** No error, no
   warning — text past the window simply does not exist as far as the vector is
   concerned. A full eligibility block alone usually exceeds that.

## Options considered

| Option | Verdict |
|---|---|
| **One point per trial, overview text only** | **Chosen for phase 1.** |
| One point per trial, overview **+ full eligibility text** | Rejected. Eligibility prose is dominated by negations ("no prior therapy with…") and boilerplate that embeddings blur, so it dilutes the signal that identifies the disease — and at 512 tokens it pushes the title and conditions out of the window entirely. |
| **One point per rule**, each prefixed with a contextual header (trial title + condition) | Rejected for phase 1, kept as a measured ablation. It multiplies the index by ~20–50×, and more importantly it needs a rule→trial aggregation (max? mean? top-k sum?) that is itself a tunable confound sitting between the retriever and the metric. |
| Hierarchical: rule-level index, aggregate to trial, then rerank | Rejected as phase-1 scope. Strictly more machinery than option 3 with the same unresolved aggregation question. |

## Decision

**One Qdrant point per trial**, with two named vectors on that one point
(`dense` from `bge-small-en-v1.5`, `sparse` from `Qdrant/bm25`) and a filterable
payload (`min_age_years`, `max_age_years`, `sex`, `phase`, `status`, `nct_id`).
Idempotent by NCT ID.

The dense vector is built from `overview_text` = title + conditions +
interventions + brief summary, assembled **most-discriminative-field-first**
precisely because of the silent 512-token truncation. Ingestion records
`overview_token_count` and `overview_truncated` on every `trials` row, so the
truncation rate is a number we can report rather than a thing we hope about.

**Eligibility text goes to Postgres and is not indexed at all.** It is split
into rules *lazily*, for retrieved trials only, and cached in `trial_criteria`
(one row per rule, unique on `(nct_id, criterion_id)`), so each trial is parsed
once and reused for every patient thereafter.

### There are three splitting decisions here, not one chunking strategy

This is the part worth saying out loud, because "what's your chunking strategy?"
has three different answers in Beacon:

| Text | Split how | Why |
|---|---|---|
| Trial overview | **not chunked** | the retrieval unit is a trial; a trial with two chunks would compete with itself in the ranking |
| Trial eligibility | **structure-aware rule splitting** (inclusion/exclusion headings, numbering, bullets), lazily, cached | a rule is the unit the LLM judges and the unit a citation attaches to |
| Patient note | **numbered sentences** | the sentence ID is the *citation* unit, so code can verify an ID exists before any LLM is asked whether it supports the label |

The patient-note numbering is explicitly not a retrieval decision —
`NumberedNote`'s docstring says so: "Numbering is for *citation*, not
retrieval."

## Consequences

- **A patient whose only discriminative signal lives in a rule is retrieved
  worse.** If the thing that makes a trial right is "requires EGFR exon 20
  insertion", and that string appears only in the eligibility block, then it is
  not in either vector. This is the real cost, and it is the hypothesis the
  phase-2 rule-level ablation exists to test. Recall@100 is the metric that
  would show it, and the figure is **{recall@100}** — not yet measured.
- **Rule parsing sits on the hot path**, lazily. `Beacon_System_Design.pdf` §6
  ranks this bottleneck **#5**: fine at current churn, not forever. The fix is
  to precompute all rules offline, which is the phase-2 rule-level index anyway.
- **Retrieval cannot be asked to judge eligibility, and is not.** A candidate is
  never a suggestion (`TrialCandidate`'s docstring). Eligibility is decided by
  typed payload filters and by the LLM rule checker, which is why indexing the
  rules is an optimisation rather than a correctness requirement.
- The index stays small — 375,580 points, ≈576 MB of dense vectors — which keeps
  a full re-index to one overnight CPU pass and makes re-running ingestion a
  cheap recovery action rather than a project.
- The phase-2 comparison is **gated on an ablation row, not an assumption**. If
  rule-level chunks with contextual headers do not beat trial-level on
  held-out 2022, they do not ship.
