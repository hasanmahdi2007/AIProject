# TODO after 15 Nov 2026: weekly live trial sync

**Status:** not started. Decided on 14 Nov to build it **after** the
application deadline. Until then the work is the ingest, the free benchmark
numbers, and the paid headline run.

## Why
Beacon's index is the **TREC Clinical Trials 2021** corpus: a frozen copy of
ClinicalTrials.gov from April 2021 (375,580 trials, downloaded once through
`ir_datasets`). It's right for the benchmark and wrong for a real user: many of
those trials have closed, and nothing recruiting today is in it.

## The rule that must not break
The benchmark index is **never updated**. TREC's expert answer key (which trials
are eligible for each test patient) was made against that exact 2021 snapshot.
Change it and Beacon's scores are no longer comparable with TrialGPT or
anyone else.

So live data goes into a **second, separate index**.

## Design
- **Source:** the ClinicalTrials.gov API v2, `https://clinicaltrials.gov/api/v2/studies`.
  It's public, needs no key, and returns JSON. Keep only `RECRUITING` and
  `NOT_YET_RECRUITING` trials.
- **Schedule:** weekly. Trials change slowly (status flips, new
  registrations), so daily adds little. Each run asks only for studies whose
  last-update date is after the previous run's date, so a weekly run is a
  small, incremental pull, not 500k records.
- **Storage:**
  - a new Qdrant collection, `trials_live` (same dense + BM25 + payload shape)
  - Postgres `trials` rows tagged `source = 'ctgov_live'`, never mixed with
    `source = 'trec'`
  - closed trials are deleted from the live index, or marked closed
- **Reuse, don't rewrite:** the ingestion pipeline already does
  parse → derive (age/sex parsing, overview text) → embed → dual write, with
  idempotent upserts keyed by NCT ID (`services/ingestion/`). The new piece is
  only a reader for the API's JSON in place of the TREC XML (`corpus.py`),
  plus a "since" checkpoint.
- **Console:** a toggle between "Benchmark data (2021)" and "Live recruiting
  trials". Retrieval takes the collection name from the request or settings.
- **Running it:** a `.\tasks.ps1 sync-live` task, scheduled weekly with
  Windows Task Scheduler (or a cron job in deployment).
- **Cost:** $0. The API is free and embedding runs on local CPU.

## Related cleanup to do at the same time
There are two eligibility splitters: `services/ingestion/src/ingestion/criteria.py`
(robust) and `services/match_worker/src/match_worker/rules.py` (simpler). On
14 Nov the worker one was found filing "Inclusion Criteria/Exclusion
Criteria:" lists as exclusions, and was fixed. Merge them into one shared
splitter in `beacon_core` so they can't disagree again.
