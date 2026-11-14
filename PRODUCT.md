# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Engineers and interviewers evaluating Beacon as an AI-engineering portfolio project. They open the console during a demo or a project deep-dive, paste a synthetic patient note, and judge in a few minutes whether the system is well engineered and whether its answers can be trusted. The console is a working demo of the engine, not a clinical product with real users.

## Product Purpose
Beacon matches a free-text patient note to clinical trials (ClinicalTrials.gov, TREC CT 2021 snapshot) and ranks them best to worst. Success on this surface: a viewer understands within one look that every eligibility decision is checked rule by rule and cites the exact sentence of the note it relied on.

## Positioning
Every verdict is cited. Each rule judgment (met / not met / not stated / not applicable) points to numbered sentences of the patient note; the citations are checked by code, and the trial verdict is derived from the rule judgments by code, not by the model. Unknowns stay unknown ("not stated in the note") and become a list of facts to ask for.

## Operating Context
- Next.js console at :3000 talking to a FastAPI gateway; a 7-step pipeline (profile, queries, retrieve, rerank, check rules, verify, rank and explain) streams progress live over SSE.
- One run takes about 3-4 minutes and costs about $0.10 of the owner's Anthropic credits (Haiku); viewers mostly watch, then read results.
- Pages: the console (note input, live pipeline, patient summary, ranked trials with rule-by-rule evidence, chat about a run) and "How to read this".

## Capabilities and Constraints
- Results: 20 ranked trials, each with verdict (eligible, likely eligible, insufficient info, excluded), score, rules confirmed fraction, rule list with labels, cited sentences, and missing facts.
- Score = verdict weight (eligible 3, likely 2, insufficient 1, excluded 0) + fraction of rules confirmed + a tiny retrieval tie-breaker.
- Behaviour and wording of the current console are kept; redesigns change the look, not the function.

## Brand Commitments
- Name: Beacon.
- The disclaimer "Decision support, not medical advice" and the synthetic-data notice appear on every page, rendered from the layout. Non-negotiable.

## Evidence on Hand
- Real engine output from live runs (Postgres `match_runs`, `trial_matches`, `criterion_judgments`).
- All patient data is synthetic (TREC topics or typed demo notes). No real patients, customers, testimonials or clinical validation exist; none may be claimed.

## Product Principles
1. Show the evidence, not the confidence: a claim without a cited sentence is not shown as a claim.
2. An unknown is a real answer; never fill it in.
3. The engineering is the product: the pipeline, the checks and the costs are visible, not hidden.
4. Decision support only; never imply clinical validity.
