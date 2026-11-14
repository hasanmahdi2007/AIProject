## Free (no-LLM) ablations on TREC 2021 (tuning)

> FULL CORPUS: all 375,580 TREC 2021 trials indexed (--indexed-trials).

- Split: **tune** (TREC 2021), 75 topics
- Retrieval backend: `run-files`
- Corpus: 375580 / 375,580 trials indexed
- Bootstrap: 10,000 resamples over topics, seed 20261109, 95% percentile intervals
- Baseline for deltas: `bm25_only`
- Code: `f0fa0d6`, generated 2026-11-14T14:30:26+00:00

### Metric definitions

- `recall@100` - Recall@100 (grade >= 1, incl. excluded)
- `recall@100_eligible_only` - Recall@100 (grade >= 2, eligible only)
- `ndcg@10` - nDCG@10 (graded, linear gains)
- `ndcg@10_exp` - nDCG@10 (graded, exponential gains)
- `p@10` - P@10 (grade >= 1, incl. excluded)
- `mrr` - MRR (grade >= 1, incl. excluded)

Relevance grades are TREC CT's: **2 = eligible, 1 = excluded, 0 = not relevant**. Grade 1 means the trial is right for the patient's disease but the patient fails a criterion, which is *partially correct retrieval* - so nDCG scores it with graded gain, and the binary metrics count it as a hit at the default threshold of 1. The stricter grade-2-only recall is reported beside it. Documents the pool never judged count as grade 0, so recall is recall *over the judged pool*.

### Results

| configuration | recall@100 | recall@100_eligible_only | ndcg@10 | ndcg@10_exp | p@10 | mrr | vs `bm25_only` |
|---|---|---|---|---|---|---|---|
| `bm25_only` | 0.114 [0.095, 0.134] | 0.113 [0.091, 0.134] | 0.254 [0.214, 0.293] | 0.222 [0.185, 0.259] | 0.316 [0.267, 0.365] | 0.628 [0.543, 0.711] | - |
| `dense_only` | 0.160 [0.129, 0.194] | 0.170 [0.136, 0.208] | 0.283 [0.238, 0.330] | 0.253 [0.211, 0.295] | 0.353 [0.297, 0.412] | 0.617 [0.532, 0.704] | +0.046 |
| `hybrid_rrf` | 0.155 [0.129, 0.184] | 0.159 [0.129, 0.189] | 0.340 [0.297, 0.382] | 0.300 [0.260, 0.340] | 0.433 [0.380, 0.487] | 0.692 [0.616, 0.766] | +0.042 |
| `plus_rerank_no_llm` | 0.155 [0.129, 0.184] | 0.159 [0.129, 0.189] | 0.236 [0.190, 0.282] | 0.208 [0.166, 0.252] | 0.311 [0.253, 0.369] | 0.477 [0.397, 0.561] | +0.042 |
| `plus_age_sex_filters_no_llm` | not measured | not measured | not measured | not measured | not measured | not measured | - |

Each cell is a point estimate with its 95% bootstrap interval over topics. `(n.s.)` on a delta means the paired bootstrap interval includes zero: on 75 topics that difference is indistinguishable from noise, and reporting it as an improvement would be unsupportable.

### Rows not measured

- `plus_age_sex_filters_no_llm`: no run file for configuration 'plus_age_sex_filters_no_llm' at evals\runs\2021\plus_age_sex_filters_no_llm.run. Available: ('bm25_only', 'dense_only', 'hybrid_rrf', 'plus_rerank_no_llm')

### Independent cross-check (`ranx`)

Each metric recomputed by `ranx` on the baseline run and compared to this harness's own implementation. A disagreement here means one of the two is wrong and no number in the table can be trusted.

| metric | this harness | ranx | abs diff | verdict |
|---|---|---|---|---|
| `recall@100` | 0.113905 | 0.113905 | 0.00e+00 | agree |
| `recall@100_eligible_only` | 0.112544 | 0.112544 | 0.00e+00 | agree |
| `ndcg@10` | 0.254003 | 0.254003 | 0.00e+00 | agree |
| `ndcg@10_exp` | 0.222209 | 0.222209 | 8.33e-17 | agree |
| `p@10` | 0.316000 | 0.316000 | 0.00e+00 | agree |
| `mrr` | 0.628021 | 0.628021 | 0.00e+00 | agree |

