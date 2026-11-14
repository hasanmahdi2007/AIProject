"""Write TREC run files for the free (no-LLM) ablation rows.

``python scripts/make_runs.py --year 2021 --out evals/runs/2021``

The eval harness scores run files and never opens a socket
(:class:`evals.backends.RunFileBackend`); this script is the other half. It
drives the **production** retrieval code - the same ``RetrievalService`` the
``POST /search`` route calls, built the same way - so a number in the README is
a number about the code that serves the console, not about a re-implementation.

One file per configuration, named ``<config>.run`` to match
:data:`evals.ablations.FREE_ABLATION_TABLE`:

- ``bm25_only`` / ``dense_only``: one branch of the hybrid searcher, top 100.
  ``SearchRequest`` has no switch to disable a branch (and should not grow
  one), so these read the searcher's per-branch lists directly - the same
  Qdrant requests the service issues, before fusion.
- ``hybrid_rrf``: the service with ``use_rerank=False``.
- ``plus_rerank_no_llm``: the service with ``use_rerank=True``. The
  cross-encoder reorders all 100 fused candidates
  (``rerank_top_k = fuse_top_k = 100``) so Recall@100 is comparable with the
  rows above; in production it keeps 20.

The query is the topic text itself (no LLM query generation), and no age/sex
filter is applied: that row needs the patient's age and sex, which only the
LLM profiler extracts. Nothing here calls Anthropic.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from beacon_core.schemas import RankedQuery, SearchRequest
from evals.backends import write_run_file
from evals.trec import load_topics
from retrieval.deps import provide_service

DEPTH = 100
CONFIGS = ("bm25_only", "dense_only", "hybrid_rrf", "plus_rerank_no_llm")


async def build_runs(year: int, configs: tuple[str, ...]) -> dict[str, dict[str, list[str]]]:
    service = provide_service()
    report = await service.verify()
    if not report.ok:
        raise SystemExit(f"retrieval collection is not usable: {report}")

    topics = load_topics(year)
    runs: dict[str, dict[str, list[str]]] = {name: {} for name in configs}
    started = time.perf_counter()
    for index, topic in enumerate(topics, start=1):
        query = RankedQuery(rank=1, text=topic.text)

        if {"bm25_only", "dense_only"} & set(configs):
            encoded = service.searcher.encode([query])
            (hits,) = await service.searcher.search(encoded, search_filter=None, limit=DEPTH)
            if "bm25_only" in configs:
                runs["bm25_only"][topic.topic_id] = hits.sparse_ids()[:DEPTH]
            if "dense_only" in configs:
                runs["dense_only"][topic.topic_id] = hits.dense_ids()[:DEPTH]

        for name, rerank in (("hybrid_rrf", False), ("plus_rerank_no_llm", True)):
            if name not in configs:
                continue
            response = await service.search(
                SearchRequest(
                    queries=[query],
                    fuse_top_k=DEPTH,
                    rerank_top_k=DEPTH,
                    use_rerank=rerank,
                )
            )
            runs[name][topic.topic_id] = [c.nct_id for c in response.candidates]

        elapsed = time.perf_counter() - started
        print(f"[{year}] topic {index}/{len(topics)} ({topic.topic_id}) {elapsed:.0f}s", flush=True)
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, choices=[2021, 2022], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    args = parser.parse_args(argv)

    runs = asyncio.run(build_runs(args.year, tuple(args.configs)))
    for name, run in runs.items():
        path = args.out / f"{name}.run"
        write_run_file(path, run, tag=name)
        print(f"wrote {path} ({len(run)} topics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
