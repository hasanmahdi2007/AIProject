"""``python -m ingestion.run`` -- the TREC corpus into Postgres and Qdrant.

    uv run python -m ingestion.run --limit 1000   # fast smoke path
    uv run python -m ingestion.run                # full run, ~30-90 min on CPU

The loop is deliberately sequential. Embedding 375,580 overviews on a CPU is
the wall clock of this job and onnxruntime already uses every core, so a
producer/consumer arrangement around it would add a failure mode (a half-
written batch whose ``indexed_at`` was set by the other thread) to buy very
little. What *is* overlapped is the only thing worth overlapping: derivation is
pure and runs over the whole corpus even on a resumed run, so the reported
truncation and fallback rates always describe all 375,580 records rather than
whatever happened to be left to do.

Resuming
--------
``trials.indexed_at`` is the marker and the only one; see
:mod:`ingestion.store`. A run reads the set of already-indexed ``nct_id`` once
and skips those records, so an interrupted job restarts with no bookkeeping and
a completed job re-run does no writes at all.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from beacon_core.db import dispose_engine, session_scope
from beacon_core.logging import get_logger
from beacon_core.settings import Settings, get_settings
from ingestion import store
from ingestion.corpus import EXPECTED_TRIAL_COUNT, iter_raw_trials
from ingestion.embedding import Encoder
from ingestion.overview import OverviewBuilder
from ingestion.pipeline import DerivedTrial, IngestStats, derive_trial
from ingestion.store import QdrantIndex, TrialPoint

__all__ = ["RunConfig", "build_parser", "ingest", "main"]

logger = get_logger("ingestion.run")

DEFAULT_CRITERIA_LIMIT = 10_000
"""How many trials get their criteria persisted by default.

``Trial`` says rules are split lazily, for trials that actually get retrieved,
and 375,580 trials would be roughly 7 million ``trial_criteria`` rows that
nothing reads. The splitter nevertheless runs over **every** record so the
fallback rate is a corpus-wide measurement; only the persisting is capped.
``--criteria-limit all`` writes them all, ``0`` writes none.
"""


@dataclass(frozen=True, slots=True)
class RunConfig:
    limit: int | None = None
    skip: int = 0
    batch_size: int = 256
    embed_batch: int = 64
    criteria_limit: int | None = DEFAULT_CRITERIA_LIMIT
    force: bool = False
    recreate_collection: bool = False
    dry_run: bool = False
    progress_every: int = 20
    threads: int | None = None


def _batched[T](items: Iterator[T], size: int) -> Iterator[list[T]]:
    """Group a lazy stream into fixed-size lists, order preserved.

    Order is load-bearing: :func:`_write_batch` pairs the i-th vector with the
    i-th trial.
    """
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _write_batch(
    batch: Sequence[DerivedTrial],
    encoder: Encoder,
    index: QdrantIndex,
    *,
    embed_batch: int,
    criteria_for: Sequence[DerivedTrial],
) -> None:
    """One batch through the dual write: Postgres, then Qdrant, then the marker."""
    texts = [d.overview.text for d in batch]
    dense, sparse = encoder.encode(texts, batch_size=embed_batch)

    # 1. Postgres first. The row exists but is not yet claimed to be indexed.
    async with session_scope() as session:
        await store.upsert_trials(session, [d.trial_row() for d in batch])
        if criteria_for:
            criteria = [c for d in criteria_for for c in d.criteria]
            if criteria:
                await store.upsert_criteria(session, criteria)
            empty = [d.nct_id for d in criteria_for if not d.criteria]
            if empty:
                await store.clear_criteria(session, empty)

    # 2. Qdrant, waiting for the acknowledgement.
    await index.upsert(
        TrialPoint(
            nct_id=d.nct_id,
            dense=dense[i],
            sparse=sparse[i],
            payload=d.qdrant_payload(),
        )
        for i, d in enumerate(batch)
    )

    # 3. Only now do both stores hold the trial.
    async with session_scope() as session:
        await store.mark_indexed(session, [d.nct_id for d in batch])


async def ingest(config: RunConfig, settings: Settings | None = None) -> IngestStats:
    """Run the ingest and return the measurements.

    The returned :class:`IngestStats` covers the records this run **processed**
    -- records skipped because they were already indexed are counted in
    ``skipped`` but not derived, so they contribute no statistics. A full first
    run therefore reports corpus-wide rates; a resumed run reports the rates of
    the remainder, and ``--force`` or ``--dry-run`` re-measures everything.
    """
    cfg = settings or get_settings()
    stats = IngestStats()
    started = time.monotonic()

    logger.info(
        "ingest.start",
        limit=config.limit,
        skip=config.skip,
        batch_size=config.batch_size,
        criteria_limit=config.criteria_limit,
        force=config.force,
        dry_run=config.dry_run,
        collection=cfg.qdrant.collection,
        dense_model=cfg.qdrant.dense_model,
        sparse_model=cfg.qdrant.sparse_model,
        sparse_idf_modifier=cfg.qdrant.sparse_idf_modifier,
    )

    encoder = Encoder(cfg.qdrant, threads=config.threads)
    builder = OverviewBuilder.from_fastembed(encoder.dense, max_tokens=cfg.qdrant.dense_max_tokens)
    index = QdrantIndex(cfg.qdrant)

    try:
        if not config.dry_run:
            await index.ensure_collection(recreate=config.recreate_collection)

        already: set[str] = set()
        if not config.force and not config.dry_run:
            async with session_scope() as session:
                already = await store.indexed_nct_ids(session)
            logger.info("ingest.resume", already_indexed=len(already))

        # -1 means "no cap"; see DEFAULT_CRITERIA_LIMIT.
        criteria_budget = -1 if config.criteria_limit is None else config.criteria_limit
        written = 0
        skipped = 0
        errors = 0

        def derived_stream() -> Iterator[DerivedTrial]:
            nonlocal errors, skipped
            for raw in iter_raw_trials(skip=config.skip, limit=config.limit):
                # Skip *before* deriving. Deriving is 9 ms a record -- the
                # tokenizer pass that measures truncation plus the splitter --
                # so deriving the whole corpus just to throw it away would make
                # a no-op re-run take an hour. The cost is that IngestStats
                # describes the records this run actually processed; the
                # corpus-wide rates come from the full first run, or from
                # --force, or from --dry-run.
                if raw.nct_id in already:
                    skipped += 1
                    continue
                try:
                    derived = derive_trial(raw, builder)
                except Exception:
                    errors += 1
                    logger.exception("ingest.derive_failed", nct_id=raw.nct_id)
                    continue
                stats.observe(derived)
                yield derived

        for batches, batch in enumerate(_batched(derived_stream(), config.batch_size), start=1):
            if config.dry_run:
                written += len(batch)
            else:
                take = len(batch) if criteria_budget < 0 else min(criteria_budget, len(batch))
                if criteria_budget > 0:
                    criteria_budget -= take
                await _write_batch(
                    batch,
                    encoder,
                    index,
                    embed_batch=config.embed_batch,
                    criteria_for=batch[:take],
                )
                written += len(batch)
            if batches % config.progress_every == 0:
                elapsed = time.monotonic() - started
                rate = written / elapsed if elapsed else 0.0
                # Skipped records are already past, so they count towards
                # progress; only derived ones count towards the rate.
                remaining = (config.limit or EXPECTED_TRIAL_COUNT) - stats.trials - skipped
                logger.info(
                    "ingest.progress",
                    scanned=stats.trials + skipped,
                    written=written,
                    skipped=skipped,
                    errors=errors,
                    rate_per_s=round(rate, 1),
                    eta_min=round(remaining / rate / 60, 1) if rate else None,
                    truncation_rate=round(stats.truncation_rate, 5),
                )

        elapsed = time.monotonic() - started
        summary = stats.as_dict()
        summary.update(
            written=written,
            skipped=skipped,
            errors=errors,
            elapsed_s=round(elapsed, 1),
            rate_per_s=round(written / elapsed, 1) if elapsed else 0.0,
        )
        if not config.dry_run:
            summary["qdrant_points"] = await index.count()
            async with session_scope() as session:
                summary["pg_trials"] = await store.count_trials(session)
                summary["pg_trials_indexed"] = await store.count_trials(session, indexed_only=True)
                summary["pg_criteria"] = await store.count_criteria(session)
        logger.info("ingest.done", **summary)
        return stats
    finally:
        await index.close()
        await dispose_engine()


def _criteria_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--criteria-limit must be >= 0, or 'all'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.run",
        description="Ingest the TREC Clinical Trials corpus into Postgres and Qdrant.",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N records")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N records")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="trials per dual-write batch (default 256)"
    )
    parser.add_argument(
        "--embed-batch", type=int, default=64, help="texts per FastEmbed batch (default 64)"
    )
    parser.add_argument(
        "--criteria-limit",
        type=_criteria_limit,
        default=DEFAULT_CRITERIA_LIMIT,
        metavar="N|all",
        help=(
            "persist trial_criteria rows for at most N trials "
            f"(default {DEFAULT_CRITERIA_LIMIT}; 0 for none, 'all' for every trial). "
            "The splitter always runs over every record; this caps only the writing."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-write every record, ignoring indexed_at (still idempotent)",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="drop and recreate the Qdrant collection first",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="derive and measure only; touch neither Postgres nor Qdrant",
    )
    parser.add_argument("--progress-every", type=int, default=20, help="log every N batches")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="onnxruntime threads per model (default: let onnxruntime decide)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        limit=args.limit,
        skip=args.skip,
        batch_size=args.batch_size,
        embed_batch=args.embed_batch,
        criteria_limit=args.criteria_limit,
        force=args.force,
        recreate_collection=args.recreate_collection,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
        threads=args.threads,
    )
    try:
        asyncio.run(ingest(config))
    except KeyboardInterrupt:
        logger.warning("ingest.interrupted", note="re-run to resume; indexed_at is the marker")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
