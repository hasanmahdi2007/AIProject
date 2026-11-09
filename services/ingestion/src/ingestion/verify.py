"""``python -m ingestion.verify`` -- prove the two stores agree.

    uv run python -m ingestion.verify                 # counts + collection info
    uv run python -m ingestion.verify --hand-check 10 # rows beside payloads

Three things are checked, each of which can fail silently in production and
therefore has to be looked at rather than assumed:

1. **The collection's shape.** Both named vectors exist, the dense one is 384-d
   cosine, and the sparse one declares ``modifier: IDF``. A collection missing
   the modifier serves queries perfectly happily and merely ranks badly.
2. **The counts.** Qdrant points, ``trials`` rows, and rows whose
   ``indexed_at`` is set. All three must match; a gap between the second and
   the third is exactly the dual-write divergence the marker exists to expose.
3. **A hand-check.** For a sample of trials, the Postgres row and the Qdrant
   payload are printed side by side and compared field by field, including the
   two vectors' presence and width. This is the check that catches a payload
   built from the wrong variable -- something no count will ever reveal.

Output is plain text on stdout, not structured logs, because its purpose is to
be read by a person and pasted into a review.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from qdrant_client import models
from sqlalchemy import select

from beacon_core.db import dispose_engine, session_scope
from beacon_core.models import Trial
from beacon_core.settings import Settings, get_settings
from ingestion import store
from ingestion.corpus import EXPECTED_TRIAL_COUNT
from ingestion.store import QdrantIndex, point_id

__all__ = ["main", "verify"]

PAYLOAD_TO_COLUMN = {
    "nct_id": "nct_id",
    "min_age_years": "min_age_years",
    "max_age_years": "max_age_years",
    "sex": "sex",
    "phase": "phase",
    "status": "status",
}
"""Every payload key and the ``trials`` column it must equal.

The payload is a projection of the row, so this mapping is the contract the
hand-check enforces. If a key is ever added to the payload without being added
here, the hand-check reports it as unchecked rather than passing quietly.
"""


def _describe_collection(info: Any, settings: Settings) -> list[str]:
    params = info.config.params
    lines = [
        f"collection            : {settings.qdrant.collection}",
        f"points_count          : {info.points_count}",
        f"status                : {info.status}",
    ]
    vectors = params.vectors or {}
    if isinstance(vectors, dict):
        for name, config in sorted(vectors.items()):
            lines.append(f"vector[{name}]          : size={config.size} distance={config.distance}")
    else:  # a single unnamed vector -- wrong shape for Beacon, so say so loudly
        lines.append(f"vector[<unnamed>]     : {vectors!r}  <-- EXPECTED NAMED VECTORS")
    for name, config in sorted((params.sparse_vectors or {}).items()):
        modifier = config.modifier or models.Modifier.NONE
        verdict = "OK" if modifier == models.Modifier.IDF else "*** NOT IDF ***"
        lines.append(f"sparse[{name}]         : modifier={modifier}  {verdict}")
    schema = info.payload_schema or {}
    for name in sorted(schema):
        entry = schema[name]
        data_type = getattr(entry, "data_type", entry)
        points = getattr(entry, "points", "?")
        lines.append(f"payload_index[{name}] : {data_type} over {points} points")
    return lines


def _compare(row: Any, payload: dict[str, Any]) -> list[str]:
    out = []
    for key, column in PAYLOAD_TO_COLUMN.items():
        expected = getattr(row, column)
        actual = payload.get(key)
        same = expected == actual or (expected is None and actual is None)
        verdict = "OK" if same else "MISMATCH"
        out.append(f"    {key:<16} pg={expected!r:<28} qdrant={actual!r:<28} {verdict}")
    unchecked = set(payload) - set(PAYLOAD_TO_COLUMN)
    if unchecked:
        out.append(f"    UNCHECKED payload keys: {sorted(unchecked)}")
    return out


async def verify(
    *, hand_check: int = 0, nct_ids: list[str] | None = None, settings: Settings | None = None
) -> int:
    """Print the evidence. Returns the number of problems found."""
    cfg = settings or get_settings()
    index = QdrantIndex(cfg.qdrant)
    problems = 0
    try:
        info = await index.info()
        exact = await index.count()
        print("=== Qdrant collection ===")
        for line in _describe_collection(info, cfg):
            print(line)
        print(f"exact count           : {exact}")

        sparse = (info.config.params.sparse_vectors or {}).get(cfg.qdrant.sparse_vector_name)
        if sparse is None or (sparse.modifier or models.Modifier.NONE) != models.Modifier.IDF:
            problems += 1
            print("PROBLEM: the sparse vector field does not declare modifier: IDF")

        async with session_scope() as session:
            total = await store.count_trials(session)
            indexed = await store.count_trials(session, indexed_only=True)
            criteria = await store.count_criteria(session)
        print("\n=== Postgres ===")
        print(f"trials rows           : {total}")
        print(f"trials with indexed_at: {indexed}")
        print(f"trial_criteria rows   : {criteria}")
        print(f"corpus size (expected): {EXPECTED_TRIAL_COUNT}")

        print("\n=== Agreement ===")
        for label, left, right in (
            ("qdrant points == pg rows", exact, total),
            ("pg rows == pg indexed", total, indexed),
        ):
            ok = left == right
            problems += 0 if ok else 1
            print(f"{label:<26}: {left} vs {right}  {'OK' if ok else 'MISMATCH'}")

        if hand_check or nct_ids:
            chosen = nct_ids
            if not chosen:
                async with session_scope() as session:
                    result = await session.execute(
                        select(Trial.nct_id).order_by(Trial.nct_id).limit(hand_check)
                    )
                    chosen = list(result.scalars().all())
            print(f"\n=== Hand-check ({len(chosen)} trials) ===")
            async with session_scope() as session:
                rows = {
                    row.nct_id: row
                    for row in (
                        await session.execute(select(Trial).where(Trial.nct_id.in_(chosen)))
                    ).scalars()
                }
            records = {record.id: record for record in await index.retrieve(chosen)}
            for nct_id in chosen:
                row = rows.get(nct_id)
                record = records.get(point_id(nct_id))
                print(f"\n  {nct_id}  (point {point_id(nct_id)})")
                if row is None or record is None:
                    problems += 1
                    print(
                        f"    MISSING: pg_row={row is not None} qdrant_point={record is not None}"
                    )
                    continue
                vectors = record.vector or {}
                named = vectors if isinstance(vectors, dict) else {}
                raw_dense = named.get(cfg.qdrant.dense_vector_name)
                raw_sparse = named.get(cfg.qdrant.sparse_vector_name)
                dense_dim = len(raw_dense) if isinstance(raw_dense, list) else None
                sparse_nnz = (
                    len(raw_sparse.indices) if isinstance(raw_sparse, models.SparseVector) else None
                )
                lines = _compare(row, record.payload or {})
                problems += sum(1 for line in lines if "MISMATCH" in line)
                for line in lines:
                    print(line)
                dense_ok = dense_dim == cfg.qdrant.dense_dim
                sparse_ok = sparse_nnz is not None and sparse_nnz > 0
                problems += 0 if dense_ok else 1
                problems += 0 if sparse_ok else 1
                verdict = "OK" if dense_ok and sparse_ok else "PROBLEM"
                print(f"    vectors          dense={dense_dim}d sparse_nnz={sparse_nnz} {verdict}")
                print(
                    f"    overview         tokens={row.overview_token_count} "
                    f"truncated={row.overview_truncated} "
                    f"title={row.title[:60]!r}"
                )
                print(
                    f"    raw ages         raw_min={row.raw_min_age!r} raw_max={row.raw_max_age!r}"
                )
                payload_text = repr(record.payload)
                if row.eligibility_text and row.eligibility_text[:40] in payload_text:
                    problems += 1
                    print("    PROBLEM: eligibility text leaked into the Qdrant payload")

        print(f"\nproblems: {problems}")
        return problems
    finally:
        await index.close()
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.verify",
        description="Check that Postgres and Qdrant agree about the indexed trials.",
    )
    parser.add_argument(
        "--hand-check",
        type=int,
        default=0,
        metavar="N",
        help="print N trials' Postgres row beside their Qdrant payload",
    )
    parser.add_argument(
        "--nct-id",
        action="append",
        dest="nct_ids",
        metavar="NCT…",
        help="hand-check this specific trial (repeatable)",
    )
    args = parser.parse_args(argv)
    problems = asyncio.run(verify(hand_check=args.hand_check, nct_ids=args.nct_ids))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
