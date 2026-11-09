"""The CLI surface and the batching the dual write depends on."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ingestion.run import DEFAULT_CRITERIA_LIMIT, _batched, build_parser


def test_defaults_match_the_documented_run() -> None:
    args = build_parser().parse_args([])
    assert args.limit is None
    assert args.skip == 0
    assert args.batch_size == 256
    assert args.criteria_limit == DEFAULT_CRITERIA_LIMIT
    assert args.force is False
    assert args.recreate_collection is False
    assert args.dry_run is False


def test_smoke_path_arguments() -> None:
    args = build_parser().parse_args(["--limit", "1000"])
    assert args.limit == 1000


def test_criteria_limit_accepts_all_and_zero() -> None:
    assert build_parser().parse_args(["--criteria-limit", "all"]).criteria_limit is None
    assert build_parser().parse_args(["--criteria-limit", "0"]).criteria_limit == 0
    assert build_parser().parse_args(["--criteria-limit", "500"]).criteria_limit == 500


def test_negative_criteria_limit_is_refused() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--criteria-limit", "-5"])


def test_batched_preserves_order_and_yields_the_remainder() -> None:
    """Order matters: the dual write pairs a vector with a trial by index."""
    numbers: Iterator[Any] = iter(range(7))
    assert list(_batched(numbers, 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_batched_on_an_empty_stream_yields_nothing() -> None:
    empty: Iterator[Any] = iter(())
    assert list(_batched(empty, 3)) == []
