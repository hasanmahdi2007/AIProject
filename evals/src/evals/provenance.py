"""What a number was measured on, carried alongside the number itself.

The problem this solves
-----------------------
A Recall@100 of 0.62 measured against **130,024 indexed trials** and a
Recall@100 of 0.62 measured against the full **375,580** are different claims,
and only the second is comparable to a published TREC Clinical Trials or
TrialGPT figure. A partial index cannot contain some of the relevant trials the
qrels judge, so recall against it is bounded above by the fraction of judged
relevant documents that were indexed at all - the number is not merely noisier,
it is measuring a different thing.

The failure mode is specific and entirely plausible: a number is computed
during development against whatever is indexed at the time, written into a
notebook or a chat message, and three weeks later copied into ``README.md`` as
the headline. Nothing about the digits says which corpus produced them.

So every result this harness produces carries a :class:`CorpusCoverage`, and
anything short of a complete index makes the result **not publishable** -
:attr:`MeasurementProvenance.publishable` goes ``False``, the report stamps a
banner on it, and :mod:`evals.report` refuses to present it as a headline.
Coverage that was never established is treated exactly as harshly as coverage
that is known to be partial: :data:`UNKNOWN_COVERAGE` is not publishable
either, because "we did not check" is not evidence of completeness.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from evals.metrics import MetricSpec

__all__ = [
    "FULL_CORPUS_TRIALS",
    "PARTIAL_INDEX_2026_11_09",
    "UNKNOWN_COVERAGE",
    "UNKNOWN_COVERAGE_SENTINEL",
    "CorpusCoverage",
    "MeasurementProvenance",
    "git_commit",
]

FULL_CORPUS_TRIALS: Final = 375_580
"""Documents in the TREC CT 2021 collection, which both tracks share.

Matches ``ir_datasets``' own metadata for ``clinicaltrials/2021``, which
``evals/tests/test_trec.py`` asserts independently. A number measured against
fewer than this is a partial-corpus number.
"""

UNKNOWN_COVERAGE_SENTINEL: Final = -1


@dataclass(frozen=True, slots=True)
class CorpusCoverage:
    """How much of the TREC collection was actually searchable for a run.

    Args:
        indexed_trials: documents present in the index the run searched, or
            :data:`UNKNOWN_COVERAGE_SENTINEL` when it was not established.
        total_trials: the collection size, normally
            :data:`FULL_CORPUS_TRIALS`.
        source: where ``indexed_trials`` came from, verbatim, so a reader can
            re-check it. "Qdrant collection 'trials' count" is a source; "we
            think it finished" is not.
        measured_at: when the count was taken. An ingest that is still running
            invalidates an older count.
    """

    indexed_trials: int
    total_trials: int = FULL_CORPUS_TRIALS
    source: str = "not established"
    measured_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.total_trials < 1:
            raise ValueError(f"total_trials must be positive, got {self.total_trials}")
        if self.indexed_trials < 0 and self.indexed_trials != UNKNOWN_COVERAGE_SENTINEL:
            raise ValueError(f"indexed_trials must be >= 0, got {self.indexed_trials}")
        if self.indexed_trials > self.total_trials:
            raise ValueError(
                f"indexed {self.indexed_trials} > collection size {self.total_trials}: "
                "the index holds documents the TREC collection does not, so the "
                "corpus is not the one the qrels judge"
            )

    @property
    def is_known(self) -> bool:
        return self.indexed_trials != UNKNOWN_COVERAGE_SENTINEL

    @property
    def is_complete(self) -> bool:
        """Whether the whole collection was searchable. The only publishable state."""
        return self.is_known and self.indexed_trials == self.total_trials

    @property
    def fraction(self) -> float | None:
        if not self.is_known:
            return None
        return self.indexed_trials / self.total_trials

    def banner(self) -> str:
        """A one-line warning to stamp on any result this coverage produced."""
        if self.is_complete:
            return (
                f"FULL CORPUS: all {self.total_trials:,} TREC 2021 trials indexed ({self.source})."
            )
        if not self.is_known:
            return (
                "CORPUS COVERAGE NOT ESTABLISHED. This number is not comparable to "
                "any published figure and must not be published. Count the indexed "
                "documents and re-run."
            )
        pct = f"{(self.fraction or 0.0) * 100:.1f}%"
        return (
            f"PARTIAL CORPUS: {self.indexed_trials:,} of {self.total_trials:,} trials "
            f"indexed ({pct}; {self.source}). Recall is bounded above by the share of "
            "judged relevant trials that were indexed at all, so these numbers are "
            "NOT comparable to published TREC or TrialGPT figures and must not be "
            "published."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexed_trials": self.indexed_trials if self.is_known else None,
            "total_trials": self.total_trials,
            "fraction": self.fraction,
            "is_complete": self.is_complete,
            "source": self.source,
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
        }


UNKNOWN_COVERAGE = CorpusCoverage(indexed_trials=UNKNOWN_COVERAGE_SENTINEL)
"""The default. Deliberately not publishable: an unchecked corpus is not a
full one, and the harness must never assume completeness it did not verify."""

PARTIAL_INDEX_2026_11_09 = CorpusCoverage(
    indexed_trials=130_024,
    source=(
        "Qdrant collection 'trials', counted by the orchestrator 2026-11-09; "
        "ingest stopped early under host memory pressure, ~7h of work remaining"
    ),
    measured_at=datetime(2026, 11, 9, tzinfo=UTC),
)
"""The index as it stood when this harness was written: 130,024 of 375,580
(34.6%).

Recorded as a constant so a run against that index can state its coverage
exactly rather than approximately, and so the figure has one home instead of
being retyped. It is **not** a default anywhere - a later, larger index would
make it silently wrong. Pass the count you actually measured.
"""


def git_commit(repo_root: Path | None = None) -> str | None:
    """The current commit, for the report header, or ``None`` if unavailable.

    A report that cannot say which code produced it is hard to trust later.
    Failure is not an error: a tarball with no ``.git`` should still produce a
    report.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    """Everything needed to decide whether a set of numbers can be trusted.

    Attached to every :class:`evals.ablations.AblationReport`, serialised into
    the JSON output and printed at the top of the Markdown report.
    """

    split_name: str
    split_year: int
    n_topics: int
    coverage: CorpusCoverage = UNKNOWN_COVERAGE
    backend: str = "unavailable"
    metric_specs: tuple[MetricSpec, ...] = ()
    resamples: int = 0
    seed: int = 0
    relevance_threshold: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    commit: str | None = field(default_factory=git_commit)
    held_out_reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_full_split(self) -> bool:
        """Whether the whole track split was scored, not a subset of it."""
        return self.split_name in {"tune", "held-out"}

    @property
    def publishable(self) -> bool:
        """Whether a number from this measurement may be quoted as a result.

        Three conditions, all necessary:

        * the corpus was complete - a partial index measures something else;
        * the full track split was scored, not a convenience subset;
        * a real retrieval backend produced the run.
        """
        return (
            self.coverage.is_complete
            and self.is_full_split
            and self.backend not in {"unavailable", "scripted"}
        )

    def blockers(self) -> tuple[str, ...]:
        """Why :attr:`publishable` is ``False``, in words. Empty when it is ``True``."""
        reasons: list[str] = []
        if not self.coverage.is_complete:
            reasons.append(self.coverage.banner())
        if not self.is_full_split:
            reasons.append(
                f"split {self.split_name!r} is not a complete track split "
                f"({self.n_topics} topics scored)"
            )
        if self.backend == "unavailable":
            reasons.append("no retrieval backend produced this run")
        elif self.backend == "scripted":
            reasons.append("the run came from a scripted test backend, not from a retrieval system")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": {"name": self.split_name, "year": self.split_year, "topics": self.n_topics},
            "coverage": self.coverage.to_dict(),
            "backend": self.backend,
            "metrics": [
                {
                    "name": s.name,
                    "label": s.label,
                    "kind": s.kind,
                    "k": s.k,
                    "threshold": s.threshold,
                    "gain": s.gain,
                }
                for s in self.metric_specs
            ],
            "bootstrap": {"resamples": self.resamples, "seed": self.seed},
            "created_at": self.created_at.isoformat(),
            "commit": self.commit,
            "held_out_reasons": list(self.held_out_reasons),
            "publishable": self.publishable,
            "blockers": list(self.blockers()),
            "notes": list(self.notes),
        }
