"""Config-driven ablations and the query-count sweep.

An ablation is a *configuration*, not a code path
-------------------------------------------------
Every row of the README's ablation table is one :class:`RunConfig` -
"BM25 only", "dense only", "hybrid RRF", "+ LLM query generation", and so on -
and the only thing that distinguishes two rows is field values. Nothing here
branches on a row's name. That matters because an ablation whose rows run
different code is not an ablation: it cannot tell you whether the dense
retriever helped or whether the dense code path happened to also fix a bug.

:meth:`RunConfig.to_search_request` renders a config into the frozen
``beacon_core.schemas.SearchRequest`` contract, so what the ablation describes
and what the retrieval service is asked for cannot drift apart.

The query-count sweep, and what N actually costs
------------------------------------------------
``N`` is the number of generated search queries, swept over
:data:`QUERY_COUNT_SWEEP` = (1, 3, 5, 10). N=1 doubles as the "does LLM query
generation help at all?" ablation.

**N is not a cost lever, and no report may claim it is.** The queries come from
a single generation call whose output is a handful of short strings, so going
from 3 queries to 10 adds roughly 7e-05 USD per patient - about seven cents per
thousand patients. Anyone can check that arithmetic from the published token
prices, and a report that claims "we chose N=3 to save money" is both false and
trivially falsifiable. The real argument for a small N is **diminishing returns
and noise**: more queries retrieve more near-duplicates of what the first few
already found, each additional query dilutes the RRF fusion, and the measured
differences between N=3, N=5 and N=10 are likely to sit inside each other's
confidence intervals. :func:`run_query_count_sweep` exists so that whichever N
is chosen is a *measured* choice with an interval attached, and so that
"N=3 and N=10 are indistinguishable on 75 topics" is a reportable finding
rather than an assumption.

The sweep runs on 2021 only, enforced by :func:`evals.split.assert_tuning_only`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

from beacon_core.schemas import AgeSexFilter, RankedQuery, SearchRequest

from evals.backends import (
    RetrievalBackend,
    RetrievalUnavailableError,
    RunFileBackend,
    UnavailableBackend,
)
from evals.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ConfidenceInterval,
    PairedComparison,
    bootstrap_mean_ci,
    paired_bootstrap,
)
from evals.metrics import (
    HEADLINE_METRICS,
    CrossCheck,
    MetricSpec,
    RunScores,
    crosscheck_with_ranx,
    score_run,
)
from evals.provenance import UNKNOWN_COVERAGE, CorpusCoverage, MeasurementProvenance
from evals.split import (
    HELD_OUT_ACCESS_LOG,
    TUNE_YEAR,
    EvalSplit,
    assert_tuning_only,
    load_held_out_split,
    load_tuning_split,
    unlock_held_out,
)

__all__ = [
    "ABLATION_TABLE",
    "AGE_FILTER_CENSUS",
    "BASELINE_CONFIG",
    "FULL_SYSTEM",
    "QUERY_COUNT_SWEEP",
    "AblationReport",
    "AblationRow",
    "FusionMode",
    "RunConfig",
    "main",
    "run_ablations",
    "run_query_count_sweep",
    "sweep_configs",
]

logger = logging.getLogger(__name__)

FusionMode = Literal["rrf", "none"]

QUERY_COUNT_SWEEP: Final[tuple[int, ...]] = (1, 3, 5, 10)
"""The N values the sweep measures. The chosen N must come from this table with
an interval attached, never from an assumption."""

AGE_FILTER_CENSUS: Final[dict[str, str]] = {
    "max_age_years_null": "49.8%",
    "min_age_years_null": "9.6%",
    "dense_truncation_rate": "2.350% (3,056 of 130,024)",
}
"""Measured facts about the indexed trials, counted in Postgres by the
orchestrator on 2026-11-09 over the 130,024 trials indexed at that point.

They belong in the ablation discussion rather than in a footnote, because they
predict the result: with ``max_age_years`` absent on **half** the trials, the
age filter can only ever exclude on positive evidence of ineligibility (which
is what ``AgeSexFilter.include_unknown`` enforces), so its measured effect is
bounded by the half of the corpus where the bound is known at all. That is why
the ``+ age/sex filters`` row is allowed to come out flat or negative, and why
that result would be a finding about ClinicalTrials.gov metadata rather than a
bug in the filter.

Counted over the partial index, so treat them as representative rather than
final.
"""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One ablation row: a retrieval configuration with a name.

    Raises:
        ValueError: on a configuration that cannot mean anything - no
            retriever enabled, a non-positive N, or LLM query generation
            switched off while N > 1 (without a generator there is only the
            note itself to search with, so N is necessarily 1).
    """

    name: str
    description: str
    use_sparse: bool = True
    use_dense: bool = True
    fuse: FusionMode = "rrf"
    llm_query_generation: bool = False
    query_count: int = 1
    use_rerank: bool = False
    apply_age_sex_filters: bool = False
    llm_rule_matching: bool = False
    fuse_top_k: int = 100
    rerank_top_k: int = 20
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if not (self.use_sparse or self.use_dense):
            raise ValueError(f"{self.name}: no retriever enabled")
        if self.query_count < 1:
            raise ValueError(f"{self.name}: query_count must be >= 1, got {self.query_count}")
        if not self.llm_query_generation and self.query_count != 1:
            raise ValueError(
                f"{self.name}: query_count={self.query_count} without "
                "llm_query_generation. With no generator the only query available is "
                "the patient note itself, so N must be 1; a config claiming "
                "otherwise would report a query-generation effect it never ran."
            )
        if self.fuse == "rrf" and not (self.use_sparse and self.use_dense):
            raise ValueError(
                f"{self.name}: fuse='rrf' needs both retrievers - fusing one ranked "
                "list with nothing is just that list, and labelling it 'RRF' would "
                "attribute a single-retriever result to fusion. Use fuse='none'."
            )
        if self.rerank_top_k > self.fuse_top_k:
            raise ValueError(
                f"{self.name}: rerank_top_k ({self.rerank_top_k}) > fuse_top_k "
                f"({self.fuse_top_k}); the reranker cannot see more than it is given"
            )

    def to_search_request(self, queries: Sequence[RankedQuery], filters: AgeSexFilter) -> SearchRequest:
        """Render this config into the frozen retrieval contract.

        The one place a config becomes a request, so the ablation table and the
        service can never describe different things. ``filters`` is passed in
        rather than built here because the age and sex come from the patient,
        not from the configuration; what the config decides is only *whether*
        they are applied.
        """
        if len(queries) != self.query_count:
            raise ValueError(
                f"{self.name}: expected {self.query_count} quer(ies), got {len(queries)}"
            )
        applied = filters if self.apply_age_sex_filters else AgeSexFilter()
        return SearchRequest(
            queries=list(queries),
            filters=applied,
            fuse_top_k=self.fuse_top_k,
            rerank_top_k=self.rerank_top_k,
            rrf_k=self.rrf_k,
            use_rerank=self.use_rerank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "use_sparse": self.use_sparse,
            "use_dense": self.use_dense,
            "fuse": self.fuse,
            "llm_query_generation": self.llm_query_generation,
            "query_count": self.query_count,
            "use_rerank": self.use_rerank,
            "apply_age_sex_filters": self.apply_age_sex_filters,
            "llm_rule_matching": self.llm_rule_matching,
            "fuse_top_k": self.fuse_top_k,
            "rerank_top_k": self.rerank_top_k,
            "rrf_k": self.rrf_k,
        }


BM25_ONLY = RunConfig(
    name="bm25_only",
    description="BM25 sparse retrieval only, patient note as the query",
    use_dense=False,
    fuse="none",
)
DENSE_ONLY = RunConfig(
    name="dense_only",
    description="bge-small dense retrieval only, patient note as the query",
    use_sparse=False,
    fuse="none",
)
HYBRID_RRF = RunConfig(
    name="hybrid_rrf",
    description="dense + BM25 fused with RRF (k=60), patient note as the query",
)
HYBRID_QUERIES_N1 = RunConfig(
    name="hybrid_rrf_llm_n1",
    description="hybrid RRF + LLM query generation, N=1 (does generation help at all?)",
    llm_query_generation=True,
    query_count=1,
)
HYBRID_QUERIES_N3 = RunConfig(
    name="hybrid_rrf_llm_n3",
    description="hybrid RRF + LLM query generation, N=3 facet-diverse queries",
    llm_query_generation=True,
    query_count=3,
)
PLUS_RERANK = RunConfig(
    name="plus_rerank",
    description="... + bge-reranker-base cross-encoder rerank to 20",
    llm_query_generation=True,
    query_count=3,
    use_rerank=True,
)
PLUS_FILTERS = RunConfig(
    name="plus_age_sex_filters",
    description="... + age/sex hard filters applied inside the index traversal",
    llm_query_generation=True,
    query_count=3,
    use_rerank=True,
    apply_age_sex_filters=True,
)
FULL_SYSTEM = RunConfig(
    name="full_system",
    description="... + LLM rule matching (the complete pipeline)",
    llm_query_generation=True,
    query_count=3,
    use_rerank=True,
    apply_age_sex_filters=True,
    llm_rule_matching=True,
)

ABLATION_TABLE: Final[tuple[RunConfig, ...]] = (
    BM25_ONLY,
    DENSE_ONLY,
    HYBRID_RRF,
    HYBRID_QUERIES_N1,
    HYBRID_QUERIES_N3,
    PLUS_RERANK,
    PLUS_FILTERS,
    FULL_SYSTEM,
)
"""The README's ablation rows, cumulative from the fourth onward.

Rows 1-3 are alternatives to each other; rows 4-8 each add one thing to the row
above, which is what makes the paired comparison in :class:`AblationRow`
interpretable as "what did *this* component contribute".
"""

BASELINE_CONFIG: Final = BM25_ONLY
"""What every row is compared against by default.

BM25 alone is the honest baseline: it is what the system would be with no
embeddings, no LLM and no reranker, and it is a genuinely strong baseline on
clinical text. Comparing against a weaker straw man would inflate every delta
in the table.
"""


def sweep_configs(counts: Iterable[int] = QUERY_COUNT_SWEEP) -> tuple[RunConfig, ...]:
    """Configs for the N sweep: the full retrieval stack at each N.

    Everything except ``query_count`` is held fixed, which is the only way the
    sweep measures N rather than measuring N plus whatever else changed.
    """
    configs: list[RunConfig] = []
    for n in counts:
        if n < 1:
            raise ValueError(f"query count must be >= 1, got {n}")
        configs.append(
            replace(
                PLUS_RERANK,
                name=f"sweep_n{n}",
                description=f"hybrid RRF + rerank, N={n} generated queries",
                llm_query_generation=True,
                query_count=n,
            )
        )
    return tuple(configs)


# ------------------------------------------------------------------ results --


@dataclass(frozen=True, slots=True)
class AblationRow:
    """One configuration's results on one split."""

    config: RunConfig
    intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)
    """``{metric_name: interval}``. Empty when the row was not measured."""

    n_scored: dict[str, int] = field(default_factory=dict)
    undefined: dict[str, tuple[str, ...]] = field(default_factory=dict)
    comparisons: dict[str, PairedComparison] = field(default_factory=dict)
    """``{metric_name: paired comparison against the baseline}``."""

    not_measured: str | None = None
    """Why this row has no numbers, or ``None`` if it does. A row that could
    not be run is reported as *not measured*; it is never scored as zero,
    which would read as "this configuration retrieves nothing"."""

    @property
    def measured(self) -> bool:
        return self.not_measured is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "measured": self.measured,
            "not_measured": self.not_measured,
            "metrics": {
                name: {
                    "point": ci.point,
                    "low": ci.low,
                    "high": ci.high,
                    "confidence": ci.confidence,
                    "n_topics": ci.n_topics,
                }
                for name, ci in self.intervals.items()
            },
            "n_scored": dict(self.n_scored),
            "undefined_topics": {k: list(v) for k, v in self.undefined.items()},
            "comparisons": {
                name: {
                    "baseline": c.baseline,
                    "delta": c.delta.point,
                    "delta_low": c.delta.low,
                    "delta_high": c.delta.high,
                    "p_value": c.p_value,
                    "significant": c.significant,
                }
                for name, c in self.comparisons.items()
            },
        }


@dataclass(frozen=True, slots=True)
class AblationReport:
    """A whole ablation table, with the provenance needed to trust it."""

    title: str
    provenance: MeasurementProvenance
    rows: tuple[AblationRow, ...]
    baseline: str
    crosschecks: tuple[CrossCheck, ...] = ()
    """``ranx`` recomputation of the baseline row, as insurance against a
    metric that computes the wrong thing."""

    @property
    def measured_rows(self) -> tuple[AblationRow, ...]:
        return tuple(r for r in self.rows if r.measured)

    @property
    def any_measured(self) -> bool:
        return bool(self.measured_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "provenance": self.provenance.to_dict(),
            "baseline": self.baseline,
            "rows": [r.to_dict() for r in self.rows],
            "crosschecks": [
                {
                    "metric": c.metric,
                    "mine": c.mine,
                    "ranx": c.ranx,
                    "abs_diff": c.abs_diff,
                    "agrees": c.agrees,
                    "n_topics": c.n_topics,
                }
                for c in self.crosschecks
            ],
        }


def _run_for_config(
    backend: RetrievalBackend, split: EvalSplit, config: RunConfig
) -> dict[str, list[str]]:
    """Collect one ranked list per topic, or raise ``RetrievalUnavailableError``.

    A :class:`~evals.backends.RunFileBackend` is asked for its whole run in one
    read rather than once per topic; that is both faster and the only way a
    partially-written run file is caught as a unit.
    """
    if isinstance(backend, RunFileBackend):
        run = backend.run_for(config.name)
        missing = [t for t in split.topic_ids if t not in run]
        if missing:
            raise RetrievalUnavailableError(
                f"run file for {config.name!r} is missing {len(missing)} of "
                f"{len(split.topic_ids)} topics (e.g. {missing[:5]}). Scoring the "
                "rest would average over the topics that happened to succeed."
            )
        return {t: list(run[t]) for t in split.topic_ids}
    return {topic.topic_id: backend.search(topic, config) for topic in split.topics}


def run_ablations(
    backend: RetrievalBackend,
    split: EvalSplit,
    configs: Iterable[RunConfig] = ABLATION_TABLE,
    specs: Iterable[MetricSpec] = HEADLINE_METRICS,
    *,
    coverage: CorpusCoverage = UNKNOWN_COVERAGE,
    baseline: RunConfig = BASELINE_CONFIG,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    title: str = "Ablations",
    crosscheck: bool = True,
) -> AblationReport:
    """Score every configuration on ``split`` and compare each to the baseline.

    A configuration the backend cannot serve becomes a *not measured* row with
    the reason attached; it does not abort the table and it is not scored zero.

    Args:
        coverage: how much of the corpus the backend could actually search.
            Defaults to :data:`~evals.provenance.UNKNOWN_COVERAGE`, which marks
            the whole report unpublishable - pass the count you measured.
        crosscheck: recompute the baseline row's metrics with ``ranx`` and
            record whether the two implementations agree.
    """
    configs = tuple(configs)
    specs = tuple(specs)
    if not configs:
        raise ValueError("no ablation configurations to run")

    scores: dict[str, RunScores] = {}
    failures: dict[str, str] = {}
    runs: dict[str, dict[str, list[str]]] = {}
    for config in configs:
        try:
            run = _run_for_config(backend, split, config)
        except RetrievalUnavailableError as exc:
            logger.warning("ablation row %s not measured: %s", config.name, exc)
            failures[config.name] = str(exc)
            continue
        runs[config.name] = run
        scores[config.name] = score_run(run, split.qrels, specs)

    baseline_scores = scores.get(baseline.name)
    rows: list[AblationRow] = []
    for config in configs:
        if config.name in failures:
            rows.append(AblationRow(config=config, not_measured=failures[config.name]))
            continue
        run_scores = scores[config.name]
        intervals: dict[str, ConfidenceInterval] = {}
        comparisons: dict[str, PairedComparison] = {}
        for spec in specs:
            values = run_scores.values(spec.name)
            if not values:
                continue
            intervals[spec.name] = bootstrap_mean_ci(
                values, resamples=resamples, confidence=confidence, seed=seed
            )
            if baseline_scores is not None and config.name != baseline.name:
                base_values = baseline_scores.values(spec.name)
                shared = sorted(set(values) & set(base_values))
                if shared:
                    comparisons[spec.name] = paired_bootstrap(
                        {t: base_values[t] for t in shared},
                        {t: values[t] for t in shared},
                        metric=spec.name,
                        baseline_name=baseline.name,
                        variant_name=config.name,
                        resamples=resamples,
                        confidence=confidence,
                        seed=seed,
                    )
        rows.append(
            AblationRow(
                config=config,
                intervals=intervals,
                n_scored={s.name: run_scores.n_scored(s.name) for s in specs},
                undefined=dict(run_scores.undefined),
                comparisons=comparisons,
            )
        )

    checks: tuple[CrossCheck, ...] = ()
    if crosscheck and baseline.name in runs:
        checks = crosscheck_with_ranx(runs[baseline.name], split.qrels, specs)

    provenance = MeasurementProvenance(
        split_name=split.name,
        split_year=split.year,
        n_topics=len(split.topics),
        coverage=coverage,
        backend=getattr(backend, "name", type(backend).__name__),
        metric_specs=specs,
        resamples=resamples,
        seed=seed,
        held_out_reasons=tuple(u.reason for u in HELD_OUT_ACCESS_LOG),
    )
    return AblationReport(
        title=title,
        provenance=provenance,
        rows=tuple(rows),
        baseline=baseline.name,
        crosschecks=checks,
    )


def run_query_count_sweep(
    backend: RetrievalBackend,
    split: EvalSplit,
    counts: Iterable[int] = QUERY_COUNT_SWEEP,
    specs: Iterable[MetricSpec] = HEADLINE_METRICS,
    *,
    coverage: CorpusCoverage = UNKNOWN_COVERAGE,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> AblationReport:
    """Sweep N over ``counts`` on the **tuning** split and compare each to N=1.

    Raises:
        HeldOutAccessError: if ``split`` is not TREC 2021. Choosing N on the
            held-out split would make the headline number a tuned number, so
            the guard is here rather than in a comment.
    """
    assert_tuning_only(split, what="the query-count sweep")
    configs = sweep_configs(counts)
    return run_ablations(
        backend,
        split,
        configs,
        specs,
        coverage=coverage,
        baseline=configs[0],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        title=f"Query-count sweep (N in {tuple(counts)}) on TREC {split.year}",
    )


# ----------------------------------------------------------------------- CLI --

_HELD_OUT_FLAG = "--i-have-finished-tuning"


def _build_backend(runs_dir: Path | None) -> RetrievalBackend:
    if runs_dir is None:
        return UnavailableBackend()
    backend = RunFileBackend(directory=runs_dir)
    available = backend.available()
    if not available:
        return UnavailableBackend(
            reason=(
                f"no *.run files in {runs_dir}. The retrieval service (agent B2) "
                "should write one TREC run file per ablation configuration, named "
                "<config>.run"
            )
        )
    logger.info("run files found for: %s", ", ".join(available))
    return backend


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m evals.ablations --tune 2021 [--runs DIR]``.

    Evaluating on the held-out split takes two flags, not one: ``--test 2022``
    *and* ``--i-have-finished-tuning "<reason>"``. The brief's
    ``--tune 2021 --test 2022`` therefore fails with an explanation, which is
    the intended behaviour - "measure the held-out set" should be an act
    someone performs deliberately and can be held to, not a default anyone can
    wander into.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.ablations",
        description="Run the Beacon ablation table and the query-count sweep.",
    )
    parser.add_argument(
        "--tune", type=int, choices=[TUNE_YEAR], default=TUNE_YEAR,
        help=f"tuning split (only {TUNE_YEAR} is a tuning split)",
    )
    parser.add_argument(
        "--test", type=int, choices=[2022], default=None,
        help="also measure the held-out split; requires " + _HELD_OUT_FLAG,
    )
    parser.add_argument(
        _HELD_OUT_FLAG, dest="unlock_reason", default=None, metavar="REASON",
        help="written reason (>=20 chars) for reading the held-out 2022 labels",
    )
    parser.add_argument("--runs", type=Path, default=None, help="directory of TREC run files")
    parser.add_argument("--sweep", action="store_true", help="run the query-count sweep too")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--indexed-trials", type=int, default=None,
        help="documents in the index that produced the runs; omitting it marks "
             "the results unpublishable",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.test is not None and not args.unlock_reason:
        parser.error(
            f"--test {args.test} needs {_HELD_OUT_FLAG} \"<reason>\": the 2022 labels "
            "are the headline measurement and are read once, after tuning on "
            f"{TUNE_YEAR} is frozen."
        )

    coverage = (
        CorpusCoverage(indexed_trials=args.indexed_trials, source="--indexed-trials")
        if args.indexed_trials is not None
        else UNKNOWN_COVERAGE
    )
    backend = _build_backend(args.runs)

    from evals.report import render_markdown  # local import: report imports this module

    reports: list[AblationReport] = []
    tune_split = load_tuning_split()
    reports.append(
        run_ablations(
            backend, tune_split, coverage=coverage,
            resamples=args.resamples, seed=args.seed,
            title=f"Ablations on TREC {tune_split.year} (tuning)",
        )
    )
    if args.sweep:
        reports.append(
            run_query_count_sweep(
                backend, tune_split, coverage=coverage,
                resamples=args.resamples, seed=args.seed,
            )
        )
    if args.test is not None:
        held_out = load_held_out_split(unlock_held_out(args.unlock_reason))
        reports.append(
            run_ablations(
                backend, held_out, coverage=coverage,
                resamples=args.resamples, seed=args.seed,
                title=f"HELD-OUT measurement on TREC {held_out.year}",
            )
        )

    for report in reports:
        sys.stdout.write(render_markdown(report))
        sys.stdout.write("\n")
    return 0 if any(r.any_measured for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
