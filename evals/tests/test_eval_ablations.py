"""Ablation runner, N sweep, provenance and report tests.

Everything here runs on a hand-built split with scripted rankings, so the whole
pipeline - config, run, metrics, bootstrap, provenance, Markdown - is exercised
with no service, no network and no corpus. The resample counts are tiny on
purpose; the statistics themselves are tested in ``test_eval_bootstrap.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.ablations import (
    ABLATION_TABLE,
    AGE_FILTER_CENSUS,
    BASELINE_CONFIG,
    FULL_SYSTEM,
    QUERY_COUNT_SWEEP,
    RunConfig,
    main,
    run_ablations,
    run_query_count_sweep,
    sweep_configs,
)
from evals.backends import RunFileBackend, ScriptedBackend, UnavailableBackend, write_run_file
from evals.metrics import RECALL_AT_100, MetricSpec
from evals.provenance import (
    FULL_CORPUS_TRIALS,
    PARTIAL_INDEX_2026_11_09,
    UNKNOWN_COVERAGE,
    CorpusCoverage,
    MeasurementProvenance,
)
from evals.report import DISCLAIMER, ReportBundle, render_markdown, write_report
from evals.split import EvalSplit, HeldOutAccessError
from evals.trec import Topic

FAST = 200
NDCG = MetricSpec(name="ndcg@10", kind="ndcg", k=10)
SPECS = (RECALL_AT_100, NDCG)


def make_split(name: str = "tune-fixture", year: int = 2021) -> EvalSplit:
    """Three topics, each with one relevant and one irrelevant trial."""
    topics = tuple(Topic(topic_id=str(i), text=f"patient {i}") for i in (1, 2, 3))
    qrels = {
        "1": {"NCT1": 2, "NCT9": 0},
        "2": {"NCT2": 2, "NCT8": 0},
        "3": {"NCT3": 1, "NCT7": 0},
    }
    return EvalSplit(name=name, year=year, topics=topics, qrels=qrels)


GOOD = ScriptedBackend(rankings={"1": ["NCT1"], "2": ["NCT2"], "3": ["NCT3"]})
BAD = ScriptedBackend(rankings={"1": ["NCT9"], "2": ["NCT8"], "3": ["NCT7"]})


# ------------------------------------------------------------- RunConfig ----


def test_the_ablation_table_covers_the_readme_rows() -> None:
    names = [c.name for c in ABLATION_TABLE]
    assert names == [
        "bm25_only",
        "dense_only",
        "hybrid_rrf",
        "hybrid_rrf_llm_n1",
        "hybrid_rrf_llm_n3",
        "plus_rerank",
        "plus_age_sex_filters",
        "full_system",
    ]


def test_the_table_is_cumulative_from_the_query_generation_row_on() -> None:
    """Each later row adds exactly one component to the one above, which is
    what makes a paired delta readable as that component's contribution."""
    assert not ABLATION_TABLE[3].use_rerank
    assert ABLATION_TABLE[5].use_rerank
    assert not ABLATION_TABLE[5].apply_age_sex_filters
    assert ABLATION_TABLE[6].apply_age_sex_filters
    assert not ABLATION_TABLE[6].llm_rule_matching
    assert FULL_SYSTEM.llm_rule_matching


def test_bm25_is_the_baseline() -> None:
    """A strong, LLM-free baseline; a weaker one would inflate every delta."""
    assert BASELINE_CONFIG.name == "bm25_only"
    assert BASELINE_CONFIG.use_sparse and not BASELINE_CONFIG.use_dense


def test_a_config_with_no_retriever_is_rejected() -> None:
    with pytest.raises(ValueError, match="no retriever enabled"):
        RunConfig(name="x", description="", use_sparse=False, use_dense=False)


def test_rrf_over_a_single_retriever_is_rejected() -> None:
    """Fusing one list with nothing is that list; calling it RRF would credit
    fusion for a single-retriever result."""
    with pytest.raises(ValueError, match="needs both retrievers"):
        RunConfig(name="x", description="", use_dense=False, fuse="rrf")


def test_a_query_count_above_one_without_a_generator_is_rejected() -> None:
    """With no generator the only available query is the note itself."""
    with pytest.raises(ValueError, match="without llm_query_generation"):
        RunConfig(name="x", description="", query_count=3)


def test_reranking_more_than_was_fused_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot see more than it is given"):
        RunConfig(name="x", description="", fuse_top_k=10, rerank_top_k=20)


def test_the_pipeline_shape_is_375k_to_100_to_20() -> None:
    assert FULL_SYSTEM.fuse_top_k == 100
    assert FULL_SYSTEM.rerank_top_k == 20
    assert FULL_SYSTEM.rrf_k == 60


def test_to_search_request_renders_the_frozen_contract() -> None:
    from beacon_core.schemas import AgeSexFilter, RankedQuery

    queries = [RankedQuery(rank=i + 1, text=f"q{i}") for i in range(3)]
    filters = AgeSexFilter(age=45, sex="male")
    request = FULL_SYSTEM.to_search_request(queries, filters)
    assert request.fuse_top_k == 100
    assert request.rerank_top_k == 20
    assert request.use_rerank is True
    assert request.filters.age == 45
    assert request.filters.include_unknown is True


def test_to_search_request_drops_the_filters_when_the_row_disables_them() -> None:
    """The age/sex ablation row must actually not filter, or the row measures
    nothing."""
    from beacon_core.schemas import AgeSexFilter, RankedQuery

    config = RunConfig(name="x", description="", apply_age_sex_filters=False)
    request = config.to_search_request(
        [RankedQuery(rank=1, text="q")], AgeSexFilter(age=45, sex="male")
    )
    assert request.filters.age is None
    assert request.filters.sex is None


def test_to_search_request_rejects_the_wrong_number_of_queries() -> None:
    from beacon_core.schemas import AgeSexFilter, RankedQuery

    with pytest.raises(ValueError, match="expected 3 quer"):
        FULL_SYSTEM.to_search_request([RankedQuery(rank=1, text="q")], AgeSexFilter())


# ----------------------------------------------------------------- sweep ----


def test_the_sweep_covers_the_documented_n_values() -> None:
    assert QUERY_COUNT_SWEEP == (1, 3, 5, 10)
    assert [c.query_count for c in sweep_configs()] == [1, 3, 5, 10]


def test_the_sweep_holds_everything_except_n_fixed() -> None:
    """Otherwise it measures N plus whatever else moved."""
    configs = sweep_configs()
    varying = {(c.use_sparse, c.use_dense, c.fuse, c.use_rerank, c.rrf_k) for c in configs}
    assert len(varying) == 1


def test_the_sweep_refuses_to_run_on_the_held_out_split() -> None:
    """Choosing N on 2022 would make the headline number a tuned number."""
    with pytest.raises(HeldOutAccessError, match="query-count sweep"):
        run_query_count_sweep(GOOD, make_split("held-out-fixture", 2022), resamples=FAST)


def test_the_sweep_runs_on_the_tuning_split_and_baselines_on_n1() -> None:
    report = run_query_count_sweep(
        ScriptedBackend(rankings={t: ["NCT1", "NCT2", "NCT3"] for t in ("1", "2", "3")}),
        make_split(),
        specs=SPECS,
        resamples=FAST,
    )
    assert report.baseline == "sweep_n1"
    assert [r.config.query_count for r in report.rows] == [1, 3, 5, 10]


def test_sweep_configs_rejects_a_non_positive_n() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        sweep_configs([0])


# ------------------------------------------------------------- the runner ----


def test_a_perfect_run_scores_one_and_a_useless_run_scores_zero() -> None:
    """Hand values: each topic has exactly one relevant trial, so retrieving
    it is Recall@100 = 1.0 and retrieving only the grade-0 trial is 0.0."""
    good = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    bad = run_ablations(BAD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    assert good.rows[0].intervals["recall@100"].point == pytest.approx(1.0)
    assert bad.rows[0].intervals["recall@100"].point == pytest.approx(0.0)


def test_a_grade_one_only_topic_still_counts_at_the_default_threshold() -> None:
    """Topic 3's only relevant trial is grade 1 ("excluded"), and the default
    threshold counts it - that is the documented policy, asserted here on a
    whole-split run rather than only on a single metric call."""
    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    assert report.rows[0].n_scored["recall@100"] == 3


def test_an_unservable_row_is_not_measured_rather_than_scored_zero() -> None:
    """The distinction the whole report depends on: "we could not run this" is
    not "this configuration retrieves nothing"."""
    report = run_ablations(
        UnavailableBackend(), make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST
    )
    row = report.rows[0]
    assert not row.measured
    assert row.intervals == {}
    assert row.not_measured is not None
    assert "no retrieval backend" in row.not_measured
    assert not report.any_measured


def test_one_unservable_row_does_not_abort_the_table(tmp_path: Path) -> None:
    write_run_file(
        tmp_path / "bm25_only.run",
        {"1": ["NCT1"], "2": ["NCT2"], "3": ["NCT3"]},
        tag="t",
    )
    report = run_ablations(
        RunFileBackend(directory=tmp_path),
        make_split(),
        [BASELINE_CONFIG, FULL_SYSTEM],
        SPECS,
        resamples=FAST,
    )
    assert report.rows[0].measured
    assert not report.rows[1].measured
    assert report.any_measured


def test_a_run_file_missing_topics_is_refused(tmp_path: Path) -> None:
    """Scoring the topics that happened to succeed inflates the mean."""
    write_run_file(tmp_path / "bm25_only.run", {"1": ["NCT1"]}, tag="t")
    report = run_ablations(
        RunFileBackend(directory=tmp_path),
        make_split(),
        [BASELINE_CONFIG],
        SPECS,
        resamples=FAST,
    )
    assert not report.rows[0].measured
    assert "missing 2 of 3 topics" in (report.rows[0].not_measured or "")


def test_each_row_is_paired_against_the_baseline() -> None:
    report = run_ablations(
        ScriptedBackend(rankings={"1": ["NCT1"], "2": ["NCT2"], "3": ["NCT3"]}),
        make_split(),
        [BASELINE_CONFIG, RunConfig(name="variant", description="")],
        SPECS,
        resamples=FAST,
    )
    assert "recall@100" not in report.rows[0].comparisons  # baseline vs itself
    comparison = report.rows[1].comparisons["recall@100"]
    assert comparison.baseline == "bm25_only"
    assert comparison.delta.point == pytest.approx(0.0)
    assert not comparison.significant


def test_the_runner_cross_checks_the_baseline_against_ranx() -> None:
    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    assert report.crosschecks
    for check in report.crosschecks:
        assert check.agrees, str(check)


def test_the_runner_rejects_an_empty_config_list() -> None:
    with pytest.raises(ValueError, match="no ablation configurations"):
        run_ablations(GOOD, make_split(), [], SPECS)


def test_the_census_facts_are_recorded_for_the_filter_discussion() -> None:
    """49.8% of trials have no upper age bound, which is why the age/sex row is
    allowed to come out negative."""
    assert AGE_FILTER_CENSUS["max_age_years_null"] == "49.8%"
    assert AGE_FILTER_CENSUS["min_age_years_null"] == "9.6%"


# ------------------------------------------------------------ provenance ----


def test_a_result_records_the_corpus_it_was_measured_against() -> None:
    report = run_ablations(
        GOOD,
        make_split(),
        [BASELINE_CONFIG],
        SPECS,
        coverage=PARTIAL_INDEX_2026_11_09,
        resamples=FAST,
    )
    assert report.provenance.coverage.indexed_trials == 130_024
    assert report.provenance.coverage.total_trials == FULL_CORPUS_TRIALS


def test_a_partial_corpus_result_is_not_publishable() -> None:
    """The failure this guard exists for: a number measured against a third of
    the collection quoted as if it were comparable to a published figure."""
    coverage = CorpusCoverage(indexed_trials=130_024, source="qdrant")
    assert not coverage.is_complete
    assert coverage.fraction == pytest.approx(0.3462, abs=1e-4)
    assert "PARTIAL CORPUS" in coverage.banner()
    assert "NOT comparable" in coverage.banner()


def test_unestablished_coverage_is_treated_as_harshly_as_partial() -> None:
    """ "We did not check" is not evidence of completeness."""
    assert not UNKNOWN_COVERAGE.is_known
    assert not UNKNOWN_COVERAGE.is_complete
    assert UNKNOWN_COVERAGE.fraction is None
    assert "NOT ESTABLISHED" in UNKNOWN_COVERAGE.banner()


def test_a_complete_corpus_is_publishable_with_a_real_backend() -> None:
    full = CorpusCoverage(indexed_trials=FULL_CORPUS_TRIALS, source="qdrant count")
    assert full.is_complete
    prov = MeasurementProvenance(
        split_name="held-out",
        split_year=2022,
        n_topics=50,
        coverage=full,
        backend="run-files",
    )
    assert prov.publishable
    assert prov.blockers() == ()


def test_a_scripted_backend_is_never_publishable() -> None:
    """A test fixture's numbers must not be able to reach a README."""
    prov = MeasurementProvenance(
        split_name="held-out",
        split_year=2022,
        n_topics=50,
        coverage=CorpusCoverage(indexed_trials=FULL_CORPUS_TRIALS, source="x"),
        backend="scripted",
    )
    assert not prov.publishable
    assert any("scripted" in b for b in prov.blockers())


def test_a_subset_split_is_never_publishable() -> None:
    prov = MeasurementProvenance(
        split_name="tune-subset",
        split_year=2021,
        n_topics=5,
        coverage=CorpusCoverage(indexed_trials=FULL_CORPUS_TRIALS, source="x"),
        backend="run-files",
    )
    assert not prov.publishable
    assert any("not a complete track split" in b for b in prov.blockers())


def test_an_index_larger_than_the_collection_is_rejected() -> None:
    """Such an index is not the corpus the qrels judge."""
    with pytest.raises(ValueError, match="not the one the qrels judge"):
        CorpusCoverage(indexed_trials=FULL_CORPUS_TRIALS + 1)


# ---------------------------------------------------------------- report ----


def test_the_markdown_leads_with_the_publishability_verdict() -> None:
    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    markdown = render_markdown(report)
    assert "NOT PUBLISHABLE" in markdown
    assert markdown.index("NOT PUBLISHABLE") < markdown.index("### Results")


def test_the_markdown_states_the_relevance_threshold_and_the_grade_scheme() -> None:
    """A Recall@100 without a stated threshold is uninterpretable."""
    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    markdown = render_markdown(report)
    assert "2 = eligible, 1 = excluded" in markdown
    assert "partially correct retrieval" in markdown
    assert "grade >= 1" in markdown
    assert "judged pool" in markdown


def test_an_unmeasured_row_renders_as_not_measured_never_as_zero() -> None:
    report = run_ablations(
        UnavailableBackend(), make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST
    )
    markdown = render_markdown(report)
    assert "not measured" in markdown
    assert "0.000" not in markdown
    assert "### Rows not measured" in markdown


def test_an_insignificant_delta_is_marked_in_the_table() -> None:
    report = run_ablations(
        ScriptedBackend(rankings={"1": ["NCT1"], "2": ["NCT2"], "3": ["NCT3"]}),
        make_split(),
        [BASELINE_CONFIG, RunConfig(name="variant", description="")],
        SPECS,
        resamples=FAST,
    )
    assert "(n.s.)" in render_markdown(report)


def test_the_report_bundle_carries_the_disclaimer_and_the_n_argument() -> None:
    """Two things that must never be dropped: the clinical framing, and the
    fact that N is not a cost lever."""
    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    sweep = run_query_count_sweep(
        ScriptedBackend(rankings={t: ["NCT1"] for t in ("1", "2", "3")}),
        make_split(),
        specs=SPECS,
        resamples=FAST,
    )
    markdown = ReportBundle(ablations=report, sweep=sweep).to_markdown()
    assert "not medical advice" in markdown
    assert "synthetic" in markdown
    assert "no clinical validity" in markdown
    assert "N is not a cost lever" in markdown
    assert "diminishing returns and noise" in markdown
    assert DISCLAIMER.strip() in markdown


def test_an_uncalibrated_judge_is_reported_as_uncalibrated() -> None:
    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    markdown = ReportBundle(ablations=report, kappa_reason="no hand labels yet").to_markdown()
    assert "Cohen's kappa" in markdown
    assert "no live judge call has been made" in markdown
    assert "claude-opus-5-5" in markdown


def test_writing_a_report_produces_markdown_and_json(tmp_path: Path) -> None:
    import json

    report = run_ablations(GOOD, make_split(), [BASELINE_CONFIG], SPECS, resamples=FAST)
    md_path, json_path = write_report(ReportBundle(ablations=report), tmp_path)
    assert md_path.is_file() and json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ablations"]["provenance"]["publishable"] is False
    assert payload["ablations"]["rows"][0]["measured"] is True
    assert payload["judge_calibration"] is None


# ------------------------------------------------------------------- CLI ----


def test_the_cli_requires_an_explicit_flag_to_touch_the_held_out_split() -> None:
    """The brief's ``--tune 2021 --test 2022`` must fail: evaluating on the
    held-out split is an act someone performs deliberately."""
    with pytest.raises(SystemExit) as exc:
        main(["--tune", "2021", "--test", "2022"])
    assert exc.value.code == 2


def test_the_cli_rejects_tuning_on_2022() -> None:
    with pytest.raises(SystemExit):
        main(["--tune", "2022"])


def test_the_cli_rejects_a_short_unlock_reason(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        main(["--test", "2022", "--i-have-finished-tuning", "ok"])
    except (SystemExit, HeldOutAccessError, OSError):
        pass
    else:  # pragma: no cover - would mean the guard did not fire
        pytest.fail("a two-character reason should not unlock the held-out split")
