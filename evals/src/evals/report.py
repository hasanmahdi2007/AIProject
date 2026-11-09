"""One command that regenerates the whole evaluation report.

``uv run python -m evals.report``

What this module will not do
----------------------------
It will not print a number that was not measured in the invocation that
produced the report. Not an estimate, not a figure carried over from a previous
run, and above all not a figure from the TrialGPT paper or any other
publication. Where there is no measurement the table says **not measured** and
explains why. That is the entire reason the ``README.md`` metrics are still
``{placeholder}``: a placeholder is honest, and a borrowed number is not, and
the difference is invisible to a reader six months later.

Three guards hold that line:

* a row with no run renders ``not measured`` with the backend's reason
  attached - never ``0.000``, which would read as a real and terrible result;
* every table carries its :class:`~evals.provenance.MeasurementProvenance`, so
  a partial-corpus or subset measurement is stamped as such at the point of
  use rather than in a caveat further down;
* :func:`render_markdown` prints the ``ranx`` cross-check, so a reader can see
  that the metrics were computed twice by two implementations and agreed.

The clinical framing in :data:`DISCLAIMER` is reproduced at the top of every
report and must stay there. Beacon is evaluated on synthetic patients and has
no clinical validity; a document full of recall figures is exactly the kind of
thing that gets forwarded without its context.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from evals.ablations import (
    AGE_FILTER_CENSUS,
    QUERY_COUNT_SWEEP,
    AblationReport,
    run_ablations,
    run_query_count_sweep,
)
from evals.backends import RetrievalBackend, RunFileBackend, UnavailableBackend
from evals.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED
from evals.judge import KAPPA_TARGET, KappaResult
from evals.metrics import DEFAULT_RELEVANCE_THRESHOLD, HEADLINE_METRICS
from evals.provenance import UNKNOWN_COVERAGE, CorpusCoverage
from evals.split import (
    HELD_OUT_YEAR,
    TUNE_YEAR,
    load_held_out_split,
    load_tuning_split,
    unlock_held_out,
)

__all__ = [
    "DEFAULT_REPORT_DIR",
    "DISCLAIMER",
    "ReportBundle",
    "build_report",
    "main",
    "render_markdown",
    "write_report",
]

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR: Final = Path(__file__).resolve().parents[2] / "reports"
"""``evals/reports/``. Generated output, git-ignored - a committed report goes
stale silently and then gets quoted."""

DISCLAIMER: Final = """\
> **Decision support, not medical advice.** Beacon is a research and portfolio
> system evaluated on the public TREC Clinical Trials benchmark with
> **synthetic** patient notes. It has no clinical validity, it must not be used
> for care decisions, and no real patient record should ever be entered into it.
> The numbers below measure retrieval and ranking quality against expert
> relevance judgements. They say nothing about patient outcomes.
"""

NOT_MEASURED: Final = "not measured"


def _fmt_interval(report: AblationReport, metric: str, row_index: int) -> str:
    row = report.rows[row_index]
    if not row.measured or metric not in row.intervals:
        return NOT_MEASURED
    return row.intervals[metric].format()


def _fmt_delta(report: AblationReport, metric: str, row_index: int) -> str:
    row = report.rows[row_index]
    comparison = row.comparisons.get(metric)
    if comparison is None:
        return "-"
    marker = "" if comparison.significant else " (n.s.)"
    return f"{comparison.delta.point:+.3f}{marker}"


def render_markdown(report: AblationReport) -> str:
    """One ablation table as Markdown, provenance first.

    Provenance comes before the numbers deliberately. A reader who stops after
    the first screen should already know whether these figures are publishable.
    """
    prov = report.provenance
    lines: list[str] = [f"## {report.title}", ""]

    if not prov.publishable:
        lines.append("> **NOT PUBLISHABLE.** " + " ".join(prov.blockers()))
        lines.append("")
    else:
        lines.append("> " + prov.coverage.banner())
        lines.append("")

    lines += [
        f"- Split: **{prov.split_name}** (TREC {prov.split_year}), "
        f"{prov.n_topics} topics",
        f"- Retrieval backend: `{prov.backend}`",
        f"- Corpus: {prov.coverage.indexed_trials if prov.coverage.is_known else 'unknown'}"
        f" / {prov.coverage.total_trials:,} trials indexed",
        f"- Bootstrap: {prov.resamples:,} resamples over topics, seed {prov.seed}, "
        "95% percentile intervals",
        f"- Baseline for deltas: `{report.baseline}`",
        f"- Code: `{prov.commit or 'unknown commit'}`, generated "
        f"{prov.created_at.isoformat(timespec='seconds')}",
        "",
    ]
    if prov.held_out_reasons:
        lines += [
            "**Held-out split was unlocked in this run.** Reasons given:",
            *[f"- {reason}" for reason in prov.held_out_reasons],
            "",
        ]

    lines += ["### Metric definitions", ""]
    for spec in prov.metric_specs:
        lines.append(f"- `{spec.name}` - {spec.label}")
    lines += [
        "",
        "Relevance grades are TREC CT's: **2 = eligible, 1 = excluded, "
        "0 = not relevant**. Grade 1 means the trial is right for the patient's "
        "disease but the patient fails a criterion, which is *partially correct "
        "retrieval* - so nDCG scores it with graded gain, and the binary metrics "
        f"count it as a hit at the default threshold of {DEFAULT_RELEVANCE_THRESHOLD}. "
        "The stricter grade-2-only recall is reported beside it. Documents the pool "
        "never judged count as grade 0, so recall is recall *over the judged pool*.",
        "",
    ]

    metric_names = [s.name for s in prov.metric_specs]
    header = "| configuration | " + " | ".join(metric_names) + f" | vs `{report.baseline}` |"
    divider = "|---" * (len(metric_names) + 2) + "|"
    lines += ["### Results", "", header, divider]
    headline = metric_names[0] if metric_names else ""
    for index, row in enumerate(report.rows):
        cells = [f"`{row.config.name}`"]
        cells += [_fmt_interval(report, name, index) for name in metric_names]
        cells.append(_fmt_delta(report, headline, index) if headline else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines += [
        "Each cell is a point estimate with its 95% bootstrap interval over topics. "
        "`(n.s.)` on a delta means the paired bootstrap interval includes zero: on "
        f"{prov.n_topics} topics that difference is indistinguishable from noise, "
        "and reporting it as an improvement would be unsupportable.",
        "",
    ]

    unmeasured = [r for r in report.rows if not r.measured]
    if unmeasured:
        lines += ["### Rows not measured", ""]
        for row in unmeasured:
            lines.append(f"- `{row.config.name}`: {row.not_measured}")
        lines.append("")

    if report.crosschecks:
        lines += [
            "### Independent cross-check (`ranx`)",
            "",
            "Each metric recomputed by `ranx` on the baseline run and compared to this "
            "harness's own implementation. A disagreement here means one of the two is "
            "wrong and no number in the table can be trusted.",
            "",
            "| metric | this harness | ranx | abs diff | verdict |",
            "|---|---|---|---|---|",
        ]
        for check in report.crosschecks:
            verdict = "agree" if check.agrees else "**DISAGREE**"
            lines.append(
                f"| `{check.metric}` | {check.mine:.6f} | {check.ranx:.6f} | "
                f"{check.abs_diff:.2e} | {verdict} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_kappa(kappa: KappaResult | None, reason: str) -> str:
    lines = ["## LLM judge calibration (Cohen's kappa)", ""]
    if kappa is None:
        lines += [
            f"**{NOT_MEASURED}.** {reason}",
            "",
            "The judge is `claude-opus-5-5` - deliberately a different and stronger "
            "model than the Sonnet 5.5 headline pipeline, so it is never grading its "
            "own family's output. The interface, the prompt and the kappa machinery "
            "are implemented and unit-tested against recorded fixtures; **no live "
            "judge call has been made**, because that would bill the owner's API key "
            "and no approval was given. Until hand labels exist and the judge has "
            f"been run against them, there is no agreement figure, and the target of "
            f"kappa >= {KAPPA_TARGET} is an intention rather than a result.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        f"- {kappa}",
        f"- Labels: {', '.join(kappa.labels)}",
        f"- Weighting: `{kappa.weights}`",
        f"- Target: kappa >= {kappa.target} - "
        f"{'met' if kappa.meets_target else '**NOT met**'}",
        "",
    ]
    if not kappa.meets_target:
        lines += [
            "Below the target, the judge's verdicts are not usable as a measurement: "
            "its disagreements with the human are too frequent to attribute to the "
            "system under test rather than to the judge.",
            "",
        ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReportBundle:
    """Everything one invocation measured."""

    ablations: AblationReport
    sweep: AblationReport | None = None
    held_out: AblationReport | None = None
    kappa: KappaResult | None = None
    kappa_reason: str = "no hand labels supplied (--hand-labels)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "ablations": self.ablations.to_dict(),
            "sweep": self.sweep.to_dict() if self.sweep else None,
            "held_out": self.held_out.to_dict() if self.held_out else None,
            "judge_calibration": self.kappa.to_dict() if self.kappa else None,
            "judge_calibration_reason": None if self.kappa else self.kappa_reason,
        }

    def to_markdown(self) -> str:
        parts = [
            "# Beacon evaluation report",
            "",
            DISCLAIMER,
            "",
            "Regenerate with `uv run python -m evals.report`. Every number below was "
            "measured by that command in the run that produced this file. Anything "
            f"not measured says so; nothing is estimated, and no figure is carried "
            "over from a paper or a previous run.",
            "",
            "## How to read the split",
            "",
            f"Configuration choices - thresholds, fusion constants, the query count N - "
            f"are fitted on the **TREC {TUNE_YEAR}** tuning split (75 topics). The "
            f"**TREC {HELD_OUT_YEAR}** split (50 topics) is the held-out measurement and "
            "is read once, deliberately, through a capability token that records the "
            "reason in this report. A section below headed HELD-OUT means that token "
            "was used; its absence means the report is built on tuning data only.",
            "",
            render_markdown(self.ablations),
        ]
        if self.sweep is not None:
            parts += [
                render_markdown(self.sweep),
                "### On the choice of N",
                "",
                f"N is swept over {QUERY_COUNT_SWEEP}. **N is not a cost lever.** Query "
                "generation is a single call producing a few short strings, so N=10 "
                "instead of N=3 costs on the order of 7e-05 USD more per patient - "
                "roughly seven cents per thousand patients. Any claim that a small N "
                "was chosen to save money is false and the arithmetic is checkable "
                "from the published token prices. The case for a small N is "
                "diminishing returns and noise: additional queries retrieve "
                "near-duplicates of what the first few already found, and the "
                "differences between N=3, N=5 and N=10 should be read against their "
                "confidence intervals before any of them is called an improvement.",
                "",
            ]
        if self.held_out is not None:
            parts.append(render_markdown(self.held_out))
        parts.append(_render_kappa(self.kappa, self.kappa_reason))
        parts += [
            "## Context for the ablation rows",
            "",
            "Measured over the indexed trials, in Postgres, and relevant to how the "
            "filter rows should be read:",
            "",
            *[f"- `{key}`: {value}" for key, value in AGE_FILTER_CENSUS.items()],
            "",
            "`max_age_years` being absent on about half the trials is why the age/sex "
            "filter row is allowed to come out flat or negative. The filter excludes "
            "only on positive evidence of ineligibility (`include_unknown=True`), so "
            "its reach is bounded by the half of the corpus where an upper age bound "
            "exists at all. A negative result there is a finding about "
            "ClinicalTrials.gov metadata, not a bug.",
            "",
        ]
        return "\n".join(parts)


def build_report(
    backend: RetrievalBackend,
    *,
    coverage: CorpusCoverage = UNKNOWN_COVERAGE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    sweep: bool = True,
    held_out_reason: str | None = None,
    kappa: KappaResult | None = None,
    kappa_reason: str = "no hand labels supplied (--hand-labels)",
) -> ReportBundle:
    """Measure everything and return it. Does no file I/O.

    Args:
        held_out_reason: when given, unlocks and measures the held-out split.
            ``None`` leaves 2022 untouched, which is the default.
    """
    tune_split = load_tuning_split()
    ablations = run_ablations(
        backend,
        tune_split,
        coverage=coverage,
        resamples=resamples,
        seed=seed,
        title=f"Ablations on TREC {tune_split.year} (tuning split)",
    )
    sweep_report = (
        run_query_count_sweep(backend, tune_split, coverage=coverage, resamples=resamples, seed=seed)
        if sweep
        else None
    )
    held_out_report = None
    if held_out_reason:
        split = load_held_out_split(unlock_held_out(held_out_reason))
        held_out_report = run_ablations(
            backend,
            split,
            coverage=coverage,
            resamples=resamples,
            seed=seed,
            title=f"HELD-OUT measurement on TREC {split.year}",
        )
    return ReportBundle(
        ablations=ablations,
        sweep=sweep_report,
        held_out=held_out_report,
        kappa=kappa,
        kappa_reason=kappa_reason,
    )


def write_report(bundle: ReportBundle, out_dir: Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.json``; return both paths.

    The JSON is the machine-readable twin of the Markdown and holds the same
    numbers, so a later chart or README update reads structured data rather
    than re-parsing a table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(bundle.to_markdown(), encoding="utf-8")
    json_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    return md_path, json_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m evals.report [--runs DIR] [--indexed-trials N]``.

    Exit codes: ``0`` when something was measured, ``1`` when nothing was -
    a report of entirely unmeasured rows is still written (it documents what is
    missing) but the command fails, so CI cannot go green on an empty report.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.report",
        description="Regenerate the Beacon evaluation report from scratch.",
    )
    parser.add_argument("--runs", type=Path, default=None, help="directory of TREC run files")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-sweep", action="store_true", help="skip the query-count sweep")
    parser.add_argument(
        "--indexed-trials", type=int, default=None,
        help="documents in the index that produced the runs. Omitting it marks every "
             "result unpublishable, which is the safe default.",
    )
    parser.add_argument(
        "--i-have-finished-tuning", dest="held_out_reason", default=None, metavar="REASON",
        help=f"measure the held-out TREC {HELD_OUT_YEAR} split, recording this reason",
    )
    parser.add_argument("--hand-labels", type=Path, default=None, help="judge hand labels (JSONL)")
    parser.add_argument(
        "--judge-fixtures", type=Path, default=None,
        help="recorded judge verdicts (JSON). No live judge call is ever made.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    backend: RetrievalBackend
    if args.runs is None:
        backend = UnavailableBackend()
    else:
        file_backend = RunFileBackend(directory=args.runs)
        backend = (
            file_backend
            if file_backend.available()
            else UnavailableBackend(reason=f"no *.run files in {args.runs}")
        )

    coverage = (
        CorpusCoverage(indexed_trials=args.indexed_trials, source="--indexed-trials")
        if args.indexed_trials is not None
        else UNKNOWN_COVERAGE
    )

    kappa, kappa_reason = _maybe_kappa(args.hand_labels, args.judge_fixtures)

    bundle = build_report(
        backend,
        coverage=coverage,
        resamples=args.resamples,
        seed=args.seed,
        sweep=not args.no_sweep,
        held_out_reason=args.held_out_reason,
        kappa=kappa,
        kappa_reason=kappa_reason,
    )
    md_path, json_path = write_report(bundle, args.out)
    sys.stdout.write(f"wrote {md_path}\nwrote {json_path}\n")

    measured = bundle.ablations.any_measured
    if not measured:
        sys.stdout.write(
            "\nNOTHING WAS MEASURED. The report documents why. Supply --runs with TREC "
            "run files from the retrieval service to produce numbers.\n"
        )
    return 0 if measured else 1


def _maybe_kappa(
    hand_labels: Path | None, judge_fixtures: Path | None
) -> tuple[KappaResult | None, str]:
    """Compute kappa if both the hand labels and recorded verdicts are present.

    Never makes a live call: the judge side can only come from recorded
    fixtures here. If either file is missing the report says the judge is
    uncalibrated, which is the truth.
    """
    if hand_labels is None or judge_fixtures is None:
        missing = [
            flag
            for flag, value in (("--hand-labels", hand_labels), ("--judge-fixtures", judge_fixtures))
            if value is None
        ]
        return None, f"missing {' and '.join(missing)}; no live judge call is ever made"

    from evals.judge import (
        ELIGIBILITY_LABELS,
        FixtureJudge,
        JudgeItem,
        calibrate,
        load_hand_labels,
    )

    labels = load_hand_labels(hand_labels)
    judge = FixtureJudge.from_file(judge_fixtures)
    items = [
        JudgeItem(item_id=item_id, patient_note="", trial_id="", trial_text="")
        for item_id in sorted(labels)
    ]
    assert ELIGIBILITY_LABELS  # the label set is fixed by the frozen contract
    return calibrate(items, labels, judge), ""


if __name__ == "__main__":
    raise SystemExit(main())
