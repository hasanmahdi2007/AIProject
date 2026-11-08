"""Turning one raw XML record into the rows and the point that get written.

This module is pure: no database, no Qdrant, no model downloads beyond the
tokenizer the :class:`~ingestion.overview.OverviewBuilder` already holds. That
is what makes the derivation testable against real messy records without any
infrastructure, and what keeps the numbers in :class:`IngestStats` honest --
they are counted at derivation time, over every record seen, whether or not the
write that follows succeeds.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ingestion.corpus import RawTrial
from ingestion.criteria import Criterion, SplitResult, split_eligibility
from ingestion.overview import Overview, OverviewBuilder
from ingestion.parsers import (
    AgeBounds,
    normalise_phase,
    normalise_status,
    parse_age_bounds,
    parse_sex,
)

__all__ = ["DerivedTrial", "IngestStats", "derive_trial"]


@dataclass(frozen=True, slots=True)
class DerivedTrial:
    """Everything one trial contributes to both stores.

    ``payload`` is the Qdrant payload and carries no eligibility text: the
    retrieval unit is a trial, and the rules live in Postgres where a judgment
    can cite them by ``criterion_id``.
    """

    raw: RawTrial
    ages: AgeBounds
    sex: str | None
    phase: str | None
    status: str | None
    overview: Overview
    split: SplitResult

    @property
    def nct_id(self) -> str:
        return self.raw.nct_id

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return self.split.criteria

    def trial_row(self) -> dict[str, Any]:
        """The ``trials`` row, minus ``indexed_at``.

        ``indexed_at`` is set only after the Qdrant point exists, so it is the
        marker that both stores agree about this trial -- see
        :mod:`ingestion.store`.
        """
        return {
            "nct_id": self.raw.nct_id,
            "title": self.raw.title,
            "conditions": list(self.raw.conditions),
            "interventions": list(self.raw.interventions),
            "phase": self.phase,
            "status": self.status,
            "min_age_years": self.ages.min_years,
            "max_age_years": self.ages.max_years,
            "sex": self.sex,
            "raw_min_age": self.ages.raw_min,
            "raw_max_age": self.ages.raw_max,
            "locations": list(self.raw.locations),
            "brief_summary": self.raw.brief_summary,
            "eligibility_text": self.raw.eligibility_text,
            "overview_text": self.overview.text,
            "overview_token_count": self.overview.token_count,
            "overview_truncated": self.overview.truncated,
            "source": "trec",
        }

    def qdrant_payload(self) -> dict[str, Any]:
        """The filterable payload.

        Unknown age and sex are written as explicit ``None`` rather than being
        omitted. Qdrant's ``IsEmpty`` condition matches a null value and a
        missing key alike, so B2's NULL-safe filter behaves the same either
        way; writing the key makes a hand-check against the Postgres row a
        direct comparison instead of an argument about absence.
        """
        return {
            "nct_id": self.raw.nct_id,
            "min_age_years": self.ages.min_years,
            "max_age_years": self.ages.max_years,
            "sex": self.sex,
            "phase": self.phase,
            "status": self.status,
        }


@dataclass
class IngestStats:
    """Counters an ingest run reports, accumulated over every record derived.

    These are measurements, not estimates: ``overview_truncated`` divided by
    ``trials`` is the truncation rate the brief asks for as an actual number.
    """

    trials: int = 0
    overview_truncated: int = 0
    overview_tokens_total: int = 0
    overview_tokens_dropped: int = 0
    min_age_null: int = 0
    max_age_null: int = 0
    age_unparsed: int = 0
    age_inverted: int = 0
    sex_null: int = 0
    phase_null: int = 0
    status_null: int = 0
    no_eligibility_text: int = 0
    no_criteria_extracted: int = 0
    criteria_total: int = 0
    criteria_truncated_trials: int = 0
    trials_without_headers: int = 0
    fallback_trials: int = 0
    parser_counts: dict[str, int] = field(default_factory=dict)
    marker_styles: dict[str, int] = field(default_factory=dict)

    def observe(self, derived: DerivedTrial) -> None:
        self.trials += 1
        self.overview_tokens_total += derived.overview.token_count
        if derived.overview.truncated:
            self.overview_truncated += 1
            self.overview_tokens_dropped += derived.overview.dropped_tokens
        if derived.ages.min_years is None:
            self.min_age_null += 1
        if derived.ages.max_years is None:
            self.max_age_null += 1
        if derived.ages.unparsed:
            self.age_unparsed += 1
        if derived.ages.inverted:
            self.age_inverted += 1
        if derived.sex is None:
            self.sex_null += 1
        if derived.phase is None:
            self.phase_null += 1
        if derived.status is None:
            self.status_null += 1
        if not derived.raw.eligibility_text:
            self.no_eligibility_text += 1
        if not derived.criteria:
            self.no_criteria_extracted += 1
        self.criteria_total += len(derived.criteria)
        if derived.split.truncated:
            self.criteria_truncated_trials += 1
        if not derived.split.had_headers:
            self.trials_without_headers += 1
        if derived.split.is_fallback:
            self.fallback_trials += 1
        for parser, count in derived.split.parser_counts.items():
            self.parser_counts[parser] = self.parser_counts.get(parser, 0) + count
        style = derived.split.marker_style
        self.marker_styles[style] = self.marker_styles.get(style, 0) + 1

    def observe_all(self, derived: Iterable[DerivedTrial]) -> None:
        for item in derived:
            self.observe(item)

    @property
    def truncation_rate(self) -> float:
        """Share of trials whose overview exceeded the 512-token window."""
        return 0.0 if not self.trials else self.overview_truncated / self.trials

    @property
    def fallback_rate(self) -> float:
        """Share of trials whose criteria came only from the prose fallback."""
        return 0.0 if not self.trials else self.fallback_trials / self.trials

    def as_dict(self) -> dict[str, Any]:
        """Flat, log-friendly summary with the derived rates included."""
        return {
            "trials": self.trials,
            "overview_truncated": self.overview_truncated,
            "truncation_rate": round(self.truncation_rate, 6),
            "overview_tokens_mean": (
                0.0 if not self.trials else round(self.overview_tokens_total / self.trials, 2)
            ),
            "overview_tokens_dropped": self.overview_tokens_dropped,
            "min_age_null": self.min_age_null,
            "max_age_null": self.max_age_null,
            "age_unparsed": self.age_unparsed,
            "age_inverted": self.age_inverted,
            "sex_null": self.sex_null,
            "phase_null": self.phase_null,
            "status_null": self.status_null,
            "no_eligibility_text": self.no_eligibility_text,
            "no_criteria_extracted": self.no_criteria_extracted,
            "criteria_total": self.criteria_total,
            "criteria_truncated_trials": self.criteria_truncated_trials,
            "trials_without_headers": self.trials_without_headers,
            "fallback_trials": self.fallback_trials,
            "fallback_rate": round(self.fallback_rate, 6),
            "parser_counts": dict(sorted(self.parser_counts.items())),
            "marker_styles": dict(sorted(self.marker_styles.items())),
        }


def derive_trial(raw: RawTrial, builder: OverviewBuilder) -> DerivedTrial:
    """Parse, split and build the overview for one raw record."""
    return DerivedTrial(
        raw=raw,
        ages=parse_age_bounds(raw.raw_min_age, raw.raw_max_age),
        sex=parse_sex(raw.raw_gender),
        phase=normalise_phase(raw.phase),
        status=normalise_status(raw.status),
        overview=builder.build(
            raw.title, raw.conditions, raw.interventions, raw.brief_summary
        ),
        split=split_eligibility(raw.nct_id, raw.eligibility_text),
    )
