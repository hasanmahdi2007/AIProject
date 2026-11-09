"""Structured-output schemas for the worker's Claude calls.

Each is a thin wrapper around a frozen ``beacon_core`` contract, because
``output_config.format`` needs a single top-level object and several nodes want
a list. The wrappers are the worker's own; the contracts inside them are not,
and in particular :class:`TrialRuleCheck` nests ``CriterionJudgment`` unchanged
- its five fields are frozen, and a sixth would change the schema sent to
Claude, bust the cached prefix and invalidate every recorded fixture.

These are plain ``BaseModel`` rather than ``Contract``: an unexpected extra key
in a model response should be ignored, not turned into a hard failure of a
twenty-trial run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from beacon_core.schemas import CriterionJudgment, RankedQuery

__all__ = [
    "GeneratedQueries",
    "TrialExplanation",
    "TrialRuleCheck",
    "TrialVerification",
    "VerifierVerdict",
]


class GeneratedQueries(BaseModel):
    """Output of ``generate_queries``: N facet-diverse retrieval queries."""

    queries: list[RankedQuery] = Field(default_factory=list)


class TrialRuleCheck(BaseModel):
    """Output of one ``check_rules`` call: every rule of one trial, judged."""

    judgments: list[CriterionJudgment] = Field(default_factory=list)


class VerifierVerdict(BaseModel):
    """Does the cited sentence actually support the label it was cited for?"""

    criterion_id: str
    verdict: Literal["supported", "unsupported", "unclear"]
    note: str = ""


class TrialVerification(BaseModel):
    """Output of the cheap verifier pass over one trial's judgments."""

    verdicts: list[VerifierVerdict] = Field(default_factory=list)


class TrialExplanation(BaseModel):
    """Output of ``rank_and_explain`` for one trial: the patient-facing text.

    The verdict itself is **not** here. ``category`` and
    ``confirmed_fraction`` are derived in code from the judgments, so the model
    writes prose about a decision it cannot change.
    """

    summary: str = ""
