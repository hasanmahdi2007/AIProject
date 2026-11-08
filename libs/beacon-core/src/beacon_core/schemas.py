"""Frozen data contracts for Beacon.

Owned by Wave 0 and **read-only to every later agent**. If a shape here has to
change, the orchestrator changes it once, in this file, for everyone.

Three rules this file encodes, each of which is a correctness issue elsewhere:

1. ``None`` on a filterable field means *unknown*, which means *possibly
   eligible*. It never means "exclude". ClinicalTrials.gov age and sex fields
   are frequently absent or unparseable, and a wrongly excluded trial never
   appears anywhere for anyone to notice.
2. ``CriterionJudgment`` is an LLM structured-output schema. Adding a field
   changes the JSON schema sent to Claude, busts the cached prompt prefix and
   invalidates recorded fixtures. Treat its five fields as fixed.
3. ``evidence_sentence_ids`` must point at real sentences in the
   ``NumberedNote`` for the same request. The verifier enforces that; the type
   only makes it expressible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AgeSexFilter",
    "Contract",
    "CriterionJudgment",
    "CriterionKind",
    "CriterionLabel",
    "EligibilityCategory",
    "MatchState",
    "MatchStatus",
    "NoteSentence",
    "NumberedNote",
    "PatientProfile",
    "PipelineStep",
    "QueryFacet",
    "RankedQuery",
    "RankedTrial",
    "SearchRequest",
    "SearchResponse",
    "Sex",
    "TrialCandidate",
    "TrialJudgment",
]


class Contract(BaseModel):
    """Base for every contract model: reject unknown fields loudly."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------- vocabularies --

Sex = Literal["male", "female", "all"]
"""Normalised sex. ``all`` is the trial-side value for "no sex restriction"."""

CriterionKind = Literal["inclusion", "exclusion"]

CriterionLabel = Literal["met", "not_met", "insufficient_info", "not_applicable"]
"""``insufficient_info`` is a success state: it is how the system abstains
instead of guessing, and it feeds the missing-facts list."""

EligibilityCategory = Literal["eligible", "likely_eligible", "insufficient_info", "excluded"]

MatchStatus = Literal["queued", "running", "completed", "failed"]

PipelineStep = Literal[
    "profile_patient",
    "generate_queries",
    "retrieve",
    "check_rules",
    "verify",
    "rank_and_explain",
]

QueryFacet = Literal["diagnosis", "biomarker", "prior_therapy", "setting", "other"]
"""With only N=3 query slots, the generator is instructed to spend one per
distinct clinical facet rather than emitting three paraphrases of one."""


# ------------------------------------------------------------- patient side --


class NoteSentence(Contract):
    """One sentence of the patient note, with the ID that judgments cite."""

    sentence_id: int = Field(ge=1)
    text: str = Field(min_length=1)


class NumberedNote(Contract):
    """The patient note split into numbered sentences.

    Numbering is for *citation*, not retrieval. These IDs are the only thing a
    ``CriterionJudgment`` may point at, which is what lets code check the
    evidence before any LLM looks at it.
    """

    sentences: list[NoteSentence]

    @property
    def valid_ids(self) -> frozenset[int]:
        return frozenset(s.sentence_id for s in self.sentences)


class PatientProfile(Contract):
    """Structured profile extracted from a free-text note.

    ``age`` is a typed ``int`` precisely because embeddings cannot compare
    numbers: every numeric eligibility decision is made against this field or
    against the trial's typed age bounds, never against a similarity score.
    """

    age: int | None = Field(default=None, ge=0, le=120)
    sex: Sex | None = None
    conditions: list[str] = Field(default_factory=list)
    biomarkers: list[str] = Field(default_factory=list)
    prior_treatments: list[str] = Field(default_factory=list)
    labs: dict[str, str] = Field(default_factory=dict)
    performance_status: int | None = Field(default=None, ge=0, le=5, description="ECOG 0-5")
    location: str | None = None


class RankedQuery(Contract):
    """One generated search query. ``rank`` 1 is the most important."""

    rank: int = Field(ge=1)
    text: str = Field(min_length=1)
    facet: QueryFacet = "other"


# --------------------------------------------------------------- trial side --


class TrialCandidate(Contract):
    """A retrieved trial.

    A candidate is **never** a suggestion. Eligibility is decided later, by the
    hard filters and the rule checker; retrieval only answers "is this trial
    about the right disease?".
    """

    nct_id: str = Field(min_length=3)
    title: str = ""
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    phase: str | None = None
    status: str | None = None

    # Typed bounds, in years. None means unknown, so the trial survives filters.
    min_age_years: float | None = None
    max_age_years: float | None = None
    sex: Sex | None = None

    # Scores, each set by the stage that computes it.
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    retrieval_rank: int | None = Field(default=None, ge=1)


class CriterionJudgment(Contract):
    """One eligibility rule, judged against the patient note.

    **Frozen LLM output schema - exactly these five fields.** This is what
    ``client.messages.parse()`` is given, so any change alters the schema sent
    to Claude, busts the cached prompt prefix and invalidates recorded fixtures.
    """

    criterion_id: str
    kind: CriterionKind
    label: CriterionLabel
    evidence_sentence_ids: list[int] = Field(default_factory=list)
    explanation: str


class TrialJudgment(Contract):
    """All rule judgments for one trial, plus the verdict derived from them.

    A trial is ``excluded`` if any inclusion rule is ``not_met`` or any
    exclusion rule is ``met``. Otherwise it is ranked by the share of rules
    confirmed, then by retrieval score.
    """

    nct_id: str
    judgments: list[CriterionJudgment] = Field(default_factory=list)
    category: EligibilityCategory = "insufficient_info"
    confirmed_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_facts: list[str] = Field(default_factory=list)
    summary: str | None = None


class RankedTrial(Contract):
    """One row of the final answer."""

    rank: int = Field(ge=1)
    candidate: TrialCandidate
    judgment: TrialJudgment
    score: float


# ----------------------------------------------- retrieval service contract --


class AgeSexFilter(Contract):
    """Hard filters, applied **inside** both searches, never afterwards.

    Post-filtering the top-k is the overfiltering bug: with ``ef_search=40``
    and a filter matching 10% of rows you average four survivors out of forty,
    so "top 20" silently returns 4.

    ``include_unknown`` must stay ``True``. It is what makes the filter exclude
    only on *positive evidence of ineligibility*: a trial whose ``min_age`` is
    absent or unparseable survives and lets the rule checker decide.
    """

    age: int | None = Field(default=None, ge=0, le=120)
    sex: Sex | None = None
    include_unknown: bool = True


class SearchRequest(Contract):
    """Request body for ``POST /search`` on the retrieval service."""

    queries: list[RankedQuery] = Field(min_length=1)
    filters: AgeSexFilter = Field(default_factory=AgeSexFilter)
    fuse_top_k: int = Field(default=100, ge=1, description="candidates kept after RRF")
    rerank_top_k: int = Field(default=20, ge=1, description="candidates kept after rerank")
    rrf_k: int = Field(default=60, ge=1, description="RRF constant; rank-based, not score-based")
    use_rerank: bool = True


class SearchResponse(Contract):
    """Response body for ``POST /search``."""

    candidates: list[TrialCandidate]
    total_before_rerank: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------- worker pipeline state --


class MatchState(Contract):
    """LangGraph state for one match run, checkpointed to Postgres per node.

    Checkpointing after every node is what makes a worker killed during step 5
    resume at step 5 instead of step 1.
    """

    match_id: str
    patient_note: str
    config_version: str = "v1"

    profile: PatientProfile | None = None
    note: NumberedNote | None = None
    queries: list[RankedQuery] = Field(default_factory=list)
    candidates: list[TrialCandidate] = Field(default_factory=list)
    judgments: list[TrialJudgment] = Field(default_factory=list)
    ranked: list[RankedTrial] = Field(default_factory=list)

    status: MatchStatus = "queued"
    current_step: PipelineStep | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_cost_usd: float = 0.0
