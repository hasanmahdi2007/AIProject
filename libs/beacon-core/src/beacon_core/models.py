"""SQLAlchemy 2 models for the ten Beacon tables.

Frozen in Wave 0, read-only afterwards. These live in ``beacon_core`` rather
than per service on purpose: four agents mapping the same tables independently
is exactly the drift the contract freeze exists to prevent.

Postgres is the source of truth. Qdrant holds a *derived* index, which is why
ingestion writes Postgres first and Qdrant second, idempotently by ``nct_id``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "Base",
    "CriterionJudgmentRow",
    "EvalRun",
    "Feedback",
    "LlmCall",
    "MatchRun",
    "Outbox",
    "Patient",
    "Trial",
    "TrialCriterion",
    "TrialMatch",
]

# JSONB on Postgres, plain JSON elsewhere (so SQLite-backed unit tests work).
Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Trial(Base):
    """One clinical trial. ``nct_id`` is the natural key everywhere in Beacon.

    The eligibility criteria are stored here as **raw text** and deliberately
    not indexed in Qdrant: the retrieval unit is a trial, not a rule. Rules are
    split lazily into ``trial_criteria`` only for trials that get retrieved.
    """

    __tablename__ = "trials"

    nct_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text, default="")
    conditions: Mapped[list[str]] = mapped_column(Json, default=list)
    interventions: Mapped[list[str]] = mapped_column(Json, default=list)
    phase: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str | None] = mapped_column(String(64))

    # Typed bounds in years. NULL means unknown, which means possibly eligible.
    min_age_years: Mapped[float | None] = mapped_column(Float)
    max_age_years: Mapped[float | None] = mapped_column(Float)
    sex: Mapped[str | None] = mapped_column(String(8))

    raw_min_age: Mapped[str | None] = mapped_column(String(64))
    raw_max_age: Mapped[str | None] = mapped_column(String(64))
    """Kept verbatim so a parser bug is diagnosable after the fact."""

    locations: Mapped[list[dict[str, Any]]] = mapped_column(Json, default=list)
    brief_summary: Mapped[str | None] = mapped_column(Text)
    eligibility_text: Mapped[str | None] = mapped_column(Text)

    overview_text: Mapped[str | None] = mapped_column(Text)
    """Exactly what was embedded: title + conditions + interventions + summary,
    in that order. Stored so the index can be rebuilt without re-deriving it."""

    overview_token_count: Mapped[int | None] = mapped_column(Integer)
    overview_truncated: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    """True when the overview exceeded the embedding model's 512-token window.
    The share of trials with this set is a reportable finding, not a guess."""

    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(32), default="trec", server_default="trec")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("sex IS NULL OR sex IN ('male','female','all')", name="ck_trials_sex"),
        Index("ix_trials_status", "status"),
        Index("ix_trials_age", "min_age_years", "max_age_years"),
    )


class TrialCriterion(Base):
    """One eligibility rule. Parsed once per trial, reused for every patient.

    Split structure-aware (section header, bullet, numbering), never
    fixed-size: a rule cut mid-sentence makes its judgment meaningless.
    """

    __tablename__ = "trial_criteria"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    nct_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("trials.nct_id", ondelete="CASCADE"), nullable=False
    )
    criterion_id: Mapped[str] = mapped_column(String(64), nullable=False)
    """Stable, human-readable: ``NCT01234567:inc:03``. Cited in judgments."""

    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    parser: Mapped[str] = mapped_column(String(32), default="rules", server_default="rules")
    """``rules`` or ``unstructured`` - so the fallback rate is measurable.

    ``rules`` is the structure-aware split: a section header, then bullets or
    numbering. ``unstructured`` is the regex fallback for the records whose
    shape defeats that.

    Not ``llm_fallback``, which this docstring claimed until 9 Nov. No LLM is
    called anywhere in rule splitting, so that value would have made this very
    metric a lie. B1 declined to write the documented value for exactly that
    reason and used ``unstructured`` instead, which was the right call: the
    name is reserved for a real LLM pass, whose value over the regex path is
    then measurable as the difference between the two.

    Measured over the first 130,024 trials: 21,308 of 164,602 rows, 12.9%,
    took the ``unstructured`` path."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("nct_id", "criterion_id", name="uq_trial_criteria_nct_criterion"),
        CheckConstraint("kind IN ('inclusion','exclusion')", name="ck_trial_criteria_kind"),
        Index("ix_trial_criteria_nct", "nct_id"),
    )


class Patient(Base):
    """A synthetic patient note, its numbered sentences and extracted profile.

    Synthetic only. No real patient records ever enter this table.
    """

    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default="trec", server_default="trec")
    topic_year: Mapped[int | None] = mapped_column(Integer)
    note_text: Mapped[str] = mapped_column(Text, nullable=False)
    sentences: Mapped[list[dict[str, Any]]] = mapped_column(Json, default=list)
    """``NumberedNote.sentences`` - the citation targets."""

    profile: Mapped[dict[str, Any] | None] = mapped_column(Json)
    """``PatientProfile`` as JSON."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MatchRun(Base):
    """One match request: status, timings, tokens, cost, config version."""

    __tablename__ = "match_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("patients.id"))
    status: Mapped[str] = mapped_column(
        String(16), default="queued", server_default="queued", nullable=False
    )
    current_step: Mapped[str | None] = mapped_column(String(32))
    config_version: Mapped[str] = mapped_column(String(32), default="v1", server_default="v1")
    model: Mapped[str | None] = mapped_column(String(64))

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','completed','failed')", name="ck_match_runs_status"
        ),
        Index("ix_match_runs_status", "status"),
    )


class TrialMatch(Base):
    """One patient-trial pair: retrieval rank, final score, verdict, summary."""

    __tablename__ = "trial_matches"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("match_runs.id", ondelete="CASCADE"), nullable=False
    )
    nct_id: Mapped[str] = mapped_column(String(32), nullable=False)

    retrieval_rank: Mapped[int | None] = mapped_column(Integer)
    rrf_score: Mapped[float | None] = mapped_column(Float)
    rerank_score: Mapped[float | None] = mapped_column(Float)
    final_rank: Mapped[int | None] = mapped_column(Integer)
    final_score: Mapped[float | None] = mapped_column(Float)

    category: Mapped[str | None] = mapped_column(String(24))
    confirmed_fraction: Mapped[float | None] = mapped_column(Float)
    missing_facts: Mapped[list[str]] = mapped_column(Json, default=list)
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("match_run_id", "nct_id", name="uq_trial_matches_run_nct"),
        CheckConstraint(
            "category IS NULL OR category IN "
            "('eligible','likely_eligible','insufficient_info','excluded')",
            name="ck_trial_matches_category",
        ),
        Index("ix_trial_matches_run", "match_run_id"),
    )


class CriterionJudgmentRow(Base):
    """One judged rule: label, cited sentence IDs, explanation, verifier verdict.

    Named with a ``Row`` suffix so it never shadows the Pydantic
    ``CriterionJudgment`` contract, which is the LLM output schema.
    """

    __tablename__ = "criterion_judgments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("match_runs.id", ondelete="CASCADE"), nullable=False
    )
    nct_id: Mapped[str] = mapped_column(String(32), nullable=False)
    criterion_id: Mapped[str] = mapped_column(String(64), nullable=False)

    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str] = mapped_column(String(24), nullable=False)
    evidence_sentence_ids: Mapped[list[int]] = mapped_column(Json, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="", server_default="")

    citations_valid: Mapped[bool | None] = mapped_column(Boolean)
    """Set by the code check: do all cited sentence IDs exist? Must be 100%."""

    verifier_verdict: Mapped[str | None] = mapped_column(String(24))
    """Set by the cheap LLM pass: does the cited sentence support the label?
    The share that pass is the faithfulness number."""

    verifier_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "match_run_id", "nct_id", "criterion_id", name="uq_criterion_judgments_identity"
        ),
        CheckConstraint("kind IN ('inclusion','exclusion')", name="ck_criterion_judgments_kind"),
        CheckConstraint(
            "label IN ('met','not_met','insufficient_info','not_applicable')",
            name="ck_criterion_judgments_label",
        ),
        Index("ix_criterion_judgments_run_nct", "match_run_id", "nct_id"),
    )


class LlmCall(Base):
    """Audit log for every Claude call. Every cost and latency number in the
    README is a query against this table, so it is written unconditionally.

    ``billed_tier`` exists because Haiku 5.5 prices 5x higher above 100K prompt
    tokens: recording the tier per call is how that cliff stays visible.
    """

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("match_runs.id", ondelete="SET NULL")
    )
    step: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    effort: Mapped[str | None] = mapped_column(String(16))

    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    billed_tier: Mapped[str | None] = mapped_column(String(24))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    stop_reason: Mapped[str | None] = mapped_column(String(32))
    request_id: Mapped[str | None] = mapped_column(String(128))
    batch_id: Mapped[str | None] = mapped_column(String(128))
    trace_id: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_llm_calls_run", "match_run_id"),
        Index("ix_llm_calls_prompt_hash", "prompt_hash"),
        Index("ix_llm_calls_created", "created_at"),
    )


class Feedback(Base):
    """Reviewer corrections, which become future evaluation data.

    This table is the data flywheel: the phase-2 feature that turns usage into
    an asset rather than a log.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("match_runs.id", ondelete="SET NULL")
    )
    nct_id: Mapped[str | None] = mapped_column(String(32))
    criterion_id: Mapped[str | None] = mapped_column(String(64))

    corrected_label: Mapped[str | None] = mapped_column(String(24))
    corrected_category: Mapped[str | None] = mapped_column(String(24))
    comment: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(64))
    promoted_to_eval: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_feedback_run", "match_run_id"),)


class EvalRun(Base):
    """Benchmark results per configuration, so runs compare over time.

    ``split`` is load-bearing: no number measured on 2021 may be reported as a
    headline result. Tuning happens on 2021, reporting on held-out 2022.
    """

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    config_version: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(Json, default=dict)
    ablation: Mapped[str | None] = mapped_column(String(64))

    model: Mapped[str | None] = mapped_column(String(64))
    n_topics: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(Json, default=dict)
    """recall@100, nDCG@10, precision@10, faithfulness, invalid-citation rate,
    plus bootstrap confidence intervals."""

    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    git_sha: Mapped[str | None] = mapped_column(String(40))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("split IN ('2021','2022','dev','subset')", name="ck_eval_runs_split"),
        Index("ix_eval_runs_split", "split"),
    )


class Outbox(Base):
    """Events waiting to be published to Kafka - the transactional outbox.

    "Save to Postgres, then publish to Kafka" is two operations that can
    half-fail. Writing the row and the pending event in **one** transaction,
    then publishing from this table, makes them impossible to desynchronise.
    """

    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    topic: Mapped[str] = mapped_column(String(64), nullable=False)
    partition_key: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)

    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_outbox_unpublished", "published_at", "id"),)
