"""Shared test material for the match worker. **Not a conftest, on purpose.**

The repository has exactly one ``conftest.py``, at ``tests/conftest.py``. A
second one anywhere - including ``services/match_worker/tests/`` - makes
``mypy .`` fail with *"Duplicate module named conftest"*, because both resolve
to the bare module name ``conftest``; marking the test directory as a package
trades that for a pytest *"Plugin already registered under a different name"*.
The real fixes (``explicit_package_bases``, ``consider_namespace_packages``)
live in the root ``pyproject.toml``, which belongs to the orchestrator. So
shared material lives here, in the package, under a unique module name -
``beacon_llm/_testing.py`` is the precedent.

Two things in here deserve explanation.

**There are no committed fixture JSON files.** Every test that calls Claude
runs the *real* ``AsyncLlmClient`` in replay mode against a fixture directory
this module writes into ``tmp_path`` first, from the same prompt builders the
nodes use. That means the tests make zero network calls (replay has no
code path to the API), they exercise the real guard, the real pricing and the
real structured-output validation, and - crucially - they keep working when
the owner rewrites the rule-check prompt, because the fixture key is recomputed
from the rewritten prompt instead of being frozen in a checked-in file.

**The responses are hand-authored, not recorded.** They are written to match
the frozen ``CriterionJudgment`` schema and carry plausible ``usage`` numbers,
including a cache read on every call after the first, so the cost assertions
have something real to price. They are deterministic by construction, which is
what the crash, idempotency, partial-budget and failing-trial tests need: a
recorded model response would be no more deterministic and would have cost
money to obtain.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from anthropic.types import Message, Usage
from pydantic import BaseModel

from beacon_core.schemas import (
    MatchStatus,
    NumberedNote,
    PatientProfile,
    PipelineStep,
    RankedQuery,
    SearchResponse,
    Sex,
    TrialCandidate,
)
from beacon_core.settings import Settings
from beacon_llm import AsyncLlmClient, CachedPrompt, FixtureMode, FixtureStore, build_cached_prompt
from match_worker.llm import SupportsCall
from match_worker.notes import to_numbered_note
from match_worker.outputs import (
    GeneratedQueries,
    TrialExplanation,
    TrialRuleCheck,
    TrialVerification,
)
from match_worker.prompts import (
    build_explain_prompt,
    build_profile_prompt,
    build_query_prompt,
    build_verify_prompt,
)
from match_worker.repository import RunInput, TrialRules
from match_worker.rule_check_prompt import render_trial_rules, rule_check_instructions
from match_worker.rules import ParsedCriterion, split_eligibility

__all__ = [
    "MATCH_ID",
    "PATIENT_ID",
    "SAMPLE_NOTE_TEXT",
    "TRIALS",
    "FlakyClient",
    "InMemoryStore",
    "TrialFixture",
    "candidates_for",
    "hand_written_message",
    "numbered_note",
    "replay_client",
    "search_response",
    "seed_pipeline_fixtures",
]

MATCH_ID = "match-c1-0001"
PATIENT_ID = "patient-c1-0001"

SAMPLE_NOTE_TEXT = """\
A 58-year-old woman presented to the oncology clinic for evaluation of a newly diagnosed left \
breast mass. Core needle biopsy showed invasive ductal carcinoma, grade 2. Immunohistochemistry \
was estrogen receptor positive at 90 percent, progesterone receptor positive at 60 percent, and \
HER2 negative by both IHC and FISH. Staging CT of the chest, abdomen and pelvis demonstrated \
three hepatic lesions consistent with metastatic disease. Bone scan was negative for osseous \
metastases. The patient has had no prior systemic therapy for breast cancer. Her past medical \
history includes well controlled type 2 diabetes mellitus and essential hypertension. ECOG \
performance status is 1. Laboratory studies showed hemoglobin 11.8 g/dL and platelet count 231 \
thousand per microliter. Creatinine was 0.9 mg/dL. Total bilirubin was 0.6 mg/dL, AST 28 U/L and \
ALT 31 U/L. Left ventricular ejection fraction by echocardiogram was 61 percent. The patient is \
postmenopausal, with her last menstrual period six years ago. She reports no history of \
thromboembolic disease and no active infection. She lives independently and works full time as a \
schoolteacher. The patient is interested in clinical trial participation.\
"""


@dataclass(frozen=True, slots=True)
class TrialFixture:
    """One trial as it would sit in Postgres after ingestion."""

    nct_id: str
    title: str
    eligibility_text: str
    conditions: tuple[str, ...] = ()
    min_age_years: float | None = None
    max_age_years: float | None = None
    sex: Sex | None = None

    @property
    def criteria(self) -> list[ParsedCriterion]:
        return split_eligibility(self.nct_id, self.eligibility_text)


TRIALS: tuple[TrialFixture, ...] = (
    TrialFixture(
        nct_id="NCT04000001",
        title="Phase 2 study of an oral CDK4/6 inhibitor in HR-positive metastatic breast cancer",
        eligibility_text=(
            "Inclusion Criteria:\n"
            "- Histologically confirmed hormone receptor positive, HER2 negative metastatic "
            "breast cancer\n"
            "- Age 18 years or older\n"
            "- ECOG performance status 0 or 1\n"
            "Exclusion Criteria:\n"
            "- Prior systemic cytotoxic chemotherapy for metastatic disease\n"
            "- Known active central nervous system metastases\n"
        ),
        conditions=("Breast Cancer",),
        min_age_years=18.0,
        sex="female",
    ),
    TrialFixture(
        nct_id="NCT04000002",
        title="Randomised study of an investigational antibody drug conjugate after chemotherapy",
        eligibility_text=(
            "INCLUSION CRITERIA\n"
            "1. Metastatic breast cancer with measurable disease\n"
            "2. At least one prior line of systemic chemotherapy for metastatic disease\n"
            "EXCLUSION CRITERIA\n"
            "1. Left ventricular ejection fraction below 50 percent\n"
        ),
        conditions=("Breast Cancer",),
        min_age_years=18.0,
    ),
    TrialFixture(
        nct_id="NCT04000003",
        title="Observational registry of postmenopausal women with metastatic breast cancer",
        eligibility_text=(
            "Eligibility:\n"
            "Postmenopausal women with a diagnosis of metastatic breast cancer. "
            "Adequate hepatic function defined as total bilirubin below 1.5 times the upper "
            "limit of normal. Willing to complete quality of life questionnaires at each visit.\n"
        ),
        conditions=("Breast Cancer",),
        sex="female",
    ),
)
"""Three trials, which is enough to show a fan-out, a per-trial failure and a
survivor in the same run while keeping a test readable."""


def numbered_note(text: str = SAMPLE_NOTE_TEXT) -> NumberedNote:
    return to_numbered_note(text)


SAMPLE_PROFILE = PatientProfile(
    age=58,
    sex="female",
    conditions=["invasive ductal carcinoma", "metastatic breast cancer", "type 2 diabetes"],
    biomarkers=["ER positive", "PR positive", "HER2 negative"],
    prior_treatments=[],
    labs={"hemoglobin": "11.8 g/dL", "creatinine": "0.9 mg/dL", "LVEF": "61 percent"},
    performance_status=1,
    location=None,
)

SAMPLE_QUERIES: tuple[RankedQuery, ...] = (
    RankedQuery(
        rank=1,
        text="hormone receptor positive HER2 negative metastatic breast cancer",
        facet="diagnosis",
    ),
    RankedQuery(
        rank=2, text="ER positive PR positive HER2 negative breast carcinoma", facet="biomarker"
    ),
    RankedQuery(
        rank=3,
        text="treatment naive first line metastatic breast cancer trial",
        facet="prior_therapy",
    ),
)


def candidates_for(trials: Sequence[TrialFixture] = TRIALS) -> list[TrialCandidate]:
    return [
        TrialCandidate(
            nct_id=trial.nct_id,
            title=trial.title,
            conditions=list(trial.conditions),
            min_age_years=trial.min_age_years,
            max_age_years=trial.max_age_years,
            sex=trial.sex,
            rrf_score=1.0 / (index + 1),
            rerank_score=1.0 - index / 10,
            retrieval_rank=index + 1,
        )
        for index, trial in enumerate(trials)
    ]


def search_response(trials: Sequence[TrialFixture] = TRIALS) -> SearchResponse:
    """What the retrieval service would return. Stubbed at the HTTP layer.

    ``services/retrieval`` is another agent's and does not exist yet; the
    contract does, so the tests speak the contract.
    """
    return SearchResponse(
        candidates=candidates_for(trials),
        total_before_rerank=len(trials) * 10,
        timings_ms={"dense": 12.5, "sparse": 8.0, "rerank": 41.0},
    )


# ----------------------------------------------------------- fixture writing --


def hand_written_message(
    payload: BaseModel | dict[str, Any],
    *,
    model: str = "claude-haiku-5-5",
    input_tokens: int = 120,
    output_tokens: int = 180,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    stop_reason: str = "end_turn",
    message_id: str = "msg_handwritten",
) -> Message:
    """A response shaped exactly like the API's, authored rather than recorded.

    ``usage`` carries real numbers because the cost meter reads them: a fixture
    with zero tokens would price every call at $0 and the budget tests would
    assert nothing.
    """
    text = payload.model_dump_json() if isinstance(payload, BaseModel) else json.dumps(payload)
    return Message.model_validate(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": Usage.model_validate(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read_tokens,
                    "cache_creation_input_tokens": cache_write_tokens,
                }
            ).model_dump(),
        }
    )


def _save(store: FixtureStore, prompt: CachedPrompt, message: Message) -> str:
    key = prompt.fingerprint()
    store.save(
        key, message=message, request={"note": "hand-authored"}, latency_ms=0.0, request_id=None
    )
    return key


def judgments_for(
    criteria: Sequence[dict[str, Any]],
    *,
    labels: dict[str, str] | None = None,
    cite: int = 3,
) -> TrialRuleCheck:
    """A plausible judgment per criterion, with a citable sentence ID.

    Default label is ``met`` for inclusion and ``not_met`` for exclusion, i.e.
    an eligible trial; pass ``labels`` keyed by ``criterion_id`` to steer one
    rule and exercise the exclusion or abstention paths.
    """
    chosen = labels or {}
    out = []
    for criterion in criteria:
        label = chosen.get(
            criterion["criterion_id"],
            "met" if criterion["kind"] == "inclusion" else "not_met",
        )
        out.append(
            {
                "criterion_id": criterion["criterion_id"],
                "kind": criterion["kind"],
                "label": label,
                "evidence_sentence_ids": [] if label == "not_applicable" else [cite],
                "explanation": (
                    f"Sentence {cite} of the note bears on this criterion; recorded as {label}."
                ),
            }
        )
    return TrialRuleCheck.model_validate({"judgments": out})


@dataclass
class SeededFixtures:
    """Where the fixtures were written, and what they covered."""

    directory: Path
    keys: dict[str, str] = field(default_factory=dict)

    def settings(self, **overrides: Any) -> Settings:
        return Settings(llm_fixtures_dir=self.directory, **overrides)


def seed_pipeline_fixtures(
    directory: Path,
    *,
    note_text: str = SAMPLE_NOTE_TEXT,
    trials: Sequence[TrialFixture] = TRIALS,
    profile: PatientProfile = SAMPLE_PROFILE,
    queries: Sequence[RankedQuery] = SAMPLE_QUERIES,
    query_count: int = 3,
    labels: dict[str, dict[str, str]] | None = None,
    skip_rule_check: Iterable[str] = (),
    verify: bool = True,
    explain_top_k: int = 5,
    output_tokens: int = 180,
) -> SeededFixtures:
    """Author one fixture per call the pipeline is about to make.

    Built from the production prompt builders, so a prompt edit moves the
    fixture key and the fixtures still match. ``skip_rule_check`` deliberately
    leaves a trial's fixture *absent*, which is how the "a trial fails and the
    others continue" test produces a failure without a network call.
    """
    store = FixtureStore(directory)
    seeded = SeededFixtures(directory=directory)
    note = to_numbered_note(note_text)
    skip = set(skip_rule_check)

    seeded.keys["profile_patient"] = _save(
        store,
        build_profile_prompt(note),
        hand_written_message(profile, input_tokens=900, output_tokens=output_tokens),
    )
    seeded.keys["generate_queries"] = _save(
        store,
        build_query_prompt(note, profile=profile, count=query_count),
        hand_written_message(
            GeneratedQueries(queries=list(queries)[:query_count]),
            input_tokens=950,
            output_tokens=output_tokens,
        ),
    )

    for index, trial in enumerate(trials):
        criteria = [
            {"criterion_id": c.criterion_id, "kind": c.kind, "text": c.text} for c in trial.criteria
        ]
        if not criteria:
            continue
        check = judgments_for(criteria, labels=(labels or {}).get(trial.nct_id))
        if trial.nct_id not in skip:
            seeded.keys[f"check_rules:{trial.nct_id}"] = _save(
                store,
                build_cached_prompt(
                    instructions=rule_check_instructions(),
                    note=note,
                    variable=render_trial_rules(
                        nct_id=trial.nct_id, title=trial.title, criteria=criteria
                    ),
                ),
                hand_written_message(
                    check,
                    # The first call writes the shared prefix; the rest read it.
                    # That is the behaviour the cache ordering exists to get, so
                    # the fixtures model it rather than pretending every call
                    # paid full price.
                    input_tokens=120 if index else 1800,
                    cache_read_tokens=1680 if index else 0,
                    cache_write_tokens=0 if index else 1680,
                    output_tokens=output_tokens,
                ),
            )
        if verify:
            described = [
                {
                    "criterion_id": item.criterion_id,
                    "kind": item.kind,
                    "criterion_text": next(
                        c["text"] for c in criteria if c["criterion_id"] == item.criterion_id
                    ),
                    "label": item.label,
                    "evidence_sentence_ids": list(item.evidence_sentence_ids),
                    "explanation": item.explanation,
                }
                for item in check.judgments
            ]
            seeded.keys[f"verify:{trial.nct_id}"] = _save(
                store,
                build_verify_prompt(note, nct_id=trial.nct_id, judgments=described),
                hand_written_message(
                    TrialVerification.model_validate(
                        {
                            "verdicts": [
                                {
                                    "criterion_id": item.criterion_id,
                                    "verdict": "supported",
                                    "note": "",
                                }
                                for item in check.judgments
                            ]
                        }
                    ),
                    input_tokens=140,
                    cache_read_tokens=1500,
                    output_tokens=output_tokens,
                ),
            )
    return seeded


def seed_explanation_fixtures(
    directory: Path,
    *,
    note_text: str,
    explanations: Sequence[dict[str, Any]],
    output_tokens: int = 180,
) -> None:
    """Author the ``rank_and_explain`` fixtures.

    Separate from :func:`seed_pipeline_fixtures` because the explain prompt
    contains the *derived* verdict, which is only known once the judgments have
    been run through ``match_worker.verdict`` - a test seeds these after it has
    the ranking, or lets the summaries be skipped.
    """
    store = FixtureStore(directory)
    note = to_numbered_note(note_text)
    for item in explanations:
        _save(
            store,
            build_explain_prompt(
                note,
                nct_id=item["nct_id"],
                title=item["title"],
                category=item["category"],
                confirmed_fraction=item["confirmed_fraction"],
                judgments=item["judgments"],
                missing_facts=item.get("missing_facts", []),
            ),
            hand_written_message(
                TrialExplanation(summary=item.get("summary", "Summary of this trial.")),
                input_tokens=150,
                cache_read_tokens=1500,
                output_tokens=output_tokens,
            ),
        )


def replay_client(directory: Path) -> AsyncLlmClient:
    """The real client, in replay mode, against a seeded directory.

    Replay has no code path to the network: a missing fixture raises. So a test
    built on this cannot spend money even if it is wrong.
    """
    return AsyncLlmClient(settings=Settings(llm_fixtures_dir=directory), mode=FixtureMode.REPLAY)


# ------------------------------------------------------------------- doubles --


@dataclass
class FlakyClient:
    """A real replay client that fails the calls a test wants failed.

    Used for the "one trial fails twice, the other nineteen finish" case. The
    failure is a transport error because that is what
    :func:`match_worker.backoff.is_retryable` retries - so the trial really
    does get retried and really does run out of attempts, rather than being
    rejected once on a non-retryable error.
    """

    inner: SupportsCall
    fail_when_variable_contains: dict[str, int] = field(default_factory=dict)
    attempts: list[str] = field(default_factory=list)

    async def call[OutputT: BaseModel](
        self,
        *,
        prompt: CachedPrompt,
        output_model: type[OutputT],
        model: str | None = None,
        effort: Any = None,
        max_tokens: int | None = None,
        verify_capabilities: bool = True,
    ) -> Any:
        for needle, remaining in list(self.fail_when_variable_contains.items()):
            if needle in prompt.variable and remaining > 0:
                self.fail_when_variable_contains[needle] = remaining - 1
                self.attempts.append(needle)
                # A transport error, not an API outcome: this is what
                # match_worker.backoff.is_retryable treats as worth retrying,
                # so the trial really does burn its attempts.
                raise httpx.ConnectError("stub transport failure (match_worker._testing)")
        return await self.inner.call(
            prompt=prompt,
            output_model=output_model,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            verify_capabilities=verify_capabilities,
        )


@dataclass
class InMemoryStore:
    """A ``MatchStore`` that enforces the same unique keys as Postgres.

    Keyed exactly like the real constraints -
    ``uq_trial_matches_run_nct``, ``uq_criterion_judgments_identity``,
    ``outbox.event_id``, ``uq_trial_criteria_nct_criterion`` - so a unit test
    can demonstrate convergence rather than duplication. It also counts
    *attempted* writes, which is the only way to tell "we wrote once" from "we
    wrote twice and the constraint saved us": the second is what a redelivery
    does, and it is the behaviour being claimed.

    It is not evidence about Postgres. The integration tests, including the
    crash test, run the real ``PostgresStore`` against the real constraints.
    """

    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    patients: dict[str, dict[str, Any]] = field(default_factory=dict)
    trials: dict[str, TrialFixture] = field(default_factory=dict)
    criteria: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    trial_matches: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    judgments: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    outbox: dict[str, dict[str, Any]] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    write_attempts: dict[str, int] = field(default_factory=dict)
    conflicts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def seeded(
        cls,
        *,
        match_id: str = MATCH_ID,
        patient_id: str | None = PATIENT_ID,
        note_text: str = SAMPLE_NOTE_TEXT,
        trials: Sequence[TrialFixture] = TRIALS,
        pre_parse: bool = False,
    ) -> InMemoryStore:
        store = cls()
        store.runs[match_id] = {
            "patient_id": patient_id,
            "status": "queued",
            "config_version": "v1",
            "current_step": None,
            "started_at": None,
        }
        if patient_id:
            store.patients[patient_id] = {"note_text": note_text}
        for trial in trials:
            store.trials[trial.nct_id] = trial
            if pre_parse:
                for criterion in trial.criteria:
                    store.criteria[(trial.nct_id, criterion.criterion_id)] = {
                        "criterion_id": criterion.criterion_id,
                        "kind": criterion.kind,
                        "text": criterion.text,
                        "position": criterion.position,
                    }
        return store

    def _count(self, name: str) -> None:
        self.write_attempts[name] = self.write_attempts.get(name, 0) + 1

    # ------------------------------------------------------------- reading --

    async def load_run(self, match_id: str) -> RunInput | None:
        run = self.runs.get(match_id)
        if run is None:
            return None
        patient = self.patients.get(run["patient_id"] or "", {})
        return RunInput(
            match_id=match_id,
            patient_id=run["patient_id"],
            patient_note=patient.get("note_text", ""),
            config_version=run["config_version"],
            status=run["status"],
        )

    async def load_trial_rules(self, nct_ids: Sequence[str]) -> dict[str, TrialRules]:
        wanted = set(nct_ids)
        grouped: dict[str, TrialRules] = {}
        for (nct_id, _), row in sorted(self.criteria.items(), key=lambda kv: kv[1]["position"]):
            if nct_id not in wanted:
                continue
            title = self.trials[nct_id].title if nct_id in self.trials else ""
            rules = grouped.setdefault(nct_id, TrialRules(nct_id, title, []))
            rules.criteria.append(
                {"criterion_id": row["criterion_id"], "kind": row["kind"], "text": row["text"]}
            )
        return grouped

    async def load_eligibility_text(self, nct_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
        return {
            nct_id: (self.trials[nct_id].title, self.trials[nct_id].eligibility_text)
            for nct_id in nct_ids
            if nct_id in self.trials
        }

    async def count_rows(self, match_id: str) -> dict[str, int]:
        return {
            "trial_matches": sum(1 for key in self.trial_matches if key[0] == match_id),
            "criterion_judgments": sum(1 for key in self.judgments if key[0] == match_id),
            "llm_calls": sum(1 for row in self.llm_calls if row.get("match_run_id") == match_id),
            "outbox": sum(1 for row in self.outbox.values() if row["partition_key"] == match_id),
        }

    # ------------------------------------------------------------- writing --

    async def mark_running(self, match_id: str, *, step: PipelineStep | None = None) -> None:
        self._count("mark_running")
        run = self.runs.get(match_id)
        if run is None:
            return
        run["status"] = "running"
        run["started_at"] = run["started_at"] or "now"
        if step:
            run["current_step"] = step

    async def set_current_step(self, match_id: str, step: PipelineStep) -> None:
        if match_id in self.runs:
            self.runs[match_id]["current_step"] = step

    async def finish_run(
        self,
        match_id: str,
        *,
        status: MatchStatus,
        error: str | None,
        total_cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        model: str | None,
        duration_ms: float | None,
    ) -> None:
        self._count("finish_run")
        run = self.runs.setdefault(match_id, {})
        run.update(
            status=status,
            error=error,
            total_cost_usd=total_cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model=model,
            duration_ms=duration_ms,
        )

    async def enqueue_event(
        self, *, event_id: str, topic: str, partition_key: str | None, payload: dict[str, Any]
    ) -> bool:
        self._count("enqueue_event")
        if event_id in self.outbox:
            self.conflicts["outbox"] = self.conflicts.get("outbox", 0) + 1
            return False
        self.outbox[event_id] = {
            "event_id": event_id,
            "topic": topic,
            "partition_key": partition_key,
            "payload": payload,
            "published_at": None,
            "id": len(self.outbox) + 1,
            "attempts": 0,
        }
        return True

    async def save_patient_profile(
        self, patient_id: str | None, *, sentences: list[dict[str, Any]], profile: dict[str, Any]
    ) -> None:
        if patient_id and patient_id in self.patients:
            self.patients[patient_id]["sentences"] = sentences
            self.patients[patient_id]["profile"] = profile

    async def save_parsed_criteria(self, nct_id: str, criteria: Sequence[ParsedCriterion]) -> None:
        self._count("save_parsed_criteria")
        for criterion in criteria:
            key = (nct_id, criterion.criterion_id)
            if key in self.criteria:
                self.conflicts["trial_criteria"] = self.conflicts.get("trial_criteria", 0) + 1
                continue
            self.criteria[key] = {
                "criterion_id": criterion.criterion_id,
                "kind": criterion.kind,
                "text": criterion.text,
                "position": criterion.position,
            }

    async def upsert_trial_match(self, match_id: str, row: dict[str, Any]) -> None:
        self._count("upsert_trial_match")
        key = (match_id, row["nct_id"])
        if key in self.trial_matches:
            self.conflicts["trial_matches"] = self.conflicts.get("trial_matches", 0) + 1
        self.trial_matches[key] = {**self.trial_matches.get(key, {}), **row}

    async def save_judgments(
        self, match_id: str, nct_id: str, rows: Sequence[dict[str, Any]]
    ) -> None:
        self._count("save_judgments")
        for row in rows:
            key = (match_id, nct_id, row["criterion_id"])
            if key in self.judgments:
                self.conflicts["criterion_judgments"] = (
                    self.conflicts.get("criterion_judgments", 0) + 1
                )
            self.judgments[key] = {**self.judgments.get(key, {}), **row}

    async def record_llm_call(self, row: dict[str, Any]) -> None:
        """Append-only, deliberately - see ``PostgresStore.record_llm_call``.

        Every other write in this double deduplicates, because its real
        counterpart has a unique constraint. This one does not: the audit log
        must show one row per call actually billed, so a second row is either
        real spend or a checkpointing defect, and both need to be visible.
        """
        self._count("record_llm_call")
        self.llm_calls.append(row)

    # --------------------------------------------------------- relay source --

    async def unpublished_events(self, limit: int) -> list[dict[str, Any]]:
        return [row for row in self.outbox.values() if row["published_at"] is None][:limit]

    async def mark_published(self, ids: Any) -> None:
        wanted = set(ids)
        for row in self.outbox.values():
            if row["id"] in wanted:
                row["published_at"] = "now"

    async def mark_publish_failed(self, row_id: int, error: str) -> None:
        for row in self.outbox.values():
            if row["id"] == row_id:
                row["attempts"] += 1
                row["last_error"] = error
