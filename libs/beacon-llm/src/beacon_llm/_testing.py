"""Shared test material: one patient, two trials, one set of instructions.

It lives in the package rather than in ``libs/beacon-llm/tests/conftest.py``
for a specific reason worth recording, because the next agent to add a
per-package test suite will hit the same wall:

``libs/beacon-llm/tests/conftest.py`` and the repository's ``tests/conftest.py``
both resolve to a module named ``conftest``, and ``mypy .`` refuses to check
either - "Duplicate module named conftest". Marking the test directory as a
package fixes mypy and immediately breaks pytest, which then derives the package
name ``tests`` for both and refuses to register the second plugin. Both proper
fixes - ``explicit_package_bases`` for mypy, ``consider_namespace_packages`` for
pytest - live in the root ``pyproject.toml``, which belongs to the orchestrator
and not to an agent. So the material lives here under a unique module name and
the test modules import it directly, as plain constants rather than fixtures.

The instructions and note below are long on purpose: the cacheable prefix has
to clear 512 tokens or prompt caching silently does nothing, and a smoke test
built on a 50-token prefix would "pass" while proving the opposite of what it
claims. The recorded live call put the prefix at 2032 tokens.

This is test scaffolding, not the production prompt - the rule-check prompt
belongs to the match worker (agent C1), not to this library. It is shaped like
the real thing only so that the cache and cost numbers it produces are
realistic.
"""

from __future__ import annotations

from pathlib import Path

from beacon_core.schemas import NoteSentence, NumberedNote

__all__ = [
    "COMMITTED_FIXTURES",
    "RULE_CHECK_INSTRUCTIONS",
    "SAMPLE_NOTE",
    "SAMPLE_TRIAL_RULES",
    "SECOND_TRIAL_RULES",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
COMMITTED_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "llm"
"""The fixture directory checked into the repository.

Four levels up: ``libs/beacon-llm/src/beacon_llm/_testing.py``. Same walk as
``beacon_core.settings.REPO_ROOT``, which is how the two agree on where
``llm_fixtures_dir`` points by default.
"""

SECOND_TRIAL_RULES = (
    "Trial NCT04000002, criterion EXC-1, presented as an exclusion criterion:\n"
    "Any prior systemic cytotoxic chemotherapy for metastatic disease.\n"
    "Judge criterion EXC-1 against the numbered note above."
)
"""A second trial for the same patient - the other half of the cache pair."""

RULE_CHECK_INSTRUCTIONS = """\
You are a clinical trial eligibility assistant. You are given a patient note \
split into numbered sentences, followed by one eligibility criterion from one \
clinical trial. Judge that single criterion against that note and return one \
JSON object describing your judgment.

Rules of judgment, in order of precedence:

1. Judge only the criterion you are given. Do not comment on any other \
criterion, do not summarise the trial, and do not state an overall eligibility \
verdict. A separate step aggregates your judgments.
2. Use only facts stated in the numbered note. Do not infer a diagnosis from a \
medication, do not infer a stage from a treatment line, and do not assume a \
negative finding because a positive one is absent. The note is the only \
evidence you have.
3. If the note does not contain the fact the criterion turns on, label the \
criterion insufficient_info. This is a correct and expected answer, not a \
failure. It is how a missing fact becomes a question for a human rather than a \
guess with a confident explanation attached. Abstaining is always preferable \
to inferring.
4. If the criterion does not apply to this patient at all - a pregnancy \
criterion for a male patient, a prior-therapy criterion for a treatment-naive \
patient - label it not_applicable rather than met or not_met.
5. Label a criterion met only when the note states the fact positively. Label \
it not_met only when the note states a fact that positively contradicts it. \
Silence is never contradiction.

Evidence requirements:

Every judgment other than not_applicable must cite at least one sentence ID \
from the numbered note, in evidence_sentence_ids. Cite the sentence that \
carries the fact you relied on, not the sentence that merely mentions the same \
topic. Cite several IDs when the judgment rests on several facts. The IDs you \
cite are checked against the note by code before your answer is used, so an ID \
that is not in the note invalidates the whole judgment.

The explanation field is one or two sentences of plain clinical English, \
addressed to a research coordinator who will read it next to the criterion \
text. State the fact you relied on and the conclusion you drew from it. Do not \
restate the criterion, do not hedge with phrases like "it appears that", and \
do not mention these instructions, the note's numbering, or your own \
reasoning process.

The criterion_id field must repeat the criterion identifier exactly as it was \
given to you, and kind must repeat whether the criterion was presented as an \
inclusion or an exclusion rule. Do not re-derive either field from the \
criterion text.
"""

_NOTE_SENTENCES = [
    "A 58-year-old woman presented to the oncology clinic for evaluation of a newly "
    "diagnosed left breast mass.",
    "Core needle biopsy showed invasive ductal carcinoma, grade 2.",
    "Immunohistochemistry was reported as estrogen receptor positive at 90 percent, "
    "progesterone receptor positive at 60 percent, and HER2 negative by both IHC and FISH.",
    "Staging CT of the chest, abdomen and pelvis demonstrated three hepatic lesions "
    "consistent with metastatic disease.",
    "Bone scan was negative for osseous metastases.",
    "The patient has no prior systemic therapy for breast cancer.",
    "She underwent an appendectomy at age 24 and a cesarean section at age 31.",
    "Her past medical history includes well controlled type 2 diabetes mellitus "
    "and essential hypertension.",
    "Current medications are metformin 1000 mg twice daily, lisinopril 10 mg daily, "
    "and atorvastatin 20 mg daily.",
    "She has no known drug allergies.",
    "ECOG performance status is 1.",
    "Laboratory studies showed hemoglobin 11.8 g/dL, white blood cell count 6.2 "
    "thousand per microliter, and platelet count 231 thousand per microliter.",
    "Creatinine was 0.9 mg/dL with an estimated glomerular filtration rate above "
    "60 mL/min per 1.73 square meters.",
    "Total bilirubin was 0.6 mg/dL, AST 28 U/L and ALT 31 U/L.",
    "Left ventricular ejection fraction by echocardiogram was 61 percent.",
    "The patient is postmenopausal, with her last menstrual period six years ago.",
    "She reports no history of thromboembolic disease and no active infection.",
    "She has never smoked and drinks alcohol socially.",
    "She lives independently and works full time as a schoolteacher.",
    "The patient is interested in clinical trial participation and asked to be "
    "considered for studies available at this center.",
]


SAMPLE_NOTE = NumberedNote(
    sentences=[NoteSentence(sentence_id=i, text=text) for i, text in enumerate(_NOTE_SENTENCES, 1)]
)
"""The note above, numbered from 1.

A module constant rather than a pytest fixture, deliberately: the fixtures that
prove the cache works have to produce a *byte-identical* prefix across calls,
and a per-test factory is one refactor away from quietly producing two different
notes in one test.
"""

SAMPLE_TRIAL_RULES = (
    "Trial NCT04000001, criterion INC-3, presented as an inclusion criterion:\n"
    "Histologically confirmed hormone receptor positive, HER2 negative "
    "metastatic breast cancer.\n"
    "Judge criterion INC-3 against the numbered note above."
)
"""One trial for the patient above - half of the cache pair."""
