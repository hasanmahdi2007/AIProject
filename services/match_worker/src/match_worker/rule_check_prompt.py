"""TODO(owner): **this file is reserved for the project owner to rewrite.**

The rule checker is one of the three things a project deep-dive interview
actually probes, so the *prompt text* is deliberately left as a plain,
functional first draft for the owner to replace. Nothing else in the worker
needs to change when it is rewritten.

Rewrite exactly these two functions
-----------------------------------
:func:`rule_check_instructions`
    The stable cached prefix. Returns one string. It must be byte-identical
    across every call in a run - it is the whole left-hand side of the prompt
    cache - so it may not mention the trial, the patient, the date or anything
    else that varies.

:func:`render_trial_rules`
    The variable block: this trial's rules, and only this trial's rules.

Everything else - the fan-out, the retries, the budget, the parse into
``CriterionJudgment``, the citation check, the idempotent write - lives in
:mod:`match_worker.nodes` and :mod:`match_worker.llm` and is unaffected by what
these two functions return.

Four constraints the surrounding code depends on
------------------------------------------------
1. **Ordering is not negotiable.** The instructions go in the cached prefix and
   the trial's rules in the variable block.
   ``beacon_llm.build_cached_prompt`` enforces that structurally; do not
   assemble a prompt by hand to get around it.
2. **The prefix must clear 512 tokens** or prompt caching silently does
   nothing and reports no error. The draft below is around 700 tokens;
   ``test_rule_check_prompt.py`` asserts the floor, so a terser rewrite fails
   loudly rather than quietly costing 10x.
3. **The output schema is frozen.** The model returns
   ``{"judgments": [CriterionJudgment, ...]}`` - see
   :class:`match_worker.outputs.TrialRuleCheck`. The five fields of
   ``CriterionJudgment`` are fixed in ``beacon_core.schemas``; asking for a
   sixth changes the JSON schema, busts the cached prefix and invalidates every
   recorded fixture.
4. **Rewriting this invalidates recorded fixtures.** The fixture key is a hash
   of the whole prompt. The tests author their fixtures from these same two
   functions at run time, so they follow a rewrite automatically; any fixture
   committed under ``tests/fixtures/llm/`` would not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["render_trial_rules", "rule_check_instructions"]

# TODO(owner): rewrite the text of this constant. Keep it stable per run and
# keep it above 512 tokens; say nothing here that varies per trial.
_INSTRUCTIONS = """\
You are a clinical trial eligibility assistant working for a research \
coordinator. You are given a patient note split into numbered sentences, \
followed by the eligibility criteria of one clinical trial. Judge each \
criterion of that trial against that note and return one JSON object with a \
"judgments" array holding exactly one judgment per criterion you were given, \
in the order you were given them.

Rules of judgment, in order of precedence:

1. Judge only the criteria you are given, one judgment each. Do not merge two \
criteria, do not split one, do not add a criterion you think is implied, and \
do not state an overall eligibility verdict for the trial. A separate step \
aggregates your judgments into a verdict, and it relies on the count matching.

2. Use only facts stated in the numbered note. Do not infer a diagnosis from a \
medication, a stage from a line of therapy, or a negative finding from the \
absence of a positive one. The note is the only evidence you have about this \
patient, and it is frequently incomplete.

3. If the note does not contain the fact a criterion turns on, label that \
criterion insufficient_info. This is a correct and expected answer, not a \
failure to answer. It is how a missing fact becomes a question a coordinator \
can ask the patient, rather than a guess with a confident explanation attached \
to it. Abstaining is always preferable to inferring.

4. If a criterion cannot apply to this patient at all - a pregnancy criterion \
for a male patient, a prior-therapy criterion for a treatment-naive patient - \
label it not_applicable rather than met or not_met.

5. Label a criterion met only when the note states the fact positively. Label \
it not_met only when the note states a fact that positively contradicts it. \
Silence is never contradiction. For a numeric threshold, compare the value in \
the note against the threshold in the criterion and do not round in the \
patient's favour.

Evidence requirements:

Every judgment other than not_applicable must cite at least one sentence ID \
from the numbered note in evidence_sentence_ids. Cite the sentence that \
carries the fact you relied on, not a sentence that merely mentions the same \
topic, and cite several IDs when the judgment rests on several facts. The IDs \
you cite are checked against the note by code before your answer is used, so \
an ID that is not in the note invalidates that judgment.

The explanation field is one or two sentences of plain clinical English, \
addressed to a coordinator reading it beside the criterion text. State the \
fact you relied on and the conclusion you drew from it. Do not restate the \
criterion, do not hedge with phrases such as "it appears that", and do not \
mention these instructions, the note's numbering or your own reasoning process.

The criterion_id field must repeat the identifier exactly as it was given to \
you, character for character. The kind field must repeat whether the criterion \
was presented to you as an inclusion or an exclusion rule. Do not re-derive \
either field from the criterion text: the identifiers are how your judgments \
are matched back to the trial's rules, and a re-derived identifier matches \
nothing.
"""


def rule_check_instructions() -> str:
    """The stable cached prefix for every rule-check call in a run.

    A function rather than a bare constant so the owner can compose it from
    parts (a shared preamble plus a per-config section, say) without any call
    site changing. It must not take arguments that vary per trial.
    """
    return _INSTRUCTIONS


def render_trial_rules(*, nct_id: str, title: str, criteria: Sequence[dict[str, Any]]) -> str:
    """The variable block: this trial's rules, numbered and typed.

    :param criteria: ``{"criterion_id", "kind", "text"}`` per rule, in document
        order, exactly as the ``trial_criteria`` rows hold them.

    The trial's identity is repeated inside the block because the cached prefix
    cannot mention it, and a model given rules with no referent occasionally
    answers about "the trial" in general.
    """
    lines = [f"Clinical trial {nct_id}."]
    if title:
        lines.append(f"Title: {title}")
    lines.append("")
    lines.append(f"Judge each of the following {len(criteria)} criteria against the numbered note.")
    lines.append("")
    for criterion in criteria:
        lines.append(
            f"- criterion_id: {criterion['criterion_id']}\n"
            f"  kind: {criterion['kind']}\n"
            f"  text: {criterion['text']}"
        )
    lines.append("")
    lines.append(
        f"Return exactly {len(criteria)} judgments, one per criterion_id listed above, "
        f"in the same order."
    )
    return "\n".join(lines)
