"""Turning judgments into a verdict - in code, never by asking a model.

The rule is the one in ``TrialJudgment``'s own docstring: a trial is
``excluded`` if any inclusion rule is ``not_met`` or any exclusion rule is
``met``; otherwise it is ranked by the share of rules confirmed. Keeping that
in code rather than in a prompt buys three things that matter more than the
flexibility it costs:

* the verdict is **auditable** - it is a function of the rows in
  ``criterion_judgments``, so anyone can recompute it from the database;
* it is **stable** - the same judgments produce the same verdict on every
  re-run, which is what makes the evaluation harness measure prompt changes
  rather than sampling noise;
* a model cannot promote a trial it likes. ``rank_and_explain`` asks Claude to
  *explain* this verdict, with the verdict already in the prompt.

Citations gate the verdict
--------------------------
A ``met`` or ``not_met`` whose cited sentence IDs are not in the note is
demoted to ``insufficient_info`` for the purposes of the verdict. It is not
deleted - the row keeps its label and gets ``citations_valid = false``, because
the invalid-citation rate is a reportable number - but an assertion whose
evidence cannot be located must not be allowed to exclude a trial.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from beacon_core.schemas import (
    CriterionJudgment,
    EligibilityCategory,
    NumberedNote,
    TrialJudgment,
)

__all__ = [
    "CATEGORY_WEIGHT",
    "citations_valid",
    "derive_trial_judgment",
    "final_score",
    "missing_facts_from",
]

CATEGORY_WEIGHT: dict[EligibilityCategory, float] = {
    "eligible": 3.0,
    "likely_eligible": 2.0,
    "insufficient_info": 1.0,
    "excluded": 0.0,
}
"""Ranking order. Excluded trials are kept and ranked last rather than dropped:
"we checked and it does not fit, here is why" is an answer, and hiding it
invites the same trial being asked about again."""

_MAX_MISSING_FACTS = 10


def citations_valid(judgment: CriterionJudgment, note: NumberedNote) -> bool:
    """Do all cited sentence IDs exist in this run's numbered note?

    Recomputed wherever it is needed rather than carried in state: it is a pure
    function of the judgment and the note, both of which are checkpointed, so
    carrying it would only create something that could disagree with them.
    """
    return all(sentence_id in note.valid_ids for sentence_id in judgment.evidence_sentence_ids)


def _effective_label(judgment: CriterionJudgment, note: NumberedNote) -> str:
    if judgment.label in {"met", "not_met"} and not citations_valid(judgment, note):
        return "insufficient_info"
    return judgment.label


def missing_facts_from(judgments: Iterable[CriterionJudgment]) -> list[str]:
    """The questions a coordinator should ask, from the abstentions.

    The explanation of an ``insufficient_info`` judgment is where the model
    said which fact the criterion turned on and the note did not state. That
    sentence is the most actionable thing the pipeline produces, so it is
    surfaced as a list rather than left inside twenty rule rows.
    """
    facts: dict[str, None] = {}
    for judgment in judgments:
        if judgment.label != "insufficient_info":
            continue
        text = judgment.explanation.strip()
        if text:
            facts.setdefault(text)
    return list(facts)[:_MAX_MISSING_FACTS]


def derive_trial_judgment(
    *,
    nct_id: str,
    judgments: Sequence[CriterionJudgment],
    note: NumberedNote,
    summary: str | None = None,
) -> TrialJudgment:
    """The verdict for one trial, derived from its rule judgments."""
    if not judgments:
        return TrialJudgment(
            nct_id=nct_id,
            judgments=[],
            category="insufficient_info",
            confirmed_fraction=0.0,
            missing_facts=[],
            summary=summary,
        )

    blocked = False
    confirmed = 0
    abstained = 0
    for judgment in judgments:
        label = _effective_label(judgment, note)
        inclusion = judgment.kind == "inclusion"
        # Blocking: the two labels that mean "this patient does not qualify".
        if label == ("not_met" if inclusion else "met"):
            blocked = True
        # Confirmed: the rule was checked and came out in the patient's favour.
        # A not_applicable rule counts as confirmed - it cannot fail - which
        # keeps the fraction comparable across trials with different rule mixes.
        if label == "not_applicable" or label == ("met" if inclusion else "not_met"):
            confirmed += 1
        elif label == "insufficient_info":
            abstained += 1

    total = len(judgments)
    fraction = confirmed / total

    category: EligibilityCategory
    if blocked:
        category = "excluded"
    elif abstained == 0:
        category = "eligible"
    elif fraction >= 0.5:
        category = "likely_eligible"
    else:
        category = "insufficient_info"

    return TrialJudgment(
        nct_id=nct_id,
        judgments=list(judgments),
        category=category,
        confirmed_fraction=round(fraction, 4),
        missing_facts=missing_facts_from(judgments),
        summary=summary,
    )


def final_score(judgment: TrialJudgment, *, retrieval_rank: int | None) -> float:
    """Ranking score: verdict first, then evidence, then retrieval order.

    Retrieval contributes a deliberately tiny term. It decides ties and nothing
    else: a similarity score is evidence that a trial is about the right
    disease, never evidence that the patient qualifies for it.
    """
    rank_term = 0.0 if retrieval_rank is None else 1.0 / (60.0 + retrieval_rank)
    return round(CATEGORY_WEIGHT[judgment.category] + judgment.confirmed_fraction + rank_term, 6)
