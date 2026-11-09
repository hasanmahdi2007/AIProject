"""The graph state, and the markers that make a partial answer legible.

:class:`GraphState` is a ``MatchState`` - a subclass, not a copy - because the
contract is frozen in ``beacon_core`` and must stay the single description of a
match run. The subclass adds exactly one thing: reducers.

Why reducers are not optional here
----------------------------------
``check_rules`` fans out with ``Send``, so twenty tasks finish in one superstep
and each returns a judgment for its own trial. Two tasks writing the same state
key in one superstep is an error in LangGraph unless the key says how to
combine them (``InvalidUpdateError: At key 'judgments': Can receive only one
value per step``). ``operator.add`` on a list is that answer, and it is also
what makes a crash mid-fan-out resumable: LangGraph replays the writes of the
tasks that finished and re-runs only the ones that did not, and each write is
added exactly once.

``total_cost_usd`` carries the same reducer, which means **every node returns
its own delta, never a running total.** A node that returned the total would
have it added to the total already in state.

Why "partial" and "could not evaluate" are string markers
---------------------------------------------------------
``MatchState`` has ``extra="forbid"`` and no ``partial`` flag, and
``EligibilityCategory`` has no "could not evaluate" member. Both omissions are
in the frozen contract, so the markers live in the fields that do exist:

* a trial that could not be judged gets ``category="insufficient_info"`` and a
  ``summary`` starting with :data:`COULD_NOT_EVALUATE`;
* a run cut short by the budget gets ``status="completed"`` and an ``error``
  starting with :data:`PARTIAL_PREFIX`.

Deriving "is this run partial?" from the summaries rather than from an
in-memory flag is deliberate: the summaries are checkpointed, so the answer
survives the restart that the flag would not.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import Field

from beacon_core.schemas import Contract, MatchState, NumberedNote, TrialJudgment

__all__ = [
    "BUDGET_EXHAUSTED",
    "COULD_NOT_EVALUATE",
    "PARTIAL_PREFIX",
    "GraphState",
    "TrialTask",
    "could_not_evaluate",
    "is_partial",
    "merge_judgments",
    "partial_reasons",
]

COULD_NOT_EVALUATE = "could not evaluate"
"""Prefix of ``TrialJudgment.summary`` for a trial that was not judged.

A trial that fails twice is marked and skipped; it is never silently dropped,
and it is never presented as excluded - "we could not tell" and "not eligible"
are different answers to a coordinator.
"""

BUDGET_EXHAUSTED = f"{COULD_NOT_EVALUATE}: per-match budget exhausted"

PARTIAL_PREFIX = "partial_results:"
"""Prefix of ``MatchState.error`` when the answer is incomplete but usable."""


class TrialTask(Contract):
    """The payload of one ``Send`` into ``check_rules``.

    One trial, its parsed rules, and the numbered note. The note is carried
    rather than read from state because a ``Send`` target is handed its payload
    and not the run's state - and rather than re-read from Postgres because the
    sentence IDs a judgment cites must be the IDs this run numbered, not the
    ones a later re-split would produce.

    ``Send`` payloads are passed to the node verbatim, so this model is
    validated inside the node rather than by the graph.
    """

    match_id: str
    nct_id: str
    title: str = ""
    note: NumberedNote
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    """``{"criterion_id", "kind", "text"}`` per rule, in document order."""


def merge_judgments(
    current: list[TrialJudgment], update: list[TrialJudgment]
) -> list[TrialJudgment]:
    """Reducer for ``GraphState.judgments``: merge by trial, last write wins.

    ``operator.add`` would be enough for the fan-out itself, but it makes the
    two steps *after* the fan-out impossible: ``verify`` and
    ``rank_and_explain`` both rewrite judgments, and appending a rewritten list
    to the original leaves forty entries for twenty trials. Keying by
    ``nct_id`` gives both behaviours at once - twenty parallel tasks each add
    their own trial, a later node replaces one - and it makes a re-run of a
    fan-out task after a crash update its trial in place instead of
    duplicating it.
    """
    merged = {judgment.nct_id: judgment for judgment in current}
    for judgment in update:
        merged[judgment.nct_id] = judgment
    return list(merged.values())


class GraphState(MatchState):
    """``MatchState`` plus the two reducers the fan-out needs.

    Subclassing keeps the frozen contract authoritative: anything that accepts
    a ``MatchState`` accepts this, and ``MatchState.model_validate`` of a
    finished run still works.
    """

    judgments: Annotated[list[TrialJudgment], merge_judgments] = Field(default_factory=list)
    total_cost_usd: Annotated[float, operator.add] = 0.0
    """Every node returns its own **delta**, never a running total: the reducer
    adds what it is given. A node that returned ``budget.spent_usd`` would have
    the whole run's spend added to the whole run's spend."""


def could_not_evaluate(nct_id: str, reason: str) -> TrialJudgment:
    """A judgment-shaped placeholder for a trial that could not be judged."""
    return TrialJudgment(
        nct_id=nct_id,
        judgments=[],
        category="insufficient_info",
        confirmed_fraction=0.0,
        missing_facts=[],
        summary=f"{COULD_NOT_EVALUATE}: {reason}",
    )


def partial_reasons(judgments: list[TrialJudgment]) -> list[str]:
    """Distinct reasons, in first-seen order, why this answer is incomplete."""
    seen: dict[str, None] = {}
    for judgment in judgments:
        summary = judgment.summary or ""
        if summary.startswith(COULD_NOT_EVALUATE):
            seen.setdefault(summary[len(COULD_NOT_EVALUATE) :].lstrip(": ").strip() or "unknown")
    return list(seen)


def is_partial(judgments: list[TrialJudgment]) -> bool:
    return bool(partial_reasons(judgments))
