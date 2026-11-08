"""Beacon shared contracts.

Everything in this package is **frozen in Wave 0 and read-only to every later
agent**. An agent that needs a shape changed asks the orchestrator; it does not
edit this package. That rule is what lets several agents build in parallel
without disagreeing about data.

Import the submodules directly (``from beacon_core.schemas import
PatientProfile``). Only the schema contracts are re-exported here, so importing
``beacon_core`` does not drag in database or broker drivers.
"""

from __future__ import annotations

from beacon_core.schemas import (
    AgeSexFilter,
    CriterionJudgment,
    MatchState,
    NoteSentence,
    NumberedNote,
    PatientProfile,
    RankedQuery,
    RankedTrial,
    SearchRequest,
    SearchResponse,
    TrialCandidate,
    TrialJudgment,
)

__version__ = "0.1.0"

__all__ = [
    "AgeSexFilter",
    "CriterionJudgment",
    "MatchState",
    "NoteSentence",
    "NumberedNote",
    "PatientProfile",
    "RankedQuery",
    "RankedTrial",
    "SearchRequest",
    "SearchResponse",
    "TrialCandidate",
    "TrialJudgment",
    "__version__",
]
