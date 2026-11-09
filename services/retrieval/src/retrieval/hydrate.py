"""Where the trial *text* comes from, and why it does not come from Qdrant.

B1's Qdrant payload is six scalar fields - ``nct_id``, ``min_age_years``,
``max_age_years``, ``sex``, ``phase``, ``status`` - and no text at all. That is
the right call for the index (payload is read on every visited point during a
filtered walk, so it should hold exactly the facts the filter needs and nothing
else), but it leaves this service with two needs it cannot meet from a search
response:

1. ``TrialCandidate`` has ``title``, ``conditions`` and ``interventions``, and a
   candidate list where those are blank is unusable by the console and by the
   worker's rule checker.
2. **The cross-encoder needs the document text.** It scores a ``(query, trial)``
   pair by reading both; with no trial text there is nothing to rerank, and the
   most accurate stage of the funnel would silently become a no-op that just
   re-sorted the RRF order.

So the text is fetched from Postgres, which is the system of record for trials
and already holds ``overview_text`` - the exact string B1 embedded, assembled
most-discriminative-field-first. This is the standard split (vector store holds
vectors plus filterable facts; document store holds documents) and it is
affordable here for one specific reason: **hydration happens after fusion, on
the fused top ``fuse_top_k`` only.** It is one indexed ``nct_id IN (...)`` query
for about 100 primary-key lookups, not a join against 375,580 rows, and it is
off the index-traversal path entirely.

A trial present in Qdrant but missing from Postgres is a dual-write divergence -
the known cost of keeping two stores. It is logged and the candidate **survives**
with an empty title: dropping it here would be a silent deletion decided by an
infrastructure inconsistency, which is precisely the class of bug this project
is organised around avoiding. It simply has nothing for the cross-encoder to
read, so it ranks on whatever text it does have.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from beacon_core.models import Trial

__all__ = [
    "MappingTrialTextStore",
    "PostgresTrialTextStore",
    "TrialText",
    "TrialTextStore",
    "rerank_document",
]


@dataclass(frozen=True, slots=True)
class TrialText:
    """The text half of a trial: display fields plus what was embedded."""

    nct_id: str
    title: str = ""
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    overview_text: str | None = None


class TrialTextStore(Protocol):
    """Fetch the text for a set of NCT IDs.

    A protocol rather than a concrete class so the unit suite can inject a
    dictionary and exercise the whole search pipeline - fusion, hydration,
    rerank ordering - with no Postgres running.
    """

    async def fetch(self, nct_ids: Sequence[str]) -> Mapping[str, TrialText]:
        """Return the subset that exists. Missing IDs are simply absent."""
        ...


class PostgresTrialTextStore:
    """Reads ``trials`` by primary key, in one statement.

    Only the five columns needed are selected. ``eligibility_text`` in
    particular is deliberately *not* loaded: it is the largest column on the
    table, retrieval has no use for it, and the lazy ``parse_rules`` node reads
    it later for the ~20 trials that survive.
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def fetch(self, nct_ids: Sequence[str]) -> Mapping[str, TrialText]:
        if not nct_ids:
            return {}
        statement = select(
            Trial.nct_id,
            Trial.title,
            Trial.conditions,
            Trial.interventions,
            Trial.overview_text,
        ).where(Trial.nct_id.in_(list(nct_ids)))
        async with self._sessionmaker() as session:
            rows = (await session.execute(statement)).all()
        return {
            row.nct_id: TrialText(
                nct_id=row.nct_id,
                title=row.title or "",
                conditions=list(row.conditions or []),
                interventions=list(row.interventions or []),
                overview_text=row.overview_text,
            )
            for row in rows
        }


class MappingTrialTextStore:
    """An in-memory :class:`TrialTextStore`, for tests and fixtures."""

    def __init__(self, texts: Mapping[str, TrialText]) -> None:
        self._texts = dict(texts)

    async def fetch(self, nct_ids: Sequence[str]) -> Mapping[str, TrialText]:
        return {nct_id: self._texts[nct_id] for nct_id in nct_ids if nct_id in self._texts}


def rerank_document(text: TrialText | None) -> str:
    """The string the cross-encoder reads for one trial.

    ``overview_text`` when present, because it is byte-identical to what the
    dense model embedded: the reranker then re-judges the same text the first
    stage ranked, so an improvement is attributable to the model rather than to
    the reranker having been handed better input.

    When it is absent (an older row, or a divergence) the display fields are
    assembled in the same most-discriminative-first order B1 uses, so the
    fallback degrades gracefully rather than changing the field order under the
    model. An empty string is returned for a trial with no text at all; the
    cross-encoder scores it low, which is the honest outcome - we have nothing
    to judge it on - and it is not dropped.
    """
    if text is None:
        return ""
    if text.overview_text:
        return text.overview_text
    parts = [text.title, "; ".join(text.conditions), "; ".join(text.interventions)]
    return "\n".join(part for part in parts if part)
