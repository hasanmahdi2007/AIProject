"""The eligibility-text splitter: one rule in, one rule out, never half a rule.

This is the ``parse_rules`` node's content. It runs lazily, on the twenty
trials that were actually retrieved, and writes ``trial_criteria`` rows that
every later patient reuses - which is why the row carries a ``parser`` column
and a stable ``criterion_id``.

The one rule that matters: **never fixed-size chunk.** A criterion cut mid
sentence ("Patients must have adequate organ function as defined by" / "a
creatinine clearance above 60 mL/min") produces two judgments, both of which
are unanswerable, and the model will answer them anyway. So the split follows
structure - section headers, bullets, numbering - and falls back to sentence
boundaries, which are also structure, before it ever considers length.

ClinicalTrials.gov eligibility text is free text with no schema. The shapes
this handles, all of which appear in TREC:

* ``Inclusion Criteria:`` / ``Exclusion Criteria:`` headers, in any case, with
  or without a colon, sometimes prefixed ("Key Inclusion Criteria").
* Bulleted items: ``-``, ``*``, ``•``, ``o``.
* Numbered items: ``1.``, ``1)``, ``(1)``, ``i.``, ``a)``.
* No markers at all - one criterion per line, or one per sentence.
* Continuation lines indented under their bullet.

Text before any header is treated as **inclusion**, which is the safe default:
a misfiled inclusion rule gets judged and can come back
``insufficient_info``, whereas a misfiled exclusion rule can exclude a trial
the patient was eligible for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from beacon_core.schemas import CriterionKind
from match_worker.notes import split_sentences

__all__ = ["ParsedCriterion", "criterion_id_for", "split_eligibility"]

_INCLUSION_HEADER = re.compile(
    r"^\s*(?:key\s+|main\s+|principal\s+)?inclusion(?:\s+criteri(?:a|on))?\s*:?\s*$",
    re.IGNORECASE,
)
_EXCLUSION_HEADER = re.compile(
    r"^\s*(?:key\s+|main\s+|principal\s+)?exclusion(?:\s+criteri(?:a|on))?\s*:?\s*$",
    re.IGNORECASE,
)
_GENERIC_HEADER = re.compile(
    r"^\s*(?:eligibility|patient\s+selection)(?:\s+criteri(?:a|on))?\s*:?\s*$", re.IGNORECASE
)
_BULLET = re.compile(r"^\s*(?:[-*•·●]|o\s)\s*")
_NUMBERED = re.compile(r"^\s*\(?(?:\d{1,2}|[ivxIVX]{1,4}|[a-hA-H])[.)]\s+")
_INLINE_HEADER = re.compile(
    r"(?P<kind>inclusion|exclusion)(?:\s+criteri(?:a|on))?\s*:", re.IGNORECASE
)
_WHITESPACE = re.compile(r"\s+")
_MIN_CRITERION_CHARS = 8
"""Below this a line is a fragment ("Age:", "Both") rather than a rule."""


@dataclass(frozen=True, slots=True)
class ParsedCriterion:
    """One rule, ready to become a ``TrialCriterion`` row."""

    criterion_id: str
    kind: CriterionKind
    position: int
    text: str


def criterion_id_for(nct_id: str, kind: CriterionKind, ordinal: int) -> str:
    """``NCT01234567:inc:03`` - stable, human-readable, cited in judgments.

    Stability is the point: the ID appears in ``criterion_judgments`` rows and
    in the unique constraint that makes a redelivered Kafka message a no-op, so
    it must be derived from the trial and the rule's position, never from a
    sequence or a UUID.
    """
    return f"{nct_id}:{'inc' if kind == 'inclusion' else 'exc'}:{ordinal:02d}"


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip(" \t;,")


def _is_item_start(line: str) -> bool:
    return bool(_BULLET.match(line) or _NUMBERED.match(line))


def _strip_marker(line: str) -> str:
    without_bullet = _BULLET.sub("", line, count=1)
    return _NUMBERED.sub("", without_bullet, count=1)


def _explode_inline_headers(text: str) -> str:
    """Put a run-on ``... Exclusion Criteria: ...`` header on its own line.

    Plenty of records carry the whole eligibility section as one paragraph. If
    the header stays inline, every exclusion rule in the record is filed as an
    inclusion rule, which inverts the verdict for that trial.
    """
    return _INLINE_HEADER.sub(lambda m: f"\n{m.group('kind')} criteria:\n", text)


def split_eligibility(nct_id: str, eligibility_text: str | None) -> list[ParsedCriterion]:
    """Split one trial's eligibility text into individually judgeable rules.

    Returns an empty list when there is nothing to split; the caller marks that
    trial "could not evaluate" rather than treating a trial with no rules as
    one every patient passes.
    """
    if not eligibility_text or not eligibility_text.strip():
        return []

    kind: CriterionKind = "inclusion"
    buckets: dict[CriterionKind, list[str]] = {"inclusion": [], "exclusion": []}
    pending: list[str] = []

    def flush() -> None:
        if pending:
            joined = _normalise(" ".join(pending))
            if len(joined) >= _MIN_CRITERION_CHARS:
                buckets[kind].append(joined)
            pending.clear()

    for raw_line in _explode_inline_headers(eligibility_text).splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        if _INCLUSION_HEADER.match(line) or _GENERIC_HEADER.match(line):
            flush()
            kind = "inclusion"
            continue
        if _EXCLUSION_HEADER.match(line):
            flush()
            kind = "exclusion"
            continue
        if _is_item_start(line):
            flush()
            pending.append(_strip_marker(line))
            continue
        if pending:
            # An indented or wrapped continuation of the current item. Joining
            # is what keeps "...as defined by" attached to its own condition.
            pending.append(line.strip())
            continue
        # No markers anywhere in this section: the line itself is the unit, and
        # a line holding several sentences splits on sentence boundaries.
        for sentence in split_sentences(line):
            candidate = _normalise(sentence)
            if len(candidate) >= _MIN_CRITERION_CHARS:
                buckets[kind].append(candidate)
    flush()

    parsed: list[ParsedCriterion] = []
    position = 0
    for bucket_kind in ("inclusion", "exclusion"):
        typed_kind: CriterionKind = "inclusion" if bucket_kind == "inclusion" else "exclusion"
        for ordinal, text in enumerate(buckets[typed_kind], start=1):
            position += 1
            parsed.append(
                ParsedCriterion(
                    criterion_id=criterion_id_for(nct_id, typed_kind, ordinal),
                    kind=typed_kind,
                    position=position,
                    text=text,
                )
            )
    return parsed
