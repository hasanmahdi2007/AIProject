"""Sentence numbering, in code and never by a model.

Every ``CriterionJudgment`` cites sentence IDs, and the verifier checks those
IDs against the note. If the numbering came from the model it could differ
between the call that produced a citation and the call that checked it, and the
mismatch would look like a hallucinated citation. So the split is deterministic,
happens exactly once per run, and the numbered note is checkpointed.

The splitter is intentionally conservative. Clinical notes are full of tokens
that look like sentence ends and are not: ``1.5 mg``, ``e.g.``, ``Dr.``,
``p.o.``, ``T2N0M0.`` at a line end. Over-splitting is the worse failure here,
because a criterion's supporting fact ends up straddling two IDs and the model
cites one of them.
"""

from __future__ import annotations

import re

from beacon_core.schemas import NoteSentence, NumberedNote

__all__ = ["split_sentences", "to_numbered_note"]

_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "vs",
        "etc",
        "approx",
        "no",
        "fig",
        "eg",
        "ie",
        "e.g",
        "i.e",
        "p.o",
        "b.i.d",
        "t.i.d",
        "q.d",
        "q.h.s",
        "mg",
        "ml",
        "mcg",
        "kg",
        "cm",
        "mm",
        "hr",
        "min",
        "wk",
        "yr",
        "u",
    }
)

_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_WHITESPACE = re.compile(r"\s+")


def _ends_with_abbreviation(chunk: str) -> bool:
    """True when the trailing period belongs to a word, not to a sentence."""
    stripped = chunk.rstrip("\"')]")
    if not stripped.endswith("."):
        return False
    tail = stripped[:-1].split()[-1].lower() if stripped[:-1].split() else ""
    if tail in _ABBREVIATIONS:
        return True
    # A single letter plus a dot is an initial ("J. Smith"); a bare digit plus a
    # dot is a list marker or a decimal that lost its fraction.
    return len(tail) <= 1 and (tail.isalpha() or tail.isdigit())


def split_sentences(text: str) -> list[str]:
    """Split a note into sentences, one per returned string.

    Blank lines separate sentences; so does sentence-final punctuation, unless
    the period belongs to an abbreviation, an initial or a number.
    """
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n+", text.strip()):
        buffer = ""
        for chunk in _BOUNDARY.split(block):
            if not chunk.strip():
                continue
            buffer = f"{buffer} {chunk}".strip() if buffer else chunk.strip()
            if _ends_with_abbreviation(buffer):
                continue
            sentences.append(_WHITESPACE.sub(" ", buffer).strip())
            buffer = ""
        if buffer.strip():
            sentences.append(_WHITESPACE.sub(" ", buffer).strip())
    return [s for s in sentences if s]


def to_numbered_note(text: str) -> NumberedNote:
    """The note as a :class:`NumberedNote`, numbered from 1.

    Numbering from 1 rather than 0 is part of the contract: the prompt renders
    ``"<id>. <text>"`` lines and a model asked to cite "sentence 0" tends to
    cite 1 instead.
    """
    return NumberedNote(
        sentences=[
            NoteSentence(sentence_id=index, text=sentence)
            for index, sentence in enumerate(split_sentences(text), start=1)
        ]
    )
