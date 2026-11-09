"""The sentence splitter. Over-splitting is the failure that matters here.

Every judgment cites sentence IDs and the verifier checks them, so a splitter
that cuts "Creatinine was 0.9 mg/dL" into two sentences does not produce a
tidier note - it produces a citation that points at half a fact.
"""

from __future__ import annotations

from match_worker.notes import split_sentences, to_numbered_note


def test_numbers_from_one_and_covers_every_sentence() -> None:
    note = to_numbered_note("First sentence here. Second sentence here. Third one.")
    assert [s.sentence_id for s in note.sentences] == [1, 2, 3]
    assert note.valid_ids == frozenset({1, 2, 3})
    assert note.sentences[0].text == "First sentence here."


def test_does_not_split_on_decimals_or_units() -> None:
    text = "Creatinine was 0.9 mg/dL with an eGFR above 60. Bilirubin was 0.6 mg/dL."
    assert split_sentences(text) == [
        "Creatinine was 0.9 mg/dL with an eGFR above 60.",
        "Bilirubin was 0.6 mg/dL.",
    ]


def test_does_not_split_on_abbreviations_or_initials() -> None:
    text = "Seen by Dr. Patel in clinic. Metformin 1000 mg p.o. b.i.d. was continued."
    sentences = split_sentences(text)
    assert sentences[0] == "Seen by Dr. Patel in clinic."
    assert sentences[1].startswith("Metformin 1000 mg p.o.")
    assert len(sentences) == 2


def test_blank_lines_separate_sentences_and_whitespace_is_normalised() -> None:
    text = "History of\n   hypertension\n\nECOG 1"
    assert split_sentences(text) == ["History of hypertension", "ECOG 1"]


def test_empty_note_yields_no_sentences() -> None:
    assert to_numbered_note("   \n\n ").sentences == []
