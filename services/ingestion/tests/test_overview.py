"""Overview assembly, field order, and the 512-token truncation measurement."""

from __future__ import annotations

import pytest

from ingestion._testing import WordTokenizer, record
from ingestion.overview import OverviewBuilder, build_overview_text


def test_field_order_is_most_discriminative_first() -> None:
    text = build_overview_text(
        "A Study of Pembrolizumab in NSCLC",
        ["Non-Small Cell Lung Cancer", "Carcinoma"],
        ["Pembrolizumab"],
        "This study evaluates whether pembrolizumab improves survival.",
    )
    lines = text.split("\n")
    assert lines[0] == "A Study of Pembrolizumab in NSCLC"
    assert lines[1] == "Conditions: Non-Small Cell Lung Cancer; Carcinoma"
    assert lines[2] == "Interventions: Pembrolizumab"
    assert lines[3].startswith("This study evaluates")
    # The summary is last because it is what the 512-token window must cut.
    assert text.index("A Study") < text.index("Conditions:") < text.index("Interventions:")
    assert text.index("Interventions:") < text.index("This study evaluates")


def test_empty_fields_are_omitted_not_labelled() -> None:
    text = build_overview_text("Only A Title", [], [], None)
    assert text == "Only A Title"
    assert "Conditions" not in text
    assert "Interventions" not in text


def test_whitespace_in_source_fields_is_collapsed() -> None:
    text = build_overview_text("  Spaced   Title \n", [], [], "A\r\n  summary   here.")
    assert text == "Spaced Title\nA summary here."


def test_eligibility_text_is_never_part_of_the_overview() -> None:
    """The retrieval unit is a trial; rules live in Postgres only."""
    fixture = record("NCT00000105")
    text = build_overview_text(
        fixture.title, list(fixture.conditions), list(fixture.interventions), None
    )
    assert "Inclusion Criteria" not in text
    assert "Karnofsky" not in text


# --- truncation -------------------------------------------------------------


def _builder(max_tokens: int = 12) -> OverviewBuilder:
    return OverviewBuilder(WordTokenizer(), max_tokens=max_tokens)


def test_short_overview_is_not_truncated() -> None:
    overview = _builder().build("Short Title", ["Asthma"], [], None)
    assert overview.truncated is False
    assert overview.dropped_tokens == 0
    assert overview.text == "Short Title\nConditions: Asthma"


def test_long_overview_is_truncated_at_a_token_boundary() -> None:
    summary = " ".join(f"word{n}" for n in range(50))
    overview = _builder(max_tokens=12).build("Title Here", ["Asthma"], [], summary)
    assert overview.truncated is True
    # 10 content tokens kept: "Title Here" (2) + "Conditions: Asthma" (2) + 6 words.
    assert overview.text.split() == [
        "Title",
        "Here",
        "Conditions:",
        "Asthma",
        "word0",
        "word1",
        "word2",
        "word3",
        "word4",
        "word5",
    ]
    # The title survives; that is the entire reason for the field order.
    assert overview.text.startswith("Title Here")


def test_token_count_is_the_untruncated_length() -> None:
    """``overview_token_count`` must quantify the overflow, not the kept part."""
    summary = " ".join(f"word{n}" for n in range(50))
    overview = _builder(max_tokens=12).build("Title Here", [], [], summary)
    assert overview.token_count == 2 + 2 + 50  # specials + title + summary words
    assert overview.token_count > overview.max_tokens
    assert overview.dropped_tokens == overview.token_count - 12


def test_truncated_text_is_a_prefix_of_the_full_text() -> None:
    summary = " ".join(f"word{n}" for n in range(50))
    builder = _builder(max_tokens=12)
    full = build_overview_text("Title Here", [], [], summary)
    overview = builder.build("Title Here", [], [], summary)
    assert full.startswith(overview.text)


def test_truncation_boundary_is_exact() -> None:
    """A text of exactly ``max_tokens`` is kept whole; one more is cut."""
    builder = _builder(max_tokens=12)
    exact = " ".join(f"w{n}" for n in range(10))  # 10 content + 2 specials
    assert builder.count_tokens(exact) == 12
    text, count, truncated = builder.truncate(exact)
    assert (text, count, truncated) == (exact, 12, False)

    one_more = exact + " overflow"
    text, count, truncated = builder.truncate(one_more)
    assert (count, truncated) == (13, True)
    assert text == exact


def test_max_tokens_must_leave_room_for_the_special_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        OverviewBuilder(WordTokenizer(), max_tokens=2)


def test_from_fastembed_refuses_a_model_without_a_tokenizer() -> None:
    """Better to fail than to silently estimate the truncation rate."""

    class NoTokenizer:
        pass

    with pytest.raises(TypeError, match="truncation rate"):
        OverviewBuilder.from_fastembed(NoTokenizer())
