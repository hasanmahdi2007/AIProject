"""Judge and Cohen's kappa tests. No API call is made anywhere in this file.

The kappa cases are computed by hand in the docstrings, which is the only way
to be sure the implementation is right: kappa is easy to write plausibly and
wrongly (the classic error is computing expected agreement from one rater's
marginals twice, which silently inflates it).

The judge side is tested through :class:`~evals.judge.ScriptedJudge` and
:class:`~evals.judge.FixtureJudge`. :class:`~evals.judge.LlmJudge` is
constructed - to prove the self-grading and spend interlocks fire - but
:meth:`~evals.judge.LlmJudge.judge` is only ever called through a fake
:class:`~evals.judge.JudgeLlm` that returns a canned string, so nothing
reaches the network. There is deliberately no attempt to mock the Anthropic
SDK at the HTTP layer: ``respx`` cannot intercept it (the SDK is built on
``httpx2``), so such a test would pass while asserting nothing at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals.judge import (
    ELIGIBILITY_LABELS,
    KAPPA_TARGET,
    FixtureJudge,
    JudgeItem,
    JudgeVerdict,
    LlmJudge,
    ScriptedJudge,
    SelfGradingError,
    SpendNotApprovedError,
    agreement_table,
    calibrate,
    cohens_kappa,
    load_hand_labels,
    model_family,
)


class FakeJudgeLlm:
    """A :class:`~evals.judge.JudgeLlm` that returns a fixed answer.

    Records its calls so a test can assert the prompt was built, without any
    network involved.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, model: str) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        return self.answer


# ----------------------------------------------------------- model families --


def test_model_family_extracts_the_family_token() -> None:
    assert model_family("claude-opus-5-5") == "opus"
    assert model_family("claude-sonnet-5-5") == "sonnet"
    assert model_family("claude-haiku-5-5") == "haiku"


def test_model_family_ignores_a_date_suffix() -> None:
    """Comparing full ids would let a dated snapshot judge its own base model."""
    assert model_family("claude-sonnet-5-5-20260301") == "sonnet"


def test_opus_judging_sonnet_is_allowed() -> None:
    judge = LlmJudge(
        llm=FakeJudgeLlm("{}"),
        model="claude-opus-5-5",
        model_under_test="claude-sonnet-5-5",
    )
    assert judge.model == "claude-opus-5-5"


@pytest.mark.parametrize(
    ("judge_model", "tested"),
    [
        ("claude-sonnet-5-5", "claude-sonnet-5-5"),
        ("claude-sonnet-5-5", "claude-sonnet-5-5-20260301"),
        ("claude-haiku-5-5", "claude-haiku-5-5"),
    ],
)
def test_a_model_may_not_grade_its_own_family(judge_model: str, tested: str) -> None:
    """Self-grading measures self-consistency and reports it as accuracy."""
    with pytest.raises(SelfGradingError, match="family"):
        LlmJudge(llm=FakeJudgeLlm("{}"), model=judge_model, model_under_test=tested)


# -------------------------------------------------------- the spend interlock --


def test_the_judge_refuses_to_spend_without_explicit_approval() -> None:
    """The owner's standing rule, encoded: nothing bills his key by accident."""
    judge = LlmJudge(
        llm=FakeJudgeLlm('{"label": "eligible"}'),
        model="claude-opus-5-5",
        model_under_test="claude-sonnet-5-5",
    )
    item = JudgeItem(item_id="1:NCT1", patient_note="n", trial_id="NCT1", trial_text="t")
    with pytest.raises(SpendNotApprovedError, match="would bill"):
        judge.judge(item)


def test_an_approved_judge_parses_a_verdict_without_touching_the_network() -> None:
    llm = FakeJudgeLlm('{"label": "excluded", "rationale": "age 24 < 25"}')
    judge = LlmJudge(
        llm=llm,
        model="claude-opus-5-5",
        model_under_test="claude-sonnet-5-5",
        spend_approved=True,
    )
    verdict = judge.judge(
        JudgeItem(item_id="1:NCT1", patient_note="24yo", trial_id="NCT1", trial_text="25+")
    )
    assert verdict.label == "excluded"
    assert verdict.rationale == "age 24 < 25"
    assert verdict.model == "claude-opus-5-5"
    assert llm.calls[0]["model"] == "claude-opus-5-5"
    assert "24yo" in llm.calls[0]["user"]


def test_a_non_json_judge_answer_raises_rather_than_defaulting() -> None:
    judge = LlmJudge(
        llm=FakeJudgeLlm("I think probably eligible"),
        model="claude-opus-5-5",
        model_under_test="claude-sonnet-5-5",
        spend_approved=True,
    )
    with pytest.raises(ValueError, match="did not return JSON"):
        judge.judge(JudgeItem(item_id="x", patient_note="", trial_id="", trial_text=""))


def test_a_judge_answer_with_no_label_raises() -> None:
    judge = LlmJudge(
        llm=FakeJudgeLlm('{"rationale": "hmm"}'),
        model="claude-opus-5-5",
        model_under_test="claude-sonnet-5-5",
        spend_approved=True,
    )
    with pytest.raises(ValueError, match="no 'label'"):
        judge.judge(JudgeItem(item_id="x", patient_note="", trial_id="", trial_text=""))


# ------------------------------------------------------------------ verdicts --


def test_a_verdict_must_carry_a_contract_label() -> None:
    """The label set is the frozen ``EligibilityCategory``, not free text."""
    with pytest.raises(ValueError, match="not an EligibilityCategory"):
        JudgeVerdict(item_id="x", label="maybe")  # type: ignore[arg-type]


def test_the_label_set_comes_from_the_frozen_contract() -> None:
    assert set(ELIGIBILITY_LABELS) == {
        "eligible",
        "likely_eligible",
        "insufficient_info",
        "excluded",
    }


def test_item_ids_join_a_topic_to_a_trial() -> None:
    assert JudgeItem.make_id("12", "NCT00000102") == "12:NCT00000102"


def test_a_scripted_judge_raises_on_an_unknown_item() -> None:
    judge = ScriptedJudge(verdicts={"a": "eligible"})
    with pytest.raises(KeyError, match="no verdict for"):
        judge.judge(JudgeItem(item_id="b", patient_note="", trial_id="", trial_text=""))


def test_a_fixture_judge_replays_from_disk_and_never_falls_through(tmp_path: Path) -> None:
    """A fall-through to a live call is how a free test suite starts billing."""
    path = tmp_path / "verdicts.json"
    path.write_text(
        json.dumps([{"item_id": "1:NCT1", "label": "eligible", "rationale": "fits"}]),
        encoding="utf-8",
    )
    judge = FixtureJudge.from_file(path)
    verdict = judge.judge(JudgeItem(item_id="1:NCT1", patient_note="", trial_id="", trial_text=""))
    assert verdict.label == "eligible"
    with pytest.raises(KeyError, match="never falls through"):
        judge.judge(JudgeItem(item_id="nope", patient_note="", trial_id="", trial_text=""))


def test_a_fixture_file_that_is_not_a_list_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"item_id": "a", "label": "eligible"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON list"):
        FixtureJudge.from_file(path)


# -------------------------------------------------------------- Cohen's kappa --


def test_kappa_of_perfect_agreement_is_one() -> None:
    """A = B = [x, y, x, y]. Po = 1.0, Pe = 0.5*0.5 + 0.5*0.5 = 0.5,
    kappa = (1 - 0.5) / (1 - 0.5) = 1.0."""
    labels = ["eligible", "excluded", "eligible", "excluded"]
    result = cohens_kappa(labels, labels, labels=["eligible", "excluded"])
    assert result.kappa == pytest.approx(1.0)
    assert result.observed_agreement == pytest.approx(1.0)
    assert result.expected_agreement == pytest.approx(0.5)
    assert result.meets_target


def test_kappa_of_chance_agreement_is_zero() -> None:
    """A = [E, E, X, X], B = [E, X, E, X].

    Po = 2/4 = 0.5. Both marginals are 0.5/0.5, so
    Pe = 0.5*0.5 + 0.5*0.5 = 0.5, and kappa = (0.5 - 0.5) / 0.5 = 0.0.
    """
    a = ["eligible", "eligible", "excluded", "excluded"]
    b = ["eligible", "excluded", "eligible", "excluded"]
    result = cohens_kappa(a, b, labels=["eligible", "excluded"])
    assert result.observed_agreement == pytest.approx(0.5)
    assert result.expected_agreement == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.0)
    assert not result.meets_target


def test_kappa_matches_the_textbook_two_by_two_case() -> None:
    """The standard worked example, confusion matrix:

        A\\B   E   X   | row
        E     4   2   |  6
        X     1   3   |  4
        ---------------
        col   5   5   | 10

    Po = (4 + 3) / 10 = 0.70
    Pe = (6/10)(5/10) + (4/10)(5/10) = 0.30 + 0.20 = 0.50
    kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
    """
    a = ["eligible"] * 6 + ["excluded"] * 4
    b = ["eligible"] * 4 + ["excluded"] * 2 + ["eligible"] * 1 + ["excluded"] * 3
    result = cohens_kappa(a, b, labels=["eligible", "excluded"])
    assert result.observed_agreement == pytest.approx(0.70)
    assert result.expected_agreement == pytest.approx(0.50)
    assert result.kappa == pytest.approx(0.40)
    assert result.n == 10
    assert result.confusion == ((4, 2), (1, 3))
    assert not result.meets_target  # 0.40 < 0.6


def test_kappa_of_total_disagreement_is_minus_one() -> None:
    """Confusion ((0, 5), (5, 0)): Po = 0, Pe = 0.5, kappa = -1.0."""
    a = ["eligible"] * 5 + ["excluded"] * 5
    b = ["excluded"] * 5 + ["eligible"] * 5
    result = cohens_kappa(a, b, labels=["eligible", "excluded"])
    assert result.observed_agreement == pytest.approx(0.0)
    assert result.kappa == pytest.approx(-1.0)


def test_kappa_is_undefined_when_both_raters_use_one_label() -> None:
    """Pe = 1.0, so kappa is 0/0. It must be nan, never a flattering 1.0.

    This is the case that would otherwise let a judge that answers
    "insufficient_info" to everything report perfect agreement with a human
    who happened to do the same.
    """
    a = ["eligible"] * 8
    result = cohens_kappa(a, list(a), labels=ELIGIBILITY_LABELS)
    assert result.degenerate
    assert math.isnan(result.kappa)
    assert not result.meets_target
    assert result.observed_agreement == pytest.approx(1.0)
    assert "undefined" in str(result)


def test_kappa_target_is_the_documented_threshold() -> None:
    assert KAPPA_TARGET == 0.6
    high = cohens_kappa(
        ["eligible"] * 9 + ["excluded"],
        ["eligible"] * 9 + ["excluded"],
        labels=["eligible", "excluded"],
    )
    assert high.kappa == pytest.approx(1.0)
    assert high.meets_target


def test_unused_labels_still_contribute_their_marginals() -> None:
    """Passing the full contract label set must not change Po, and must give
    the same kappa as the two-label call when the extra labels are unused: a
    zero marginal contributes zero to Pe."""
    a = ["eligible"] * 6 + ["excluded"] * 4
    b = ["eligible"] * 4 + ["excluded"] * 2 + ["eligible"] * 1 + ["excluded"] * 3
    narrow = cohens_kappa(a, b, labels=["eligible", "excluded"])
    wide = cohens_kappa(a, b, labels=ELIGIBILITY_LABELS)
    assert wide.kappa == pytest.approx(narrow.kappa)
    assert len(wide.confusion) == 4


def test_linear_weighted_kappa_on_ordinal_grades() -> None:
    """Hand case on the ordinal TREC grades ["0", "1", "2"] (k = 3).

    Linear weights are ``w_ij = 1 - |i - j| / (k - 1)``, so w(0,0)=1,
    w(0,1)=0.5, w(0,2)=0.

    A = [0, 1], B = [1, 0]:
        Po_w = (w(0,1) + w(1,0)) / 2 = (0.5 + 0.5) / 2 = 0.5
        marginals: A -> 0:0.5, 1:0.5 ; B -> 0:0.5, 1:0.5
        Pe_w = 0.25*(w(0,0) + w(0,1) + w(1,0) + w(1,1))
             = 0.25*(1 + 0.5 + 0.5 + 1) = 0.75
        kappa_w = (0.5 - 0.75) / (1 - 0.75) = -1.0
    """
    result = cohens_kappa(["0", "1"], ["1", "0"], labels=["0", "1", "2"], weights="linear")
    assert result.observed_agreement == pytest.approx(0.5)
    assert result.expected_agreement == pytest.approx(0.75)
    assert result.kappa == pytest.approx(-1.0)


def test_weighted_kappa_credits_a_near_miss_more_than_a_far_one() -> None:
    """Confusing grade 2 with grade 1 is a smaller error than with grade 0.

    Unweighted kappa cannot see the difference; this is the only reason the
    weighted variants exist, and they are used only for the genuinely ordinal
    relevance grades, never for the eligibility categories.
    """
    grades = ["0", "1", "2"]
    near = cohens_kappa(["2", "0"], ["1", "0"], labels=grades, weights="linear")
    far = cohens_kappa(["2", "0"], ["0", "0"], labels=grades, weights="linear")
    assert near.observed_agreement > far.observed_agreement


def test_kappa_rejects_mismatched_rater_lengths() -> None:
    with pytest.raises(ValueError, match="same items"):
        cohens_kappa(["eligible"], ["eligible", "excluded"])


def test_kappa_rejects_no_items() -> None:
    with pytest.raises(ValueError, match="no items"):
        cohens_kappa([], [])


def test_kappa_rejects_a_label_outside_the_declared_set() -> None:
    with pytest.raises(ValueError, match="not in the declared label set"):
        cohens_kappa(["eligible"], ["banana"], labels=ELIGIBILITY_LABELS)


# --------------------------------------------------------------- alignment --


def test_agreement_table_sorts_and_pairs_by_item_id() -> None:
    ids, hand, judged = agreement_table(
        {"b": "excluded", "a": "eligible"}, {"a": "eligible", "b": "insufficient_info"}
    )
    assert ids == ("a", "b")
    assert hand == ("eligible", "excluded")
    assert judged == ("eligible", "insufficient_info")


def test_agreement_table_rejects_a_partial_overlap() -> None:
    """Silently intersecting is how a calibration set shrinks to easy items."""
    with pytest.raises(ValueError, match="same items"):
        agreement_table({"a": "eligible", "b": "excluded"}, {"a": "eligible"})


# ------------------------------------------------------------- hand labels --


def test_load_hand_labels_reads_jsonl_and_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "hand.jsonl"
    path.write_text(
        "// the owner's labels\n"
        '{"item_id": "1:NCT1", "label": "eligible"}\n'
        "\n"
        '{"item_id": "1:NCT2", "label": "excluded"}\n',
        encoding="utf-8",
    )
    assert load_hand_labels(path) == {"1:NCT1": "eligible", "1:NCT2": "excluded"}


def test_load_hand_labels_rejects_an_unknown_label(tmp_path: Path) -> None:
    path = tmp_path / "hand.jsonl"
    path.write_text('{"item_id": "a", "label": "probably"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="is not one of"):
        load_hand_labels(path)


def test_load_hand_labels_rejects_a_duplicate_item(tmp_path: Path) -> None:
    path = tmp_path / "hand.jsonl"
    path.write_text(
        '{"item_id": "a", "label": "eligible"}\n{"item_id": "a", "label": "excluded"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate item_id"):
        load_hand_labels(path)


def test_missing_hand_labels_explain_why_there_is_no_kappa(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nothing to calibrate against"):
        load_hand_labels(tmp_path / "absent.jsonl")


# -------------------------------------------------------------- calibration --


def test_calibrate_runs_a_judge_over_hand_labelled_items() -> None:
    """End-to-end kappa with a scripted judge: 4 items, 3 agreements.

        hand  = [eligible, eligible, excluded,  insufficient_info]
        judge = [eligible, eligible, excluded,  excluded]

    Po = 3/4 = 0.75. Marginals over the 4-label contract set:
        hand : eligible 0.5, excluded 0.25, insufficient_info 0.25
        judge: eligible 0.5, excluded 0.5
        Pe = 0.5*0.5 + 0.25*0.5 = 0.25 + 0.125 = 0.375
        kappa = (0.75 - 0.375) / (1 - 0.375) = 0.375 / 0.625 = 0.6
    """
    hand = {
        "i1": "eligible",
        "i2": "eligible",
        "i3": "excluded",
        "i4": "insufficient_info",
    }
    judge = ScriptedJudge(
        verdicts={
            "i1": "eligible",
            "i2": "eligible",
            "i3": "excluded",
            "i4": "excluded",
        }
    )
    items = [
        JudgeItem(item_id=i, patient_note="", trial_id="", trial_text="") for i in sorted(hand)
    ]
    result = calibrate(items, hand, judge)  # type: ignore[arg-type]
    assert result.observed_agreement == pytest.approx(0.75)
    assert result.expected_agreement == pytest.approx(0.375)
    assert result.kappa == pytest.approx(0.6)
    assert result.meets_target  # exactly at the target counts as meeting it


def test_calibrate_refuses_when_the_judge_covers_different_items() -> None:
    hand = {"i1": "eligible", "i2": "excluded"}
    judge = ScriptedJudge(verdicts={"i1": "eligible", "i2": "excluded"})
    items = [JudgeItem(item_id="i1", patient_note="", trial_id="", trial_text="")]
    with pytest.raises(ValueError, match="same items"):
        calibrate(items, hand, judge)  # type: ignore[arg-type]


def test_kappa_result_serialises_for_the_report() -> None:
    result = cohens_kappa(
        ["eligible", "excluded"], ["eligible", "excluded"], labels=ELIGIBILITY_LABELS
    )
    payload = result.to_dict()
    assert payload["kappa"] == pytest.approx(1.0)
    assert payload["meets_target"] is True
    assert payload["weights"] == "none"
    assert payload["target"] == KAPPA_TARGET
