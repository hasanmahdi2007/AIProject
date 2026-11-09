"""LLM-judge interface and Cohen's kappa calibration. **No live calls.**

Two independent ideas live here, and only the second one has ever been run.

**1. The judge seam.** Eligibility judgements are graded by a *different,
stronger* model than the one being graded: the headline pipeline runs Sonnet
5.5, so the judge is Opus 5.5 (``settings.anthropic.judge_model``). A model
grading its own output measures self-consistency and reports it as accuracy,
which is worse than not measuring at all because it produces a high number.
:class:`LlmJudge` refuses to be constructed when the judge and the model under
test come from the same family - see :func:`model_family` and
:class:`SelfGradingError`.

**2. Cohen's kappa.** An LLM judge is only worth anything once it has been
shown to agree with a human on the same items. :func:`cohens_kappa` is
implemented here from the definition (rather than pulled from ``sklearn``,
which is not a dependency) and is validated in the tests against
hand-computed 2x2 cases - including the exact textbook case
``Po = 0.70, Pe = 0.50, kappa = 0.40``. The target is
:data:`KAPPA_TARGET` = 0.6; below that the judge's verdicts are not usable as
a measurement and the harness says so instead of quoting them.

Why nothing here has made an API call
-------------------------------------
The owner's standing instruction is that nothing bills his key without his
explicit prior approval, and that approval has not been given, so this agent's
API budget is **zero**. That is not a reason to leave the judge unbuilt; it is
a reason to build it so it can be tested without calls:

* :class:`LlmJudge` depends on :class:`JudgeLlm`, a two-method protocol, not on
  ``beacon_llm`` or on ``anthropic``. ``evals`` declares neither as a
  dependency and imports neither. The real client is injected by whoever owns
  that dependency, in about ten lines.
* :class:`FixtureJudge` and :class:`ScriptedJudge` satisfy the same protocol
  from recorded or hand-written verdicts, and they are what the tests use.
* :class:`LlmJudge` additionally carries a spend interlock:
  :attr:`LlmJudge.spend_approved` defaults to ``False`` and
  :meth:`LlmJudge.judge` raises :class:`SpendNotApprovedError` until it is set
  deliberately. A stray call in a notebook fails instead of billing.

A note for whoever wires the real client in: do **not** try to prove the wiring
with a mocked HTTP layer. ``respx`` cannot intercept the Anthropic SDK (it is
built on ``httpx2``), so such a test passes while asserting nothing. Test
against :class:`FixtureJudge`, and prove the live path once, manually, with the
owner watching the bill.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

from beacon_core.schemas import EligibilityCategory

__all__ = [
    "ELIGIBILITY_LABELS",
    "JUDGE_SYSTEM_PROMPT",
    "KAPPA_TARGET",
    "FixtureJudge",
    "JudgeBackend",
    "JudgeItem",
    "JudgeLlm",
    "JudgeVerdict",
    "KappaResult",
    "KappaWeights",
    "LlmJudge",
    "ScriptedJudge",
    "SelfGradingError",
    "SpendNotApprovedError",
    "agreement_table",
    "calibrate",
    "cohens_kappa",
    "load_hand_labels",
    "model_family",
]

KAPPA_TARGET: Final = 0.6
"""The agreement floor below which the judge is not used as a measurement.

0.6 is the conventional boundary between "moderate" and "substantial"
agreement on the Landis-Koch scale. It is a convention, not a law, which is
why :class:`KappaResult` reports the raw kappa, the observed agreement and the
expected agreement side by side: a reader who disagrees with the threshold can
apply their own.
"""

ELIGIBILITY_LABELS: Final[tuple[EligibilityCategory, ...]] = (
    "eligible",
    "likely_eligible",
    "insufficient_info",
    "excluded",
)
"""The label set, from the frozen contract ``beacon_core.schemas``.

Fixed and explicit rather than derived from the data, because a label that
appears in neither rater's answers still belongs in the confusion matrix: the
marginals that produce the expected agreement are wrong if a category silently
vanishes.
"""

KappaWeights = Literal["none", "linear", "quadratic"]
"""Disagreement weighting.

``none`` is the reported default and the only one used for the Beacon label
set. The four eligibility categories are **not cleanly ordinal** -
``insufficient_info`` is an abstention, not a midpoint between ``eligible``
and ``excluded`` - so a weighted kappa over them would encode an ordering that
does not exist. The weighted variants are implemented for the one label set
that *is* ordinal, the TREC relevance grades 0/1/2, where confusing grade 2
with grade 1 genuinely is a smaller error than confusing it with grade 0.
"""


class SelfGradingError(RuntimeError):
    """Raised when a judge would grade its own family's output."""


class SpendNotApprovedError(RuntimeError):
    """Raised when a live judge call is attempted without explicit approval."""


def model_family(model: str) -> str:
    """The family token of a model id: ``claude-opus-5-5`` -> ``opus``.

    Used to catch self-grading that a plain ``!=`` would miss. Comparing full
    ids would let ``claude-sonnet-5-5`` judge ``claude-sonnet-5-5-20260301``,
    which is the same model grading itself.
    """
    parts = [p for p in model.lower().replace("_", "-").split("-") if p]
    for part in parts:
        if part in {"opus", "sonnet", "haiku", "fable"}:
            return part
    return model.lower()


# ------------------------------------------------------------- judge types --


@dataclass(frozen=True, slots=True)
class JudgeItem:
    """One (patient, trial) pair to be judged.

    ``item_id`` is the join key between the judge's verdicts and the owner's
    hand labels, so it must be stable across runs - ``f"{topic_id}:{nct_id}"``
    is the convention :func:`load_hand_labels` expects.
    """

    item_id: str
    patient_note: str
    trial_id: str
    trial_text: str

    @staticmethod
    def make_id(topic_id: str, nct_id: str) -> str:
        return f"{topic_id}:{nct_id}"


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One verdict, from either the judge or a human."""

    item_id: str
    label: EligibilityCategory
    rationale: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if self.label not in ELIGIBILITY_LABELS:
            raise ValueError(
                f"{self.item_id}: {self.label!r} is not an EligibilityCategory; "
                f"expected one of {ELIGIBILITY_LABELS}"
            )


@runtime_checkable
class JudgeLlm(Protocol):
    """The minimum :class:`LlmJudge` needs from a Claude client.

    ``beacon_llm.LlmClient`` satisfies this through a small adapter, which
    belongs wherever ``beacon-llm`` is a declared dependency - not here::

        class BeaconJudgeLlm:
            def __init__(self, client: LlmClient) -> None:
                self._client = client

            def complete(self, *, system: str, user: str, model: str) -> str:
                prompt = build_cached_prompt(instructions=system, note="", variable=user)
                return self._client.call(
                    prompt=prompt, output_model=JudgeOutput, model=model, effort="high"
                ).parsed.model_dump_json()
    """

    def complete(self, *, system: str, user: str, model: str) -> str:
        """Return the model's raw JSON answer for one item."""
        ...


@runtime_checkable
class JudgeBackend(Protocol):
    """Anything that can produce a verdict for an item."""

    model: str

    def judge(self, item: JudgeItem) -> JudgeVerdict: ...


@dataclass(frozen=True, slots=True)
class ScriptedJudge:
    """Verdicts from a mapping. The unit suite's judge.

    Args:
        verdicts: ``{item_id: label}``.
    """

    verdicts: Mapping[str, EligibilityCategory]
    model: str = "scripted-judge"

    def judge(self, item: JudgeItem) -> JudgeVerdict:
        try:
            label = self.verdicts[item.item_id]
        except KeyError:
            raise KeyError(
                f"scripted judge has no verdict for {item.item_id!r}; "
                f"it covers {sorted(self.verdicts)[:5]}..."
            ) from None
        return JudgeVerdict(item_id=item.item_id, label=label, model=self.model)


@dataclass(frozen=True, slots=True)
class FixtureJudge:
    """Replays recorded verdicts from a JSON file. Never reaches the network.

    File format - a list of objects, each with ``item_id`` and ``label`` and
    optionally ``rationale``::

        [{"item_id": "1:NCT00000102", "label": "excluded", "rationale": "..."}]

    A missing item **raises**, exactly as ``beacon_llm``'s replay mode does: a
    fall-through to a live call is how a "free, deterministic" test suite
    quietly starts billing, and a fall-through to a default label is how a
    calibration number becomes fiction.
    """

    verdicts: Mapping[str, JudgeVerdict]
    model: str = "fixture-judge"
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path, *, model: str = "fixture-judge") -> FixtureJudge:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a JSON list of verdict objects")
        verdicts: dict[str, JudgeVerdict] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: every entry must be an object, got {type(entry)}")
            item_id = str(entry["item_id"])
            verdicts[item_id] = JudgeVerdict(
                item_id=item_id,
                label=entry["label"],
                rationale=str(entry.get("rationale", "")),
                model=model,
            )
        if not verdicts:
            raise ValueError(f"{path}: no verdicts")
        return cls(verdicts=verdicts, model=model, path=path)

    def judge(self, item: JudgeItem) -> JudgeVerdict:
        try:
            return self.verdicts[item.item_id]
        except KeyError:
            raise KeyError(
                f"no recorded verdict for {item.item_id!r} in {self.path or 'fixtures'}; "
                "record it or remove the item from the calibration set. The judge "
                "never falls through to a live call."
            ) from None


JUDGE_SYSTEM_PROMPT = """\
You are auditing a clinical-trial matching system. You are shown a synthetic \
patient note and one clinical trial's eligibility criteria. Decide whether the \
patient described in the note would be eligible for the trial.

Answer with JSON only: {"label": <one of eligible, likely_eligible, \
insufficient_info, excluded>, "rationale": "<one sentence>"}.

Use "insufficient_info" when the note does not contain the facts the criteria \
require. Abstaining is correct behaviour and is scored as such; guessing is not. \
Judge only what the note states - do not infer unstated comorbidities, and do \
not reason about whether the trial is a good idea for the patient.
"""
"""The judge's instructions.

Written and reviewed but **never sent**: no call has been made with this
prompt, so it is unvalidated text. Expect to iterate on it once spend is
approved, and expect the first kappa to be lower than the second.
"""


@dataclass(frozen=True, slots=True)
class LlmJudge:
    """The real judge. Constructed in this harness; never yet called.

    Args:
        llm: injected client satisfying :class:`JudgeLlm`.
        model: the judging model, normally
            ``settings.anthropic.judge_model`` (``claude-opus-5-5``).
        model_under_test: the model whose output is being judged, normally the
            Sonnet 5.5 headline model. Used only to refuse self-grading.
        spend_approved: must be set ``True`` deliberately. Defaults to
            ``False``, which makes :meth:`judge` raise.

    Raises:
        SelfGradingError: when ``model`` and ``model_under_test`` share a
            family.
    """

    llm: JudgeLlm
    model: str
    model_under_test: str
    spend_approved: bool = False

    def __post_init__(self) -> None:
        if model_family(self.model) == model_family(self.model_under_test):
            raise SelfGradingError(
                f"judge {self.model!r} and model under test {self.model_under_test!r} "
                f"are both the {model_family(self.model)!r} family. A model grading "
                "its own output measures self-consistency and reports it as accuracy. "
                "The headline run is Sonnet 5.5, so the judge must be Opus 5.5."
            )

    def judge(self, item: JudgeItem) -> JudgeVerdict:
        """Judge one item. Raises unless spend was explicitly approved.

        Raises:
            SpendNotApprovedError: while :attr:`spend_approved` is ``False``.
            ValueError: if the model's answer is not the expected JSON shape.
        """
        if not self.spend_approved:
            raise SpendNotApprovedError(
                "LlmJudge.judge() would bill the owner's Anthropic key. Construct it "
                "with spend_approved=True only after the owner has approved the spend "
                "for this specific run. Use FixtureJudge for tests."
            )
        raw = self.llm.complete(
            system=JUDGE_SYSTEM_PROMPT, user=self._render(item), model=self.model
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{item.item_id}: judge did not return JSON: {raw[:200]!r}") from exc
        if not isinstance(payload, dict) or "label" not in payload:
            raise ValueError(f"{item.item_id}: judge answer has no 'label': {raw[:200]!r}")
        return JudgeVerdict(
            item_id=item.item_id,
            label=payload["label"],
            rationale=str(payload.get("rationale", "")),
            model=self.model,
        )

    @staticmethod
    def _render(item: JudgeItem) -> str:
        return (
            f"PATIENT NOTE (synthetic):\n{item.patient_note}\n\n"
            f"TRIAL {item.trial_id} ELIGIBILITY CRITERIA:\n{item.trial_text}\n"
        )


# -------------------------------------------------------------- Cohen kappa --


@dataclass(frozen=True, slots=True)
class KappaResult:
    """Cohen's kappa with the quantities it was derived from.

    Reporting ``observed_agreement`` and ``expected_agreement`` alongside
    kappa is not decoration: a kappa of 0.4 on labels that are 95% one class
    means something very different from a kappa of 0.4 on balanced labels, and
    only the components show which case you are in.
    """

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n: int
    labels: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...] = field(repr=False, default=())
    weights: KappaWeights = "none"
    degenerate: bool = False
    """``True`` when expected agreement is 1.0 and kappa is 0/0 - every item
    got the same label from both raters, so there is no variation to agree
    about. :attr:`kappa` is ``nan`` there, never 1.0."""

    target: float = KAPPA_TARGET

    @property
    def meets_target(self) -> bool:
        """Whether the judge agrees with the human well enough to be used."""
        return not self.degenerate and self.kappa >= self.target

    def __str__(self) -> str:
        if self.degenerate:
            return (
                f"kappa=undefined (every item identical, n={self.n}); "
                f"observed agreement {self.observed_agreement:.3f}"
            )
        verdict = "meets" if self.meets_target else "BELOW"
        return (
            f"kappa={self.kappa:.3f} ({verdict} target {self.target}), "
            f"Po={self.observed_agreement:.3f} Pe={self.expected_agreement:.3f}, n={self.n}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kappa": None if self.degenerate else self.kappa,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "n": self.n,
            "labels": list(self.labels),
            "confusion": [list(row) for row in self.confusion],
            "weights": self.weights,
            "degenerate": self.degenerate,
            "target": self.target,
            "meets_target": self.meets_target,
        }


def _weight(i: int, j: int, k: int, scheme: KappaWeights) -> float:
    """Agreement weight between label index ``i`` and ``j`` out of ``k``.

    1.0 on the diagonal, decreasing with ordinal distance. ``k == 1`` is the
    degenerate single-label case, where every pair is the diagonal.
    """
    if scheme == "none":
        return 1.0 if i == j else 0.0
    if k < 2:
        return 1.0
    distance = abs(i - j) / (k - 1)
    if scheme == "linear":
        return 1.0 - distance
    return 1.0 - distance**2


def cohens_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    weights: KappaWeights = "none",
    target: float = KAPPA_TARGET,
) -> KappaResult:
    """Cohen's kappa between two raters' label sequences.

    ``kappa = (Po - Pe) / (1 - Pe)`` where ``Po`` is observed agreement and
    ``Pe`` is the agreement expected from the two raters' marginal label
    distributions. Weighted variants replace the 0/1 agreement indicator with
    :func:`_weight`.

    Args:
        rater_a: one label per item.
        rater_b: the other rater's labels, same length and order.
        labels: the full label set. Defaults to the sorted union of labels
            actually used, which is the right default for ad-hoc label sets but
            *not* for the Beacon categories - pass
            :data:`ELIGIBILITY_LABELS` there so that an unused category still
            contributes its (zero) marginal.
        weights: see :data:`KappaWeights`; ``none`` is what gets reported.

    Raises:
        ValueError: on mismatched lengths, no items, or a label outside
            ``labels``.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"raters must label the same items: {len(rater_a)} vs {len(rater_b)}"
        )
    if not rater_a:
        raise ValueError("no items to compare")

    label_list = tuple(labels) if labels is not None else tuple(sorted(set(rater_a) | set(rater_b)))
    if len(set(label_list)) != len(label_list):
        raise ValueError(f"duplicate labels in label set: {label_list}")
    index = {label: i for i, label in enumerate(label_list)}
    unknown = sorted({lab for lab in (*rater_a, *rater_b) if lab not in index})
    if unknown:
        raise ValueError(f"labels not in the declared label set: {unknown}")

    k = len(label_list)
    n = len(rater_a)
    confusion = [[0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b, strict=True):
        confusion[index[a]][index[b]] += 1

    marginal_a = [sum(confusion[i]) / n for i in range(k)]
    marginal_b = [sum(confusion[i][j] for i in range(k)) / n for j in range(k)]

    observed = sum(
        confusion[i][j] * _weight(i, j, k, weights) for i in range(k) for j in range(k)
    ) / n
    expected = sum(
        marginal_a[i] * marginal_b[j] * _weight(i, j, k, weights)
        for i in range(k)
        for j in range(k)
    )

    degenerate = abs(1.0 - expected) < 1e-12
    kappa = float("nan") if degenerate else (observed - expected) / (1.0 - expected)
    return KappaResult(
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        n=n,
        labels=label_list,
        confusion=tuple(tuple(row) for row in confusion),
        weights=weights,
        degenerate=degenerate,
        target=target,
    )


def agreement_table(
    hand: Mapping[str, str], judge: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Align two label mappings by item id.

    Returns:
        ``(item_ids, hand_labels, judge_labels)`` over the items **both** have,
        sorted by item id so the pairing is reproducible.

    Raises:
        ValueError: when the two mappings do not cover the same items. Scoring
            kappa over an intersection chosen by accident is how a calibration
            set silently shrinks to the easy items.
    """
    if set(hand) != set(judge):
        only_hand = sorted(set(hand) - set(judge))[:5]
        only_judge = sorted(set(judge) - set(hand))[:5]
        raise ValueError(
            "kappa needs both raters on the same items; "
            f"{len(set(hand) - set(judge))} only hand-labelled (e.g. {only_hand}), "
            f"{len(set(judge) - set(hand))} only judged (e.g. {only_judge})"
        )
    ids = tuple(sorted(hand))
    return ids, tuple(hand[i] for i in ids), tuple(judge[i] for i in ids)


def load_hand_labels(path: Path) -> dict[str, EligibilityCategory]:
    """Load the owner's hand labels from JSONL.

    One object per line, ``{"item_id": "...", "label": "..."}``. Comment lines
    starting ``//`` and blank lines are skipped, so the template file can carry
    instructions.

    Raises:
        FileNotFoundError: if the file is absent.
        ValueError: on a bad line, an unknown label, or a duplicate item id.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no hand labels at {path}. Cohen's kappa compares the judge to a human; "
            "without hand labels there is nothing to calibrate against and the "
            "harness reports the judge as uncalibrated rather than inventing a number."
        )
    labels: dict[str, EligibilityCategory] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON: {line[:100]!r}") from exc
        if not isinstance(entry, dict) or "item_id" not in entry or "label" not in entry:
            raise ValueError(f"{path}:{lineno}: needs 'item_id' and 'label': {line[:100]!r}")
        item_id = str(entry["item_id"])
        label = entry["label"]
        if label not in ELIGIBILITY_LABELS:
            raise ValueError(
                f"{path}:{lineno}: label {label!r} is not one of {ELIGIBILITY_LABELS}"
            )
        if item_id in labels:
            raise ValueError(f"{path}:{lineno}: duplicate item_id {item_id!r}")
        labels[item_id] = label
    if not labels:
        raise ValueError(f"{path}: no hand labels found")
    return labels


def calibrate(
    items: Iterable[JudgeItem],
    hand_labels: Mapping[str, EligibilityCategory],
    judge: JudgeBackend,
    *,
    weights: KappaWeights = "none",
    target: float = KAPPA_TARGET,
) -> KappaResult:
    """Run a judge over hand-labelled items and measure agreement.

    The judge is called once per item through :class:`JudgeBackend`, so this
    function is identical whether the backend is :class:`FixtureJudge` (free,
    deterministic, what the tests use) or :class:`LlmJudge` (billable, not yet
    run).

    Raises:
        ValueError: when the items and the hand labels disagree about which
            items are in the calibration set.
    """
    item_list = list(items)
    verdicts = {item.item_id: judge.judge(item).label for item in item_list}
    _, hand, judged = agreement_table(dict(hand_labels), verdicts)
    return cohens_kappa(
        hand, judged, labels=ELIGIBILITY_LABELS, weights=weights, target=target
    )
