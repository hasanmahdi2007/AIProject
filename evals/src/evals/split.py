"""The tune/held-out split, enforced by the type system rather than by habit.

2021 tunes. 2022 is measured once, at the end.
-----------------------------------------------
Every threshold, weight, fusion constant and value of N is chosen on the **75
TREC 2021 topics**. The **50 TREC 2022 topics** are touched once, to produce
the headline number, after tuning is finished. Looking at 2022 while tuning -
even once, even informally, even "just to see" - turns the held-out number into
a training number and quietly destroys the only claim the evaluation makes.

Convention does not survive a tired evening, so this module makes the mistake
*structurally hard*:

1. :func:`load_tuning_split` takes no arguments and returns 2021. It is the
   path of least resistance, which is the point.
2. :func:`load_held_out_split` **requires** a :class:`HeldOutUnlock` as a
   positional argument. There is no default, so under ``mypy strict`` a call
   that omits it fails type-checking, not merely at runtime.
3. The only way to build a :class:`HeldOutUnlock` is :func:`unlock_held_out`,
   which demands a written reason of at least
   :data:`MIN_UNLOCK_REASON_CHARS` characters. You cannot produce one by
   accident, and you cannot produce one without saying why.
4. Every unlock is appended to :data:`HELD_OUT_ACCESS_LOG` and logged at
   WARNING. :mod:`evals.report` prints the log, so a report built on the
   held-out set says so on its face, with the reason attached.
5. :func:`assert_tuning_only` refuses a non-2021 split outright. The query
   count sweep and every ablation-tuning entry point call it, so "sweep N on
   the test set" raises instead of running.

What this cannot stop: someone calling :func:`unlock_held_out` with the reason
"because" and iterating against 2022. Nothing in code can stop that. What it
can do - and does - is ensure the act is deliberate, attributable and visible
in the output.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, Literal

from evals.trec import Qrels, Topic, load_qrels, load_topics

__all__ = [
    "EXPECTED_TOPIC_COUNTS",
    "HELD_OUT_ACCESS_LOG",
    "HELD_OUT_YEAR",
    "MIN_UNLOCK_REASON_CHARS",
    "TUNE_YEAR",
    "EvalSplit",
    "HeldOutAccessError",
    "HeldOutUnlock",
    "SplitName",
    "assert_tuning_only",
    "load_held_out_split",
    "load_tuning_split",
    "unlock_held_out",
]

logger = logging.getLogger(__name__)

TUNE_YEAR: Final = 2021
HELD_OUT_YEAR: Final = 2022

EXPECTED_TOPIC_COUNTS: Final[dict[int, int]] = {2021: 75, 2022: 50}
"""Topic counts from the track overviews, re-verified by A4 against the files.

Checked on load. A truncated or half-downloaded topics file would otherwise
produce a plausible-looking number over 40 topics, and nothing downstream would
notice.
"""

MIN_UNLOCK_REASON_CHARS: Final = 20

SplitName = Literal["tune", "held-out"]
"""The two full splits. ``EvalSplit.name`` is a plain ``str`` because
:meth:`EvalSplit.subset` produces names outside this pair on purpose."""

FULL_SPLIT_NAMES: Final[frozenset[str]] = frozenset({"tune", "held-out"})
"""Names whose topic count is validated against the track. A split named
anything else is a subset and is not a publishable measurement."""


class HeldOutAccessError(RuntimeError):
    """Raised when the held-out split is reached for without an unlock."""


@dataclass(frozen=True, slots=True)
class HeldOutUnlock:
    """A capability token that permits one read of the 2022 labels.

    Construct it only through :func:`unlock_held_out`. It exists to be a
    *required argument* of :func:`load_held_out_split`, so that reading the
    held-out labels is a visible act at the call site and a type error to
    forget.
    """

    reason: str
    requested_at: datetime

    def __post_init__(self) -> None:
        if len(self.reason.strip()) < MIN_UNLOCK_REASON_CHARS:
            raise HeldOutAccessError(
                "unlocking the held-out 2022 split needs a written reason of at "
                f"least {MIN_UNLOCK_REASON_CHARS} characters (got "
                f"{len(self.reason.strip())}). State what is finished being tuned "
                "on 2021 and why the headline measurement is being taken now."
            )


HELD_OUT_ACCESS_LOG: Final[list[HeldOutUnlock]] = []
"""Every held-out unlock taken in this process, in order.

Read by :mod:`evals.report`, which prints it in the report header. A report
whose log is empty was built on tuning data only; one with entries carries the
reasons.
"""


def unlock_held_out(reason: str) -> HeldOutUnlock:
    """Deliberately permit a read of the held-out 2022 labels.

    Args:
        reason: at least :data:`MIN_UNLOCK_REASON_CHARS` characters saying why
            tuning is finished. It is recorded in the report.

    Raises:
        HeldOutAccessError: if the reason is too short to be a real one.
    """
    unlock = HeldOutUnlock(reason=reason.strip(), requested_at=datetime.now(UTC))
    HELD_OUT_ACCESS_LOG.append(unlock)
    logger.warning(
        "HELD-OUT SPLIT UNLOCKED: reading the TREC %d labels. Reason: %s. "
        "Every configuration choice must already be frozen on %d.",
        HELD_OUT_YEAR,
        unlock.reason,
        TUNE_YEAR,
    )
    return unlock


@dataclass(frozen=True, slots=True)
class EvalSplit:
    """Topics and graded qrels for one track year.

    Validated on construction: the topic count must match
    :data:`EXPECTED_TOPIC_COUNTS` and every topic must carry judgements, since
    a topic present in the run but absent from the qrels is dropped by every
    scorer in silence.
    """

    name: str
    year: int
    topics: tuple[Topic, ...]
    qrels: Qrels = field(repr=False)

    def __post_init__(self) -> None:
        if not self.topics:
            raise ValueError(f"{self.name}: a split with no topics scores nothing")
        expected = EXPECTED_TOPIC_COUNTS.get(self.year)
        if self.name in FULL_SPLIT_NAMES and expected is not None and len(self.topics) != expected:
            raise ValueError(
                f"TREC {self.year} must have {expected} topics, loaded "
                f"{len(self.topics)}. A partial topics file would skew every "
                "number computed from it; clear the ir_datasets cache and retry."
            )
        unjudged = sorted(t.topic_id for t in self.topics if not self.qrels.get(t.topic_id))
        if unjudged:
            raise ValueError(
                f"TREC {self.year}: {len(unjudged)} topic(s) have no judgements "
                f"(e.g. {unjudged[:5]}); they would be scored against nothing."
            )

    @property
    def topic_ids(self) -> tuple[str, ...]:
        return tuple(t.topic_id for t in self.topics)

    @property
    def is_held_out(self) -> bool:
        return self.year == HELD_OUT_YEAR

    def grade_counts(self) -> dict[int, int]:
        """How many judgements fall at each grade, for the report's provenance."""
        counts: dict[int, int] = {}
        for judged in self.qrels.values():
            for grade in judged.values():
                counts[grade] = counts.get(grade, 0) + 1
        return dict(sorted(counts.items()))

    def subset(self, topic_ids: Sequence[str]) -> EvalSplit:
        """A split restricted to some topics, for a cheap smoke run.

        The result is named ``<name>-subset``, which both exempts it from the
        topic-count validation and marks it in every report it reaches: a
        number measured over 5 of 75 topics must never be mistaken for the
        split's number.
        """
        wanted = set(topic_ids)
        topics = tuple(t for t in self.topics if t.topic_id in wanted)
        missing = wanted - {t.topic_id for t in topics}
        if missing:
            raise ValueError(f"no such topics in {self.name}: {sorted(missing)[:5]}")
        return EvalSplit(
            name=f"{self.name}-subset",
            year=self.year,
            topics=topics,
            qrels={t.topic_id: self.qrels[t.topic_id] for t in topics},
        )

    @property
    def is_full_split(self) -> bool:
        """Whether this is a complete track split and so publishable."""
        return self.name in FULL_SPLIT_NAMES


def _load(name: SplitName, year: int) -> EvalSplit:
    topics = tuple(load_topics(year))
    qrels = load_qrels(year)
    return EvalSplit(name=name, year=year, topics=topics, qrels=qrels)


def load_tuning_split() -> EvalSplit:
    """The 75 TREC 2021 topics and their graded qrels.

    Takes no arguments, because there is only one thing to tune on.
    """
    return _load("tune", TUNE_YEAR)


def load_held_out_split(unlock: HeldOutUnlock) -> EvalSplit:
    """The 50 TREC 2022 topics and their graded qrels. Measured once.

    Args:
        unlock: a token from :func:`unlock_held_out`. Required, with no
            default, so that forgetting it is a ``mypy`` error rather than a
            quiet default to the test set.

    Raises:
        HeldOutAccessError: if ``unlock`` is not a real token. Guarded at
            runtime as well as in the types, because the CLI builds its
            arguments from strings and loses the static check.
    """
    if not isinstance(unlock, HeldOutUnlock):
        raise HeldOutAccessError(
            "load_held_out_split() needs a HeldOutUnlock from unlock_held_out(reason=...); "
            f"got {type(unlock).__name__}. The 2022 labels are the headline measurement "
            "and are not available to tuning code."
        )
    return _load("held-out", HELD_OUT_YEAR)


def assert_tuning_only(split: EvalSplit, *, what: str = "this procedure") -> None:
    """Refuse to run a tuning procedure on anything but 2021.

    Called by the query-count sweep and by every ablation path whose output is
    used to *choose* a configuration. The sweep picking N on the held-out set
    would be the single most expensive mistake available here, and it would
    leave no trace in the numbers.

    Raises:
        HeldOutAccessError: if ``split`` is not the 2021 tuning split.
    """
    if split.year != TUNE_YEAR:
        raise HeldOutAccessError(
            f"{what} selects a configuration and must run on the TREC {TUNE_YEAR} "
            f"tuning split; got {split.name!r} (TREC {split.year}). Tuning on the "
            "held-out split makes the headline number meaningless."
        )
