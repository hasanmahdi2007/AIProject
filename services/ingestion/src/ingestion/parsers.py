"""Age and sex parsing, plus phase/status normalisation.

The one rule that matters
-------------------------
**An age that cannot be parsed is ``None``, never a number.** ``None`` reaches
Qdrant as a payload field with no value, which B2's filters treat as *possibly
eligible*, so the trial still surfaces and a human can rule on it. A guessed
``0`` or ``999`` instead makes the trial fail or pass a numeric filter
silently, and a wrongly excluded trial never appears anywhere for anybody to
notice. That is why :func:`parse_age` returns ``None`` on anything it does not
recognise and the caller stores the original string in ``raw_min_age`` /
``raw_max_age`` alongside.

What the corpus actually contains
---------------------------------
Measured over all 375,580 records of ``clinicaltrials/2021`` on 2026-11-08 by
regex-scanning ``<minimum_age>`` and ``<maximum_age>`` in the source XML. Both
fields are present for every record that has an ``<eligibility>`` element, and
**every** value is either ``N/A`` or ``<number> <unit>``; there are zero other
shapes. Pooling the two fields, the unit distribution is::

    years   523,514      year     2,667
    N/A     209,086      month    1,110
    months    8,453      day         73
    weeks     2,224      week        32
    days      1,735      hour        20
    hours       417      minute      14
    minutes      89

863 records carry no ``<eligibility>`` element at all, so for those both ages
and the sex are absent -- which is exactly the ``None`` case, not an error.

Sub-year units are not a curiosity: 8,453 ``months`` values are neonatal and
paediatric trials, and rounding ``6 Months`` up to 1 year or down to 0 would
misfile them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "AGE_PATTERN",
    "DAYS_PER_YEAR",
    "SEX_VALUES",
    "UNIT_YEARS",
    "AgeBounds",
    "normalise_phase",
    "normalise_status",
    "parse_age",
    "parse_age_bounds",
    "parse_sex",
]

DAYS_PER_YEAR = 365.25
"""Julian year. Chosen over 365 so that a 'days' bound and a 'years' bound of
the same real duration do not disagree at the leap-day boundary."""

UNIT_YEARS: dict[str, float] = {
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 7.0 / DAYS_PER_YEAR,
    "day": 1.0 / DAYS_PER_YEAR,
    "hour": 1.0 / (DAYS_PER_YEAR * 24.0),
    "minute": 1.0 / (DAYS_PER_YEAR * 24.0 * 60.0),
}
"""Conversion to years for every unit the corpus uses, singular stem.

Months are 1/12 of a year exactly rather than 30.44 days, so ``6 Months``
is 0.5 and not 0.5003 -- the value a human would hand-check against.
"""

AGE_PATTERN = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>year|month|week|day|hour|minute)s?$",
    re.IGNORECASE,
)
"""``"18 Years"``, ``"6 Months"``, ``"20 Weeks"``, ``"1 Year"``.

Anchored at both ends on purpose. A loose search would read ``"at least 18
Years"`` as 18 and ``"18-55 Years"`` as 18, inventing a bound the record never
stated; anchoring means such a value falls through to ``None`` and stays
visible in ``raw_min_age``.
"""

SEX_VALUES: dict[str, str] = {
    "all": "all",
    "both": "all",
    "male": "male",
    "female": "female",
}
"""The ``trials.sex`` check constraint allows ``male``/``female``/``all``/NULL.

The corpus only ever says ``All``, ``Female`` or ``Male`` (320,948 / 36,496 /
17,273, plus 863 records with no eligibility element). ``both`` is accepted
because older ClinicalTrials.gov exports used it and the mapping is
unambiguous; anything else returns ``None`` rather than being coerced.
"""

_NOT_APPLICABLE = {"n/a", "na", "none", "not applicable", "unknown", "-", ""}
"""Values that explicitly mean "no bound stated". They parse to ``None``, which
is the same outcome as an unrecognised value -- the distinction is kept in the
raw column, not in the typed one."""


@dataclass(frozen=True, slots=True)
class AgeBounds:
    """A trial's age window in years, with the source strings kept.

    ``min_years``/``max_years`` are ``None`` whenever the bound is absent,
    ``N/A`` or unparseable. ``unparsed`` names the fields that held a non-empty
    string the parser could not read, so an ingest run can report a real
    parse-failure rate instead of conflating it with ``N/A``.
    """

    min_years: float | None
    max_years: float | None
    raw_min: str | None
    raw_max: str | None
    unparsed: tuple[str, ...] = ()

    @property
    def inverted(self) -> bool:
        """True when the stated minimum exceeds the stated maximum.

        A data error in the source record. The bounds are kept as published --
        repairing them here would hide it -- and the ingest run counts them.
        """
        return (
            self.min_years is not None
            and self.max_years is not None
            and self.min_years > self.max_years
        )


def parse_age(raw: str | None) -> float | None:
    """Age in years, or ``None`` if the value states no usable bound.

    ``None`` is returned for a missing field, for an explicit ``N/A``, and for
    anything that does not match :data:`AGE_PATTERN`. The caller must store
    ``raw`` as well; see the module docstring.

    >>> parse_age("18 Years")
    18.0
    >>> parse_age("6 Months")
    0.5
    >>> parse_age("N/A") is None
    True
    >>> parse_age(None) is None
    True
    """
    if raw is None:
        return None
    text = raw.strip()
    if text.lower() in _NOT_APPLICABLE:
        return None
    match = AGE_PATTERN.match(text)
    if match is None:
        return None
    years = float(match["value"]) * UNIT_YEARS[match["unit"].lower()]
    # Round at the 6th decimal so "90 Days" is stable across platforms and a
    # hand-check of the Postgres row against the Qdrant payload compares equal.
    return round(years, 6)


def _is_unparsed(raw: str | None, parsed: float | None) -> bool:
    """True when a non-empty, non-``N/A`` string failed to parse."""
    if parsed is not None or raw is None:
        return False
    return raw.strip().lower() not in _NOT_APPLICABLE


def parse_age_bounds(raw_min: str | None, raw_max: str | None) -> AgeBounds:
    """Both bounds at once, with the raw strings and a parse-failure record."""
    min_years = parse_age(raw_min)
    max_years = parse_age(raw_max)
    unparsed: list[str] = []
    if _is_unparsed(raw_min, min_years):
        unparsed.append("min")
    if _is_unparsed(raw_max, max_years):
        unparsed.append("max")
    return AgeBounds(
        min_years=min_years,
        max_years=max_years,
        raw_min=raw_min,
        raw_max=raw_max,
        unparsed=tuple(unparsed),
    )


def parse_sex(raw: str | None) -> str | None:
    """``"male"``, ``"female"``, ``"all"`` or ``None``.

    ``None`` means the record did not say, which -- exactly as with age --
    must widen the candidate set rather than narrow it.
    """
    if raw is None:
        return None
    return SEX_VALUES.get(raw.strip().lower())


def normalise_phase(raw: str | None) -> str | None:
    """Trial phase, whitespace-normalised, or ``None`` when absent.

    ``"N/A"`` is **kept**, not mapped to ``None``: 127,333 records state it and
    it means "phase does not apply to this study" (observational and expanded
    access records), which is different from the 82,931 that simply have no
    ``<phase>`` element. Collapsing the two would destroy a distinction the
    source makes, and ``trials.phase`` is wide enough to hold it.
    """
    if raw is None:
        return None
    value = " ".join(raw.split())
    return value or None


def normalise_status(raw: str | None) -> str | None:
    """Recruitment status, whitespace-normalised, or ``None`` when absent.

    Kept verbatim (``"Completed"``, ``"Recruiting"``, ``"Active, not
    recruiting"``, ...). There are 13 distinct values across the corpus and no
    coarser bucketing is imposed here: B2 decides what counts as recruitable,
    and a lossy mapping made during ingestion could not be revisited without a
    full re-index.
    """
    if raw is None:
        return None
    value = " ".join(raw.split())
    return value or None
