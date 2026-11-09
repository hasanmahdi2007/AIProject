"""NULL-safe Qdrant payload filters, built as data so they can be unit-tested.

Two independent correctness rules are encoded here, and both of them fail
*silently* when broken - no exception, no log line, just quietly wrong results.

**Rule 1: the filter goes inside the search, not after it.** This module only
builds the :class:`~qdrant_client.models.Filter`; :mod:`retrieval.search`
passes it to Qdrant as ``QueryRequest.filter`` so it constrains the HNSW graph
walk itself. Nothing in this service ever filters a result list. Taking the top
20 and then dropping the rows that fail the filter is the overfiltering bug:
with ``hnsw_ef=40`` and a filter that matches 10% of the corpus, the 40 points
the walk visited were chosen without reference to the filter, so about 4 of them
survive. You asked for 20 candidates, you got 4, and nothing raised.

**Rule 2: NULL means "possibly eligible", never "exclude".** The age and sex
fields on ClinicalTrials.gov are frequently absent ("N/A") or unparseable, and
B1 writes those as ``None``. A filter may exclude a trial only on *positive
evidence of ineligibility* - a parsed ``min_age_years`` of 65 against a
45-year-old patient. The naive clause

    must=[FieldCondition(key="min_age_years", range=Range(lte=45))]

is wrong, and wrong in the dangerous direction: a range condition in Qdrant can
only be satisfied by a point that *has* a comparable value there, so every
trial with an unknown lower age bound disappears. A wrongly excluded trial never
appears anywhere for anyone to notice. So each positive condition is wrapped in
a ``should`` group together with the two "no value here" conditions, which turns
"min age is at most 45" into "min age is at most 45, **or** we do not know the
min age".

Both ``IsNullCondition`` and ``IsEmptyCondition`` are included in that group.
They are not the same predicate - ``is_null`` matches a payload key whose value
is JSON ``null``, ``is_empty`` matches a key that is missing or holds an empty
array - and which one applies depends on whether the writer stored the key with
a null value or omitted it. B1 currently writes the key explicitly as ``None``,
but that is B1's choice to change, not a thing this service should depend on.
Listing both costs one extra disjunct and makes the filter correct either way.
``services/retrieval/tests/test_integration_qdrant.py`` asserts the NULL case
against a live Qdrant rather than trusting this reasoning.

What is allowed to be a filter at all: cheap, exact, structured facts. Age, sex,
and later ``status == "recruiting"``. **"No brain metastases" and "ECOG 0-1" are
not filters.** They are soft criteria over free text that the LLM rule checker
judges with evidence and an ``insufficient_info`` escape hatch. Promoting one to
a hard filter here would silently delete every good trial whose eligibility text
phrases it differently.
"""

from __future__ import annotations

from qdrant_client import models

from beacon_core.schemas import AgeSexFilter, Sex

__all__ = [
    "MAX_AGE_FIELD",
    "MIN_AGE_FIELD",
    "SEX_FIELD",
    "UNRESTRICTED_SEX",
    "build_filter",
    "nullable_condition",
]

# Payload keys, matching what B1's ``DerivedTrial.qdrant_payload()`` writes and
# what the live collection reports a payload index for. They are constants here
# rather than literals at the call sites so a rename is one edit, and the
# integration test asserts the live collection actually indexes these three.
MIN_AGE_FIELD = "min_age_years"
MAX_AGE_FIELD = "max_age_years"
SEX_FIELD = "sex"

UNRESTRICTED_SEX = "all"
"""The trial-side value for "no sex restriction" (``Sex`` in the contract).

A trial with ``sex == "all"`` is eligible for any patient, so it must be matched
*in addition to* the patient's own sex. Omitting it would delete the ~95% of
trials that enrol both sexes - the single most expensive possible version of
this bug.
"""


def nullable_condition(
    key: str,
    positive: models.Condition,
    *,
    include_unknown: bool,
) -> models.Condition:
    """Wrap a positive condition so that "no value at ``key``" also passes.

    Args:
        key: the payload field the positive condition reads.
        positive: the condition that constitutes positive evidence of
            *eligibility* - e.g. ``min_age_years <= 45``.
        include_unknown: when ``True`` (the contract default, and the only value
            that should be used in production), a point with no usable value at
            ``key`` passes. When ``False`` the condition is returned unwrapped,
            which is the strict variant kept only so the
            ``apply_age_sex_filters`` ablation can measure what NULL-tolerance
            costs - that row is allowed to come out negative.

    Returns:
        A nested :class:`Filter` acting as an OR group, or ``positive``
        unchanged. Qdrant's ``Condition`` union includes ``Filter``, so a
        nested filter is a legal member of an outer ``must`` list and is the
        documented way to express OR-inside-AND.
    """
    if not include_unknown:
        return positive
    return models.Filter(
        should=[
            positive,
            # Key present, value JSON null.
            models.IsNullCondition(is_null=models.PayloadField(key=key)),
            # Key absent, or present with an empty array.
            models.IsEmptyCondition(is_empty=models.PayloadField(key=key)),
        ]
    )


def _sex_condition(patient_sex: Sex) -> models.Condition:
    """Trial sex matches the patient's, or is unrestricted."""
    return models.FieldCondition(
        key=SEX_FIELD,
        match=models.MatchAny(any=[patient_sex, UNRESTRICTED_SEX]),
    )


def build_filter(filters: AgeSexFilter, *, enabled: bool = True) -> models.Filter | None:
    """Build the hard filter for one search, or ``None`` for "no constraint".

    ``None`` is returned - rather than an empty :class:`Filter` - when there is
    nothing to constrain, so Qdrant takes its unfiltered path instead of
    evaluating a vacuous filter on every visited point.

    Args:
        filters: the request's :class:`AgeSexFilter`. Fields left ``None`` mean
            the *patient's* value is unknown, which is also not grounds to
            exclude anything: if we do not know the patient's age we cannot have
            positive evidence that any trial's age range excludes them, so no
            age clause is emitted at all.
        enabled: ``settings.retrieval.apply_age_sex_filters``. ``False`` is the
            "no hard filters" ablation row and must be reachable by config
            alone.

    Returns:
        ``Filter(must=[...])`` where each element is one AND-ed clause:

        * age -> two clauses, because the two bounds are independently
          unknown: a trial can have ``min_age_years == 18`` and
          ``max_age_years is None``, and it must pass on the strength of the
          bound it does have. One combined clause would make an unknown upper
          bound discard a known lower bound.
        * sex -> one clause.

        Note what is *not* here: no clause excludes a trial for having a NULL,
        and ``must_not`` is never used. ``must_not`` is the tempting shape -
        "exclude trials whose min age is above the patient" - and it is subtly
        different: ``must_not(range(gt=45))`` happens to pass NULLs, so it looks
        equivalent, but it also passes points whose value is a *non-numeric*
        type and it inverts the reading of the clause, so a later edit that adds
        a second condition to the group silently changes which trials survive.
        A positive condition OR-ed with the unknown cases says what is meant.
    """
    if not enabled:
        return None

    clauses: list[models.Condition] = []

    if filters.age is not None:
        age = float(filters.age)
        # The trial's floor is at or below the patient's age...
        clauses.append(
            nullable_condition(
                MIN_AGE_FIELD,
                models.FieldCondition(key=MIN_AGE_FIELD, range=models.Range(lte=age)),
                include_unknown=filters.include_unknown,
            )
        )
        # ...and its ceiling is at or above it. Inclusive on both sides: an
        # "18 to 65 Years" trial enrols a 65-year-old.
        clauses.append(
            nullable_condition(
                MAX_AGE_FIELD,
                models.FieldCondition(key=MAX_AGE_FIELD, range=models.Range(gte=age)),
                include_unknown=filters.include_unknown,
            )
        )

    # ``all`` on the patient side is not a sex, it is the trial-side value for
    # "unrestricted". A request carrying it tells us nothing about the patient,
    # so it cannot be evidence that any trial excludes them.
    if filters.sex is not None and filters.sex != UNRESTRICTED_SEX:
        clauses.append(
            nullable_condition(
                SEX_FIELD,
                _sex_condition(filters.sex),
                include_unknown=filters.include_unknown,
            )
        )

    if not clauses:
        return None
    return models.Filter(must=clauses)
