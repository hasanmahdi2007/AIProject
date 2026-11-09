"""NULL-safe filter construction.

**What these tests do and do not prove.** They assert the *shape* of the filter
handed to Qdrant: that every positive condition is OR-ed with the two
"no value here" conditions, that ``must_not`` is never used, that the bounds are
inclusive. They cannot prove Qdrant then *interprets* that shape as intended -
only Qdrant can say that, and
``test_retrieval_integration_qdrant.py::TestNullSafety`` asserts it against a live engine
with a real NULL-age point. Both halves are necessary: the structural test
catches a wrong filter being built, the integration test catches a correct
filter being understood differently than assumed.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from beacon_core.schemas import AgeSexFilter
from retrieval.filters import (
    MAX_AGE_FIELD,
    MIN_AGE_FIELD,
    SEX_FIELD,
    build_filter,
    nullable_condition,
)


def _clauses(built: models.Filter | None) -> list[models.Condition]:
    assert built is not None
    assert built.must is not None
    assert isinstance(built.must, list)
    assert built.must_not is None, "must_not is never the right shape here"
    assert built.should is None, "top level is a conjunction of clauses"
    return built.must


def _or_group(clause: models.Condition) -> list[models.Condition]:
    """A clause that tolerates unknowns is a nested Filter with a ``should``."""
    assert isinstance(clause, models.Filter), f"expected an OR group, got {type(clause).__name__}"
    assert clause.should is not None
    assert isinstance(clause.should, list)
    assert clause.must is None
    assert clause.must_not is None
    return clause.should


def _keys_of_unknown_conditions(group: list[models.Condition]) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for condition in group:
        if isinstance(condition, models.IsNullCondition):
            found.add(("is_null", condition.is_null.key))
        elif isinstance(condition, models.IsEmptyCondition):
            found.add(("is_empty", condition.is_empty.key))
    return found


class TestContractDefault:
    def test_include_unknown_defaults_to_true(self) -> None:
        """The frozen contract's default, re-asserted here because the whole
        NULL-safety story rests on it and a flipped default would be a one-word
        diff in another package with no local test to catch it."""
        assert AgeSexFilter().include_unknown is True

    def test_an_empty_filter_constrains_nothing(self) -> None:
        """No patient age and no patient sex is not grounds to exclude anything.

        ``None`` rather than an empty ``Filter``, so Qdrant takes its unfiltered
        path instead of evaluating a vacuous condition on every visited point.
        """
        assert build_filter(AgeSexFilter()) is None


class TestAgeIsNullSafe:
    FILTERS = AgeSexFilter(age=45)

    def test_two_clauses_one_per_bound(self) -> None:
        """The bounds are independently unknown, so they are independent clauses.

        A trial can have ``min_age_years=18`` and ``max_age_years=None``. One
        combined clause would let the unknown upper bound discard the known
        lower bound - or, worse, let the known lower bound make the unknown
        upper bound look like evidence.
        """
        clauses = _clauses(build_filter(self.FILTERS))

        assert len(clauses) == 2

    def test_each_bound_tolerates_a_missing_value(self) -> None:
        min_clause, max_clause = _clauses(build_filter(self.FILTERS))

        assert _keys_of_unknown_conditions(_or_group(min_clause)) == {
            ("is_null", MIN_AGE_FIELD),
            ("is_empty", MIN_AGE_FIELD),
        }
        assert _keys_of_unknown_conditions(_or_group(max_clause)) == {
            ("is_null", MAX_AGE_FIELD),
            ("is_empty", MAX_AGE_FIELD),
        }

    def test_the_positive_conditions_are_inclusive_ranges(self) -> None:
        """``lte``/``gte``, not ``lt``/``gt``: an "18 to 65 Years" trial enrols a
        65-year-old, and a 45-year-old is eligible for a "45 years and older"
        trial. Strict bounds would delete exactly the boundary patients."""
        min_clause, max_clause = _clauses(build_filter(self.FILTERS))

        lower = _or_group(min_clause)[0]
        upper = _or_group(max_clause)[0]
        assert isinstance(lower, models.FieldCondition)
        assert isinstance(upper, models.FieldCondition)
        assert lower.key == MIN_AGE_FIELD
        assert upper.key == MAX_AGE_FIELD
        assert lower.range == models.Range(lte=45.0)
        assert upper.range == models.Range(gte=45.0)

    def test_no_clause_anywhere_uses_must_not(self) -> None:
        """``must_not(range(gt=45))`` also happens to pass NULLs, which is why
        it is tempting. It is rejected because it inverts the reading of the
        clause: a later edit adding a second condition to the group changes
        which trials survive, in the silent direction."""
        for clause in _clauses(build_filter(AgeSexFilter(age=45, sex="female"))):
            assert isinstance(clause, models.Filter)
            assert clause.must_not is None

    def test_a_patient_with_no_age_gets_no_age_clause(self) -> None:
        """Unknown on the *patient* side is also not evidence: without an age we
        cannot know that any trial's range excludes them."""
        clauses = _clauses(build_filter(AgeSexFilter(sex="female")))

        assert len(clauses) == 1
        group = _or_group(clauses[0])
        assert all(
            not (isinstance(c, models.FieldCondition) and c.key.endswith("age_years"))
            for c in group
        )


class TestSexIsNullSafe:
    def test_matches_the_patient_sex_and_the_unrestricted_value(self) -> None:
        """Omitting ``all`` would delete the ~95% of trials that enrol both
        sexes - the most expensive possible version of this bug."""
        clauses = _clauses(build_filter(AgeSexFilter(sex="female")))

        condition = _or_group(clauses[0])[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == SEX_FIELD
        assert condition.match == models.MatchAny(any=["female", "all"])

    def test_tolerates_a_missing_sex(self) -> None:
        clauses = _clauses(build_filter(AgeSexFilter(sex="male")))

        assert _keys_of_unknown_conditions(_or_group(clauses[0])) == {
            ("is_null", SEX_FIELD),
            ("is_empty", SEX_FIELD),
        }

    def test_a_patient_sex_of_all_is_not_a_constraint(self) -> None:
        """``all`` is the *trial*-side value for "unrestricted". On the patient
        side it says nothing about the patient, so it cannot be evidence that
        any trial excludes them."""
        assert build_filter(AgeSexFilter(sex="all")) is None

    def test_a_patient_with_no_sex_gets_no_sex_clause(self) -> None:
        clauses = _clauses(build_filter(AgeSexFilter(age=30)))

        assert len(clauses) == 2


class TestStrictMode:
    """``include_unknown=False`` - the ablation variant, never the default."""

    def test_drops_the_unknown_branches(self) -> None:
        clauses = _clauses(build_filter(AgeSexFilter(age=45, sex="male", include_unknown=False)))

        assert len(clauses) == 3
        for clause in clauses:
            assert isinstance(clause, models.FieldCondition), (
                "strict mode emits bare conditions, with no tolerance for unknowns"
            )

    def test_nullable_condition_passes_through_when_strict(self) -> None:
        positive = models.FieldCondition(key="x", range=models.Range(lte=1.0))

        assert nullable_condition("x", positive, include_unknown=False) is positive


class TestDisabled:
    def test_apply_age_sex_filters_false_removes_every_filter(self) -> None:
        """The "no hard filters" ablation row, reachable by config alone."""
        assert build_filter(AgeSexFilter(age=45, sex="female"), enabled=False) is None


class TestWholeFilter:
    @pytest.mark.parametrize(
        ("filters", "expected_clauses"),
        [
            (AgeSexFilter(), 0),
            (AgeSexFilter(age=45), 2),
            (AgeSexFilter(sex="female"), 1),
            (AgeSexFilter(age=45, sex="female"), 3),
            (AgeSexFilter(age=0), 2),
        ],
    )
    def test_clause_count(self, filters: AgeSexFilter, expected_clauses: int) -> None:
        built = build_filter(filters)

        if expected_clauses == 0:
            assert built is None
        else:
            assert len(_clauses(built)) == expected_clauses

    def test_age_zero_is_a_real_age_and_not_a_missing_one(self) -> None:
        """A newborn is age 0. ``if filters.age:`` would silently drop the
        filter for them, which is the classic falsy-zero bug and here it means a
        neonatal patient is matched against adult-only trials."""
        clauses = _clauses(build_filter(AgeSexFilter(age=0)))

        lower = _or_group(clauses[0])[0]
        assert isinstance(lower, models.FieldCondition)
        assert lower.range == models.Range(lte=0.0)
