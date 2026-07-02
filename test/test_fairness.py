"""Unit tests for `helpers.fairness`.

The fairness module hosts two deliberately separate notions that share the
word "protected": raw input columns that are themselves protected
attributes, and feature-name prefixes derived from protected attributes.
Conflating them would let a perturbation plan poke a protected column, or
let a proxy audit wave a tautological association through. These tests pin
each predicate independently, lock the two published constant tuples, and
hand-verify the Cramer's V proxy-association statistic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers.fairness import (
    PROTECTED_ATTRIBUTE_COLUMNS,
    PROTECTED_FEATURE_PREFIXES,
    cramers_v,
    is_protected_attribute,
    is_protected_feature,
)


# Published constants


def test_protected_attribute_columns_membership():
    """The raw-column allowlist must contain every documented member.

    The drift-simulation perturbation plans use this tuple to decide which
    columns a perturbation plan may not touch. If a column silently dropped out
    of the allowlist, a fairness-sensitive field could be perturbed without
    operator opt-in, so we assert the full documented set is present.
    """
    expected = {
        "race",
        "gender",
        "age",
        "weight",
        "diag_1",
        "diag_2",
        "diag_3",
        "payer_code",
        "medical_specialty",
        "patient_nbr",
    }
    assert set(PROTECTED_ATTRIBUTE_COLUMNS) == expected


def test_protected_feature_prefixes_membership():
    """The feature-prefix set must contain every documented prefix.

    NB08's proxy audit uses these prefixes to exclude tautological
    feature-importance entries. A missing prefix would let a one-hot
    expansion of a protected attribute count as a top influential feature
    against itself.
    """
    expected = {
        "age_band",
        "age_mid",
        "age=",
        "age_",
        "race",
        "gender",
    }
    assert set(PROTECTED_FEATURE_PREFIXES) == expected


def test_protected_constants_are_tuples():
    """Both published constants must stay immutable tuples.

    They are module-level allowlists imported across the codebase. Exporting
    a mutable list would invite a caller to mutate the shared guardrail in
    place, so the tuple type is itself part of the contract.
    """
    assert isinstance(PROTECTED_ATTRIBUTE_COLUMNS, tuple)
    assert isinstance(PROTECTED_FEATURE_PREFIXES, tuple)


# is_protected_attribute


@pytest.mark.parametrize(
    "column",
    [
        "race",
        "gender",
        "age",
        "weight",
        "payer_code",
        "medical_specialty",
        "patient_nbr",
        # Case-insensitive: mixed and upper case must still match.
        "RACE",
        "Gender",
        "Age",
        # diag_* family exact members.
        "diag_1",
        "diag_2",
        "diag_3",
        # diag_* substring rule: re-encoded variants still caught.
        "diag_1_grouped",
        "diag_2_icd9",
        "DIAG_3_BUCKET",
    ],
)
def test_is_protected_attribute_positive(column):
    """Allowlisted columns (and diag_* variants) must be flagged protected.

    This is the guardrail's positive path. Exact members match
    case-insensitively, and the diag_* substring rule has to catch re-encoded
    columns such as `diag_1_grouped` that share the protected prefix.
    """
    assert is_protected_attribute(column) is True


@pytest.mark.parametrize(
    "column",
    [
        "time_in_hospital",
        "num_medications",
        "readmitted",
        "encounter_id",
        # "diagnosis" shares letters with "diag" but not the "diag_" prefix.
        "diagnosis",
        "diag",
        "ageband",  # no exact match and not a diag_* member
    ],
)
def test_is_protected_attribute_negative(column):
    """Ordinary feature columns must not be flagged protected.

    The negative path matters as much as the positive one: over-flagging
    would block legitimate perturbations. `diag` and `diagnosis` must
    stay unprotected because neither equals an allowlist entry nor starts
    with the literal `diag_` prefix.
    """
    assert is_protected_attribute(column) is False


@pytest.mark.parametrize("value", [None, ""])
def test_is_protected_attribute_falsy_input(value):
    """None or empty string must return False rather than raising.

    Callers feed column names pulled from possibly-empty config, so the
    predicate guards against falsy input up front. A crash here would take
    down the whole perturbation-plan validation.
    """
    assert is_protected_attribute(value) is False


def test_is_protected_attribute_strips_whitespace():
    """Surrounding whitespace must not defeat the match.

    Column names arriving from CSV headers or hand-edited config can carry
    stray padding. The predicate lower-cases and strips, so a padded
    protected name still has to be caught.
    """
    assert is_protected_attribute("  Race  ") is True


# is_protected_feature


@pytest.mark.parametrize(
    "feature_name",
    [
        "age_band",
        "age_band=80+",
        "age_mid",
        "age_mid_value",
        "age=70-80",
        "age_",
        "age_years",
        "race",
        "race=Caucasian",
        "gender",
        "gender=Female",
        # Case-insensitive matching.
        "Age_Band",
        "RACE=AfricanAmerican",
    ],
)
def test_is_protected_feature_positive(feature_name):
    """Features derived from protected attributes must be flagged.

    The proxy audit relies on this to drop one-hot expansions, binned age
    fields, and raw age columns from the top-N influential list. Each prefix
    in PROTECTED_FEATURE_PREFIXES has to be matched by equality or as a
    prefix, case-insensitively.
    """
    assert is_protected_feature(feature_name) is True


@pytest.mark.parametrize(
    "feature_name",
    [
        "num_medications",
        "time_in_hospital",
        "weight",  # protected raw column but NOT a feature prefix here
        "diag_1",  # protected column, not in the feature-prefix set
        "averaged",  # starts with "age"? no, starts with "ave" -> must not match
        "agile_score",  # "agi" prefix, not "age"
    ],
)
def test_is_protected_feature_negative(feature_name):
    """Non-derived features must not be flagged by the prefix rule.

    The feature-prefix set is intentionally narrower than the raw-column
    allowlist. `weight` and `diag_1` are protected raw columns but are not
    in PROTECTED_FEATURE_PREFIXES, so the feature predicate must leave them
    alone. `agile_score` guards against a naive `age` substring match.
    """
    assert is_protected_feature(feature_name) is False


# cramers_v


def test_cramers_v_perfect_association_2x2():
    """A perfectly associated 2x2 table must give Cramer's V == 1.

    For a 2x2 contingency where each level of a maps one-to-one onto a level
    of b, the association is total. With min(r, k) - 1 == 1 the normalisation
    reduces to sqrt(chi2 / n), and a perfect association drives that to 1.0.
    This is the anchor value for the proxy-association statistic.
    """
    # Ten rows: a and b move in lockstep -> perfect association.
    a = ["x"] * 5 + ["y"] * 5
    b = ["p"] * 5 + ["q"] * 5

    assert cramers_v(a, b) == pytest.approx(1.0)


def test_cramers_v_hand_computed_partial_association():
    """A known 2x2 table must reproduce the hand-computed Cramer's V.

    Table (correction=False):
        a\\b   p   q
         x     8   2
         y     2   8
    Row and column totals are all 10, n = 20. The uncorrected chi-squared for
    this symmetric table is 7.2, so V = sqrt(7.2 / (20 * 1)) = sqrt(0.36) =
    0.6. Pinning this exact value catches any drift in how the helper
    normalises chi-squared.
    """
    a = ["x"] * 10 + ["y"] * 10
    # Within the x block: 8 p then 2 q. Within the y block: 2 p then 8 q.
    b = (["p"] * 8 + ["q"] * 2) + (["p"] * 2 + ["q"] * 8)

    assert cramers_v(a, b) == pytest.approx(0.6, abs=1e-9)


def test_cramers_v_is_symmetric():
    """Cramer's V must be symmetric in its two arguments.

    The statistic measures association, which is a symmetric relation.
    Swapping a and b transposes the contingency table without changing
    chi-squared or min(r, k), so V(a, b) must equal V(b, a). The proxy audit
    sometimes feeds arguments in either order, so this invariant has to hold.
    """
    a = ["x"] * 8 + ["y"] * 7 + ["z"] * 5
    b = (["p", "q", "p", "q", "p", "q", "p", "q"]
         + ["q", "q", "p", "p", "q", "p", "q"]
         + ["p", "p", "q", "q", "p"])

    forward = cramers_v(a, b)
    backward = cramers_v(b, a)

    assert forward == pytest.approx(backward)


def test_cramers_v_empty_input_is_nan():
    """An empty contingency table must return NaN, not raise.

    The helper is documented to return NaN so callers can filter without
    wrapping every call in a try/except. Two empty series produce a
    zero-size crosstab, which must short-circuit to NaN.
    """
    empty = np.array([], dtype=object)

    assert math.isnan(cramers_v(empty, empty))


def test_cramers_v_degenerate_single_level_is_nan():
    """A single-level series gives a degenerate denominator -> NaN.

    When b has only one distinct level the table has k == 1, so
    min(r, k) - 1 == 0 and the denominator is zero. The helper must return
    NaN in that degenerate case rather than dividing by zero, so the proxy
    audit can drop the column cleanly.
    """
    a = ["x", "y", "x", "y", "x", "y"]
    b = ["p", "p", "p", "p", "p", "p"]  # one level only

    assert math.isnan(cramers_v(a, b))


def test_cramers_v_returns_python_float():
    """The statistic must be a plain Python float for clean serialisation.

    The proxy-audit table is rendered and logged, and a numpy float would
    serialise differently. Pinning the concrete type keeps the audit output
    consistent across callers.
    """
    a = ["x"] * 6 + ["y"] * 6
    b = ["p"] * 6 + ["q"] * 6

    result = cramers_v(a, b)

    assert type(result) is float
