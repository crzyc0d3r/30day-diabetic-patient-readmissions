"""Fairness / protected-attribute utilities.

Two distinct concepts share a name across the codebase:

1. **Protected raw input columns**: the allowlist of dataset columns
   (race, gender, age, weight, diag_1..3, payer_code, medical_specialty,
   patient_nbr) that a perturbation plan or feature transform is not
   allowed to target without explicit operator opt-in.

2. **Protected feature-name prefixes**: the prefix set used to detect
   features derived from protected attributes (e.g. one-hot-encoded
   'race=Caucasian', the binned 'age_band') so a feature-importance
   audit can exclude them from "top-10 most influential" lists. Used by
   'pipeline/08_conclusion.ipynb' §8.x proxy audit.

These are different tests and must not be conflated. A column called
'race' is a protected raw attribute, while a feature called 'age_band=80+'
is derived from a protected attribute. Keeping both helpers in one
module keeps the protected-attribute concept in one place while still
letting each caller pick the right semantic.

The Cramér's V helper rounds out the fairness audit's chi-squared
proxy-association test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# (1) raw input columns that are protected attributes themselves
# ---------------------------------------------------------------------------

PROTECTED_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
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
)


def is_protected_attribute(column: str | None) -> bool:
    """Return True if 'column' is on the fairness-guardrail allowlist.

    Matching is case-insensitive and exact for most columns, with a
    substring rule for the 'diag_*' family so re-encoded variants
    like 'diag_1_grouped' are still caught.
    """
    if not column:
        return False
    name = column.strip().lower()
    for protected in PROTECTED_ATTRIBUTE_COLUMNS:
        p = protected.lower()
        if name == p:
            return True
        # diag_* family: catch any column whose name starts with the prefix.
        if p.startswith("diag_") and name.startswith(p):
            return True
    return False


# ---------------------------------------------------------------------------
# (2) feature-name prefixes derived from protected attributes
# ---------------------------------------------------------------------------

PROTECTED_FEATURE_PREFIXES: tuple[str, ...] = (
    "age_band",
    "age_mid",
    "age=",
    "age_",
    "race",
    "gender",
)


def is_protected_feature(feature_name: str) -> bool:
    """Return True if 'feature_name' looks like it was derived from a
    protected attribute. One-hot expansions of race/gender, binned
    age fields, and raw age columns are all matched.

    Used by NB08's proxy-audit to exclude tautological associations
    (the protected attribute against itself) from the top-N
    feature-importance audit input.
    """
    s = str(feature_name).lower()
    return any(s == p or s.startswith(p) for p in PROTECTED_FEATURE_PREFIXES)


# ---------------------------------------------------------------------------
# Cramér's V: chi-squared association between two categorical series
# ---------------------------------------------------------------------------


def cramers_v(a, b) -> float:
    """Cramér's V between two categorical series.

    Returns NaN on an empty contingency table or a degenerate
    `min(r, k) - 1 == 0` denominator so callers can filter without
    catching exceptions. NB08's proxy-audit uses this against
    discretised numeric features to detect categorical back-doors to
    protected attributes.
    """
    cont = pd.crosstab(a, b)
    if cont.size == 0:
        return float("nan")
    chi2 = chi2_contingency(cont, correction=False)[0]
    n_obs = cont.values.sum()
    r, k = cont.shape
    denom = n_obs * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")
