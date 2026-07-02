"""Exploratory-analysis statistics: proportion confidence intervals, odds ratios,
per-group rates with CIs, and confounding-adjusted logistic regression.

NB03 uses these to put **effect sizes, confidence intervals, and adjusted
estimates** on the readmission EDA, replacing qualitative "modest / roughly
doubles" language with auditable numbers. Two design choices:

* Dependency-light: only numpy + scipy are required. ``adjusted_logit`` imports
  ``statsmodels`` lazily, so every other function works even if statsmodels is
  absent, and the notebook can degrade to unadjusted odds ratios with a printed
  note rather than hard-failing.
* Single source of truth: the readmission binary target is ``(readmitted ==
  "<30")`` everywhere, matching the rest of the pipeline. Categorical
  associations reuse ``helpers.fairness.cramers_v`` (not reimplemented here).

Per project conventions this file avoids em dashes and semicolons and uses the
spelling "program".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    Preferred over the normal (Wald) interval because it stays inside [0, 1] and
    behaves well at small n and near 0 or 1, which is exactly where thinly
    populated readmission cells live. Returns ``(lo, hi)``, or ``(nan, nan)``
    when ``n == 0``.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def odds_ratio_ci(a: float, b: float, c: float, d: float,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    """Odds ratio and 95% CI for a 2x2 table.

    Cell convention (exposure x outcome):
        a = exposed & event,    b = exposed & no event,
        c = unexposed & event,  d = unexposed & no event.
    OR = (a*d) / (b*c). A Haldane-Anscombe 0.5 continuity correction is applied
    when any cell is zero, so the OR and its log-scale CI stay finite. Returns
    ``(or_, ci_low, ci_high)``.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = float(norm.ppf(1 - alpha / 2))
    log_or = np.log(or_)
    return (float(or_), float(np.exp(log_or - z * se)), float(np.exp(log_or + z * se)))


def _as_binary(df: pd.DataFrame, col_or_series, name: str) -> pd.Series:
    """Coerce a column name or aligned series to an int 0/1 Series on df's index."""
    if isinstance(col_or_series, str):
        s = df[col_or_series]
    else:
        s = pd.Series(np.asarray(col_or_series), index=df.index, name=name)
    return s.astype(int)


def univariate_or(df: pd.DataFrame, target, indicator,
                  alpha: float = 0.05) -> dict:
    """Unadjusted odds ratio of one binary ``indicator`` (0/1 or bool) against a
    binary ``target``. Both may be a column name or an aligned series. Returns
    ``{or, ci_low, ci_high, n_exposed, n_unexposed}``."""
    y = _as_binary(df, target, "target")
    x = _as_binary(df, indicator, "indicator")
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    or_, lo, hi = odds_ratio_ci(a, b, c, d, alpha)
    return {"or": or_, "ci_low": lo, "ci_high": hi,
            "n_exposed": a + b, "n_unexposed": c + d}


def rate_with_ci(df: pd.DataFrame, group_col: str, target,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Per-group event rate with a Wilson CI and the group N.

    ``target`` is a column name or an aligned 0/1 series. Returns a DataFrame
    indexed by group with columns ``n, events, rate, ci_low, ci_high`` (rates
    and CI bounds are fractions in [0, 1]). Drives the CI + N annotations
    retrofitted onto NB03's readmission-rate bar charts.
    """
    y = _as_binary(df, target, "target")
    rows = []
    for group, idx in df.groupby(group_col, observed=False).groups.items():
        yy = y.loc[idx]
        n = int(len(yy))
        k = int(yy.sum())
        lo, hi = wilson_ci(k, n, alpha)
        rows.append({"group": group, "n": n, "events": k,
                     "rate": (k / n if n else float("nan")),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("group")


def adjusted_logit(df: pd.DataFrame, outcome: str, predictors: list[str],
                   *, dropna: bool = True, alpha: float = 0.05) -> pd.DataFrame:
    """Multivariable logistic regression returning a tidy odds-ratio table.

    Fits ``outcome ~ predictors`` with statsmodels. Numeric predictors enter
    linearly; non-numeric ones are treated as categorical via ``C(col)``.
    Returns a DataFrame ``[term, OR, ci_low, ci_high, p_value]`` (the intercept
    excluded), so the notebook can show adjusted ORs beside the unadjusted ones
    and read off how much each effect is confounded.

    Raises ImportError with an actionable message when statsmodels is missing, so
    the caller can fall back to the unadjusted table.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - exercised only without statsmodels
        raise ImportError(
            "adjusted_logit requires statsmodels (pip install statsmodels==0.14.6)"
        ) from exc

    data = df[[outcome, *predictors]].copy()
    if dropna:
        data = data.dropna()
    data[outcome] = data[outcome].astype(int)

    terms = [p if pd.api.types.is_numeric_dtype(data[p]) else f"C({p})" for p in predictors]
    formula = f"{outcome} ~ " + " + ".join(terms)
    model = smf.logit(formula, data=data).fit(disp=0)

    conf = model.conf_int(alpha=alpha)
    rows = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "OR": float(np.exp(model.params[term])),
            "ci_low": float(np.exp(conf.loc[term, 0])),
            "ci_high": float(np.exp(conf.loc[term, 1])),
            "p_value": float(model.pvalues[term]),
        })
    return pd.DataFrame(rows)
