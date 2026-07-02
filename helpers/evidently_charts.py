"""Render Evidently drift results as PNG charts and log them to MLflow.

MLflow's artifact viewer renders images natively and inline, but it shows
Evidently's self-contained HTML report inside a sandboxed iframe that blocks
the report's Plotly scripts — so the charts never draw. This module closes
that gap: it reads the per-column drift scores Evidently already computed
(the `ValueDrift(column=…)` entries in the saved report JSON) plus the raw
reference/current CSVs, and emits two static figures that DO render in the
MLflow UI:

  charts/drift_scores.png   per-column drift score as a multiple of its
                            threshold (>1 = drifted), one dashed line at the
                            boundary so PSI and KS columns read on one axis.
  charts/distributions.png  reference-vs-current density overlays for the
                            drifted columns (up to GRID_MAX), the visual that
                            answers "what actually shifted?".

`build_figures` is import-safe and Airflow-friendly (matplotlib Agg, no
Evidently dependency), so the same code backfills existing runs from the host
and can be called inside the DAG's log_to_mlflow task for future runs.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless: no display in the DAG worker or this script
import matplotlib.pyplot as plt
import pandas as pd

from helpers.constants import DRIFT_PSI_WARN

# At most this many drifted columns get a distribution panel. Beyond ~9 the
# grid stops being readable; the bar chart already covers the full column set.
GRID_MAX = 9


def parse_drift_scores(report_json_path: str) -> pd.DataFrame:
    """Pull per-column drift verdicts out of a saved Evidently report JSON.

    Returns a frame indexed by column with the drift `score`, the `method`
    Evidently used (psi or ks), its `threshold`, and a `drifted` bool. Sorted
    so the most-drifted column (by score/threshold ratio) is first.
    """
    with open(report_json_path) as fh:
        report = json.load(fh)

    rows = []
    for metric in report.get("metrics", []):
        cfg = metric.get("config", {})
        if cfg.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        col = cfg["column"]
        method = cfg.get("method", "psi")
        threshold = float(cfg.get("threshold", DRIFT_PSI_WARN))
        score = float(metric.get("value"))
        # Direction depends on the statistic. PSI is a divergence: bigger =
        # more drift, so it drifts when score > threshold. KS reports a
        # p-value: smaller = more drift, so it drifts when score < threshold.
        # `exceedance` folds both onto one scale where >1 means drifted, so a
        # single boundary line at 1.0 reads correctly for either method.
        if method == "ks":
            exceedance = threshold / score if score else math.inf
        else:
            exceedance = score / threshold if threshold else math.inf
        rows.append({
            "column": col,
            "method": method,
            "threshold": threshold,
            "score": score,
            "ratio": exceedance,
            "drifted": exceedance > 1.0,
        })

    df = pd.DataFrame(rows).set_index("column")
    return df.sort_values("ratio", ascending=False)


def _drift_score_figure(scores: pd.DataFrame, scenario: str):
    """Horizontal bar of score/threshold ratio. >1 (dashed line) = drifted.

    Plotting the ratio rather than the raw statistic lets PSI columns (scores
    up to ~3) and KS columns (scores in [0, 1]) share one axis with a single
    interpretable boundary at 1.0.
    """
    n = len(scores)
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.28 * n)))
    colors = ["#d6455d" if d else "#3b7dd8" for d in scores["drifted"]]
    ax.barh(range(n), scores["ratio"].clip(upper=40), color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{c}  ({m.upper()})" for c, m in zip(scores.index, scores["method"])],
        fontsize=7,
    )
    ax.invert_yaxis()  # most-drifted at top
    ax.axvline(1.0, color="#222", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("drift vs. threshold  (PSI: score÷thr, KS: thr÷p-value; "
                  "log scale, dashed line = drift boundary)")
    n_drift = int(scores["drifted"].sum())
    ax.set_title(
        f"Per-column drift — {scenario}   "
        f"({n_drift} of {n} columns drifted)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def _distribution_figure(scenario: str, drifted_cols: list[str],
                         reference_path: str, current_path: str):
    """Reference-vs-current density overlays for the drifted columns."""
    cols = drifted_cols[:GRID_MAX]
    if not cols:
        return None
    usecols = list(dict.fromkeys(cols))  # de-dupe, preserve order
    ref = pd.read_csv(reference_path, usecols=lambda c: c in usecols, low_memory=False)
    cur = pd.read_csv(current_path, usecols=lambda c: c in usecols, low_memory=False)

    ncols = min(3, len(cols))
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        r, c = ref[col].dropna(), cur[col].dropna()
        if pd.api.types.is_numeric_dtype(ref[col]):
            # Numeric: density histograms on shared bins for comparability.
            lo = float(min(r.min(), c.min()))
            hi = float(max(r.max(), c.max()))
            bins = 30 if hi > lo else 1
            rng = (lo, hi) if hi > lo else None
            ax.hist(r, bins=bins, range=rng, density=True, alpha=0.55,
                    color="#3b7dd8", label="reference")
            ax.hist(c, bins=bins, range=rng, density=True, alpha=0.55,
                    color="#d6455d", label="current")
        else:
            # Categorical: side-by-side normalized frequencies over the union
            # of categories (top by reference share, capped for readability).
            r_share = r.value_counts(normalize=True)
            c_share = c.value_counts(normalize=True)
            cats = r_share.index.union(c_share.index)
            cats = list(r_share.reindex(cats).fillna(0).sort_values(ascending=False).index[:8])
            x = range(len(cats))
            ax.bar([v - 0.2 for v in x], [r_share.get(k, 0) for k in cats],
                   width=0.4, color="#3b7dd8", label="reference")
            ax.bar([v + 0.2 for v in x], [c_share.get(k, 0) for k in cats],
                   width=0.4, color="#d6455d", label="current")
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(k)[:10] for k in cats], rotation=45,
                               ha="right", fontsize=6)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    # Blank any unused grid cells.
    for j in range(len(cols), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Reference vs current distributions — {scenario}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_figures(report_json_path: str, scenario: str,
                  reference_path: str, current_path: str) -> dict:
    """Return {artifact_name: matplotlib Figure} for a single scenario report."""
    scores = parse_drift_scores(report_json_path)
    figs = {"charts/drift_scores.png": _drift_score_figure(scores, scenario)}
    drifted = scores.index[scores["drifted"]].tolist()
    dist = _distribution_figure(scenario, drifted, reference_path, current_path)
    if dist is not None:
        figs["charts/distributions.png"] = dist
    return figs
