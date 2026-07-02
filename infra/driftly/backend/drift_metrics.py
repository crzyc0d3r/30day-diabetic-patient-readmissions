"""Per-feature drift computation for Driftly.

Orchestrates the three headline metrics over the monitored feature set, reusing
the math in ``helpers.drift_sim`` (single source of truth shared with the
Airflow pipeline) and adding the display histograms the dashboard overlays:

  * numeric    -> Wasserstein (std-normalized) + PSI + KL + a binned histogram
  * categorical-> PSI + KL + grouped category frequencies; Wasserstein is null
                  (a nominal variable has no natural ground distance)

Per-feature status is the worst band across the metrics that apply; the overall
verdict is the worst feature status. Output is plain JSON-serialisable dicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from helpers.drift_sim import (
    MONITORED_COLUMNS,
    kl_categorical,
    kl_numeric,
    psi_categorical,
    psi_numeric,
    wasserstein_numeric,
)

_ORDER = {"OK": 0, "WARN": 1, "ALERT": 2}


def _status(value: float | None, warn: float, alert: float) -> str:
    if value is None:
        return "OK"
    if value >= alert:
        return "ALERT"
    if value >= warn:
        return "WARN"
    return "OK"


def worst(statuses: list[str]) -> str:
    """The most severe status present (ALERT > WARN > OK); OK when empty."""
    return max(statuses, key=lambda s: _ORDER[s]) if statuses else "OK"


def _numeric_hist(ref: np.ndarray, cur: np.ndarray, bins: int) -> dict:
    """Equal-width histogram over the reference range, returned as *fractions*
    (densities) so reference and current overlay comparably despite different
    row counts. Current values outside the reference range are clipped into the
    edge bins, which is itself a visible drift signal."""
    lo, hi = float(np.min(ref)), float(np.max(ref))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0  # degenerate/constant column: a single placeholder bin
    edges = np.linspace(lo, hi, bins + 1)
    rc = np.histogram(np.clip(ref, lo, hi), bins=edges)[0]
    cc = np.histogram(np.clip(cur, lo, hi), bins=edges)[0]
    rf = (rc / max(rc.sum(), 1)).astype(float)
    cf = (cc / max(cc.sum(), 1)).astype(float)
    return {"bins": edges.round(4).tolist(),
            "reference": rf.round(5).tolist(),
            "current": cf.round(5).tolist()}


def _categorical_hist(ref: pd.Series, cur: pd.Series, top: int = 12) -> dict:
    """Grouped category frequencies over the reference's top categories, with a
    lumped ``(other)`` bucket so the bars always sum to 1.0 on each side."""
    ref_s, cur_s = ref.astype(str), cur.astype(str)
    cats = ref_s.value_counts().head(top).index.tolist()
    rf_all = ref_s.value_counts(normalize=True)
    cf_all = cur_s.value_counts(normalize=True)
    rf = [float(rf_all.get(c, 0.0)) for c in cats]
    cf = [float(cf_all.get(c, 0.0)) for c in cats]
    rf_other, cf_other = max(0.0, 1.0 - sum(rf)), max(0.0, 1.0 - sum(cf))
    if rf_other > 1e-9 or cf_other > 1e-9:
        cats = cats + ["(other)"]
        rf.append(rf_other)
        cf.append(cf_other)
    return {"categories": cats,
            "reference": [round(x, 5) for x in rf],
            "current": [round(x, 5) for x in cf]}


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, batch: str) -> dict:
    """Compute the full per-feature drift result for one reference/current pair.

    Only monitored columns present in *both* frames are scored, so an upload
    missing some columns degrades to the intersection rather than erroring.
    """
    cols = [c for c in MONITORED_COLUMNS
            if c in reference.columns and c in current.columns]
    features: list[dict] = []
    for col in cols:
        ref = reference[col].dropna()
        if len(ref) == 0:
            continue
        if pd.api.types.is_numeric_dtype(ref):
            # Coerce the current column to numeric so a malformed upload (a
            # string in a numeric column) drops those rows rather than crashing.
            cur = pd.to_numeric(current[col], errors="coerce").dropna()
            if len(cur) == 0:
                continue
            ftype = "numeric"
            ref_a, cur_a = ref.to_numpy(), cur.to_numpy()
            wass = wasserstein_numeric(ref_a, cur_a)
            psi = psi_numeric(ref_a, cur_a)
            kl = kl_numeric(ref_a, cur_a)
            hist = _numeric_hist(ref_a, cur_a, config.HIST_BINS)
        else:
            cur = current[col].dropna()
            if len(cur) == 0:
                continue
            ftype = "categorical"
            wass = None
            psi = psi_categorical(ref, cur)
            kl = kl_categorical(ref, cur)
            hist = _categorical_hist(ref, cur)

        statuses = [
            _status(psi, config.PSI_WARN, config.PSI_ALERT),
            _status(kl, config.KL_WARN, config.KL_ALERT),
        ]
        if wass is not None:
            statuses.append(_status(wass, config.WASSERSTEIN_WARN, config.WASSERSTEIN_ALERT))
        status = worst(statuses)

        features.append({
            "name": col,
            "type": ftype,
            "wasserstein": _round(wass),
            "psi": _round(psi),
            "kl": _round(kl),
            "status": status,
            "histogram": hist,
        })

    verdict = worst([f["status"] for f in features])
    summary = {
        "n_features": len(features),
        "ok": sum(f["status"] == "OK" for f in features),
        "warn": sum(f["status"] == "WARN" for f in features),
        "alert": sum(f["status"] == "ALERT" for f in features),
    }
    return {
        "batch": batch,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "verdict": verdict,
        "summary": summary,
        "thresholds": config.thresholds(),
        "features": features,
    }
