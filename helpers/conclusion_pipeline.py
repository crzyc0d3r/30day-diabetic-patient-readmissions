"""Conclusion orchestration for the retrain pipeline.

`run_conclusion_and_register` is the single entry point. It aggregates the
per-stage result joblibs (default-config fits, HPO winners, HPO-refit fits)
into a unified validation-F1 leaderboard, selects the champion by highest
validation F1, evaluates the champion once on the held-out test set with a
1000-resample bootstrap confidence interval on the headline metrics, writes
the deployable bundle to `data/final_model.joblib`, registers the champion
to the MLflow Model Registry, and sets the `@champion` alias the inference
API resolves at serve time.

Pickle safety note: every joblib artefact this module reads or writes is
produced and consumed inside the retrain pipeline on the same Airflow workers
(`training_models.joblib`, `tuned_models.joblib`, `default_models.joblib`,
`final_model.joblib`, `training_results.joblib`, `tuned_results.joblib`,
`default_results.joblib`). They are not loaded from untrusted sources, so
the standard pickle-arbitrary-code-execution concern does not apply here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from helpers.evaluation import best_f1_threshold, per_subgroup_metrics
from helpers.mlops_helpers import (
    enable_mlflow_autolog_and_tracing,
    init_mlflow,
    log_estimator_to_mlflow,
    log_training_dataset,
)

# Sidecar artefacts the inference API expects to find under the registered
# model version's "preprocessor/" path. Missing files are tolerated when
# running locally with only a subset present.
_SIDECAR_ARTEFACTS = (
    "ohe.joblib",
    "scaler.joblib",
    "feature_selector.joblib",
    "feature_names.csv",
    "full_inference_pipeline.joblib",
    "numeric_medians.joblib",
    "final_model.joblib",
)


def _g(res: dict, *keys: str) -> Any:
    """Return the first key present. Upstream stages use inconsistent names."""
    for k in keys:
        if k in res:
            return res[k]
    raise KeyError(keys)


def _build_leaderboard(
    default_results: dict[str, dict],
    tuned_results: dict[str, dict],
    training_results: dict[str, dict],
) -> pd.DataFrame:
    """Aggregate per-stage result dicts into one validation-F1-sorted table.

    Three stages contribute (Default, HPO-winner, HPO-refit), one row per
    (model, stage). The lookup is tolerant of both `f1` / `val_f1` and
    `auc_roc` / `val_auc_roc` key variants because the upstream stages
    use slightly different schemas.
    """
    rows = []
    for source, bag in [
        ("Default", default_results),
        ("HPO-winner", tuned_results),
        ("HPO-refit", training_results),
    ]:
        for name, res in bag.items():
            rows.append({
                "Model": name,
                "Variant": f"{name} ({source})",
                "Source": source,
                "F1": float(_g(res, "f1", "val_f1")),
                "AUC-ROC": float(_g(res, "auc_roc", "val_auc_roc")),
            })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def _bootstrap_ci(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap CI over headline metrics: resample the test set with
    replacement and recompute each metric on every resample.

    Returns `{metric_name: {point, ci_low, ci_high, beats_naive_pct}}` for the
    six metrics the model card surfaces. The `beats_naive_pct` column reports
    the fraction of resamples on which the model strictly beats the
    prevalence-aware naive baseline. 95% or higher indicates a 95% CI disjoint
    from the baseline.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)

    stats = {k: [] for k in ("F1", "Recall", "Precision", "AUC-ROC", "AUC-PR", "Brier")}
    beats = {k: 0 for k in stats}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_test[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue  # degenerate resample, skip
        yp = y_pred[idx]
        ypr = y_prob[idx]
        rate_b = float(yt.mean())

        f1_b = float(f1_score(yt, yp))
        rec_b = float(recall_score(yt, yp))
        prc_b = float(precision_score(yt, yp, zero_division=0))
        auc_b = float(roc_auc_score(yt, ypr))
        ap_b = float(average_precision_score(yt, ypr))
        br_b = float(brier_score_loss(yt, ypr))
        stats["F1"].append(f1_b); stats["Recall"].append(rec_b)
        stats["Precision"].append(prc_b); stats["AUC-ROC"].append(auc_b)
        stats["AUC-PR"].append(ap_b); stats["Brier"].append(br_b)

        if f1_b > rate_b: beats["F1"] += 1
        if rec_b > rate_b: beats["Recall"] += 1
        if prc_b > rate_b: beats["Precision"] += 1
        if auc_b > 0.5: beats["AUC-ROC"] += 1
        if ap_b > rate_b: beats["AUC-PR"] += 1
        if br_b < rate_b * (1 - rate_b): beats["Brier"] += 1

    point_panel = {
        "F1":        float(f1_score(y_test, y_pred)),
        "Recall":    float(recall_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "AUC-ROC":   float(roc_auc_score(y_test, y_prob)),
        "AUC-PR":    float(average_precision_score(y_test, y_prob)),
        "Brier":     float(brier_score_loss(y_test, y_prob)),
    }
    out: dict[str, dict[str, float]] = {}
    for k, vals in stats.items():
        arr = np.asarray(vals)
        n_eff = len(arr)
        out[k] = {
            "point": point_panel[k],
            "ci_low":  float(np.percentile(arr, 2.5)) if n_eff else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if n_eff else float("nan"),
            "beats_naive_pct": (beats[k] / n_eff * 100.0) if n_eff else float("nan"),
        }
    return out


def _load_champion_model(
    champion_entry: pd.Series,
    in_dir: Path,
) -> Any:
    """Pull the fitted champion estimator from the matching per-stage joblib."""
    source_to_artefact = {
        "Default":    "default_models.joblib",
        "HPO-winner": "tuned_models.joblib",
        "HPO-refit":  "training_models.joblib",
    }
    source = champion_entry["Source"]
    artefact = in_dir / source_to_artefact[source]
    models = joblib.load(artefact)
    return models[champion_entry["Model"]]


def _age_band(age_value: Any) -> str:
    """Bucket the UCI Diabetes-130 `age` field into broad bands.

    The raw column already arrives as a 10-year string bucket
    ('[0-10)', '[10-20)', ...). Collapse the eleven raw buckets into
    three audit bands so per-subgroup recall has enough sample per cell
    to be stable: young (<40), middle (40-70), senior (70+). Unrecognised
    inputs land in 'unknown' so the audit never drops a row silently.
    """
    if not isinstance(age_value, str):
        return "unknown"
    s = age_value.strip()
    young = {"[0-10)", "[10-20)", "[20-30)", "[30-40)"}
    middle = {"[40-50)", "[50-60)", "[60-70)"}
    senior = {"[70-80)", "[80-90)", "[90-100)"}
    if s in young:
        return "young"
    if s in middle:
        return "middle"
    if s in senior:
        return "senior"
    return "unknown"


def _compute_fairness_audit(
    in_dir: Path,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]] | None:
    """Best-effort per-subgroup recall audit on the test split.

    Loads raw demographic columns (race, gender, age) from
    `<in_dir>/features.csv` + `<in_dir>/patient_ids.csv` so the retrain
    DAG ships a populated fairness_audit dict instead of None. Returns
    None when either CSV is missing, the test-row alignment fails, or
    the demographic columns are not present. Every failure mode is
    swallowed because the bundle has to keep writing even when the raw
    demographics are not on disk (a fresh notebook smoke-test runs
    against the npz only).

    Returns
    -------
    dict | None
        '{attribute_name: {subgroup_label: {metric: value}}}' or None.
        The metric shape matches `helpers.evaluation.per_subgroup_metrics`.
    """
    features_path = in_dir / "features.csv"
    patient_ids_path = in_dir / "patient_ids.csv"
    if not features_path.exists() or not patient_ids_path.exists():
        return None
    try:
        features = pd.read_csv(features_path)
        patient_ids = pd.read_csv(patient_ids_path)
    except Exception:
        return None

    # The retrain pipeline writes patient_ids.csv with a `partition` column
    # whose value is 'test' for the held-out test rows. Without that column
    # we cannot align demographics to y_test row indices, so bail.
    if "partition" not in patient_ids.columns:
        return None
    test_mask = patient_ids["partition"].astype(str).str.lower() == "test"
    if int(test_mask.sum()) != len(y_test):
        # Row alignment unsafe: refuse to invent a join key.
        return None

    audit: dict[str, dict[str, dict[str, float]]] = {}
    test_features = features.loc[test_mask.to_numpy()].reset_index(drop=True)
    for attr_in, attr_out, transform in (
        ("race", "race", lambda s: s.astype(str)),
        ("gender", "gender", lambda s: s.astype(str)),
        ("age", "age_band", lambda s: s.map(_age_band)),
    ):
        if attr_in not in test_features.columns:
            continue
        labels = transform(test_features[attr_in]).to_numpy()
        audit[attr_out] = per_subgroup_metrics(y_test, y_test_pred, labels)
    return audit or None


def run_conclusion_and_register(
    train_test_path: str | Path = "data/train_test.npz",
    in_dir: str | Path = "data",
    out_dir: str | Path = "data",
    *,
    mlflow_experiment: str = "medi-watch-readmission",
    registered_name: str = "medi-watch-readmission",
    register: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Aggregate the leaderboard, pick a champion, evaluate on test, register.

    The fairness audit is computed best-effort inside this function:
    when `<in_dir>/features.csv` and `<in_dir>/patient_ids.csv` are
    both present and the `patient_ids.csv` carries a `partition`
    column with the value `test` for the held-out rows, the bundle
    ships a populated `fairness_audit` over race / gender / age_band.
    When either CSV is missing or the row counts do not align (a fresh
    notebook smoke-test against the npz only), the bundle falls back to
    `fairness_audit=None` so the inference API contract is preserved.

    Parameters
    ----------
    train_test_path
        Path to the preprocessed train / val / test arrays. The test
        arrays are loaded here for the first time in the retrain path.
        Every upstream stage scores on X_val only, preserving the
        test-set discipline.
    in_dir
        Directory containing the per-stage result + model joblibs from
        `run_hpo` and `train_baselines_and_refits`.
    out_dir
        Directory to write `final_model.joblib` into. Overwrites whatever
        NB07 / `training_pipeline` placed there as a placeholder champion.
    mlflow_experiment / registered_name
        MLflow experiment name and Model Registry registered-model name.
        Both default to `medi-watch-readmission`. The inference API
        resolves `models:/medi-watch-readmission@champion`.
    register
        If True (default) registers the champion to the MLflow Model
        Registry and sets the `@champion` alias. If False, only the
        local `final_model.joblib` bundle is written, useful for unit
        tests and local dry runs.
    n_bootstrap
        Resamples for the test-set CI block. Default 1000 (NB08 §8.5).

    Returns
    -------
    dict
        `{"champion_name": str, "champion_source": str,
           "champion_metrics": {...}, "registered_version": str | None,
           "bundle_path": str}`
    """
    train_test_path = Path(train_test_path)
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(train_test_path)
    X_val = data["X_val"]; y_val = data["y_val"]
    X_test = data["X_test"]; y_test = data["y_test"]
    pos_rate_test = float(y_test.mean())
    print(f"[run_conclusion] X_test={X_test.shape} "
          f"positive rate={pos_rate_test:.3%}")

    init_mlflow(experiment=mlflow_experiment)

    # Aggregate the three stages' results. Each load is wrapped because a
    # fresh clone may not have every stage produced yet (a smoke-test run
    # against the deterministic fallback leaves training_results.joblib
    # absent, for example).
    try:
        default_results = joblib.load(in_dir / "default_results.joblib")
    except FileNotFoundError:
        default_results = {}
    try:
        tuned_results = joblib.load(in_dir / "tuned_results.joblib")
    except FileNotFoundError:
        tuned_results = {}
    try:
        training_results = joblib.load(in_dir / "training_results.joblib")
    except FileNotFoundError:
        training_results = {}

    leaderboard = _build_leaderboard(default_results, tuned_results, training_results)
    if leaderboard.empty:
        raise RuntimeError(
            f"No candidate results found under {in_dir}. Run "
            "helpers.hpo_pipeline.run_hpo and "
            "helpers.training_pipeline.train_baselines_and_refits first."
        )
    print(f"[run_conclusion] leaderboard: {len(leaderboard)} candidates "
          f"({leaderboard['Source'].value_counts().to_dict()})")
    print(leaderboard.head(10).to_string(index=False))

    # Champion selection: highest validation F1, no manual tiebreak.
    champion_entry = leaderboard.iloc[0]
    champion_name = champion_entry["Model"]
    champion_source = champion_entry["Source"]
    print(f"[run_conclusion] champion = {champion_name} (from {champion_source} stage), "
          f"val F1={champion_entry['F1']:.4f}")

    champion_model = _load_champion_model(champion_entry, in_dir)

    # Test-set headline panel plus bootstrap CI on the test split.
    y_test_pred = champion_model.predict(X_test)
    y_test_prob = champion_model.predict_proba(X_test)[:, 1]
    y_val_prob = champion_model.predict_proba(X_val)[:, 1]

    # Threshold is picked on val so the test number stays honest.
    recommended_threshold, _ = best_f1_threshold(y_val, y_val_prob)
    y_test_pred_opt = (y_test_prob >= recommended_threshold).astype(int)
    test_f1_default = float(f1_score(y_test, y_test_pred))
    test_f1_opt = float(f1_score(y_test, y_test_pred_opt))
    test_auc = float(roc_auc_score(y_test, y_test_prob))
    test_ap = float(average_precision_score(y_test, y_test_prob))
    test_brier = float(brier_score_loss(y_test, y_test_prob))
    test_mcc = float(matthews_corrcoef(y_test, y_test_pred))

    bootstrap = _bootstrap_ci(y_test, y_test_pred, y_test_prob, n_boot=n_bootstrap)
    print(f"\n[run_conclusion] test panel @ default 0.5: F1={test_f1_default:.4f} "
          f"AUC={test_auc:.4f} AP={test_ap:.4f}")
    print(f"[run_conclusion] test panel @ t*={recommended_threshold:.4f}: F1={test_f1_opt:.4f}")
    print(f"[run_conclusion] bootstrap F1 95% CI = "
          f"[{bootstrap['F1']['ci_low']:.4f}, {bootstrap['F1']['ci_high']:.4f}] "
          f"(beats_naive={bootstrap['F1']['beats_naive_pct']:.1f}%)")

    # Deployable bundle. The inference API reads this file as a local
    # fallback when the MLflow Registry is unreachable, so the schema has
    # to stay stable across runs. fairness_audit is populated by
    # _compute_fairness_audit when raw demographics are present on disk
    # (the production retrain DAG writes them). When absent (a fresh
    # notebook smoke-test against the npz only) the field stays None so
    # the bundle still writes and the notebook can re-dump with the audit
    # populated later.
    fairness_audit = _compute_fairness_audit(in_dir, y_test, y_test_pred_opt)
    if fairness_audit is None:
        print("[run_conclusion] fairness_audit=None (raw demographics not "
              "found under data/; bundle ships without subgroup audit).")
    else:
        print(f"[run_conclusion] fairness_audit populated across "
              f"{len(fairness_audit)} attribute(s): {sorted(fairness_audit.keys())}")
    bundle = {
        "model": champion_model,
        "model_name": champion_name,
        "source_stage": champion_source,
        "recommended_threshold": recommended_threshold,
        "test_metrics": {
            "f1_default_threshold": test_f1_default,
            "f1_optimal_threshold": test_f1_opt,
            "auc_roc": test_auc,
            "auc_pr": test_ap,
            "brier": test_brier,
            "mcc": test_mcc,
        },
        "test_bootstrap_95ci": bootstrap,
        "test_n": int(len(y_test)),
        "prevalence_test": pos_rate_test,
        "fairness_audit": fairness_audit,
    }
    bundle_path = out_dir / "final_model.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"[run_conclusion] saved {bundle_path}")

    registered_version: str | None = None
    if not register:
        return {
            "champion_name": champion_name,
            "champion_source": champion_source,
            "champion_metrics": bundle["test_metrics"],
            "registered_version": None,
            "bundle_path": str(bundle_path),
        }

    # Registry promotion. No try/except wrapping: a failed registration
    # must raise so the failure is visible in the DAG run rather than
    # producing a silent "[WARN] skipped" log line.
    from mlflow.tracking import MlflowClient

    enable_mlflow_autolog_and_tracing()
    with mlflow.start_run(
        run_name=f"champion_promotion_{champion_name}",
        tags={"phase": "promotion", "source_stage": champion_source},
    ) as run:
        mlflow.log_params({
            "model_name": champion_name,
            "source_stage": champion_source,
            "recommended_threshold": recommended_threshold,
        })
        mlflow.log_metrics({
            "test_f1_default_threshold": test_f1_default,
            "test_f1_at_val_threshold": test_f1_opt,
            "test_auc_roc": test_auc,
            "test_auc_pr": test_ap,
            "test_brier": test_brier,
            "test_mcc": test_mcc,
        })
        test_ds = log_training_dataset(
            X_test, y_test,
            name="train_test.npz[X_test]",
            source="data/train_test.npz",
            context="test",
        )
        for sidecar in _SIDECAR_ARTEFACTS:
            sidecar_path = in_dir / sidecar
            if sidecar_path.exists():
                mlflow.log_artifact(str(sidecar_path), artifact_path="preprocessor")
        model_info = log_estimator_to_mlflow(
            champion_model,
            name=champion_name.lower().replace(" ", "_"),
            X_sample=X_test,
            registered_model_name=registered_name,
            datasets=[(test_ds, "test")],
        )
        client = MlflowClient()
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            # log_model did not surface the version directly (varies across
            # MLflow flavors). Look up the latest version owned by this run
            # instead of blindly picking the highest registry number.
            for v in client.search_model_versions(f"name='{registered_name}'"):
                if v.run_id == run.info.run_id:
                    version = v.version
                    break
        if version is None:
            raise RuntimeError(
                f"log_model did not produce a registered version for "
                f"{registered_name} in run {run.info.run_id}; cannot set "
                "@champion alias."
            )
        client.set_registered_model_alias(registered_name, "champion", str(version))
        print(f"[run_conclusion] registered {registered_name} v{version}, "
              f"alias @champion (run_id={run.info.run_id}).")
        registered_version = str(version)

    return {
        "champion_name": champion_name,
        "champion_source": champion_source,
        "champion_metrics": bundle["test_metrics"],
        "registered_version": registered_version,
        "bundle_path": str(bundle_path),
    }
