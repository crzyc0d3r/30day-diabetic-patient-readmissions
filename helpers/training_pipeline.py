"""Model training orchestration for the retrain pipeline.

`train_baselines_and_refits` is the single entry point. It loads
`data/train_test.npz` and `data/tuned_results.joblib`, builds two
populations of estimators per family (library-default configuration and
HPO-winner refit), fits each on the full `X_train` partition, scores
every fit on `X_val` through the shared `helpers.models.evaluate_model`
panel, sweeps the F1-optimal decision threshold per model, and persists
the joblib artefacts the conclusion stage reads.

Outputs written under `out_dir` (default `data/`):

- `training_models.joblib` and `training_results.joblib` for the
  HPO-refit population.
- `default_models.joblib` and `default_results.joblib` for the
  library-default population.
- `model_thresholds.joblib` (per-model F1-optimal threshold map) and
  `final_model_threshold.joblib` (the champion's threshold).
- `final_model.joblib` (the highest-validation-F1 estimator, saved as
  a starting champion bundle for serving).
- `mlp_results.joblib` with `best_config` and `best_epoch` so
  `helpers.mlp_train.nb07_best_epoch` and `nb08_best_config` resolve
  to the HPO-selected production configuration rather than the
  hand-coded fallback inside that module.

Pickle safety note: every joblib artefact this module reads or writes is
produced and consumed by the same retrain pipeline running on the same
Airflow workers. The files live under `data/` next to the rest of the
project state and are not loaded from untrusted sources, so the standard
arbitrary-code-execution concern for pickle-style formats does not apply
here. The project-wide convention is joblib for fitted estimators and
result dicts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from helpers.constants import OVERFIT_THRESHOLD
from helpers.mlops_helpers import (
    cuda_device_name,
    enable_mlflow_autolog_and_tracing,
    has_cuda,
    init_mlflow,
    log_estimator_to_mlflow,
    log_training_dataset,
)
from helpers.models import build_estimator, evaluate_model

# This module pins sklearn (not cuML) for LR/RF so the saved joblib loads in
# any environment. Airflow workers do not ship cuML, and cuML-typed
# estimators break `joblib.load` there. LR + RF train in seconds on CPU
# anyway. sample_weight is never forwarded because sklearn's MLPClassifier
# does not accept it and the other estimators read class balance from
# `class_weight="balanced"` / `scale_pos_weight` set inside `build_estimator`.


def _safe_params(model: Any) -> dict[str, Any]:
    """`get_params()` but stringify exotic types so MLflow accepts them."""
    try:
        params = model.get_params(deep=False)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _train_and_evaluate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit one candidate on X_train and score it on X_val.

    Class imbalance is already handled inside `build_estimator`:
    sklearn LR / RF use `class_weight="balanced"`, XGBoost uses
    `scale_pos_weight`, and CatBoost uses `auto_class_weights="Balanced"`.
    sklearn's MLP wrapper does not accept `sample_weight`, so this function
    does not forward one, avoiding a try/except branch around the fit call.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    metrics = evaluate_model(y_val, y_pred, y_prob, y_train, y_train_pred, y_train_prob)
    return model, y_pred, y_prob, metrics


def _compute_optimal_threshold(
    y_val: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """F1-optimal decision threshold on X_val.

    Returns `(optimal_threshold, f1_at_opt, recall_at_opt, precision_at_opt,
    y_pred_opt)`. The operating point is picked on X_val where the same set
    drove the choice. That is mildly optimistic, but it is the standard
    hold-one-split-out remedy in a two-split design. The conclusion stage
    re-applies these thresholds on X_test without retuning to give an honest
    cross-check.
    """
    if hasattr(y_prob, "get"):
        y_prob = y_prob.get()
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    # precision_recall_curve returns one more (prec, rec) than thresholds, so
    # drop the trailing point so the F1 vector aligns with the threshold grid.
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = int(np.argmax(f1s))
    best_thresh = float(thresholds[best_idx])
    y_pred_opt = (np.asarray(y_prob) >= best_thresh).astype(int)
    return (
        best_thresh,
        float(f1_score(y_val, y_pred_opt)),
        float(recall_score(y_val, y_pred_opt)),
        float(precision_score(y_val, y_pred_opt, zero_division=0)),
        y_pred_opt,
    )


def train_baselines_and_refits(
    train_test_path: str | Path = "data/train_test.npz",
    tuned_results_path: str | Path = "data/tuned_results.joblib",
    out_dir: str | Path = "data",
    *,
    mlflow_experiment: str = "medi-watch-readmission",
    autolog: bool = True,
) -> dict[str, Any]:
    """Train default + HPO-refit estimators across all five model families.

    Loads the preprocessed train and validation arrays from
    `data/train_test.npz` along with the tuned-results summary from
    `data/tuned_results.joblib`. Trains two populations of every model
    family (library defaults and HPO-winner refits) on the full `X_train`
    partition, scores each on `X_val` with the shared metric panel, picks
    an F1-optimal threshold per model, and persists the joblib artefacts
    the conclusion stage consumes.

    Parameters
    ----------
    train_test_path
        Path to the preprocessed train / val / test arrays.
    tuned_results_path
        Path to the per-model HPO summary produced by `run_hpo`.
    out_dir
        Directory to write the seven output joblibs into.
    mlflow_experiment
        MLflow experiment name. `init_mlflow` raises if the server is
        unreachable (Postgres-backed, no silent file-store fallback).
    autolog
        If True (default), call `enable_mlflow_autolog_and_tracing` so the
        Datasets / Traces tabs in the UI get populated. Set False for unit
        tests that don't want MLflow side effects on every fit.

    Returns
    -------
    dict
        `{"results": {name: {**metrics, y_pred, y_prob}, ...},
           "default_results": {name: {...}, ...},
           "fitted_models": {name: estimator, ...},
           "default_fitted_models": {name: estimator, ...},
           "threshold_results": {name: {optimal_threshold, f1_default,
                                        f1_optimized, recall_default,
                                        recall_optimized, precision_optimized,
                                        y_pred_opt}, ...},
           "model_thresholds": {name: float, ...},
           "champion_name": str,
           "champion_threshold": float}`
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(train_test_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight_ratio = float(n_neg / max(n_pos, 1))

    _HAS_CUDA = has_cuda()
    if _HAS_CUDA:
        print(f"[train_baselines] CUDA detected ({cuda_device_name() or 'unknown'})")
    print(f"[train_baselines] X_train={X_train.shape} X_val={X_val.shape} "
          f"pos_weight_ratio={pos_weight_ratio:.2f}")

    init_mlflow(experiment=mlflow_experiment)
    if autolog:
        enable_mlflow_autolog_and_tracing()

    tuned_results_summary = joblib.load(tuned_results_path)
    print(f"[train_baselines] loaded tuned results for: "
          f"{list(tuned_results_summary.keys())}")

    model_families = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "MLP"]
    default_models = {
        name: build_estimator(name, {}, pos_weight_ratio, _HAS_CUDA)
        for name in model_families
    }
    tuned_models = {
        name: build_estimator(
            name, tuned_results_summary[name]["best_params"],
            pos_weight_ratio, _HAS_CUDA,
        )
        for name in tuned_results_summary
    }

    default_results: dict[str, dict[str, Any]] = {}
    default_fitted_models: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}

    def _run_one(phase: str, name: str, model: Any) -> tuple[Any, dict[str, Any]]:
        run_name = f"{phase}_{name.lower().replace(' ', '_')}"
        with mlflow.start_run(run_name=run_name, tags={"phase": phase, "model": name}):
            fitted, y_pred, y_prob, metrics = _train_and_evaluate(
                name, model, X_train, y_train, X_val, y_val,
            )
            print(f"[{phase}/{name}] F1={metrics['f1']:.3f} "
                  f"AUC-ROC={metrics['auc_roc']:.3f}")

            mlflow.log_params(_safe_params(fitted))
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            train_ds = log_training_dataset(
                X_train, y_train,
                name="train_test.npz[X_train]",
                source="data/train_test.npz",
                context="training",
            )
            val_ds = log_training_dataset(
                X_val, y_val,
                name="train_test.npz[X_val]",
                source="data/train_test.npz",
                context="validation",
            )
            log_estimator_to_mlflow(
                fitted, name=run_name, X_sample=X_val,
                datasets=[(train_ds, "training"), (val_ds, "validation")],
            )
            return fitted, {**metrics, "y_pred": y_pred, "y_prob": y_prob}

    print("--- Training default-config models ---")
    for name, model in default_models.items():
        fitted, record = _run_one("default", name, model)
        default_fitted_models[name] = fitted
        default_results[name] = record

    print("\n--- Training HPO-winner refits ---")
    for name, model in tuned_models.items():
        fitted, record = _run_one("refit", name, model)
        fitted_models[name] = fitted
        results[name] = record

    # Overfit check, printed as a tuning prompt. The 0.15 gate is soft because
    # class-weighted fits legitimately produce small train-vs-val gaps.
    print("\nOverfit report (gate=0.15, soft; CV-fold gate=0.10 in NB06):")
    for name, record in results.items():
        f1_gap = record["train_f1"] - record["f1"]
        auc_gap = record["train_auc_roc"] - record["auc_roc"]
        flag = " !! OVERFITTING" if max(f1_gap, auc_gap) > OVERFIT_THRESHOLD else ""
        print(f"  {name:25s}  F1 gap: {f1_gap:+.4f}  AUC gap: {auc_gap:+.4f}{flag}")

    # F1-optimal decision threshold per model, computed on X_val.
    print("\nThreshold sweep (F1-optimal on X_val):")
    threshold_results: dict[str, dict[str, Any]] = {}
    for name in results:
        opt_t, f1_opt, rec_opt, prec_opt, y_pred_opt = _compute_optimal_threshold(
            y_val, results[name]["y_prob"],
        )
        threshold_results[name] = {
            "optimal_threshold": opt_t,
            "f1_default": float(results[name]["f1"]),
            "f1_optimized": f1_opt,
            "recall_default": float(results[name]["recall"]),
            "recall_optimized": rec_opt,
            "precision_optimized": prec_opt,
            "y_pred_opt": y_pred_opt,
        }
        results[name]["f1_optimized"] = f1_opt
        results[name]["recall_optimized"] = rec_opt
        results[name]["y_pred_opt"] = y_pred_opt
        gain = f1_opt - float(results[name]["f1"])
        print(f"  {name:25s}  t*={opt_t:.4f}  F1: {results[name]['f1']:.4f} -> "
              f"{f1_opt:.4f} ({gain:+.4f})  Recall: {rec_opt:.4f}")

    model_thresholds = {
        name: float(threshold_results[name]["optimal_threshold"]) for name in results
    }

    # Starting champion = highest validation F1 at the default 0.5 cut. The
    # conclusion stage may overwrite this with a different champion after the
    # held-out test pass, but we save a candidate here so the serving layer
    # has something to load even if the conclusion stage is skipped.
    champion_name = max(results, key=lambda m: results[m]["f1"])
    champion_threshold = model_thresholds[champion_name]
    champion_model = fitted_models[champion_name]

    joblib.dump(champion_model, out_dir / "final_model.joblib")
    joblib.dump(champion_threshold, out_dir / "final_model_threshold.joblib")
    joblib.dump(model_thresholds, out_dir / "model_thresholds.joblib")
    joblib.dump(fitted_models, out_dir / "training_models.joblib")
    joblib.dump(results, out_dir / "training_results.joblib")
    joblib.dump(default_fitted_models, out_dir / "default_models.joblib")
    joblib.dump(default_results, out_dir / "default_results.joblib")

    # mlp_results.joblib carries the HPO-selected hyperparameter dict plus the
    # epoch budget. helpers/mlp_train.py::nb07_best_epoch and nb08_best_config
    # read from it so the out-of-fold MLP loops align with the production MLP.
    if "MLP" in tuned_results_summary:
        mlp_best_params = tuned_results_summary["MLP"]["best_params"]
        mlp_results = {
            "best_config": {
                "lr":           float(mlp_best_params.get("lr", 1e-3)),
                "weight_decay": float(mlp_best_params.get("weight_decay", 1e-4)),
                "dropout":      float(mlp_best_params.get("dropout", 0.3)),
                "batch_size":   int(  mlp_best_params.get("batch_size", 512)),
            },
            "best_epoch": int(mlp_best_params.get("epochs", 15)),
        }
        joblib.dump(mlp_results, out_dir / "mlp_results.joblib")
        print(f"[train_baselines] saved {out_dir / 'mlp_results.joblib'} "
              f"(best_config={mlp_results['best_config']}, "
              f"best_epoch={mlp_results['best_epoch']})")

    print(f"[train_baselines] champion = {champion_name} "
          f"(F1={results[champion_name]['f1']:.4f}, t*={champion_threshold:.4f})")
    print(f"[train_baselines] saved 7 joblib artefacts to {out_dir}/")

    return {
        "results": results,
        "default_results": default_results,
        "fitted_models": fitted_models,
        "default_fitted_models": default_fitted_models,
        "threshold_results": threshold_results,
        "model_thresholds": model_thresholds,
        "champion_name": champion_name,
        "champion_threshold": champion_threshold,
    }
