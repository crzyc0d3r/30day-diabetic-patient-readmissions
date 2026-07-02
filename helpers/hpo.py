"""Hyperparameter-tuning helpers shared by NB06 §6.4 and §6.4.1.

Extracted from `pipeline/06_hyperparameter_tuning.ipynb` so the §6.4.1
cell stays under ~80 visible lines. The notebook imports these as
module-level functions and keeps the cell focused on configuration plus glue.

Three execution paths share this module:

1. `deterministic_grid_fallback`. Single-process StratifiedGroupKFold
   sweep used when ray is not installed. Deterministic, runs in
   roughly two minutes on CPU.
2. `make_tune_trial`. Factory returning the per-trial callable that
   `ray.tune.Tuner` invokes. The factory binds `train_ref`, `pos_weight`,
   and the cuda flag via closure so the trial function matches the
   one-arg `tune.Tuner` contract.
3. `sequential_cv_fallback`. Synchronous 3-fold CV sweep used when the
   Tuner reports no `mean_test_score` (the GB10 + multi-actor Ray-with-GPU
   failure mode). Same scorer as the bare-@ray.remote path in §6.4 so
   downstream §6.5+ keeps a populated `tuned_results`.

All three rely on `helpers.models.build_estimator` for the model-name to
estimator dispatch so the per-trial scorer cannot drift across paths.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from helpers.models import build_estimator
from helpers.mlops_helpers import (
    log_estimator_to_mlflow,
    log_training_dataset,
    stamp_experiment_metadata,
    stamp_run_metadata,
)


def _materialise(arr: Any) -> Any:
    """Convert a cuML/cupy array to host numpy if needed, else pass through."""
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def _cv_fit_and_score(
    name: str,
    cfg: Mapping[str, Any],
    cv_splits_iter: Iterable[tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
    *,
    report_each_fold: Callable[[int, float, float], None] | None = None,
) -> tuple[float, float]:
    """Single source of truth for "fit one HPO config across every fold".

    Iterates `cv_splits_iter`, refits a fresh `build_estimator(name, cfg, ...)`
    on each train fold, scores it on the held-out val fold, and accumulates
    train + test F1. If `report_each_fold` is supplied it is invoked after every
    fold with `(fold_idx, cumulative_test_f1, cumulative_train_f1)` so a Ray
    Tune trial can hand control back to ASHA between folds. Returns the final
    `(mean_test_f1, mean_train_f1)` across all folds seen.

    Three execution paths in this module share this scorer so the per-fold
    metric definition cannot drift between them: the sklearn fallback
    (`_cv_score_one_config` → `deterministic_grid_fallback`,
    `sequential_cv_fallback`), the Tune trial (`make_tune_trial`), and the
    bare-`@ray.remote` path in `helpers/hpo_pipeline.py`.
    """
    cv_test: list[float] = []
    cv_train: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits_iter, start=1):
        est = build_estimator(name, dict(cfg), pos_weight, _HAS_CUDA)
        est.fit(X_train[tr_idx], y_train[tr_idx])
        cv_test.append(f1_score(y_train[va_idx], est.predict(X_train[va_idx])))
        cv_train.append(f1_score(y_train[tr_idx], est.predict(X_train[tr_idx])))
        if report_each_fold is not None:
            report_each_fold(fold_idx, float(np.mean(cv_test)), float(np.mean(cv_train)))
    return float(np.mean(cv_test)), float(np.mean(cv_train))


def _cv_score_one_config(
    name: str,
    cfg: Mapping[str, Any],
    cv,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_patient_ids: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
) -> tuple[float, float]:
    """Score one HPO config across every fold of `cv` (sklearn-only path).

    Thin wrapper around `_cv_fit_and_score` that materialises the cv splits
    from the StratifiedGroupKFold the fallbacks pass in. Returns
    `(mean_test_f1, mean_train_f1)` so the caller can wrap the per-config
    record in the same shape the Ray paths produce.
    """
    return _cv_fit_and_score(
        name, cfg,
        cv.split(X_train, y_train, groups=train_patient_ids),
        X_train, y_train, pos_weight, _HAS_CUDA,
    )


def deterministic_grid_fallback(
    top_models: Sequence[str],
    search_spaces: Mapping[str, Mapping[str, Sequence[Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_patient_ids: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
    n_samples: int = 3,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Fallback HPO (used when Ray is unavailable): 15 trials per model, 2-fold CV, single process.
    Expanded search for the 85-feature regime after adding med complexity + lab risk features.

    This is the no-Ray smoke-test sweep, not a production HPO path. It
    exists so a fresh environment without `ray` installed can still
    execute downstream §6.5+ cells against a populated `tuned_results` dict
    in under 2 minutes. The production HPO path lives
    in `helpers/hpo_pipeline.py::_tuner_asha_hpo` (Ray Tune + ASHA, 3-fold
    CV, full trial budget) and `_bare_ray_hpo` (bare @ray.remote, 3-fold
    CV, full trial budget).

    Differences from the production paths that make this strictly a
    smoke-test:

      * n_splits=2 (vs 3 in production) trades variance for wall-time.
      * n_samples=3 default (vs `NUM_SAMPLES`=50 in production) trades
        coverage for wall-time.
      * No ASHA pruning, no Ray actor isolation, no per-fold reporting.

    Uses the same per-fold scorer (`_cv_fit_and_score` then
    `build_estimator` + `f1_score` on the held-out fold) as both Ray
    paths, so the metric definition cannot drift across the three paths.
    The 2-fold fold count is intentionally below the bare path's variance
    floor. Install `ray` and rerun for a real sweep before drawing
    conclusions from these numbers.
    """
    from helpers.models import build_estimator

    # smoke-test grid: n_splits=2 (not 3) trades variance for wall-time so the
    # fallback finishes in under 2 minutes on a CPU-only host. The
    # full Tuner+ASHA path uses n_splits=3.
    out: dict[str, dict[str, Any]] = {}
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)

    for name in top_models:
        if name not in search_spaces:
            continue
        rng = random.Random(seed)
        trials: list[dict[str, Any]] = []
        configs = [
            {k: rng.choice(list(v)) for k, v in search_spaces[name].items()}
            for _ in range(n_samples)
        ]
        run_name = f"hpo_fallback_{name.lower()}"
        with mlflow.start_run(
            run_name=run_name,
            tags={
                "phase": "hpo",
                "model": name,
                "hpo_backend": "sklearn_groupkfold_fallback",
                "ray_available": "false",
            },
        ) as _run, mlflow.start_span(name=f"run_lifecycle:{run_name}") as _span:
            # Stamp the parent fallback run + its experiment up front so the
            # code.version / Description columns are populated even if every
            # trial later fails before reaching 'log_estimator_to_mlflow'.
            _hpo_desc = (
                f"NB06 deterministic-grid HPO sweep for {name} "
                f"(n_configs={len(configs)}, ray_available=false). "
                "See helpers/hpo.py::deterministic_grid_fallback."
            )
            stamp_run_metadata(description=_hpo_desc)
            stamp_experiment_metadata(
                experiment_id=_run.info.experiment_id,
                description=_hpo_desc,
            )
            try:
                _span.set_attributes({
                    "mlflow.runId": _run.info.run_id,
                    "model": name,
                    "n_configs": float(len(configs)),
                })
            except Exception:
                pass
            for cfg in configs:
                mean_test, mean_train = _cv_score_one_config(
                    name, cfg, cv, X_train, y_train, train_patient_ids,
                    pos_weight, _HAS_CUDA,
                )
                trials.append(
                    {
                        "config": cfg,
                        "mean_test_score": mean_test,
                        "mean_train_score": mean_train,
                    }
                )

            best = max(trials, key=lambda t: t["mean_test_score"])
            best_cfg = best["config"]

            fitted = build_estimator(name, best_cfg, pos_weight, _HAS_CUDA)
            fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_val)
            y_prob = fitted.predict_proba(X_val)[:, 1]
            if hasattr(y_pred, "get"):
                y_pred = y_pred.get()
            if hasattr(y_prob, "get"):
                y_prob = y_prob.get()

            df = pd.DataFrame(trials)
            cv_results_compat = pd.DataFrame(
                {
                    "mean_test_score": df["mean_test_score"],
                    "mean_train_score": df["mean_train_score"],
                    "rank_test_score": df["mean_test_score"]
                    .rank(ascending=False, method="min")
                    .astype(int),
                }
            ).reset_index(drop=True)

            out[name] = {
                "best_params": best_cfg,
                "best_cv_f1": float(best["mean_test_score"]),
                "val_f1": float(f1_score(y_val, y_pred)),
                "val_auc_roc": float(roc_auc_score(y_val, y_prob)),
                "val_auc_pr": float(average_precision_score(y_val, y_prob)),
                "val_recall": float(recall_score(y_val, y_pred)),
                "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
                "y_pred": y_pred,
                "y_prob": y_prob,
                "model": fitted,
                "cv_results": cv_results_compat,
            }

            mlflow.log_params({k: str(v) for k, v in best_cfg.items()})
            mlflow.log_metrics(
                {
                    "best_cv_f1": float(out[name]["best_cv_f1"]),
                    "val_f1": float(out[name]["val_f1"]),
                    "val_auc_roc": float(out[name]["val_auc_roc"]),
                    "val_auc_pr": float(out[name]["val_auc_pr"]),
                    "val_recall": float(out[name]["val_recall"]),
                    "val_precision": float(out[name]["val_precision"]),
                    "n_trials": float(len(trials)),
                }
            )
            ds_desc = (
                f"NB06 HPO fallback ({name}) -- StratifiedGroupKFold by patient_nbr, "
                f"split from data/train_test.npz; n_train={len(y_train)}, n_val={len(y_val)}."
            )
            log_training_dataset(
                X_train, y_train,
                name="train_test.npz[X_train]",
                source="data/train_test.npz",
                context="training",
                description=ds_desc,
            )
            log_training_dataset(
                X_val, y_val,
                name="train_test.npz[X_val]",
                source="data/train_test.npz",
                context="validation",
                description=ds_desc,
            )
            log_estimator_to_mlflow(
                fitted, name=name.lower().replace(" ", "_"), X_sample=X_val,
                description=(
                    f"Deterministic-grid HPO fallback winner for {name}. "
                    f"best_cv_f1={out[name]['best_cv_f1']:.4f}, "
                    f"val_f1={out[name]['val_f1']:.4f}, "
                    f"val_auc_roc={out[name]['val_auc_roc']:.4f}. "
                    "Trained inside helpers/hpo.py::deterministic_grid_fallback "
                    "(used when Ray Tune is unavailable)."
                ),
            )

            print(
                f"[fallback/{name}] best_cv_f1={out[name]['best_cv_f1']:.4f} "
                f"val_f1={out[name]['val_f1']:.4f}"
            )

    return out


def make_tune_trial(
    train_ref: Any,
    pos_weight: float,
    _HAS_CUDA: bool,
) -> Callable[[Mapping[str, Any]], None]:
    """Build the per-trial callable that `ray.tune.Tuner` consumes.

    Closes over `train_ref`, `pos_weight`, and the cuda flag so the
    returned function matches the one-arg `tune.Tuner` contract
    (`def trial(config) -> None` reporting via `tune.report`). Same
    scorer as the bare-@ray.remote path, so a config that scores X on
    one path scores X on the other and downstream §6.5+ stays
    comparable across paths.
    """
    import ray
    from ray import tune

    from helpers.constants import N_SPLITS_HPO, SEED as _HPO_SEED

    def _tune_trial(config: Mapping[str, Any]) -> None:
        # The trainable returns None and only communicates back through
        # `tune.report(...)` with primitive floats / ints. Returning anything
        # heavier (an estimator, a numpy memmap, an open file handle, a
        # logger, an MLflow client) trips Ray's per-trial RPC serialiser with
        # `_InactiveRpcError: Failed to serialize response!` and aborts
        # `Tuner.fit()` mid-run. Keep the body strictly primitive in / out.
        name = config.pop("__model__")
        X_tr, y_tr, pids = ray.get(train_ref)
        cv = StratifiedGroupKFold(n_splits=N_SPLITS_HPO, shuffle=True, random_state=_HPO_SEED)
        # Hand `tune.report(...)` to the shared scorer as a per-fold callback so
        # ASHA still sees one rung per fold while the metric definition stays
        # in lock-step with the sklearn fallback and the bare-@ray.remote path.
        try:
            _cv_fit_and_score(
                name, config,
                cv.split(X_tr, y_tr, groups=pids),
                X_tr, y_tr, pos_weight, _HAS_CUDA,
                report_each_fold=lambda fold_idx, test_f1, train_f1: tune.report({
                    "mean_test_score": float(test_f1),
                    "mean_train_score": float(train_f1),
                    "folds_completed": int(fold_idx),
                }),
            )
        except Exception as exc:
            # Convert any native-library exception (CatBoost GPU OOM, XGBoost
            # CUDA stream death, CUDA-arena memmap detach, ...) into a plain
            # RuntimeError so Ray can pickle it. A C-typed exception falling
            # through here is exactly what surfaces as
            # `_InactiveRpcError: Failed to serialize response!`.
            raise RuntimeError(f"trial {name} failed: {type(exc).__name__}: {exc}") from None
        return None

    return _tune_trial


def to_tune_space(
    name: str,
    space: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Map the existing list-of-choices search_spaces dict to tune.search_space."""
    from ray import tune

    out: dict[str, Any] = {"__model__": name}
    for k, v in space.items():
        out[k] = tune.choice(list(v))
    return out


def refit_and_log_tuner_result(
    name: str,
    best_cfg: Mapping[str, Any],
    df_results: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
    num_samples: int,
) -> dict[str, Any]:
    """Refit the best Tuner config on full X_train, score on val, log to mlflow.

    Returns a `tuned_results`-shaped dict matching the bare-@ray.remote
    path (§6.4) so downstream §6.5+ cells consume the same record schema
    regardless of which HPO path produced it. The single refit-and-log
    site means a metric definition cannot drift across the two HPO
    paths.
    """
    @mlflow.trace(name=f"refit_tuner_winner_{name.lower().replace(' ', '_')}")
    def _refit_and_predict():
        m = build_estimator(name, best_cfg, pos_weight, _HAS_CUDA)
        m.fit(X_train, y_train)
        return m, _materialise(m.predict(X_val)), _materialise(m.predict_proba(X_val))[:, 1]

    fitted, y_pred, y_prob = _refit_and_predict()

    cv_results_compat = pd.DataFrame({
        "mean_test_score": df_results["mean_test_score"].to_numpy(),
        "mean_train_score": df_results.get(
            "mean_train_score", pd.Series(np.nan, index=df_results.index)
        ).to_numpy(),
        "rank_test_score": df_results["mean_test_score"].rank(
            ascending=False, method="min"
        ).astype(int).to_numpy(),
    }).reset_index(drop=True)

    record = {
        "best_params": best_cfg,
        "best_cv_f1": float(df_results["mean_test_score"].max()),
        "val_f1": f1_score(y_val, y_pred),
        "val_auc_roc": roc_auc_score(y_val, y_prob),
        "val_auc_pr": average_precision_score(y_val, y_prob),
        "val_recall": recall_score(y_val, y_pred),
        "val_precision": precision_score(y_val, y_pred, zero_division=0),  # pyright: ignore[reportArgumentType]
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model": fitted,
        "cv_results": cv_results_compat,
    }

    mlflow.log_params({k: str(v) for k, v in best_cfg.items()})
    mlflow.log_metrics({
        "best_cv_f1": float(record["best_cv_f1"]),
        "val_f1": float(record["val_f1"]),
        "val_auc_roc": float(record["val_auc_roc"]),
        "val_auc_pr": float(record["val_auc_pr"]),
        "val_recall": float(record["val_recall"]),
        "val_precision": float(record["val_precision"]),
        "n_trials": float(num_samples),
    })
    ds_desc = (
        f"NB06 Ray Tune refit ({name}) -- StratifiedGroupKFold by patient_nbr, "
        f"split from data/train_test.npz; n_train={len(y_train)}, n_val={len(y_val)}, "
        f"num_samples={num_samples}."
    )
    train_ds = log_training_dataset(
        X_train, y_train,
        name="train_test.npz[X_train]",
        source="data/train_test.npz",
        context="training",
        description=ds_desc,
    )
    val_ds = log_training_dataset(
        X_val, y_val,
        name="train_test.npz[X_val]",
        source="data/train_test.npz",
        context="validation",
        description=ds_desc,
    )
    log_estimator_to_mlflow(
        fitted, name=name.lower().replace(" ", "_"), X_sample=X_val,
        datasets=[(train_ds, "training"), (val_ds, "validation")],
        description=(
            f"Ray Tune ASHA winner for {name} refit on full X_train. "
            f"best_cv_f1={record['best_cv_f1']:.4f}, "
            f"val_f1={record['val_f1']:.4f}, "
            f"val_auc_roc={record['val_auc_roc']:.4f}, "
            f"val_auc_pr={record['val_auc_pr']:.4f}. "
            f"Search space: {dict(best_cfg)}. "
            "Trained inside helpers/hpo.py::refit_and_log_tuner_result."
        ),
    )

    print(
        f"[Tuner/{name}] best_cv_f1={record['best_cv_f1']:.4f}"
        f" val_f1={record['val_f1']:.4f}"
    )
    return record


def sequential_cv_fallback(
    name: str,
    n_samples: int,
    search_spaces: Mapping[str, Mapping[str, Sequence[Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_patient_ids: np.ndarray,
    pos_weight: float,
    _HAS_CUDA: bool,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """In-process 3-fold CV sweep when Tuner+ASHA trials silently fail.

    Used when the Tuner reports no `mean_test_score` column (the GB10 +
    multi-actor Ray-with-GPU failure mode where every trial errors out
    of band). Replays the same scorer the bare-@ray.remote path uses
    (`build_estimator` plus `f1_score` on a 3-fold StratifiedGroupKFold)
    so the populated `tuned_results` downstream §6.5+ consumes stays
    comparable across paths.
    """
    from helpers.constants import N_SPLITS_HPO

    rng = random.Random(seed)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS_HPO, shuffle=True, random_state=42)
    rows = []
    for _ in range(n_samples):
        cfg = {k: rng.choice(list(v)) for k, v in search_spaces[name].items()}
        mean_test, mean_train = _cv_score_one_config(
            name, cfg, cv, X_train, y_train, train_patient_ids,
            pos_weight, _HAS_CUDA,
        )
        rows.append({
            "config": cfg,
            "mean_test_score": mean_test,
            "mean_train_score": mean_train,
        })
    df = pd.DataFrame(rows)
    best_cfg = max(rows, key=lambda r: r["mean_test_score"])["config"]
    return df, best_cfg


def asha_pruning_report(
    per_model_results: Mapping[str, pd.DataFrame],
    *,
    out_path: str | None = "data/hpo_diagnostics.parquet",
    print_tables: bool = True,
) -> pd.DataFrame:
    """Audit how many folds each Tune trial finished, by model.

    Tune-with-ASHA halves trials between rungs, where each rung is one fold
    of `make_tune_trial`'s 3-fold CV. If MLP fold-1 F1 is noisier than RF's
    (10–20 epoch budget, BatchNorm cold start, no warmup), ASHA can prune
    MLP trials before they reach the catch-up rungs while RF survives more
    reliably, biasing the head-to-head leaderboard toward RF without any
    bug in the scorer or the data pipeline. This report makes that pattern
    visible.

    Parameters
    ----------
    per_model_results : Mapping[str, pd.DataFrame]
        `{model_name: result_dataframe}` where each frame is
        `results.get_dataframe()` from one model's `tune.Tuner.fit()` call.
        Tune returns the LAST per-trial report by default, which is exactly
        what "what did this trial reach?" needs. Each frame must contain at
        minimum columns `mean_test_score`, `mean_train_score`,
        `folds_completed` (the keys `make_tune_trial._tune_trial` emits via
        `tune.report(...)`).
    out_path : str | None
        Where to persist the long-form trial DataFrame (parquet). `None`
        disables persistence. Default `data/hpo_diagnostics.parquet`.
    print_tables : bool
        If True, also prints two stdout summaries: per-model counts +
        median test F1 per rung, and an MLP vs RF side-by-side line.

    Returns
    -------
    pd.DataFrame
        One row per trial with columns `model`, `folds_completed`,
        `mean_test_score`, `mean_train_score`.
    """
    from pathlib import Path

    rows: list[dict[str, Any]] = []
    for model, df in per_model_results.items():
        if df is None or len(df) == 0:
            continue
        for _, r in df.iterrows():
            fc_val = r.get("folds_completed", np.nan)
            ts_val = r.get("mean_test_score", np.nan)
            tr_val = r.get("mean_train_score", np.nan)
            rows.append({
                "model": model,
                "folds_completed": int(fc_val) if pd.notna(fc_val) else 0,
                "mean_test_score": float(ts_val) if pd.notna(ts_val) else float("nan"),
                "mean_train_score": float(tr_val) if pd.notna(tr_val) else float("nan"),
            })

    diag = pd.DataFrame(rows, columns=["model", "folds_completed", "mean_test_score", "mean_train_score"])
    if out_path is not None and len(diag) > 0:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        diag.to_parquet(p, index=False)

    if not print_tables or len(diag) == 0:
        return diag

    print("\n[ASHA pruning report] trials by (model, folds_completed):")
    per_model = (
        diag.groupby(["model", "folds_completed"])
        .agg(n_trials=("mean_test_score", "size"),
             median_test_f1=("mean_test_score", "median"))
        .reset_index()
        .sort_values(["model", "folds_completed"])
    )
    print(per_model.to_string(index=False))

    if {"MLP", "Random Forest"}.issubset(set(diag["model"].unique())):
        print("\n[ASHA pruning report] MLP vs Random Forest:")
        for m in ("MLP", "Random Forest"):
            sub = diag[diag["model"] == m]
            total = int(len(sub))
            survived = int((sub["folds_completed"] >= 3).sum())
            pct = (survived / total * 100) if total else 0.0
            rung1 = sub.loc[sub["folds_completed"] == 1, "mean_test_score"].median()
            rung3 = sub.loc[sub["folds_completed"] == 3, "mean_test_score"].median()
            lift = (rung3 - rung1) if pd.notna(rung1) and pd.notna(rung3) else float("nan")
            r1_str = f"{rung1:.4f}" if pd.notna(rung1) else "  n/a "
            r3_str = f"{rung3:.4f}" if pd.notna(rung3) else "  n/a "
            lift_str = f"{lift:+.4f}" if pd.notna(lift) else "  n/a "
            print(f"  {m:<15s}  trials={total:3d}  reach fold 3 = {survived:3d} ({pct:5.1f}%)"
                  f"  rung1 med F1={r1_str}  rung3 med F1={r3_str}  lift={lift_str}")
        print(
            "  Expected ASHA-pruning fingerprint: MLP 'reach fold 3' percentage materially below"
            " RF's AND MLP rung-3 median F1 strictly above rung-1 (catch-up signal)."
        )

    return diag
