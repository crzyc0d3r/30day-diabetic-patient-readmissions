"""Unit tests for the deterministic, in-process paths of `helpers/hpo.py`.

WHAT this file covers

`helpers/hpo.py` holds three HPO execution paths (a single-process sklearn
grid fallback, a Ray Tune trial factory, and a synchronous CV fallback) plus
two small near-pure helpers. WHY we exercise only a subset here: the Ray Tune
wiring (`make_tune_trial`) requires a live Ray cluster and the MLflow logging
branches require a tracking server, neither of which belongs in a fast unit
suite. We therefore test the deterministic sklearn code paths directly and
monkeypatch every `mlflow.*` logging call plus the `helpers.mlops_helpers`
sinks down to no-ops. That keeps the tests pure, in-process, and reproducible.

The model throughout is `"Logistic Regression"` because it fits in
milliseconds on tiny data and needs no GPU. We pass `_HAS_CUDA=False`
everywhere so `build_estimator` stays on its pure-CPU branch.

STYLE NOTE: no em dashes, no semicolons, "program" never the British spelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers import hpo


#
# Shared toy-data fixtures
#
# WHY local fixtures: the grouped StratifiedGroupKFold splitter needs three
# coordinated arrays (X, y, patient ids) plus both classes present in every
# fold. The repo-wide conftest fixtures are shaped for other modules, so we
# build a tiny dataset sized for grouped 2-fold and 3-fold CV here.


@pytest.fixture
def lr_search_space():
    """A deliberately tiny Logistic Regression grid.

    WHAT: two values of the regularisation strength C and a single penalty.
    WHY small: `deterministic_grid_fallback` samples configs by random
    choice over these lists, so a small grid keeps the sweep cheap and the
    sampled configs easy to reason about. The keys match the real
    `SEARCH_SPACES["Logistic Regression"]` so `build_estimator` accepts
    them unchanged.
    """
    return {"Logistic Regression": {"C": [0.1, 1.0], "penalty": ["l2"]}}


@pytest.fixture
def grouped_dataset(rng):
    """A small binary problem with repeating patient ids for grouped CV.

    WHAT we return: `(X_train, y_train, X_val, y_val, train_patient_ids)`.

    WHY this shape:
      * Both classes appear and stay reasonably balanced so `f1_score` is
        well defined on every held-out fold (no all-one-class folds, which
        would make F1 degenerate).
      * `train_patient_ids` repeats each id across several rows so
        `StratifiedGroupKFold` has genuine groups to keep intact. We use
        more distinct groups than the maximum fold count (3) so the splitter
        can always place whole groups on each side.
      * The signal is close to linearly separable (a single informative axis
        plus light noise) so Logistic Regression learns something and the train
        F1 is generally at least as high as the held-out test F1.
    """
    n = 90
    # One informative feature carries the label signal, plus two noise columns so
    # the model fits a non-trivial design matrix.
    y = np.array([0, 1] * (n // 2), dtype=int)
    signal = y * 2.0 + rng.normal(0.0, 0.5, size=n)
    noise1 = rng.normal(0.0, 1.0, size=n)
    noise2 = rng.normal(0.0, 1.0, size=n)
    X = np.column_stack([signal, noise1, noise2]).astype(float)

    # 18 distinct patient ids, each owning 5 consecutive rows. With both classes
    # alternating, every group carries a mix of labels, which keeps stratified
    # grouped folds populated.
    train_patient_ids = np.repeat(np.arange(18), 5)

    # A separate small validation block for the *_grid_fallback path, which
    # refits the winner on all of X_train and scores it on X_val/y_val.
    nv = 30
    yv = np.array([0, 1] * (nv // 2), dtype=int)
    signal_v = yv * 2.0 + rng.normal(0.0, 0.5, size=nv)
    Xv = np.column_stack(
        [signal_v, rng.normal(0, 1, nv), rng.normal(0, 1, nv)]
    ).astype(float)

    return X, y, Xv, yv, train_patient_ids


@pytest.fixture
def patched_mlflow(monkeypatch):
    """Neutralise every MLflow + mlops sink `deterministic_grid_fallback` touches.

    WHY: the grid fallback opens an MLflow run, a span, and logs params,
    metrics, datasets, and an estimator. None of that should hit a real
    tracking server in a unit test. We replace each call site with a no-op (or
    a tiny context-manager stand-in for the run/span) so the deterministic CV
    maths underneath runs untouched while the logging side effects vanish.
    """

    class _NullCtx:
        """A no-op stand-in for an MLflow run or span context manager.

        `deterministic_grid_fallback` reads `_run.info.run_id` and
        `_run.info.experiment_id` off the run object, so we expose a tiny
        `info` shim carrying both attributes.
        """

        class _Info:
            run_id = "test-run-id"
            experiment_id = "test-exp-id"

        info = _Info()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def set_attributes(self, *a, **k):
            # Spans accept attribute dicts. We swallow them silently.
            return None

    monkeypatch.setattr(hpo.mlflow, "start_run", lambda *a, **k: _NullCtx())
    monkeypatch.setattr(hpo.mlflow, "start_span", lambda *a, **k: _NullCtx())
    monkeypatch.setattr(hpo.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(hpo.mlflow, "log_metrics", lambda *a, **k: None)

    # The mlops helpers are imported by name into the hpo module namespace, so we
    # patch the names as the module sees them rather than the source module.
    # log_training_dataset returns a dataset handle in real life, so we hand back a
    # harmless sentinel.
    monkeypatch.setattr(hpo, "stamp_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(hpo, "stamp_experiment_metadata", lambda *a, **k: None)
    monkeypatch.setattr(hpo, "log_training_dataset", lambda *a, **k: "ds-handle")
    monkeypatch.setattr(hpo, "log_estimator_to_mlflow", lambda *a, **k: None)
    return monkeypatch


#
# _cv_fit_and_score
#


def _make_splits(X, y, groups, n_splits=2, seed=42):
    """Materialise a list of (train_idx, val_idx) tuples from grouped CV.

    WHY a concrete list and not the lazy generator: `_cv_fit_and_score`
    consumes the iterator once, and we sometimes inspect the fold count in the
    test, so we freeze the splits up front.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups=groups))


def test_cv_fit_and_score_returns_two_floats_in_unit_range(grouped_dataset):
    """The scorer returns `(mean_test_f1, mean_train_f1)` both in [0, 1].

    WHAT we assert: two plain Python floats, each a valid F1 score. WHY this
    matters: every HPO path leans on this single scorer, so its contract (two
    finite floats in the unit interval) is load-bearing for the whole module.
    """
    X, y, _, _, pids = grouped_dataset
    splits = _make_splits(X, y, pids, n_splits=2)

    mean_test, mean_train = hpo._cv_fit_and_score(
        "Logistic Regression",
        {"C": 1.0, "penalty": "l2"},
        splits,
        X,
        y,
        pos_weight=1.0,
        _HAS_CUDA=False,
    )

    assert isinstance(mean_test, float)
    assert isinstance(mean_train, float)
    assert 0.0 <= mean_test <= 1.0
    assert 0.0 <= mean_train <= 1.0


def test_cv_fit_and_score_train_f1_not_below_test_f1(grouped_dataset):
    """Train F1 is typically at least the held-out F1 for this easy problem.

    WHY: a model scored on the same rows it trained on should not do worse than
    on unseen rows for a well-behaved fit. We allow a small tolerance so
    fold-to-fold noise on a small sample cannot flake the test.
    """
    X, y, _, _, pids = grouped_dataset
    splits = _make_splits(X, y, pids, n_splits=2)

    mean_test, mean_train = hpo._cv_fit_and_score(
        "Logistic Regression",
        {"C": 1.0, "penalty": "l2"},
        splits,
        X,
        y,
        pos_weight=1.0,
        _HAS_CUDA=False,
    )

    assert mean_train >= mean_test - 0.05


def test_cv_fit_and_score_invokes_callback_once_per_fold(grouped_dataset):
    """`report_each_fold` fires once per fold with the documented args.

    The source contract (read from the docstring and body) is that the callback
    receives `(fold_idx, cumulative_mean_test_f1, cumulative_mean_train_f1)`
    after every fold, with `fold_idx` starting at 1. WHY we check this: the
    Ray Tune trial relies on this per-fold callback to hand control back to
    ASHA, so the count and argument shapes are part of the public behaviour.
    """
    X, y, _, _, pids = grouped_dataset
    splits = _make_splits(X, y, pids, n_splits=3)

    calls = []

    def _record(fold_idx, test_f1, train_f1):
        calls.append((fold_idx, test_f1, train_f1))

    hpo._cv_fit_and_score(
        "Logistic Regression",
        {"C": 1.0, "penalty": "l2"},
        splits,
        X,
        y,
        pos_weight=1.0,
        _HAS_CUDA=False,
        report_each_fold=_record,
    )

    # One callback per fold, fold indices 1..n in order.
    assert len(calls) == 3
    assert [c[0] for c in calls] == [1, 2, 3]
    # Every reported cumulative metric is a finite float in the unit range.
    for _, test_f1, train_f1 in calls:
        assert isinstance(test_f1, float)
        assert isinstance(train_f1, float)
        assert 0.0 <= test_f1 <= 1.0
        assert 0.0 <= train_f1 <= 1.0


#
# deterministic_grid_fallback
#


def test_deterministic_grid_fallback_record_shape(
    grouped_dataset, lr_search_space, patched_mlflow
):
    """The fallback returns a dict keyed by model with all documented fields.

    WHAT: one entry per requested model, each carrying `best_params`,
    `best_cv_f1`, the four `val_*` metrics, predictions, the fitted model,
    and a `cv_results` frame. WHY: downstream sections (§6.5+) consume this
    exact record schema regardless of which HPO path produced it, so a
    missing key is a real regression.
    """
    X, y, Xv, yv, pids = grouped_dataset

    out = hpo.deterministic_grid_fallback(
        top_models=["Logistic Regression"],
        search_spaces=lr_search_space,
        X_train=X,
        y_train=y,
        train_patient_ids=pids,
        X_val=Xv,
        y_val=yv,
        pos_weight=1.0,
        _HAS_CUDA=False,
        n_samples=3,
        seed=42,
    )

    assert set(out.keys()) == {"Logistic Regression"}
    rec = out["Logistic Regression"]
    expected_keys = {
        "best_params",
        "best_cv_f1",
        "val_f1",
        "val_auc_roc",
        "val_auc_pr",
        "val_recall",
        "val_precision",
        "y_pred",
        "y_prob",
        "model",
        "cv_results",
    }
    assert expected_keys.issubset(rec.keys())

    # Spot-check the scalar metric types and ranges.
    for k in ("best_cv_f1", "val_f1", "val_auc_roc", "val_auc_pr", "val_recall", "val_precision"):
        assert isinstance(rec[k], float)
        assert 0.0 <= rec[k] <= 1.0

    # cv_results is the GridSearchCV-compatible frame with a rank column.
    assert isinstance(rec["cv_results"], pd.DataFrame)
    assert {"mean_test_score", "mean_train_score", "rank_test_score"}.issubset(
        rec["cv_results"].columns
    )
    # best_params came out of the search space we passed in.
    assert rec["best_params"]["C"] in lr_search_space["Logistic Regression"]["C"]


def test_deterministic_grid_fallback_is_deterministic(
    grouped_dataset, lr_search_space, patched_mlflow
):
    """Two runs with the same seed yield identical best params and CV F1.

    WHY: the function name promises determinism. The config sampling is driven
    by `random.Random(seed)` and the CV splitter is seeded, so back-to-back
    runs must agree exactly. This guards against an accidental unseeded RNG
    sneaking into the sampling loop.
    """
    X, y, Xv, yv, pids = grouped_dataset
    kwargs = dict(
        top_models=["Logistic Regression"],
        search_spaces=lr_search_space,
        X_train=X,
        y_train=y,
        train_patient_ids=pids,
        X_val=Xv,
        y_val=yv,
        pos_weight=1.0,
        _HAS_CUDA=False,
        n_samples=3,
        seed=42,
    )

    out_a = hpo.deterministic_grid_fallback(**kwargs)
    out_b = hpo.deterministic_grid_fallback(**kwargs)

    rec_a = out_a["Logistic Regression"]
    rec_b = out_b["Logistic Regression"]
    assert rec_a["best_params"] == rec_b["best_params"]
    assert rec_a["best_cv_f1"] == rec_b["best_cv_f1"]
    assert rec_a["val_f1"] == rec_b["val_f1"]


def test_deterministic_grid_fallback_skips_unknown_models(
    grouped_dataset, lr_search_space, patched_mlflow
):
    """A requested model absent from the search space is skipped, not crashed.

    WHY: the loop guards `if name not in search_spaces: continue`. We feed a
    bogus model name alongside the real one and confirm only the known model
    appears in the output.
    """
    X, y, Xv, yv, pids = grouped_dataset

    out = hpo.deterministic_grid_fallback(
        top_models=["Logistic Regression", "Nonexistent Model"],
        search_spaces=lr_search_space,
        X_train=X,
        y_train=y,
        train_patient_ids=pids,
        X_val=Xv,
        y_val=yv,
        pos_weight=1.0,
        _HAS_CUDA=False,
        n_samples=2,
        seed=42,
    )

    assert set(out.keys()) == {"Logistic Regression"}


#
# sequential_cv_fallback
#


def test_sequential_cv_fallback_shapes_and_keys(grouped_dataset, lr_search_space):
    """The synchronous CV fallback returns `(df_results, best_cfg)`.

    WHAT we assert:
      * a DataFrame with one row per sampled config and the score columns the
        Tuner-compatible path expects, plus a raw `config` column,
      * a `best_cfg` dict whose keys came from the search space.

    WHY no MLflow patch here: `sequential_cv_fallback` does no logging at all.
    It runs `n_samples` configs through the shared scorer and returns the
    frame plus the winning config, so the deterministic sklearn path runs
    as-is.
    """
    X, y, _, _, pids = grouped_dataset
    n_samples = 4

    df, best_cfg = hpo.sequential_cv_fallback(
        name="Logistic Regression",
        n_samples=n_samples,
        search_spaces=lr_search_space,
        X_train=X,
        y_train=y,
        train_patient_ids=pids,
        pos_weight=1.0,
        _HAS_CUDA=False,
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    assert {"config", "mean_test_score", "mean_train_score"}.issubset(df.columns)
    # Scores are valid F1 values.
    assert df["mean_test_score"].between(0.0, 1.0).all()

    assert isinstance(best_cfg, dict)
    assert set(best_cfg.keys()) == set(lr_search_space["Logistic Regression"].keys())
    # The winner is the row with the highest held-out score.
    assert best_cfg == df.loc[df["mean_test_score"].idxmax(), "config"]


def test_sequential_cv_fallback_is_deterministic(grouped_dataset, lr_search_space):
    """Re-running with the same seed reproduces the same winning config.

    WHY: like the grid fallback, the config sampling here is driven by
    `random.Random(seed)` and a seeded splitter, so identical inputs must
    produce identical winners.
    """
    X, y, _, _, pids = grouped_dataset
    common = dict(
        name="Logistic Regression",
        n_samples=4,
        search_spaces=lr_search_space,
        X_train=X,
        y_train=y,
        train_patient_ids=pids,
        pos_weight=1.0,
        _HAS_CUDA=False,
        seed=42,
    )

    df_a, best_a = hpo.sequential_cv_fallback(**common)
    df_b, best_b = hpo.sequential_cv_fallback(**common)

    assert best_a == best_b
    assert df_a["mean_test_score"].tolist() == df_b["mean_test_score"].tolist()


#
# to_tune_space
#


def test_to_tune_space_maps_choices_and_stamps_model_name():
    """Each grid entry becomes a Ray Tune categorical and `__model__` is set.

    WHAT the source does: returns a dict that starts with `{"__model__":
    name}` and then converts every `list` of candidate values into a
    `tune.choice(...)` sampler. WHY we check `__model__`: the Tune trial
    pops that key back out to recover the model name, so it must be present and
    equal to the name we passed.
    """
    from ray.tune.search.sample import Categorical

    space = {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]}
    out = hpo.to_tune_space("Logistic Regression", space)

    # The model name is stamped under the reserved key.
    assert out["__model__"] == "Logistic Regression"
    # Every original grid key survives and is now a Tune sampler.
    for key in space:
        assert key in out
        assert isinstance(out[key], Categorical)
    # The reserved key plus the two grid keys, nothing extra.
    assert set(out.keys()) == {"__model__", "C", "penalty"}


#
# asha_pruning_report
#


def test_asha_pruning_report_columns_and_rows(tmp_path):
    """The audit frame has one row per trial and the four documented columns.

    WHAT we feed: a synthetic `{model: results_dataframe}` mapping shaped like
    `tune.Tuner.fit().get_dataframe()` output, carrying the keys
    `make_tune_trial` emits (`mean_test_score`, `mean_train_score`,
    `folds_completed`). WHY tmp_path: the function persists the long-form
    frame to parquet, so we point `out_path` at a temp file rather than the
    repo's `data/` directory. `print_tables=False` keeps stdout quiet.
    """
    per_model_results = {
        "MLP": pd.DataFrame(
            {
                "mean_test_score": [0.50, 0.55, 0.60],
                "mean_train_score": [0.70, 0.72, 0.75],
                "folds_completed": [1, 2, 3],
            }
        ),
        "Random Forest": pd.DataFrame(
            {
                "mean_test_score": [0.62, 0.64],
                "mean_train_score": [0.80, 0.82],
                "folds_completed": [3, 3],
            }
        ),
    }
    out_path = tmp_path / "hpo_diag.parquet"

    diag = hpo.asha_pruning_report(
        per_model_results,
        out_path=str(out_path),
        print_tables=False,
    )

    # One row per trial across both models (3 + 2 = 5).
    assert isinstance(diag, pd.DataFrame)
    assert list(diag.columns) == [
        "model",
        "folds_completed",
        "mean_test_score",
        "mean_train_score",
    ]
    assert len(diag) == 5
    assert set(diag["model"].unique()) == {"MLP", "Random Forest"}
    # folds_completed is coerced to int.
    assert diag["folds_completed"].dtype.kind in ("i", "u")
    # The parquet sidecar was written and round-trips to the same row count.
    assert out_path.exists()
    assert len(pd.read_parquet(out_path)) == 5


def test_asha_pruning_report_skips_empty_frames(tmp_path):
    """Empty or None per-model frames contribute no rows.

    WHY: the loop guards `if df is None or len(df) == 0: continue`. A model
    whose Tuner returned nothing should not appear in the audit. We also confirm
    a fully empty input yields an empty frame and writes no file (the
    persistence branch is gated on `len(diag) > 0`).
    """
    out_path = tmp_path / "empty_diag.parquet"

    diag = hpo.asha_pruning_report(
        {"MLP": None, "Random Forest": pd.DataFrame()},
        out_path=str(out_path),
        print_tables=False,
    )

    assert isinstance(diag, pd.DataFrame)
    assert len(diag) == 0
    # Nothing was persisted because there were no rows to write.
    assert not out_path.exists()


def test_asha_pruning_report_out_path_none_skips_persistence(tmp_path):
    """Passing `out_path=None` returns the frame without writing anything.

    WHY: callers that want only the in-memory audit can disable persistence.
    The function must honour that and still return a well-formed frame.
    """
    per_model_results = {
        "MLP": pd.DataFrame(
            {
                "mean_test_score": [0.5],
                "mean_train_score": [0.7],
                "folds_completed": [2],
            }
        )
    }

    diag = hpo.asha_pruning_report(
        per_model_results,
        out_path=None,
        print_tables=False,
    )

    assert len(diag) == 1
    assert diag.iloc[0]["model"] == "MLP"
    assert diag.iloc[0]["folds_completed"] == 2
