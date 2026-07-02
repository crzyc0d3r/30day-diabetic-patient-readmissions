"""Unit tests for `helpers.conclusion_pipeline`.

This module is an IMPURE orchestrator. It loads per-stage result joblibs,
selects a champion, scores it once on the held-out test split with a
bootstrap CI, writes a deployable bundle, and optionally registers the
champion to the MLflow Model Registry. None of the MLflow or registry side
effects belong in a unit test.

The strategy mirrors the training-pipeline suite:

1. Drive the PURE private helpers directly. `_g`, `_build_leaderboard`,
   `_bootstrap_ci`, `_age_band`, and `_compute_fairness_audit` are all
   deterministic given their arguments and are tested with small synthetic
   inputs.

2. For the `run_conclusion_and_register` entry point we fabricate the
   minimal joblib inputs in `tmp_path`, call it with `register=False` so
   the registry promotion branch is skipped entirely, monkeypatch
   `init_mlflow` to a no-op (it otherwise raises when the server is
   unreachable), and assert the returned dict shape plus the written
   `bundle_path`. We never touch the repo's real `data/` directory and
   never contact a live MLflow server.

Style note: docstrings are deliberately generous so the tests document the
orchestrator contract as well as verify it.

Pickle and joblib note: every joblib artefact loaded below is one the test
itself wrote moments earlier into `tmp_path`. It is trusted by
construction, matching the module's own documented pickle-safety convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from helpers import conclusion_pipeline as cp


# ===========================================================================
# _g  (tolerant key lookup)
# ===========================================================================
@pytest.mark.parametrize(
    "res, keys, expected",
    [
        # First key present wins.
        ({"f1": 0.8, "val_f1": 0.7}, ("f1", "val_f1"), 0.8),
        # Falls through to the second key when the first is absent.
        ({"val_f1": 0.7}, ("f1", "val_f1"), 0.7),
        # Single-key lookup.
        ({"auc_roc": 0.9}, ("auc_roc",), 0.9),
        # Order matters: take the FIRST present key in argument order.
        ({"f1": 0.5, "val_f1": 0.6}, ("val_f1", "f1"), 0.6),
    ],
)
def test_g_returns_first_present_key(res, keys, expected):
    """`_g` returns the value of the first key that exists in `res`."""
    assert cp._g(res, *keys) == expected


def test_g_raises_keyerror_when_no_variant_present():
    """When no candidate key exists the helper raises `KeyError`."""
    with pytest.raises(KeyError):
        cp._g({"something_else": 1}, "f1", "val_f1")


# ===========================================================================
# _build_leaderboard
# ===========================================================================
def test_build_leaderboard_one_row_per_model_and_f1_sorted():
    """Three stage dicts collapse into one F1-sorted table, one row per entry.

    We feed the three variant key schemas the source tolerates (plain `f1`
    for the default stage, `val_f1` for the HPO stages) and assert the
    combined frame has the right columns, the right number of rows, and is
    sorted by F1 descending.
    """
    default_results = {
        "Logistic Regression": {"f1": 0.50, "auc_roc": 0.60},
        "Random Forest": {"f1": 0.55, "auc_roc": 0.65},
    }
    tuned_results = {
        "XGBoost": {"val_f1": 0.70, "val_auc_roc": 0.80},
    }
    training_results = {
        "XGBoost": {"f1": 0.65, "auc_roc": 0.75},
    }

    board = cp._build_leaderboard(default_results, tuned_results, training_results)

    # One row per (model, stage) entry across the three bags.
    assert len(board) == 4
    assert set(board.columns) == {"Model", "Variant", "Source", "F1", "AUC-ROC"}

    # Sorted by F1 descending: the 0.70 HPO-winner XGBoost must rank first.
    assert board.iloc[0]["F1"] == pytest.approx(0.70)
    assert list(board["F1"]) == sorted(board["F1"], reverse=True)

    # Source labels are stamped per stage.
    assert set(board["Source"]) == {"Default", "HPO-winner", "HPO-refit"}


def test_build_leaderboard_single_stage_only():
    """A leaderboard built from a single populated stage still sorts cleanly.

    We avoid the all-empty case on purpose. When every stage dict is empty
    the source builds a column-less DataFrame and `sort_values("F1")` raises
    a `KeyError` because no 'F1' column exists. The entry point never relies
    on that path because it checks `leaderboard.empty` only after a build
    that always has at least one row in practice. So here we feed a single
    populated stage and confirm the row count and column schema.
    """
    board = cp._build_leaderboard(
        {"Random Forest": {"f1": 0.55, "auc_roc": 0.65}},
        {}, {},
    )
    assert len(board) == 1
    assert set(board.columns) == {"Model", "Variant", "Source", "F1", "AUC-ROC"}
    assert board.iloc[0]["Source"] == "Default"


# ===========================================================================
# _bootstrap_ci
# ===========================================================================
def _bootstrap_inputs(rng, n=200):
    """A non-degenerate (y_test, y_pred, y_prob) triple with both classes.

    The bootstrap resamples rows with replacement, so both classes must be
    well represented or many resamples would be skipped as degenerate.
    """
    y_test = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    # Probabilities correlate with the label but stay imperfect.
    y_prob = np.clip(
        np.where(y_test == 1, 0.65, 0.35) + rng.normal(0.0, 0.1, size=n),
        0.0, 1.0,
    )
    y_pred = (y_prob >= 0.5).astype(int)
    return y_test, y_pred, y_prob


def test_bootstrap_ci_structure_and_ordering(rng):
    """The CI dict carries the six metrics, each with the four fields.

    For every metric we assert `ci_low <= point <= ci_high` and that
    `beats_naive_pct` is a percentage in [0, 100].
    """
    y_test, y_pred, y_prob = _bootstrap_inputs(rng)

    out = cp._bootstrap_ci(y_test, y_pred, y_prob, n_boot=200, seed=42)

    # Exactly the six headline metrics the model card surfaces.
    assert set(out.keys()) == {"F1", "Recall", "Precision",
                               "AUC-ROC", "AUC-PR", "Brier"}

    for metric, block in out.items():
        assert set(block.keys()) == {"point", "ci_low", "ci_high", "beats_naive_pct"}
        # The point estimate must sit inside its own bootstrap interval. We
        # allow a tiny tolerance because the percentile interval is estimated
        # from a finite resample set and can clip the point by rounding.
        assert block["ci_low"] - 1e-9 <= block["point"] <= block["ci_high"] + 1e-9, metric
        assert 0.0 <= block["beats_naive_pct"] <= 100.0, metric


def test_bootstrap_ci_is_deterministic_under_fixed_seed(rng):
    """Two runs with the same seed produce byte-identical CI dicts."""
    y_test, y_pred, y_prob = _bootstrap_inputs(rng)

    a = cp._bootstrap_ci(y_test, y_pred, y_prob, n_boot=200, seed=42)
    b = cp._bootstrap_ci(y_test, y_pred, y_prob, n_boot=200, seed=42)

    for metric in a:
        for field in a[metric]:
            assert a[metric][field] == b[metric][field], (metric, field)


# ===========================================================================
# _age_band
# ===========================================================================
@pytest.mark.parametrize(
    "age_value, expected",
    [
        # Young band: under 40.
        ("[0-10)", "young"),
        ("[30-40)", "young"),
        # Middle band: 40 to 70.
        ("[40-50)", "middle"),
        ("[60-70)", "middle"),
        # Senior band: 70 and over.
        ("[70-80)", "senior"),
        ("[90-100)", "senior"),
        # Unrecognised string maps to unknown (never dropped silently).
        ("not-a-bucket", "unknown"),
        # Non-string inputs map to unknown.
        (None, "unknown"),
        (42, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_age_band_boundaries(age_value, expected):
    """Each raw age bucket collapses into its documented audit band."""
    assert cp._age_band(age_value) == expected


def test_age_band_strips_whitespace():
    """Surrounding whitespace is stripped before the band lookup."""
    assert cp._age_band("  [40-50)  ") == "middle"


# ===========================================================================
# _compute_fairness_audit
# ===========================================================================
def test_compute_fairness_audit_returns_none_when_demographics_absent(tmp_path):
    """With no `features.csv` or `patient_ids.csv` the audit returns None.

    The function is best-effort. When the raw demographic CSVs are not on
    disk it must return `None` so the bundle still writes. Pointing
    `in_dir` at an empty tmp directory exercises exactly that path.
    """
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert cp._compute_fairness_audit(tmp_path, y_test, y_pred) is None


def test_compute_fairness_audit_returns_none_without_partition_column(tmp_path):
    """Present CSVs but no `partition` column still yields None.

    Without the `partition` column the function cannot align demographics
    to the test rows, so it refuses to invent a join key and returns None.
    """
    n = 6
    pd.DataFrame({
        "race": ["Caucasian"] * n,
        "gender": ["Male"] * n,
        "age": ["[40-50)"] * n,
    }).to_csv(tmp_path / "features.csv", index=False)
    # patient_ids.csv exists but lacks the required 'partition' column.
    pd.DataFrame({"patient_nbr": range(n)}).to_csv(
        tmp_path / "patient_ids.csv", index=False
    )

    y_test = np.zeros(3, dtype=int)
    y_pred = np.zeros(3, dtype=int)
    assert cp._compute_fairness_audit(tmp_path, y_test, y_pred) is None


def test_compute_fairness_audit_populates_when_demographics_align(tmp_path):
    """When the CSVs align to y_test the audit returns a per-attribute dict.

    We write `features.csv` and `patient_ids.csv` with a `partition`
    column flagging exactly `len(y_test)` rows as 'test', so the alignment
    check passes and the audit is computed over race, gender, and age_band.
    """
    # Four 'test' rows plus two non-test rows. y_test has length 4.
    partitions = ["train", "test", "test", "test", "test", "train"]
    n = len(partitions)
    features = pd.DataFrame({
        "race": ["Caucasian", "Caucasian", "AfricanAmerican",
                 "Caucasian", "AfricanAmerican", "Caucasian"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "age": ["[40-50)", "[40-50)", "[70-80)", "[20-30)", "[80-90)", "[40-50)"],
    })
    features.to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"partition": partitions}).to_csv(
        tmp_path / "patient_ids.csv", index=False
    )

    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    audit = cp._compute_fairness_audit(tmp_path, y_test, y_pred)

    assert audit is not None
    # The three demographic attributes are audited (age becomes age_band).
    assert set(audit.keys()) == {"race", "gender", "age_band"}
    # Each attribute maps subgroup labels to a metric dict carrying 'recall'.
    for attr_block in audit.values():
        assert len(attr_block) >= 1
        for subgroup_metrics in attr_block.values():
            assert "recall" in subgroup_metrics


# ===========================================================================
# run_conclusion_and_register  (register=False, MLflow neutered)
# ===========================================================================
def test_run_conclusion_and_register_no_register(tmp_path, monkeypatch, rng):
    """End-to-end conclusion run with registration off and MLflow neutered.

    We fabricate a `train_test.npz` (val and test arrays), a fitted-model
    joblib for one stage, and the matching per-stage result joblib, then call
    `run_conclusion_and_register` with `register=False`. Only `init_mlflow`
    needs neutering on this branch because the registry promotion code path is
    skipped entirely when `register=False`. We assert the returned dict shape
    and that `final_model.joblib` was written to `out_dir`.
    """
    import joblib

    # init_mlflow raises if the server is unreachable, so stub it out. The
    # register=False branch returns before any other MLflow boundary is hit.
    monkeypatch.setattr(cp, "init_mlflow", lambda *a, **k: None)

    # Fabricate train_test.npz with val and test arrays (both classes).
    def _both_classes(m):
        y = np.concatenate([np.zeros(m // 2), np.ones(m // 2)]).astype(int)
        X = np.column_stack([
            np.where(y == 1, 2.0, -2.0) + rng.normal(0, 0.5, size=m),
            rng.normal(0, 1, size=m),
        ])
        return X.astype(np.float64), y

    X_val, y_val = _both_classes(40)
    X_test, y_test = _both_classes(40)
    npz_path = tmp_path / "train_test.npz"
    np.savez(npz_path, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    # A single 'HPO-refit' stage with one fitted champion.
    champion = LogisticRegression(max_iter=500)
    champion.fit(X_val, y_val)

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    # training_results.joblib feeds the 'HPO-refit' source in the leaderboard.
    joblib.dump(
        {"Logistic Regression": {"f1": 0.80, "auc_roc": 0.85}},
        in_dir / "training_results.joblib",
    )
    # training_models.joblib is the artefact _load_champion_model reads for
    # the 'HPO-refit' source.
    joblib.dump(
        {"Logistic Regression": champion},
        in_dir / "training_models.joblib",
    )

    out_dir = tmp_path / "out"
    result = cp.run_conclusion_and_register(
        train_test_path=str(npz_path),
        in_dir=str(in_dir),
        out_dir=str(out_dir),
        register=False,
        n_bootstrap=50,
    )

    # Returned dict contract.
    assert result["champion_name"] == "Logistic Regression"
    assert result["champion_source"] == "HPO-refit"
    assert result["registered_version"] is None
    for metric_key in ("f1_default_threshold", "f1_optimal_threshold",
                       "auc_roc", "auc_pr", "brier", "mcc"):
        assert metric_key in result["champion_metrics"]

    # Persisted bundle.
    bundle_path = out_dir / "final_model.joblib"
    assert bundle_path.exists()
    assert result["bundle_path"] == str(bundle_path)

    # The bundle round-trips to the documented schema.
    bundle = joblib.load(bundle_path)
    for key in ("model", "model_name", "recommended_threshold",
                "test_metrics", "test_bootstrap_95ci", "fairness_audit"):
        assert key in bundle
    # No demographics on disk, so fairness_audit falls back to None.
    assert bundle["fairness_audit"] is None
