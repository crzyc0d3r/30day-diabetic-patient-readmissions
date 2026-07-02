"""Unit tests for `helpers/model_loading.py`.

`model_loading` is the shared flavor dispatcher used by BOTH the inference API
and the retrain DAG's champion gate. Its whole reason to exist is that a model
logged with a non-sklearn flavor must not be loaded with
`mlflow.sklearn.load_model` (which raises "Model does not have the sklearn
flavor"). So the tests assert the dispatch picks the right native loader.

Strategy mirrors `test_mlops_helpers`:
- The pure parts (`read_flavors`, a YAML file read, and `predict_labels`, numpy
  label coercion) are exercised directly with `tmp_path` / arrays.
- `load_model_any_flavor` is tested by monkeypatching only the EXTERNAL
  boundary (`mlflow.artifacts.download_artifacts`, `importlib.import_module`,
  `mlflow.pyfunc.load_model`) with record-only fakes. We assert the dispatch
  decision, not the correctness of MLflow's loaders. `mlflow`/`torch` are
  optional in some envs, so those paths `importorskip`.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from helpers import model_loading as m


# --- read_flavors (pure file read) ------------------------------------------

def test_read_flavors_parses_mlmodel(tmp_path):
    (tmp_path / "MLmodel").write_text(
        "flavors:\n  xgboost:\n    xgb_version: 2.0\n  python_function:\n    loader_module: mlflow.xgboost\n"
    )
    flavors = m.read_flavors(str(tmp_path))
    assert set(flavors) == {"xgboost", "python_function"}


def test_read_flavors_missing_file_returns_empty(tmp_path):
    assert m.read_flavors(str(tmp_path)) == {}


# --- predict_labels (pure numpy coercion) -----------------------------------

class _LabelClf:
    """sklearn-API stub whose predict already returns class labels."""
    def predict(self, X):
        return np.array([0, 1, 1, 0])


class _ProbaVecClf:
    """pyfunc-style stub returning P(class=1) as a float vector."""
    def predict(self, X):
        return np.array([0.2, 0.8, 0.51, 0.49])


class _ProbaMatClf:
    """pyfunc-style stub returning a 2-column proba matrix."""
    def predict(self, X):
        return np.array([[0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])


def test_predict_labels_passthrough_for_label_estimator():
    out = m.predict_labels(_LabelClf(), "sklearn", np.zeros((4, 3)))
    assert out.tolist() == [0, 1, 1, 0]


def test_predict_labels_thresholds_proba_vector():
    out = m.predict_labels(_ProbaVecClf(), "pyfunc", np.zeros((4, 3)))
    assert out.tolist() == [0, 1, 1, 0]  # 0.51 -> 1, 0.49 -> 0


def test_predict_labels_argmaxes_proba_matrix():
    out = m.predict_labels(_ProbaMatClf(), "pyfunc", np.zeros((4, 3)))
    assert out.tolist() == [0, 1, 1, 0]


def test_predict_labels_pytorch_forward_path():
    torch = pytest.importorskip("torch")

    class _Net(torch.nn.Module):
        def forward(self, x):  # 2-logit output -> argmax
            n = x.shape[0]
            return torch.tensor([[2.0, -1.0]] * (n // 2 + n % 2) + [[-1.0, 2.0]] * (n // 2))[:n]

    out = m.predict_labels(_Net(), "pytorch", np.zeros((2, 3), dtype="float32"))
    assert out.tolist() == [0, 1]


# --- load_model_any_flavor (boundary monkeypatched) -------------------------

def test_load_model_any_flavor_dispatches_native(monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    (tmp_path / "MLmodel").write_text("flavors:\n  xgboost: {}\n  python_function: {}\n")
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", lambda uri: str(tmp_path))

    sentinel = object()
    real_import = importlib.import_module

    def fake_import(name, *a, **k):
        if name == "mlflow.xgboost":
            import types
            return types.SimpleNamespace(load_model=lambda uri: sentinel)
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    model, flavor = m.load_model_any_flavor("models:/medi-watch-readmission/7")
    assert flavor == "xgboost"
    assert model is sentinel


def test_load_model_any_flavor_picks_first_listed_native(monkeypatch, tmp_path):
    """When several native flavors are present, FLAVOR_LOADERS order decides.
    catboost should win over sklearn here (it's earlier in the tuple)."""
    mlflow = pytest.importorskip("mlflow")
    (tmp_path / "MLmodel").write_text("flavors:\n  sklearn: {}\n  catboost: {}\n  python_function: {}\n")
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", lambda uri: str(tmp_path))

    chosen = {}
    real_import = importlib.import_module

    def fake_import(name, *a, **k):
        chosen["name"] = name
        import types
        return types.SimpleNamespace(load_model=lambda uri: "M")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    model, flavor = m.load_model_any_flavor("models:/x/1")
    assert flavor == "catboost"          # earlier in FLAVOR_LOADERS than sklearn
    assert chosen["name"] == "mlflow.catboost"
