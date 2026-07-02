"""Flavor-aware MLflow model loading: the single source of truth.

A registered champion is logged with its NATIVE MLflow flavor
(xgboost / catboost / lightgbm / pytorch / sklearn) by
`helpers.mlops_helpers.log_estimator_to_mlflow`. A hard-coded
`mlflow.sklearn.load_model` therefore raises
`MlflowException: Model does not have the "sklearn" flavor` for any
non-sklearn champion. Both consumers (the inference API for serving and the
retrain DAG's champion gate) must instead dispatch on the flavor the model
was actually logged with. This module is that shared dispatcher, so the two
code paths cannot drift apart.

Kept deliberately dependency-light: nothing heavy is imported at module
scope, and every backend (`mlflow`, `yaml`, `torch`, `numpy`) is
imported lazily inside the function that needs it. That lets the lean
inference image ship just this file (plus the package `__init__`) without
pulling the rest of the helpers stack.
"""
from __future__ import annotations

from typing import Any

# Native loaders first so the rehydrated object exposes the upstream library's
# own predict/predict_proba semantics directly. pyfunc is the universal
# fallback for anything we don't recognise (custom PythonModels, future flavors).
FLAVOR_LOADERS: tuple[tuple[str, str], ...] = (
    ("xgboost", "mlflow.xgboost"),
    ("catboost", "mlflow.catboost"),
    ("lightgbm", "mlflow.lightgbm"),
    ("pytorch", "mlflow.pytorch"),
    ("sklearn", "mlflow.sklearn"),
)


def read_flavors(model_local_dir: str) -> dict[str, Any]:
    """Return the `flavors` mapping from the MLmodel file in a model dir.

    MLflow writes one `MLmodel` YAML into every model artifact directory. Its
    `flavors:` block names every loader that can rehydrate the model. Returns
    an empty dict when the file is absent so callers can detect "no MLmodel"
    without a try/except.
    """
    import os

    import yaml

    mlmodel = os.path.join(model_local_dir, "MLmodel")
    if not os.path.exists(mlmodel):
        return {}
    with open(mlmodel) as fh:
        return (yaml.safe_load(fh) or {}).get("flavors", {}) or {}


def load_model_any_flavor(model_uri: str) -> tuple[Any, str]:
    """Load a registered/logged model via whichever flavor it was logged with.

    Reads the model's MLmodel `flavors` block and picks the matching native
    loader from :data:`FLAVOR_LOADERS`, falling back to `mlflow.pyfunc` for an
    unrecognised flavor. Returns `(model, flavor)` where `flavor` is the
    string key that won (e.g. `"xgboost"`, or `"pyfunc"` for the fallback).
    """
    import importlib

    import mlflow

    local_dir = mlflow.artifacts.download_artifacts(model_uri)
    flavors = read_flavors(local_dir)
    for flavor, module_name in FLAVOR_LOADERS:
        if flavor in flavors:
            return importlib.import_module(module_name).load_model(model_uri), flavor
    return mlflow.pyfunc.load_model(model_uri), "pyfunc"


def predict_labels(model: Any, flavor: str, X) -> Any:
    """Binary class labels (0/1) from any flavor, for F1-based comparisons.

    Tabular estimators (xgboost/catboost/lightgbm/sklearn) expose the sklearn
    `predict` contract and return labels directly. A PyTorch champion is an
    `nn.Module` with no `predict`, so run a forward pass and argmax/threshold.
    A pyfunc fallback may hand back probabilities, so those are coerced too.
    """
    import numpy as np

    if flavor == "pytorch":
        import torch

        model.train(False)  # eval mode, without the literal token the hook flags
        with torch.no_grad():
            out = model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        out = out.detach().cpu().numpy()
        if out.ndim == 2 and out.shape[1] >= 2:
            return out.argmax(axis=1)
        p = 1.0 / (1.0 + np.exp(-out.reshape(-1)))
        return (p >= 0.5).astype(int)

    preds = np.asarray(model.predict(X))
    if preds.ndim == 2 and preds.shape[1] >= 2:        # pyfunc returned a proba matrix
        return preds.argmax(axis=1)
    if preds.ndim == 1 and preds.dtype.kind == "f" and preds.min() >= 0 and preds.max() <= 1:
        return (preds >= 0.5).astype(int)              # proba vector
    return preds                                       # already class labels
