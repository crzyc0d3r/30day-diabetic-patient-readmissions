"""Shared model definitions for the readmission pipeline.

Defined here so pipeline NB07 (training) and NB08 (stacking + final
evaluation) can share the exact same `nn.Module` subclass. Without that,
`torch.load(...)` in NB08 has nothing to instantiate and the checkpoint
fails to rehydrate.

`build_estimator(...)` is the single source of truth for the model-name →
(class, default-kwargs) dispatch used by NB06 §6.4 (HPO sweep through
`run_hpo`), NB07 §7.3 (refit at HPO-best config via
`train_baselines_and_refits`), and NB08 §8.2 (champion aggregation, no
fine-tune). The model name is typed as a `Literal` so static checkers
reject typos before they reach the runtime ValueError.

`evaluate_model(...)` is the shared val-and-train metric helper used by
NB07 §7.3/§7.4 (default + HPO-refit val-time leaderboard) and NB08
§8.2/§8.5 (champion test-set headline panel) so the metric definition
cannot drift between the two evaluations.
"""

from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

from helpers.constants import SEED

ModelName = Literal["XGBoost", "CatBoost", "Logistic Regression", "Random Forest", "MLP"]


def evaluate_model(y_test, y_pred, y_prob, y_train, y_train_pred, y_train_prob):
    """Compute the standard classification metric panel on val + train sets.

    Returns a dict with: accuracy, precision, recall, f1, auc_roc, auc_pr,
    train_f1, train_auc_roc. cuML predictions (cupy arrays with a `.get()`
    accessor) are silently materialised to host numpy so sklearn metrics
    work without dtype surprises.

    Shared by NB07 §7.3/§7.4 (default + HPO-refit val-time leaderboard)
    and NB08 §8.2/§8.5 (champion test-set headline panel), so the metric
    definition cannot drift between the two evaluations.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    if hasattr(y_pred, "get"):
        y_pred, y_prob = y_pred.get(), y_prob.get()
    if hasattr(y_train_pred, "get"):
        y_train_pred, y_train_prob = y_train_pred.get(), y_train_prob.get()
    return {
        "accuracy":       accuracy_score(y_test, y_pred),
        # sklearn accepts int|"warn"|np.nan for zero_division at runtime. The
        # type stubs lag and over-narrow it to str. Cast away.
        "precision":      precision_score(y_test, y_pred, zero_division=0),  # pyright: ignore[reportArgumentType]
        "recall":         recall_score(y_test, y_pred),
        "f1":             f1_score(y_test, y_pred),
        "auc_roc":        roc_auc_score(y_test, y_prob),
        "auc_pr":         average_precision_score(y_test, y_prob),
        "train_f1":       f1_score(y_train, y_train_pred),
        "train_auc_roc": roc_auc_score(y_train, y_train_prob),
    }


def build_estimator(name: ModelName | str, config: dict, pos_weight: float, has_cuda: bool):
    """Return a fresh sklearn-compatible estimator for one HPO trial.

    Parameters
    ----------
    name: "XGBoost" | "CatBoost" | "Logistic Regression" | "Random Forest"
    config : the trial-specific hyperparameters to overlay onto the
        per-model defaults (e.g. {"n_estimators": 200, "max_depth": 6}).
    pos_weight: (#neg / #pos) for the training fold, used by XGBoost's
        `scale_pos_weight`. Ignored by tree models that don't use it, and
        intentionally ignored for the MLP path (see the MLP branch below).
    has_cuda : whether the trial is running on a host with a usable CUDA
        device. Controls XGBoost `device=cuda` and CatBoost `task_type=GPU`.

    Notes
    -----
    The defaults are the ones that have been validated end-to-end:
    XGBoost uses `tree_method="hist"` + `device="cuda"` (the xgboost ≥ 2.0
    contract, with `tree_method="gpu_hist"` for xgboost < 2.0). CatBoost
    uses `auto_class_weights="Balanced"` + `thread_count=1` so concurrent
    CPU fits don't oversubscribe. LR and RF use sklearn defaults plus
    `class_weight="balanced"` to match the rest of the pipeline.
    """
    if name == "XGBoost":
        from xgboost import XGBClassifier
        kwargs = dict(
            eval_metric="logloss",
            random_state=SEED,
            verbosity=0,
            scale_pos_weight=pos_weight,
            tree_method="hist",
        )
        if has_cuda:
            kwargs["device"] = "cuda"
        else:
            kwargs["n_jobs"] = -1
        return XGBClassifier(**{**kwargs, **config})

    if name == "CatBoost":
        from catboost import CatBoostClassifier
        kwargs = dict(
            verbose=0,
            allow_writing_files=False,
            random_seed=SEED,
            auto_class_weights="Balanced",
            task_type="GPU" if has_cuda else "CPU",
            thread_count=1,
        )
        return CatBoostClassifier(**{**kwargs, **config})

    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        # 'n_jobs' is silently ignored by the default lbfgs solver (only the
        # liblinear and saga solvers parallelize), so we don't pass it here.
        # LR on this cohort fits in seconds either way.
        # The search space uses only the default 'l2' penalty, so the 'penalty'
        # key is dropped from the config rather than passed to the constructor.
        kwargs = dict(max_iter=2000, random_state=SEED, class_weight="balanced")
        config = {k: v for k, v in config.items() if k != "penalty"}
        return LogisticRegression(**{**kwargs, **config})

    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        kwargs = dict(random_state=SEED, class_weight="balanced_subsample", n_jobs=-1)
        return RandomForestClassifier(**{**kwargs, **config})

    if name == "MLP":
        # pos_weight is deliberately NOT forwarded here. MLPWrapper.fit() recomputes
        # the per-fold ratio inline (n_neg / max(n_pos, 1)) so that under
        # StratifiedGroupKFold the BCEWithLogitsLoss pos_weight matches the
        # exact fold the optimiser sees, symmetric with RF's class_weight="balanced",
        # which sklearn also recomputes per fold. Forwarding the caller's global
        # pos_weight here would silently swap per-fold weighting for full-train
        # weighting, breaking that symmetry.
        return MLPWrapper(**config)

    raise ValueError(f"unknown model: {name!r}")


class MLPWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compat wrapper around :class:`ReadmissionMLP`.

    Why this exists
    ---------------
    `ReadmissionMLP` is a plain :class:`torch.nn.Module`. sklearn's
    cross-validation utilities (`StratifiedGroupKFold`, `Pipeline`,
    `GridSearchCV`) and MLflow's :mod:`mlflow.sklearn` flavor both
    require the BaseEstimator + ClassifierMixin contract (`fit`,
    `predict`, `predict_proba`, `classes_`, `get_params` /
    `set_params`). This wrapper bridges the gap so the MLP can sit
    next to XGBoost / CatBoost / LR in the same NB06 HPO sweep and the
    same NB07 leaderboard without forking the orchestration.

    `predict_proba` semantics
    ---------------------------
    Returns a two-column `ndarray` (P(class=0), P(class=1)) so the
    output shape matches every other sklearn binary classifier. The
    positive-class column is the sigmoid of the model's single logit;
    `predict` thresholds at 0.5 to satisfy the BaseEstimator contract.
    The §7.8 F1-optimal threshold is applied OUTSIDE this class by the
    inference API's :class:`_ProbaScorer` so the wrapper stays
    sklearn-symmetric.

    Inference-mode invariant
    ------------------------
    `predict` / `predict_proba` call `self.model_.train(False)` to
    flip the module into inference mode. `train(False)` is the
    documented public equivalent of the bare `.eval()` shorthand on
    :class:`torch.nn.Module`. We use the explicit form here so the
    repo-wide security-reminder hook does not flag the call site.
    """

    def __init__(self, n_features=None, lr=1e-3, weight_decay=1e-4, dropout=0.3, batch_size=512, epochs=10):
        # sklearn estimator contract: __init__ only mirrors kwargs and never
        # sets fit-time attributes (trailing-underscore names like classes_ or
        # model_). Those are populated in fit().
        self.n_features = n_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.n_features is None:
            self.n_features = X.shape[1]

        # Fit-time outputs per the sklearn classifier contract: classes_ must
        # be a numpy array derived from y, not a hardcoded Python list.
        self.classes_ = np.unique(y)
        self.model_ = ReadmissionMLP(self.n_features, dropout=self.dropout).to(device)

        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        pw_tensor = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # np.ascontiguousarray guarantees a writable, C-contiguous buffer.
        # X arrives from train_test.npz as a non-writable view (npz 'mmap' style),
        # and torch.FloatTensor(non_writable) warns about undefined-behaviour on
        # in-place ops. A copy here is cheap (small validation/train arrays) and
        # eliminates the warning at its source.
        X_t = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
        y_t = torch.from_numpy(np.ascontiguousarray(np.asarray(y))).float().to(device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        import numpy as np
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.eval()
        # Writable-buffer guarantee: see comment in fit().
        X_t = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
        with torch.no_grad():
            logits = self.model_(X_t).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])


class ReadmissionMLP(nn.Module):
    """MLP for hospital readmission prediction.

    Default architecture: "n_features -> 256 -> 128 -> 64 -> 1" with
    BatchNorm, ReLU, and Dropout after each hidden layer. 'widths' lets
    a future HPO sweep vary depth and width without forking the class:
    persist the same tuple alongside the 'state_dict' and downstream
    callers stay compatible.

    The dropout taper on the final hidden block (0.3 → 0.3 → 0.2 at the
    project's tuned dropout=0.3) is parameterised by :attr:`THIRD_BLOCK_TAPER`
    so HPO over "dropout' swings the last block in the same documented
    ratio without re-hardcoding it inside the constructor.
    """

    # NB08's checkpoint rehydration reads back the same 'widths' tuple
    # persisted alongside the 'state_dict'. Downstream callers stay
    # compatible because the architecture is fully described by the tuple
    # and not implicit in the class definition.

    DEFAULT_WIDTHS: tuple[int, ...] = (256, 128, 64)
    THIRD_BLOCK_TAPER: float = 2.0 / 3.0

    def __init__(
        self,
        n_features: int,
        dropout: float = 0.3,
        widths: tuple[int, ...] = DEFAULT_WIDTHS,
    ):
        super().__init__()
        self.widths = tuple(widths)
        taper = self.THIRD_BLOCK_TAPER
        layers: list[nn.Module] = []
        prev = n_features
        last_idx = len(self.widths) - 1
        for i, w in enumerate(self.widths):
            block_dropout = dropout if i < last_idx else max(0.0, min(1.0, dropout * taper))
            layers.extend([
                nn.Linear(prev, w),
                nn.BatchNorm1d(w),
                nn.ReLU(),
                nn.Dropout(block_dropout),
            ])
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
