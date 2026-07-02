"""Shared single-fold MLP trainer.

`train_mlp_one_fold` is the function the stacking-OOF loop in pipeline NB08
calls once per `GroupKFold` fold so the meta-learner sees out-of-fold MLP
probabilities instead of in-sample ones. The function accepts every
hyperparameter NB06's HPO selects (lr / weight_decay / dropout / batch_size)
so the OOF fold model matches the production MLP's training distribution.
The N-epoch budget is sourced from the production MLP's saved `best_epoch`
when available (`data/mlp_results.joblib`), so the per-fold fit stops at the
same point the production MLP did instead of running an arbitrary 15-epoch
schedule.

The companion helper `nb08_best_config()` loads the full hyperparameter
dict (not just the epoch budget) so the OOF loop in NB08 can hand it
through verbatim:

    cfg = nb08_best_config()
    fold_model = train_mlp_one_fold(
        X_train[tr_idx], y_train[tr_idx],
        n_features=n_feat, device=device, pos_weight=pw_full,
        n_epochs=nb07_best_epoch(), **cfg,
    )
"""

from __future__ import annotations

from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset

from helpers.constants import SEED
from helpers.models import ReadmissionMLP


def _set_torch_determinism(seed: int = SEED) -> None:
    """Set every PyTorch knob the reproducibility docs require for bit-identical re-runs.

    All four calls are needed (see https://pytorch.org/docs/stable/notes/randomness.html):
      * `torch.manual_seed` seeds CPU + CUDA RNGs.
      * `torch.use_deterministic_algorithms(True, warn_only=True)` forces
        deterministic kernels where available and warns (rather than crashes)
        when an op has no deterministic implementation. The warn-only path
        keeps the MLP loop runnable even when a downstream caller uses an op
        that lacks a deterministic kernel on the active backend.
      * `cudnn.deterministic=True` + `cudnn.benchmark=False` are the
        cuDNN-specific knobs documented as "required for full determinism".
        Omitting either silently re-introduces non-determinism on CUDA hosts.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mlp_one_fold(
    X_tr,
    y_tr,
    n_features: int,
    device: torch.device,
    pos_weight: float,
    n_epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
) -> nn.Module:
    """Train one ReadmissionMLP fold and return the fitted model.

    The architecture, loss, optimizer, and batch size mirror pipeline NB07's
    production training path (modulo the early-stop inner-val split, which
    the outer OOF GroupKFold already replaces).
    `dropout`, `lr`, `weight_decay`, and `batch_size` are explicit
    parameters so NB08's OOF loop can pass NB06's HPO-selected `best_config`
    through. Relying on the defaults instead would produce an OOF MLP that
    does not match the production MLP, biasing the meta-learner.

    Design choices (rationale)
    --------------------------
    * **Optimizer: `torch.optim.Adam` (not `AdamW`).** The cohort is
      small (~70k rows after split) and weight decay is already applied
      via Adam's `weight_decay` kwarg. AdamW's decoupled-decay
      formulation would shift the optimum and force a re-run of NB06's
      HPO sweep. Adam keeps the validation surface comparable to the
      paper-default sklearn / XGBoost baselines.
    * **Loss: `BCEWithLogitsLoss(pos_weight=...)` (not FocalLoss).** The
      class imbalance is ~11% positive, well within the range where
      pos_weight-rescaled BCE matches FocalLoss's effective gradient
      shape, and BCE keeps the logit scale interpretable for the §7.8
      threshold sweep. FocalLoss's gamma re-shapes the score
      distribution and would invalidate the persisted F1-optimal cut.
    * **LR schedule: none.** N_EPOCHS is small (default 15) and NB06's
      HPO already selected `lr` per model, so a scheduler would
      compound with the HPO pick. Adding one is a known future
      improvement, deferred until the training-set size grows enough to
      warrant cosine-warmup or one-cycle.
    * **No early stop inside the fold.** The OOF GroupKFold replaces the
      train/val/early-stop pattern. The fold's val partition belongs to
      the meta-learner, not the inner trainer. `nb07_best_epoch()`
      supplies the production MLP's best epoch so the per-fold fits stop
      at the same point.
    """
    # Pin every torch RNG / cuDNN knob before constructing the model or
    # DataLoader so weight init + shuffle are bit-identical across re-runs.
    _set_torch_determinism()
    # DataLoader's shuffle=True consumes torch's global RNG. Passing an
    # explicit generator seeded from helpers.constants.SEED guarantees the
    # mini-batch order is reproducible even if a caller perturbed the global
    # RNG between fold fits.
    loader_gen = torch.Generator().manual_seed(SEED)
    model = ReadmissionMLP(n_features, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    if not torch.is_tensor(X_tr):
        X_tr = torch.FloatTensor(X_tr).to(device)
    if not torch.is_tensor(y_tr):
        y_tr = torch.FloatTensor(y_tr).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True,
        generator=loader_gen,
    )

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


def nb07_best_epoch(results_path: str = "../data/mlp_results.joblib", default: int = 15) -> int:
    """Return the production MLP's best epoch (1-indexed) if recorded, else `default`.

    Pipeline NB07 persists `best_epoch` into `mlp_results.joblib`. If the file
    is absent (NB07 not yet run) or the key is missing (older format), we
    fall back to `default` so the OOF loop still produces a result.
    """
    import joblib
    try:
        results = joblib.load(results_path)
    except Exception:
        return default
    val = results.get("best_epoch") if isinstance(results, dict) else None
    if isinstance(val, int) and val > 0:
        return val
    return default


def nb08_best_config(
    results_path: str = "../data/mlp_results.joblib",
) -> dict:
    """Return NB06's HPO-selected hyperparameter dict if available, else the hand-coded fallback.

    Without this helper the OOF MLP loop would call `train_mlp_one_fold`
    with the defaults `lr=1e-3, batch_size=512, dropout=0.3, weight_decay=1e-4`,
    different from the production MLP whenever NB06's HPO picked anything
    else. Loading `best_config` from `mlp_results.joblib` and passing it
    through keeps the two model populations aligned.
    """
    fallback = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 512,
    }
    import joblib
    try:
        results = joblib.load(results_path)
    except Exception:
        return fallback
    cfg = results.get("best_config") if isinstance(results, dict) else None
    if not isinstance(cfg, dict):
        return fallback
    return {
        "lr":           float(cfg.get("lr",           fallback["lr"])),
        "weight_decay": float(cfg.get("weight_decay", fallback["weight_decay"])),
        "dropout":      float(cfg.get("dropout",      fallback["dropout"])),
        "batch_size":   int(  cfg.get("batch_size",   fallback["batch_size"])),
    }
