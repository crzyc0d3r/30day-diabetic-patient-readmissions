"""Unit tests for `helpers.mlp_train`.

This module covers the single-fold MLP trainer NB08 calls inside its stacking
out-of-fold loop, plus the two small loaders that feed it the production MLP's
recorded epoch budget and hyperparameter config:

* `train_mlp_one_fold`: trains one ReadmissionMLP fold on the CPU.
* `nb07_best_epoch`: returns the recorded best epoch, or a default.
* `nb08_best_config`: returns the recorded HPO config, or a fallback.
* `_set_torch_determinism`: pins torch RNG knobs for reproducibility.

WHY these matter: the OOF loop must reproduce the production MLP exactly. If
the trainer were non-deterministic, or the loaders silently returned wrong
defaults, the meta-learner would be fed mismatched probabilities and its
scores would be biased. The tests pin determinism, the documented fallback
schemas, and the trainer's output contract.

Everything runs CPU-only with tiny epoch counts so the suite finishes in
seconds.
"""

from __future__ import annotations

import joblib
import numpy as np
import torch
from torch import nn

from helpers.mlp_train import (
    nb07_best_epoch,
    nb08_best_config,
    train_mlp_one_fold,
)

# Force every torch path onto the CPU. The test venv ships a CUDA build, but we
# run CPU-only so results never depend on a GPU being present.
CPU = torch.device("cpu")


# Small helpers
def _make_fold_data(rng, n_rows: int = 40, n_features: int = 6):
    """Build a tiny numeric (X, y) fold with both classes present.

    The trainer accepts raw numpy and converts to tensors internally, so we
    hand it float32 features and a 0/1 label vector. We force both classes in
    so the pos_weight-rescaled loss has a real gradient to follow.
    """
    X = rng.normal(0.0, 1.0, size=(n_rows, n_features)).astype(np.float32)
    signal = X @ rng.normal(0.0, 1.0, size=n_features)
    y = (signal > np.median(signal)).astype(np.float32)
    return X, y


# train_mlp_one_fold
def test_train_mlp_one_fold_returns_module_with_finite_outputs(rng):
    """The trainer returns an `nn.Module` that forwards finite, well-shaped logits.

    WHAT: we train one fold on the CPU with a tiny epoch budget, then forward a
    fresh batch through the returned model and assert the output is
    `(batch, 1)` finite logits. WHY: NB08 immediately scores the returned
    model on held-out rows, so it must be a real, usable module, not a
    half-built object or one producing NaNs.
    """
    X, y = _make_fold_data(rng)
    n_features = X.shape[1]

    model = train_mlp_one_fold(
        X, y,
        n_features=n_features,
        device=CPU,
        pos_weight=1.0,
        n_epochs=2,
        batch_size=16,
    )

    # Contract: a torch module comes back.
    assert isinstance(model, nn.Module)

    # Forward a fresh batch (>1 row, because BatchNorm needs batch stats in
    # train mode). We switch to eval to make a single deterministic pass safe.
    model.eval()
    x = torch.from_numpy(X).float().to(CPU)
    with torch.no_grad():
        out = model(x)

    # One logit per row.
    assert tuple(out.shape) == (len(X), 1)
    assert torch.all(torch.isfinite(out))


def test_train_mlp_one_fold_is_deterministic(rng):
    """Two runs with identical inputs produce identical first-batch logits.

    WHY this is the headline test: the trainer calls `_set_torch_determinism`
    and seeds the DataLoader generator from `helpers.constants.SEED` precisely
    so the OOF fold model is bit-reproducible. If determinism regressed, the
    meta-learner would see different OOF probabilities on every re-run. We
    train twice on the same data and compare a forward pass exactly.
    """
    X, y = _make_fold_data(rng)
    n_features = X.shape[1]

    def _train_and_score():
        model = train_mlp_one_fold(
            X, y,
            n_features=n_features,
            device=CPU,
            pos_weight=1.0,
            n_epochs=2,
            batch_size=16,
        )
        model.eval()
        x = torch.from_numpy(X).float().to(CPU)
        with torch.no_grad():
            return model(x)

    first = _train_and_score()
    second = _train_and_score()

    # Bit-identical: the determinism setup must make the two runs match exactly.
    assert torch.equal(first, second)


# nb07_best_epoch
def test_nb07_best_epoch_missing_file_returns_default(tmp_path):
    """A non-existent results file falls back to the supplied default.

    WHY: NB07 may not have run yet (no joblib on disk), and the OOF loop must
    still produce a result. We point the loader at a path that does not exist
    and confirm it returns the `default` rather than raising.
    """
    missing = tmp_path / "does_not_exist.joblib"
    assert nb07_best_epoch(results_path=str(missing), default=15) == 15


def test_nb07_best_epoch_reads_recorded_epoch(tmp_path):
    """A joblib file recording `best_epoch` returns that value.

    The source reads the `best_epoch` key from a dict and accepts it only
    when it is a positive int. We write exactly that schema and confirm the
    recorded value wins over the default.
    """
    results_path = tmp_path / "mlp_results.joblib"
    joblib.dump({"best_epoch": 7}, results_path)
    assert nb07_best_epoch(results_path=str(results_path), default=15) == 7


def test_nb07_best_epoch_invalid_value_falls_back(tmp_path):
    """A non-positive or non-int `best_epoch` is rejected in favour of the default.

    The source guards with `isinstance(val, int) and val > 0`. A zero or a
    string must not slip through, so we confirm the default is returned.
    """
    results_path = tmp_path / "mlp_results.joblib"
    joblib.dump({"best_epoch": 0}, results_path)
    assert nb07_best_epoch(results_path=str(results_path), default=15) == 15


# nb08_best_config
#
# The documented keys every returned config must carry.
CONFIG_KEYS = {"lr", "weight_decay", "dropout", "batch_size"}


def test_nb08_best_config_missing_file_returns_fallback(tmp_path):
    """A non-existent results file returns the hand-coded fallback dict.

    WHAT: we point the loader at a missing path and confirm it returns a dict
    carrying exactly the four documented keys (lr, weight_decay, dropout,
    batch_size). WHY: without a recorded config the OOF loop still needs a
    valid hyperparameter set to hand to `train_mlp_one_fold`.
    """
    missing = tmp_path / "does_not_exist.joblib"
    cfg = nb08_best_config(results_path=str(missing))
    assert set(cfg.keys()) == CONFIG_KEYS
    # The documented fallback values.
    assert cfg["lr"] == 1e-3
    assert cfg["weight_decay"] == 1e-4
    assert cfg["dropout"] == 0.3
    assert cfg["batch_size"] == 512


def test_nb08_best_config_reads_recorded_config(tmp_path):
    """A joblib file recording `best_config` returns those values, typed.

    The source reads the `best_config` sub-dict and coerces each field
    (lr/weight_decay/dropout to float, batch_size to int). We record a distinct
    config and confirm it overrides the fallback and is correctly typed.
    """
    results_path = tmp_path / "mlp_results.joblib"
    recorded = {
        "best_config": {
            "lr": 5e-4,
            "weight_decay": 2e-5,
            "dropout": 0.5,
            "batch_size": 256,
        }
    }
    joblib.dump(recorded, results_path)

    cfg = nb08_best_config(results_path=str(results_path))
    assert set(cfg.keys()) == CONFIG_KEYS
    assert cfg["lr"] == 5e-4
    assert cfg["weight_decay"] == 2e-5
    assert cfg["dropout"] == 0.5
    # batch_size is coerced to a plain int.
    assert cfg["batch_size"] == 256
    assert isinstance(cfg["batch_size"], int)


# _set_torch_determinism (private, imported defensively)
def test_set_torch_determinism_makes_seeded_forward_passes_match(rng):
    """`_set_torch_determinism` makes two seeded weight inits identical.

    WHY guarded import: the leading underscore marks this as private, so we
    test it only if it is importable. WHAT: after calling the setter we build
    a fresh `ReadmissionMLP` and forward a fixed batch, twice. Because the
    setter pins `torch.manual_seed`, both weight initialisations, and thus
    both forward passes, must be bit-identical.
    """
    try:
        from helpers.mlp_train import _set_torch_determinism
    except ImportError:
        import pytest

        pytest.skip("_set_torch_determinism is not importable")

    from helpers.models import ReadmissionMLP

    n_features = 6
    # A fixed input so the only source of variation is weight initialisation.
    x = torch.randn(8, n_features, device=CPU)

    def _seeded_forward():
        # Re-pin every torch RNG knob, then build a fresh model so its weights
        # are drawn from the freshly seeded RNG.
        _set_torch_determinism(seed=42)
        model = ReadmissionMLP(n_features).to(CPU)
        model.eval()
        with torch.no_grad():
            return model(x)

    first = _seeded_forward()
    second = _seeded_forward()

    # Identical determinism state must yield bit-identical outputs.
    assert torch.equal(first, second)
