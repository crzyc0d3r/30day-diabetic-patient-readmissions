"""Pipeline-parity guards shared by NB05 §5.9.1 and the retrain DAG.

Extracted from `pipeline/05_split_encode_scale_select.ipynb` so the
production `retrain_on_drift` DAG and any future programmatic caller can
import the same parity checks the notebook runs interactively. The
notebook and the DAG share this module, so the witness output is identical
across both call sites.

Two helpers:

  * `assert_pipeline_parity`. Three structural checks that the unified
    inference pipeline's selector / OHE / scaler match the standalone
    artefacts NB06-NB09 train against. Any drift aborts the cell.
  * `assert_pipeline_loads_in_fresh_process`. Round-trip the pipeline
    through joblib + a fresh subprocess to catch the
    `FunctionTransformer`/closure failure mode where a pipeline pickles
    fine in-kernel and silently fails to load inside the inference
    container.
"""

from __future__ import annotations

import os as _pp_os
import subprocess as _pp_subprocess
import sys as _pp_sys
import tempfile as _pp_tempfile

import joblib as _pp_joblib
import numpy as _pp_np


def assert_pipeline_parity(
    full_inference_pipeline,
    standalone_ohe,
    standalone_scaler,
    manual_mask,
    cat_cols,
    n_select,
):
    """Assert the pipeline OHE, scaler, and MI mask match the standalone artefacts.

    Three checks fire in order. Any failure aborts the cell with an
    `AssertionError` whose message identifies which drift triggered.
    Returns `None` on success and prints a short witness for each check.
    """
    # Parity check 1: SelectKBest mask matches the manual MI selection in §5.8.
    pipeline_mask = full_inference_pipeline.named_steps["selector"].get_support()
    overlap = (pipeline_mask & manual_mask).sum()
    print(f"Pipeline vs manual MI selection: {overlap}/{n_select} features in common")
    assert overlap == n_select, "MI selection drift between manual and pipeline path"

    # Parity check 2: pipeline OHE learned the same categories as the
    # standalone OHE from §5.6. Notebooks 6 through 9 train against the
    # standalone OHE, the inference API loads the unified pipeline. Both
    # must see the same vocabulary.
    pipeline_ohe = full_inference_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    assert len(pipeline_ohe.categories_) == len(standalone_ohe.categories_), (
        f"OHE category-count drift: pipeline={len(pipeline_ohe.categories_)} "
        f"standalone={len(standalone_ohe.categories_)}"
    )
    for i, (pcats, scats) in enumerate(zip(pipeline_ohe.categories_, standalone_ohe.categories_)):
        assert _pp_np.array_equal(pcats, scats), (
            f"OHE column {i} ({cat_cols[i]}) categories drift: "
            f"pipeline={list(pcats)[:3]}... standalone={list(scats)[:3]}..."
        )
    print(f"OHE parity: {len(pipeline_ohe.categories_)} columns match across pipeline + standalone")

    # Parity check 3: pipeline StandardScaler learned the same statistics.
    # The standalone scaler was fit on hstack([num, cat]). The pipeline
    # ColumnTransformer order is also (num, cat), so the per-column means
    # and scales align by index.
    pipeline_scaler = full_inference_pipeline.named_steps["scaler"]
    assert pipeline_scaler.mean_.shape == standalone_scaler.mean_.shape, (
        f"scaler.mean_ shape drift: pipeline={pipeline_scaler.mean_.shape} "
        f"standalone={standalone_scaler.mean_.shape}"
    )
    assert _pp_np.allclose(pipeline_scaler.mean_, standalone_scaler.mean_, rtol=1e-6, atol=1e-9), \
        "StandardScaler mean_ drift between pipeline and standalone scaler.joblib"
    assert _pp_np.allclose(pipeline_scaler.scale_, standalone_scaler.scale_, rtol=1e-6, atol=1e-9), \
        "StandardScaler scale_ drift between pipeline and standalone scaler.joblib"
    print(
        f"Scaler parity: mean_ + scale_ match across pipeline + standalone "
        f"(n={pipeline_scaler.mean_.shape[0]})"
    )


def assert_pipeline_loads_in_fresh_process(full_inference_pipeline):
    """Round-trip the pipeline through joblib + a fresh subprocess.

    Catches the `FunctionTransformer`/closure failure mode where a
    pipeline pickles fine in-kernel and silently fails to load in the
    inference container with `AttributeError: __main__.select_features`.
    Returns the subprocess stdout on success, raises on failure.
    """
    with _pp_tempfile.NamedTemporaryFile(suffix=".jl", delete=False) as f:
        tmp = f.name
    _pp_joblib.dump(full_inference_pipeline, tmp)
    res = _pp_subprocess.run(
        [_pp_sys.executable, "-c",
         f"import joblib; P = joblib.load({tmp!r}); "
         f"sel = P.named_steps['selector']; "
         f"print('LOAD_OK steps=', [n for n,_ in P.steps], 'selected=', int(sel.get_support().sum()))"],
        capture_output=True, text=True,
    )
    _pp_os.unlink(tmp)
    assert "LOAD_OK" in res.stdout, (
        f"cross-process load FAILED:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    )
    print(res.stdout.strip())
    return res.stdout
