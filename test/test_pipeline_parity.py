"""Unit tests for `helpers/pipeline_parity.py`.

WHAT this file covers

`pipeline_parity.py` ships two guards the NB05 cell and the retrain DAG both
call:

  * `assert_pipeline_parity` compares a unified inference Pipeline's selector
    mask, OneHotEncoder vocabulary, and StandardScaler statistics against the
    standalone artefacts the other notebooks train against.
  * `assert_pipeline_loads_in_fresh_process` round-trips the pipeline through
    joblib and a fresh Python subprocess to catch the closure /
    FunctionTransformer pickling failure mode.

WHY we build REAL sklearn objects: the parity guard reads concrete fitted
attributes (`selector.get_support()`, `ohe.categories_`, `scaler.mean_`,
`scaler.scale_`). Mock objects would not exercise the real comparison logic,
so we fit genuine small artefacts on shared toy data, confirm the guard passes
when everything agrees, and confirm it raises `AssertionError` when we
perturb exactly one artefact.

STYLE NOTE: no em dashes, no semicolons, "program" never the British spelling.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from helpers import pipeline_parity


#
# Toy data and artefact construction
#
# WHY a single helper that builds BOTH the pipeline and the matching standalone
# artefacts from the same numbers: the whole point of the parity guard is that
# the two paths fit on identical data. Building them together guarantees they
# agree by construction, which is the happy-path baseline we then deliberately
# perturb in the failure-path tests.

CAT_COLS = ["race", "gender"]
# The categorical block is two columns. The OHE in both the pipeline and the
# standalone artefact must learn the same vocabulary per column.
_RAW_CAT = np.array(
    [
        ["A", "M"],
        ["B", "F"],
        ["A", "F"],
        ["C", "M"],
        ["B", "M"],
        ["C", "F"],
    ],
    dtype=object,
)


def _build_artifacts(n_select=2):
    """Build a fitted inference Pipeline plus matching standalone OHE and scaler.

    WHAT the returned pipeline contains, in the order the parity guard reads its
    named steps:

      * `preprocessor`: a ColumnTransformer whose `cat` transformer is a
        OneHotEncoder fit on the categorical block. WHY name it `cat`: the
        guard does `named_transformers_["cat"]` to find the encoder.
      * `scaler`: a StandardScaler fit on the full numeric-plus-encoded
        matrix. WHY this order: the guard compares `scaler.mean_` against the
        standalone scaler index-by-index, so both must be fit on the same
        stacked column order (num, cat).
      * `selector`: a SelectKBest whose support mask is compared against the
        manual MI mask.

    Returns a dict carrying every argument `assert_pipeline_parity` needs, all
    consistent by construction.
    """
    rng = np.random.default_rng(0)
    n = _RAW_CAT.shape[0]

    # Numeric block: three columns of mild signal so SelectKBest has something to
    # rank and StandardScaler has non-degenerate spread to learn.
    num = rng.normal(size=(n, 3))
    # A close-to-separable label so f_classif produces a stable ranking.
    y = np.array([0, 1, 0, 1, 1, 0])

    #  Standalone OHE: the vocabulary NB06-NB09 train against 
    standalone_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = standalone_ohe.fit_transform(_RAW_CAT)

    # The full design matrix is (num, cat) stacked left to right. Both the
    # standalone scaler and the pipeline scaler must agree on this order.
    full = np.hstack([num, cat_encoded])

    #  Standalone scaler: fit on the same stacked matrix 
    standalone_scaler = StandardScaler().fit(full)

    #  Manual MI mask: which of the scaled columns we keep 
    # We mimic the manual selection of section 5.8 by fitting the same SelectKBest
    # the pipeline uses and reusing its support as the "manual" ground truth. WHY
    # reuse it: parity check 1 passes only when the pipeline selector overlaps the
    # manual mask on all n_select features, so the baseline must agree.
    scaled_full = standalone_scaler.transform(full)
    manual_selector = SelectKBest(score_func=f_classif, k=n_select).fit(scaled_full, y)
    manual_mask = manual_selector.get_support()

    #  The unified inference pipeline 
    # preprocessor: pass the numeric columns through untouched and one-hot the
    # categorical columns under the transformer name "cat". We fit it on a combined
    # object array so a single ColumnTransformer reproduces the same (num, cat)
    # column order as the standalone hstack above.
    n_num = num.shape[1]
    combined = np.empty((n, n_num + len(CAT_COLS)), dtype=object)
    combined[:, :n_num] = num
    combined[:, n_num:] = _RAW_CAT

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(range(n_num))),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(range(n_num, n_num + len(CAT_COLS))),
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=n_select)),
        ]
    )
    pipeline.fit(combined, y)

    return {
        "full_inference_pipeline": pipeline,
        "standalone_ohe": standalone_ohe,
        "standalone_scaler": standalone_scaler,
        "manual_mask": manual_mask,
        "cat_cols": CAT_COLS,
        "n_select": n_select,
    }


@pytest.fixture
def parity_artifacts():
    """Consistent-by-construction artefacts for the happy-path parity test."""
    return _build_artifacts(n_select=2)


#
# assert_pipeline_parity: happy path
#


def test_parity_passes_when_artifacts_agree(parity_artifacts):
    """When every artefact agrees, the guard returns None and does not raise.

    WHY None specifically: the function has no explicit return, so on success it
    falls off the end and yields None. A passing parity check is the whole
    contract, so we assert both "no exception" and "returns None".
    """
    result = pipeline_parity.assert_pipeline_parity(**parity_artifacts)
    assert result is None


#
# assert_pipeline_parity: failure paths
#


def test_parity_raises_on_mask_drift(parity_artifacts):
    """A manual mask that does not overlap the pipeline selector aborts.

    Parity check 1 asserts `overlap == n_select`. We flip the manual mask so
    its True positions no longer line up with the pipeline's selected features,
    which drops the overlap below `n_select` and must raise.
    """
    args = dict(parity_artifacts)
    # Invert the boolean mask so the kept columns no longer coincide.
    args["manual_mask"] = ~args["manual_mask"]

    with pytest.raises(AssertionError, match="MI selection drift"):
        pipeline_parity.assert_pipeline_parity(**args)


def test_parity_raises_on_ohe_category_drift(parity_artifacts):
    """A standalone OHE fit on different categories trips the vocabulary check.

    Parity check 2 compares `categories_` per column. We refit the standalone
    OHE on data containing an extra category the pipeline never saw, so at least
    one column's vocabulary differs and the guard raises.
    """
    args = dict(parity_artifacts)
    # Same two columns but with an unseen category "Z" injected, so the learnt
    # vocabulary diverges from the pipeline's encoder.
    drifted_cat = np.array(
        [["A", "M"], ["B", "F"], ["Z", "F"], ["C", "M"], ["B", "M"], ["C", "F"]],
        dtype=object,
    )
    drifted_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    drifted_ohe.fit(drifted_cat)
    args["standalone_ohe"] = drifted_ohe

    with pytest.raises(AssertionError, match="OHE"):
        pipeline_parity.assert_pipeline_parity(**args)


def test_parity_raises_on_scaler_stat_drift(parity_artifacts):
    """A scaler fit on shifted data trips the mean_/scale_ closeness check.

    Parity check 3 requires `scaler.mean_` and `scaler.scale_` to match
    within tight tolerances. We refit the standalone scaler on the same-shape
    but numerically shifted matrix so its statistics diverge from the pipeline's
    scaler, which must raise.
    """
    args = dict(parity_artifacts)
    original = args["standalone_scaler"]
    n_features = original.mean_.shape[0]

    # Fit a fresh scaler on data with the same column count but a large offset,
    # guaranteeing mean_ drifts well outside the 1e-6 tolerance.
    rng = np.random.default_rng(1)
    shifted = rng.normal(loc=100.0, scale=5.0, size=(20, n_features))
    args["standalone_scaler"] = StandardScaler().fit(shifted)

    with pytest.raises(AssertionError, match="StandardScaler"):
        pipeline_parity.assert_pipeline_parity(**args)


#
# assert_pipeline_loads_in_fresh_process
#


def test_fresh_process_load_returns_success_marker(parity_artifacts):
    """A clean, picklable pipeline loads in a fresh subprocess and returns stdout.

    WHAT we assert: the helper returns a string containing the `LOAD_OK`
    marker the subprocess prints. WHY a real subprocess is acceptable here: our
    pipeline is built entirely from importable sklearn classes with no
    `__main__` closures or FunctionTransformers, so it pickles and reloads
    cleanly in a fresh interpreter. This exercises the success branch end to
    end.
    """
    marker = pipeline_parity.assert_pipeline_loads_in_fresh_process(
        parity_artifacts["full_inference_pipeline"]
    )

    assert isinstance(marker, str)
    assert "LOAD_OK" in marker
    # The subprocess also reports the step names, so the selector survived.
    assert "selector" in marker


def test_fresh_process_load_raises_on_unloadable_pipeline(parity_artifacts):
    """A pipeline referencing a `__main__` closure fails the cross-process load.

    WHY this is the failure mode the guard exists to catch: a FunctionTransformer
    wrapping a function defined in the test module pickles by reference, not by
    value. The fresh subprocess has no such function in its `__main__`, so
    `joblib.load` raises and the helper's `assert "LOAD_OK" in res.stdout`
    fails. We wrap a trivial module-less closure to trigger exactly that.

    Two outcomes both count as "the unloadable pipeline did not silently pass":

      * `joblib.dump` itself refuses to pickle the closure by reference and
        raises a `PicklingError`. The bad pipeline never reaches disk, which
        is a legitimate way of failing closed.
      * dump succeeds but the fresh subprocess cannot resolve the closure on
        load, so `res.stdout` lacks `LOAD_OK` and the helper raises
        `AssertionError`.

    We accept either, and skip only if some pickling backend round-trips the
    closure end to end (in which case there is no failure to observe).
    """
    import pickle

    from sklearn.preprocessing import FunctionTransformer

    # A closure defined here. Pickled by reference, it points at this test
    # module's local scope, which the fresh subprocess cannot resolve, so the
    # round-trip must fail at dump or at load.
    def _local_only_transform(X):
        return X

    broken = Pipeline(
        steps=[
            ("preprocessor", FunctionTransformer(_local_only_transform)),
            ("selector", parity_artifacts["full_inference_pipeline"].named_steps["selector"]),
        ]
    )

    # The helper raises AssertionError when the subprocess cannot load. If the
    # pickle backend cannot even dump the closure, joblib raises a PicklingError
    # from inside the helper before the subprocess runs. Either is a valid
    # "failed closed" result, so we accept both exception types.
    with pytest.raises((AssertionError, pickle.PicklingError, AttributeError, TypeError)):
        pipeline_parity.assert_pipeline_loads_in_fresh_process(broken)
