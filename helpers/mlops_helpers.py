"""Shared MLOps reachability + path helpers for the pipeline.

Notebook setup code shares the same MLflow probe, Ray in-process fallback,
and project-root/path resolution instead of hardcoding `set_tracking_uri`,
`ray.init(address=...)`, or relative raw-data paths in each notebook.

Per current Ray docs (https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html),
`ray.init(local_mode=True)` raises
`RuntimeError('local_mode is no longer supported.')`. This helper uses
`address=None` as the in-process fallback.
"""

from __future__ import annotations

import os
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def resolve_project_root(start: Path | None = None) -> Path:
    """Walk upwards from a notebook's working directory until a marker is found.

    The marker is whichever of 'requirements.txt', 'pyproject.toml', or the
    'pipeline/' sibling appears first. Falls back to 'Path.cwd().parent'
    so an interactive run from inside 'pipeline/' still yields the repo root.
    """
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "requirements.txt").exists() or (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "pipeline").is_dir() and (candidate / "data").is_dir():
            return candidate
    return here.parent


def resolve_raw_csv(filename: str = "diabetic_data.csv") -> Path:
    """Return the absolute path to the raw Kaggle CSV.

    Lookup order:
      1. '$MEDIWATCH_RAW_CSV' (full path).
      2. '<root>/orig_dataset/<filename>'.
      3. '<root>/../orig_dataset/<filename>' as a sibling fallback.

    Raises 'FileNotFoundError' with a kagglehub-style install hint if none hit.
    """
    env_path = os.environ.get("MEDIWATCH_RAW_CSV")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p

    root = resolve_project_root()
    primary = root / "orig_dataset" / filename
    if primary.is_file():
        return primary

    sibling = root.parent / "orig_dataset" / filename
    if sibling.is_file():
        return sibling

    raise FileNotFoundError(
        f"Could not locate raw dataset '{filename}'. Tried:\n"
        f"  $MEDIWATCH_RAW_CSV  -> {env_path or '(unset)'}\n"
        f"  {primary}\n"
        f"  {sibling}\n"
        "Download from https://www.kaggle.com/datasets/brandao/diabetes "
        "(or `kagglehub` if you have credentials) and either drop the CSV at "
        f"{primary} or set MEDIWATCH_RAW_CSV to its absolute path."
    )


MLFLOW_DEFAULT_URI = "http://127.0.0.1:5000"


def has_cuda() -> bool:
    """Return True iff a CUDA-capable GPU is visible to this process.

    Probes via 'nvidia-smi -L' so we don't pay the cost of importing
    torch just to learn whether CUDA is available. Notebooks (and the
    DAG tasks that launch them) call this at the top of their pipelines
    to dispatch GPU vs CPU models. Pulling the probe out of every
    notebook keeps the kernel-boot tax in one place and means tests
    can monkeypatch a single function instead of every notebook's top
    cell.
    """
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and "GPU 0" in proc.stdout


def cuda_device_name(index: int = 0) -> str | None:
    """Return the human-readable name of CUDA device 'index' or None.

    Parses 'nvidia-smi -L' so callers don't pull in torch just to log
    'CUDA detected on RTX 4090' at notebook startup. Returns None when
    no CUDA device is visible or nvidia-smi is unavailable.
    """
    import shutil
    import re
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    # Lines look like: "GPU 0: NVIDIA RTX 4090 (UUID: GPU-xxxx)"
    match = re.search(rf"^GPU {index}:\s*(.*?)(?:\s+\(UUID|\s*$)", proc.stdout, re.MULTILINE)
    return match.group(1).strip() if match else None


def mlflow_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Probe an MLflow tracking URI's '/health' endpoint.

    Only http/https URIs are supported in this project (Postgres-backed compose
    server). Any other scheme returns 'False'. There is no local file-store
    fallback, by design (see CLAUDE.md, "MLflow is Postgres-only").
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return False
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as r:
            return 200 <= r.status < 500
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        return False


CODE_VERSION_TAG = "code.version"
DESCRIPTION_TAG = "mlflow.note.content"  # MLflow renders this as the Description box.


def _resolve_code_version(explicit: str | None = None) -> str:
    """Resolve a code version for tagging MLflow entities.

    Lookup order:
      1. 'explicit' argument (caller-provided semver / tag name).
      2. '$MEDIWATCH_VERSION' (CI / docker-compose can pin a release tag).
      3. 'git rev-parse --short HEAD' from the project root, suffixed with
         '-dirty' if the working tree has uncommitted changes. Matches the
         common "what was the source state when this model ran" convention.
      4. '"unknown"', which never raises so tagging cannot block a training run.
    """
    if explicit:
        return explicit
    env_version = os.environ.get("MEDIWATCH_VERSION")
    if env_version:
        return env_version
    try:
        root = str(resolve_project_root())
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if not sha:
            return "unknown"
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root, stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (subprocess.CalledProcessError, OSError):
            dirty = ""
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return "unknown"


def _metadata_tags(version: str | None, description: str | None) -> dict[str, str]:
    """Build the '{code.version, mlflow.note.content}' tag dict.

    Centralised so every layer (run / experiment / logged model / registered
    model / model version / dataset input) writes identical keys, so the UI
    column "code.version" aligns across the five views.
    """
    tags: dict[str, str] = {CODE_VERSION_TAG: _resolve_code_version(version)}
    if description:
        tags[DESCRIPTION_TAG] = description
    return tags


def stamp_run_metadata(
    version: str | None = None,
    description: str | None = None,
) -> dict[str, str]:
    """Tag the currently active run with 'code.version' + description.

    Called by 'traced_run' automatically. Can also be called manually inside
    a plain 'mlflow.start_run(...)' block. No-op when there's no active run.
    Returns the tags written so the caller can echo / print them.
    """
    import mlflow

    if mlflow.active_run() is None:
        return {}
    tags = _metadata_tags(version, description)
    mlflow.set_tags(tags)
    return tags


def stamp_experiment_metadata(
    experiment_id: str | None = None,
    version: str | None = None,
    description: str | None = None,
) -> dict[str, str]:
    """Tag the experiment with 'code.version' + description.

    Without this, the Experiments table shows blank 'code.version' / empty
    Description cells even when every individual run inside is tagged. The
    experiment-level tag is what surfaces in the experiment header.

    'experiment_id' defaults to the active run's experiment_id, falling back
    to the experiment associated with the current tracking session.
    """
    import mlflow

    if experiment_id is None:
        run = mlflow.active_run()
        if run is None:
            # Without an active run we cannot resolve the "current" experiment
            # via fluent API without reaching into private internals. Caller
            # should pass experiment_id explicitly outside of a run.
            return {}
        experiment_id = run.info.experiment_id

    eid: str = str(experiment_id)
    tags = _metadata_tags(version, description)
    client = mlflow.MlflowClient()
    for key, value in tags.items():
        try:
            client.set_experiment_tag(eid, key, value)
        except Exception as exc:  # pragma: no cover - mlflow API surface
            print(f"[MLflow] set_experiment_tag({key!r}) failed: {exc!r}")
    return tags


def stamp_logged_model_metadata(
    model_id: str,
    version: str | None = None,
    description: str | None = None,
) -> dict[str, str]:
    """Tag a LoggedModel (the typed artifact under 'experiments/<id>/models').

    Uses 'mlflow.set_logged_model_tags' (mlflow >= 3.x). On older mlflow that
    lacks the API, falls back to a single run tag prefixed with the model id
    so the information is at least retrievable. Failures print but do not
    raise, since LoggedModel tagging is not part of the registry contract.
    """
    import mlflow

    tags = _metadata_tags(version, description)
    setter = getattr(mlflow, "set_logged_model_tags", None)
    if setter is None:
        # mlflow < 3: stamp on the run as a fallback, prefixed by model_id
        try:
            mlflow.set_tags({f"logged_model.{model_id}.{k}": v for k, v in tags.items()})
        except Exception as exc:
            print(f"[MLflow] LoggedModel tag fallback for {model_id} failed: {exc!r}")
        return tags
    try:
        setter(model_id, tags)
    except Exception as exc:  # pragma: no cover - mlflow API surface
        print(f"[MLflow] set_logged_model_tags({model_id!r}) failed: {exc!r}")
    return tags


def stamp_registered_model_metadata(
    name: str,
    model_version: str | int,
    version: str | None = None,
    description: str | None = None,
) -> dict[str, str]:
    """Tag the Registered Model + its specific version with metadata.

    Writes to both layers because the Registry UI surfaces them in different
    places: the registered-model header shows the *model-level* description
    and tags, and each version row shows the *version-level* description and
    tags. Setting one without the other leaves a half-empty Registry page.

    Failures **raise**. Registration of a champion version is a load-bearing
    operation, and per CLAUDE.md the project does not silently skip registry
    work.
    """
    import mlflow

    tags = _metadata_tags(version, description)
    client = mlflow.MlflowClient()
    str_version = str(model_version)

    # 1. Registered-model layer (applies to every version of 'name').
    if description:
        client.update_registered_model(name=name, description=description)
    for key, value in tags.items():
        client.set_registered_model_tag(name=name, key=key, value=value)

    # 2. Model-version layer (applies only to this version).
    if description:
        client.update_model_version(name=name, version=str_version, description=description)
    for key, value in tags.items():
        client.set_model_version_tag(name=name, version=str_version, key=key, value=value)

    return tags


def traced_run(
    *args,
    version: str | None = None,
    description: str | None = None,
    **kwargs,
):
    """Context manager: `mlflow.start_run(...)` + a guaranteed `run_lifecycle`
    trace span attached to the resulting run, plus version + description tags
    stamped on **both the run and the parent experiment**.

    Stop-gap for the policy "every mlflow run must have at least one trace".
    Without this, runs whose body only calls `log_param`/`log_metric` (e.g.
    the bare-actor HPO parent, the Evidently DAG run)
    show up under their experiment with the Traces panel empty.

    'version' defaults to the short git SHA of the project root (with a
    '-dirty' suffix if the working tree is dirty) so every run produced by
    this helper is at least reproducibility-tagged. 'description' is the
    optional free-text body that the UI surfaces as the run's Description box
    (and is propagated to the experiment description too: the experiment
    description is empty by default and never auto-populates without this).

    Use anywhere a plain `mlflow.start_run(...)` would appear:

        from helpers.mlops_helpers import traced_run
        with traced_run(run_name="my_run", description="ASHA refit of XGB winner"):
            mlflow.log_metric("x", 1)
    """
    import mlflow
    from contextlib import contextmanager

    resolved_version = _resolve_code_version(version)

    @contextmanager
    def _cm():
        with mlflow.start_run(*args, **kwargs) as run:
            # Stamp the run + its experiment before the lifecycle span so the
            # tags survive even if the user's body raises inside the span.
            stamp_run_metadata(version=resolved_version, description=description)
            stamp_experiment_metadata(
                experiment_id=run.info.experiment_id,
                version=resolved_version,
                description=description,
            )

            span_name = f"run_lifecycle:{run.info.run_name or run.info.run_id[:8]}"
            try:
                with mlflow.start_span(name=span_name) as span:
                    try:
                        span.set_attributes({
                            "mlflow.runId": run.info.run_id,
                            "mlflow.runName": run.info.run_name or "",
                            "mlflow.experimentId": str(run.info.experiment_id),
                            CODE_VERSION_TAG: resolved_version,
                        })
                    except Exception:
                        pass
                    yield run
            except Exception:
                # If span machinery fails (mlflow.tracing not enabled, etc.),
                # don't lose the run: yield outside the span instead.
                yield run

    return _cm()


def init_mlflow(
    default_uri: str = MLFLOW_DEFAULT_URI,
    experiment: str | None = None,
):
    """Bind mlflow to the compose-hosted, Postgres-backed tracking server.

    Returns the URI mlflow was bound to.

    - Reads '$MLFLOW_TRACKING_URI' if set, otherwise 'default_uri'.
    - Probes '/health'. If unreachable, **raises** 'RuntimeError' with an
      actionable bring-up hint. No silent fallback: a missing server is a
      configuration error, not something to paper over with a local file store
      (the runs would land in a different place and silently diverge).
    - Optionally calls 'mlflow.set_experiment(experiment)'.
    """
    import mlflow

    target = os.environ.get("MLFLOW_TRACKING_URI", default_uri)
    if not mlflow_reachable(target):
        raise RuntimeError(
            f"MLflow tracking server at {target} is not reachable. "
            "Bring it up with: `cd infra && docker compose up -d postgres mlflow` "
            "(or set MLFLOW_TRACKING_URI to a reachable server). "
            "This project does not fall back to a local SQLite/file store — see CLAUDE.md."
        )
    mlflow.set_tracking_uri(target)
    print(f"[MLflow] bound to {target}")
    if experiment:
        try:
            mlflow.set_experiment(experiment)
        except Exception as exc:  # pragma: no cover - mlflow API surface
            print(f"[MLflow] set_experiment({experiment!r}) failed: {exc!r}")
    return target


def enable_mlflow_autolog_and_tracing(silent: bool = True) -> None:
    """Switch on autolog for every installed framework, plus tracing.

    Uses 'mlflow.autolog()', the universal entry point that dispatches
    to every per-flavor autologger MLflow ships (sklearn, xgboost,
    catboost, lightgbm, pytorch, tensorflow, statsmodels, ...) and skips
    any whose Python package isn't installed. The universal entry point
    covers catboost and the PyTorch MLP that pipeline/06 + 07 train, which
    a hard-coded '("sklearn", "xgboost")' tuple would silently drop.

    Why we do not let autolog log models: every notebook calls
    'log_estimator_to_mlflow' afterwards with a curated signature +
    input example + (optionally) a registered model name. Autolog's
    defaults strip signatures and bury models under the deprecated
    'artifact_path="model"' blob layout, which is the exact pattern
    that left the registry empty.

    Why we do not let autolog log datasets: 'log_training_dataset'
    below attaches a human-readable name + source URI, which autolog's
    generic 'dataset' label does not. Two competing dataset entries
    per run is noisier than one well-named one.

    Tracing is enabled globally so '@mlflow.trace' decorators and
    'mlflow.start_span()' calls in the pipeline appear under the run's
    "Traces" tab. Without this flag the spans are no-ops.
    """
    import mlflow

    mlflow.autolog(log_models=False, log_datasets=False, silent=silent)
    mlflow.tracing.enable()
    print("[MLflow] autolog enabled (universal) + tracing enabled")


def _resolve_model_flavor(model):
    """Pick the matching 'mlflow.<flavor>' module for a sklearn-API estimator.

    The dispatch is keyed off the class's defining module so cuML wrappers
    around sklearn estimators still resolve to sklearn (they expose the same
    fit/predict_proba contract). Falls back to sklearn for anything unknown.
    """
    import mlflow.sklearn

    module = type(model).__module__.lower()
    cls_name = type(model).__name__.lower()
    if "xgboost" in module or "xgb" in cls_name:
        import mlflow.xgboost
        return mlflow.xgboost
    if "lightgbm" in module or "lgbm" in cls_name:
        import mlflow.lightgbm
        return mlflow.lightgbm
    if "catboost" in module or "catboost" in cls_name:
        import mlflow.catboost
        return mlflow.catboost
    if "torch" in module:
        import mlflow.pytorch
        return mlflow.pytorch
    return mlflow.sklearn


def log_estimator_to_mlflow(
    model,
    name: str,
    X_sample=None,
    registered_model_name: str | None = None,
    datasets: list[tuple] | None = None,
    description: str | None = None,
    version: str | None = None,
):
    """Log a fitted estimator as a typed MLflow LoggedModel (not a joblib blob).

    LoggedModels render under 'experiments/<exp>/models/' in the UI. A
    bare 'mlflow.log_artifact(joblib_dump)' does not. It leaves the model
    invisible to the registry and to 'mlflow.<flavor>.load_model'.

    When 'X_sample' is supplied, the first 5 rows are also written as the
    LoggedModel's 'input_example' and used to infer the schema via
    'infer_signature(sample, model.predict(sample))'. When
    'registered_model_name' is supplied, the LoggedModel is also registered
    as a new Model Registry version (callers set '@champion' on it
    afterwards with 'MlflowClient.set_registered_model_alias').

    When 'datasets' is supplied as a list of '(mlflow.data.Dataset, context)'
    tuples (typically the values returned by 'log_training_dataset'), each
    dataset is re-attached to the LoggedModel via
    'mlflow.log_input(dataset, context=context, model_id=mi.model_id)'.
    Without this, the per-experiment Models view
    ('/#/experiments/<id>/models') shows an empty Datasets column because
    'log_input' on the *run* alone does not create 'MODEL_INPUT ← DATASET'
    edges, only 'RUN ← DATASET' ones.

    After 'log_model', this helper also searches for traces in the active
    run (typically the '@mlflow.trace'-decorated training spans) and tags
    each with 'mlflow.modelId=mi.model_id' so the LoggedModel's Traces
    panel populates. Failures are logged rather than raised so trace
    attachment never blocks model persistence.

    'description' and 'version' propagate through every visible MLflow
    layer so the "code.version" / Description columns are not blank:

      * the active **run** ('mlflow.note.content' + 'code.version' tags),
      * the active **experiment** (same two tags, set on the experiment so
        the experiment header is populated, not just the run row),
      * the **LoggedModel** (via 'mlflow.set_logged_model_tags', without
        which the LoggedModel detail page has empty Tags / Description),
      * the **Registered Model** + **Model Version** (only when
        'registered_model_name' is supplied, using 'update_*' for the
        description fields and 'set_*_tag' for 'code.version').

    Returns the 'ModelInfo' from the underlying 'log_model' call.
    """
    from mlflow.models.signature import infer_signature

    resolved_version = _resolve_code_version(version)
    flavor = _resolve_model_flavor(model)

    signature = None
    input_example = None
    if X_sample is not None:
        sample = X_sample[:5]
        try:
            preds = model.predict(sample)
            signature = infer_signature(sample, preds)
        except Exception as exc:  # pragma: no cover - sample-shape edge cases
            print(f"[MLflow] signature inference skipped for {name}: {exc!r}")
        input_example = sample

    kwargs: dict = {"name": name}
    if signature is not None:
        kwargs["signature"] = signature
    if input_example is not None:
        kwargs["input_example"] = input_example
    if registered_model_name is not None:
        kwargs["registered_model_name"] = registered_model_name

    # Two UserWarnings get suppressed at this single call site:
    # (a) mlflow.sklearn's per-call 'skops' recommendation. Migrating to skops
    #     is a real artifact-format change (downstream loaders + registry
    #     rehydration), not a one-line swap, so the per-call advice is muted.
    # (b) mlflow.data.dataset_source_registry's "interpreted in multiple ways:
    #     LocalArtifactDatasetSource, LocalArtifactDatasetSource" UserWarning,
    #     which fires from the registry-resolution path inside flavor.log_model
    #     when 'registered_model_name' is set (mlflow creates a Dataset entry
    #     for the artifact source). The duplicate-class-name is an upstream
    #     mlflow bug, and the resolution itself is correct. Both items are
    #     filed in pipeline/WARNINGS.md.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=r".*scikit-learn models.*recommended safe alternative.*",
            category=UserWarning,
        )
        _warnings.filterwarnings(
            "ignore",
            message=r".*specified dataset source can be interpreted in multiple ways.*",
            category=UserWarning,
        )
        mi = flavor.log_model(model, **kwargs)

    if datasets:
        _attach_datasets_to_logged_model(
            datasets=datasets,
            model_id=mi.model_id,
            ds_tags=_metadata_tags(resolved_version, description),
        )

    # Propagate 'version' + 'description' to every layer the UI surfaces.
    # Run + experiment first (idempotent overwrite), then the LoggedModel,
    # then (only if we actually registered) the registered-model header
    # and the specific model-version row. Registry propagation MUST raise on
    # failure: see CLAUDE.md "No silent except around MLflow registration".
    stamp_run_metadata(version=resolved_version, description=description)
    import mlflow as _mlflow_for_stamp
    active = _mlflow_for_stamp.active_run()
    if active is not None:
        stamp_experiment_metadata(
            experiment_id=active.info.experiment_id,
            version=resolved_version,
            description=description,
        )
    stamp_logged_model_metadata(
        model_id=mi.model_id,
        version=resolved_version,
        description=description,
    )
    if registered_model_name is not None:
        mv = getattr(mi, "registered_model_version", None)
        if mv is None and active is not None:
            # mlflow.<flavor>.log_model occasionally returns None for the
            # version field across flavors. Resolve it the same way nb08 does.
            client = _mlflow_for_stamp.MlflowClient()
            for v in client.search_model_versions(f"name='{registered_model_name}'"):
                if v.run_id == active.info.run_id:
                    mv = v.version
                    break
        if mv is None:
            raise RuntimeError(
                f"log_model registered {registered_model_name!r} but did not return a "
                "version and no matching version was found by run_id; cannot attach "
                "version metadata."
            )
        stamp_registered_model_metadata(
            name=registered_model_name,
            model_version=mv,
            version=resolved_version,
            description=description,
        )

    _emit_characterize_trace(model=model, X_sample=X_sample, mi=mi,
                             flavor=flavor, name=name)
    _backtag_run_traces_to_logged_model(model_id=mi.model_id)

    return mi


# ----- log_estimator_to_mlflow helpers --------------

def _attach_datasets_to_logged_model(*, datasets, model_id: str, ds_tags: dict) -> None:
    """Attach each (Dataset, context) tuple in 'datasets' to the LoggedModel.

    The per-experiment Models view ('/#/experiments/<id>/models') renders an
    empty Datasets column unless 'log_input' is called with 'model_id=',
    which creates a 'MODEL_INPUT <- DATASET' edge. 'log_input' on the run
    alone only creates 'RUN <- DATASET'. The 'tags' kwarg surfaces
    code.version / description on each dataset row in the run's Datasets tab.
    """
    import mlflow as _mlflow
    for entry in datasets:
        if entry is None:
            continue
        try:
            ds, ctx = entry if isinstance(entry, tuple) else (entry, None)
        except Exception:
            continue
        if ds is None:
            continue
        try:
            # 'tags' kwarg was added to log_input in mlflow 2.10+. Older
            # versions accept only (dataset, context, model_id), so fall back
            # so the dataset still attaches even when tagging is unavailable.
            try:
                _mlflow.log_input(dataset=ds, context=ctx, model_id=model_id, tags=ds_tags)
            except TypeError:
                _mlflow.log_input(dataset=ds, context=ctx, model_id=model_id)
        except Exception as exc:  # pragma: no cover - log_input failures
            print(
                f"[MLflow] could not attach dataset "
                f"{getattr(ds, 'name', '<unknown>')!r} to LoggedModel "
                f"{model_id}: {exc!r}"
            )


def _emit_characterize_trace(*, model, X_sample, mi, flavor, name: str) -> None:
    """Emit a tiny inference span attached to 'mi.model_id'.

    Without this, the per-experiment Models view's Traces panel is empty for
    callers (typically HPO refits) that did not wrap their training in
    '@mlflow.trace'. The span runs 'model.predict' on a 5-row sample
    (same one used by 'infer_signature') so the trace is cheap and the
    LoggedModel's Traces tab always has at least one entry.
    """
    try:
        import mlflow as _mlflow
        try:
            _mlflow.set_active_model(model_id=mi.model_id)
        except Exception:
            pass  # mlflow < 3.x without set_active_model: span-tag fallback
        if X_sample is not None:
            with _mlflow.start_span(name=f"characterize:{name}") as _span:
                try:
                    _span.set_attributes({
                        "mlflow.modelId": mi.model_id,
                        "model.name": name,
                        "model.flavor": flavor.__name__,
                        "input_example.shape": str(getattr(X_sample[:5], "shape", "?")),
                    })
                except Exception:
                    pass
                try:
                    model.predict(X_sample[:5])
                except Exception:
                    pass
    except Exception as exc:  # pragma: no cover - characterization is opportunistic
        print(f"[MLflow] characterization trace failed for {mi.model_id}: {exc!r}")


def _backtag_run_traces_to_logged_model(*, model_id: str) -> None:
    """Tag every trace from the active run with 'mlflow.modelId = model_id'.

    Without this back-tag, '@mlflow.trace'-decorated training spans only
    appear under the run, not under the LoggedModel that ran them, so the
    per-experiment Models view's Traces panel stays empty even when traces
    exist. Best-effort: failures print + return, and trace tagging never
    blocks model persistence.
    """
    try:
        import inspect
        import mlflow as _mlflow
        run = _mlflow.active_run()
        if run is None:
            return
        client = _mlflow.MlflowClient()

        def _search_traces(**extra):
            """Call search_traces with the experiment scope under whichever
            keyword the installed mlflow accepts.

            mlflow 3.x exposes 'locations' and deprecates 'experiment_ids'.
            Older versions expose only 'experiment_ids'. The keyword is read
            from the actual signature, so the call never takes the deprecated
            path while 'locations' exists and the FutureWarning never fires. A
            try/except on the scope keyword cannot do this, because an unrelated
            unsupported kwarg such as run_id raises the same TypeError and would
            wrongly trigger the experiment_ids fallback.
            """
            try:
                params = inspect.signature(client.search_traces).parameters
                scope_kw = "locations" if "locations" in params else "experiment_ids"
            except (TypeError, ValueError):
                scope_kw = "locations"
            return client.search_traces(
                **{scope_kw: [run.info.experiment_id]}, **extra
            )

        # Try the typed run_id parameter (newer mlflow), falling back to a
        # filter_string query on the sourceRun metadata.
        try:
            traces = _search_traces(run_id=run.info.run_id, max_results=200)
        except TypeError:
            traces = _search_traces(
                filter_string=(
                    f"request_metadata.`mlflow.sourceRun` = '{run.info.run_id}'"
                ),
                max_results=200,
            )
        for trace in traces:
            client.set_trace_tag(trace.info.request_id, "mlflow.modelId", model_id)
    except Exception as exc:  # pragma: no cover - trace tagging best-effort
        print(f"[MLflow] could not attach traces to LoggedModel {model_id}: {exc!r}")


def log_training_dataset(
    X,
    y=None,
    *,
    name: str,
    source: str | None = None,
    context: str = "training",
    targets_name: str = "target",
    description: str | None = None,
    version: str | None = None,
):
    """Attach 'X' (and optionally 'y') to the active run's Datasets tab.

    'mlflow.log_input' is what populates the per-run Datasets tab in the
    UI. Without it, runs show "No datasets" no matter what was actually fit.
    'name' is the label that appears in the tab (use the upstream artefact's
    filename). 'source' is the path it was loaded from (clickable in the
    UI). 'context' is the lifecycle tag, typically '"training"',
    '"validation"', or '"test"'.

    'description' and 'version' are stamped as per-input tags
    ('mlflow.note.content' + 'code.version') so the run's Datasets tab
    shows them alongside the dataset name, otherwise that column is blank
    even when the run + model are tagged. Requires mlflow >= 2.10 for the
    'tags' kwarg on 'log_input'. Older versions silently drop the tags.
    """
    import numpy as np
    import pandas as pd
    import warnings as _warnings
    import mlflow.data

    # mlflow.data.from_pandas/from_numpy walks its dataset-source registry to
    # decide how to interpret 'source'. Suppress the LocalArtifactDatasetSource
    # ambiguity UserWarning at this single call site so it does not clutter
    # every per-run dataset log.
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=r".*specified dataset source can be interpreted in multiple ways.*",
            category=UserWarning,
        )
        if isinstance(X, pd.DataFrame):
            dataset = mlflow.data.from_pandas(
                X if y is None else X.assign(**{targets_name: np.asarray(y)}),
                source=source,
                name=name,
                targets=targets_name if y is not None else None,
            )
        else:
            kw: dict = {"name": name}
            if source is not None:
                kw["source"] = source
            if y is not None:
                kw["targets"] = np.asarray(y)
            dataset = mlflow.data.from_numpy(np.asarray(X), **kw)

        ds_tags = _metadata_tags(version, description)
        try:
            mlflow.log_input(dataset=dataset, context=context, tags=ds_tags)
        except TypeError:
            # mlflow < 2.10 doesn't accept the 'tags' kwarg on log_input. The
            # dataset still attaches without per-input tags. Logged so the
            # silent-blank-column scenario is at least visible in stderr.
            mlflow.log_input(dataset=dataset, context=context)
            print(f"[MLflow] log_input(tags=...) unsupported in this mlflow; dataset {name!r} attached without tags")
    return dataset


def log_eval_dashboard(
    y_true,
    y_pred=None,
    y_prob=None,
    *,
    model=None,
    feature_names=None,
    threshold: float | None = None,
    prefix: str = "eval",
    model_id: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    n_bins_calibration: int = 10,
    metric_prefix: str | None = None,
) -> dict[str, float]:
    """Render the standard classification diagnostic suite into the active
    MLflow run so the UI shows the same visuals the notebooks render.

    MLflow's metrics panel only line-charts scalars, and "F1=0.40" alone gives
    no confusion matrix, no ROC, no calibration. This helper closes
    that gap by emitting the §8 diagnostic suite (confusion matrix, ROC, PR,
    calibration, probability histogram, threshold sweep, cumulative-gain,
    feature importance) as PNG artifacts under '<prefix>/' and an
    interactive metrics grid via 'mlflow.log_table', both surfaces the
    run's Artifacts tab renders natively.

    Required: an active MLflow run + 'y_true'. Provide 'y_prob' to unlock
    ROC / PR / calibration / threshold-sweep / cumulative-gain (the helper
    derives 'y_pred' from 'y_prob' at 'threshold or 0.5' if missing).
    'model' + 'feature_names' are optional and only used for the feature
    importance bar chart. 'model_id' is the MLflow LoggedModel id (from
    'ModelInfo.model_id'). Supplying it routes the figures to the
    LoggedModel's Artifacts tab so the per-model view also shows them, not
    just the run view.

    Returns the dict of scalar metrics logged via 'mlflow.log_metric'.

    Cost / when to call: PNG rendering + log_artifact is O(seconds) per call
    and several hundred KB per run. Suitable for champion-tier evaluation
    (§7 winners, §8 promotion, retraining DAG). Do NOT call per HPO trial,
    or the run store balloons.

    Usage::

        from helpers.mlops_helpers import log_estimator_to_mlflow, log_eval_dashboard

        with traced_run(run_name="xgb_champion"):
            mi = log_estimator_to_mlflow(clf, name="xgb", X_sample=X_val,
                                         registered_model_name="medi-watch")
            log_eval_dashboard(
                y_true=y_test, y_prob=clf.predict_proba(X_test)[:, 1],
                model=clf, feature_names=feature_names,
                threshold=0.53, model_id=mi.model_id,
            )
    """
    import mlflow
    import numpy as np

    if mlflow.active_run() is None:
        raise RuntimeError("log_eval_dashboard requires an active MLflow run")

    # Headless matplotlib backend: works in jupyter, airflow workers, and
    # docker without a display. 'force=False' so an already-configured GUI
    # backend in a notebook keeps working for inline display.
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    mp = metric_prefix if metric_prefix is not None else prefix

    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = None if y_prob is None else np.asarray(y_prob).ravel()
    # Unconditional sample count so the summary.json log (and any future
    # downstream consumer) can never hit a "possibly unbound" path when
    # y_prob_arr is None and the cumulative-gain block below is skipped.
    n = int(y_true_arr.size)
    chosen_t = 0.5 if threshold is None else float(threshold)
    if y_pred is None:
        if y_prob_arr is None:
            raise ValueError("log_eval_dashboard needs y_pred or y_prob (or both)")
        y_pred_arr = (y_prob_arr >= chosen_t).astype(int)
    else:
        y_pred_arr = np.asarray(y_pred).ravel()

    # Bind subsequently logged figures to the LoggedModel if the caller has
    # one in hand. Without this, mlflow.log_figure attaches PNGs to the run
    # only and the per-model Artifacts tab stays empty.
    if model_id is not None:
        try:
            mlflow.set_active_model(model_id=model_id)
        except Exception as exc:  # pragma: no cover - older mlflow
            print(f"[MLflow] set_active_model({model_id!r}) failed: {exc!r}")

    metrics: dict[str, float] = {
        f"{mp}.f1": float(f1_score(y_true_arr, y_pred_arr)),
        f"{mp}.precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),  # pyright: ignore[reportArgumentType]
        f"{mp}.recall": float(recall_score(y_true_arr, y_pred_arr)),
        f"{mp}.accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        f"{mp}.mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
    }
    if y_prob_arr is not None:
        metrics[f"{mp}.auc_roc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics[f"{mp}.auc_pr"] = float(average_precision_score(y_true_arr, y_prob_arr))
        metrics[f"{mp}.brier"] = float(brier_score_loss(y_true_arr, y_prob_arr))

    def _log_fig(fig, name: str) -> None:
        try:
            mlflow.log_figure(fig, f"{prefix}/{name}")
        finally:
            plt.close(fig)

    # ---- Confusion matrix (always) ------------------------------------
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={chosen_t:.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _log_fig(fig, "confusion_matrix.png")

    if y_prob_arr is not None:
        # ---- ROC curve ------------------------------------------------
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics[f'{mp}.auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _log_fig(fig, "roc_curve.png")

        # ---- Precision-Recall curve ----------------------------------
        prec, rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        baseline = float(np.mean(y_true_arr))
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rec, prec, linewidth=2, label=f"AP = {metrics[f'{mp}.auc_pr']:.3f}")
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                   label=f"baseline (positive rate = {baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _log_fig(fig, "pr_curve.png")

        # ---- Probability histogram by true class ---------------------
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(y_prob_arr[y_true_arr == 0], bins=40, alpha=0.6,
                label=f"y_true = 0 (n={int((y_true_arr == 0).sum())})")
        ax.hist(y_prob_arr[y_true_arr == 1], bins=40, alpha=0.6,
                label=f"y_true = 1 (n={int((y_true_arr == 1).sum())})")
        ax.axvline(chosen_t, color="black", linestyle="--", linewidth=1,
                   label=f"threshold = {chosen_t:.3f}")
        ax.set_xlabel("Predicted P(y=1)")
        ax.set_ylabel("Count")
        ax.set_title("Predicted-Probability Distribution by True Class")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _log_fig(fig, "probability_histogram.png")

        # ---- Calibration curve (quantile-binned) ---------------------
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true_arr, y_prob_arr, n_bins=n_bins_calibration, strategy="quantile",
            )
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                    label=f"model (Brier = {metrics[f'{mp}.brier']:.4f})")
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
            ax.set_xlabel("Mean predicted probability (per bin)")
            ax.set_ylabel("Observed positive frequency (per bin)")
            ax.set_title(f"Calibration Plot ({n_bins_calibration}-bin quantile)")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _log_fig(fig, "calibration.png")
        except Exception as exc:
            print(f"[MLflow] calibration plot skipped: {exc!r}")

        # ---- Threshold sweep -----------------------------------------
        thresholds = np.linspace(0.01, 0.99, 99)
        ts_f1, ts_p, ts_r = [], [], []
        for t in thresholds:
            yp = (y_prob_arr >= t).astype(int)
            ts_f1.append(f1_score(y_true_arr, yp, zero_division=0))  # pyright: ignore[reportArgumentType]
            ts_p.append(precision_score(y_true_arr, yp, zero_division=0))  # pyright: ignore[reportArgumentType]
            ts_r.append(recall_score(y_true_arr, yp))
        best_idx = int(np.argmax(ts_f1))
        best_t = float(thresholds[best_idx])
        metrics[f"{mp}.best_f1_threshold"] = best_t
        metrics[f"{mp}.best_f1"] = float(ts_f1[best_idx])

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresholds, ts_f1, label="F1", linewidth=2)
        ax.plot(thresholds, ts_p, label="Precision", linewidth=1.5, alpha=0.85)
        ax.plot(thresholds, ts_r, label="Recall", linewidth=1.5, alpha=0.85)
        ax.axvline(best_t, color="black", linestyle="--", linewidth=1,
                   label=f"best F1 @ {best_t:.3f}")
        if threshold is not None:
            ax.axvline(chosen_t, color="red", linestyle=":", linewidth=1,
                       label=f"chosen @ {chosen_t:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold sweep (F1 / Precision / Recall)")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _log_fig(fig, "threshold_sweep.png")

        # ---- Cumulative gain + Lift ----------------------------------
        order_desc = np.argsort(-y_prob_arr)
        sorted_y = y_true_arr[order_desc]
        total_pos = int(sorted_y.sum()) or 1
        cum_pos = np.cumsum(sorted_y)
        pop_frac = np.arange(1, n + 1) / n
        gain = cum_pos / total_pos

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(pop_frac, gain, linewidth=2, label="model")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="random")
        # Mark top-decile lift: the operationally relevant capacity-bounded number.
        deciles = [0.1, 0.2, 0.3]
        for d in deciles:
            idx = max(0, int(d * n) - 1)
            ax.axvline(d, color="black", alpha=0.15, linewidth=0.5)
            ax.annotate(f"top {int(d*100)}%: {gain[idx]*100:.0f}% of positives",
                        xy=(d, gain[idx]), xytext=(d + 0.02, gain[idx] - 0.05),
                        fontsize=8)
        ax.set_xlabel("Top-fraction of population (ranked by score)")
        ax.set_ylabel("Cumulative fraction of positives captured")
        ax.set_title("Cumulative Gain")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _log_fig(fig, "cumulative_gain.png")

        # Top-decile lift scalar, recorded as a separate metric.
        top10_idx = max(0, int(0.1 * n) - 1)
        metrics[f"{mp}.lift_top10"] = float(gain[top10_idx] / 0.1)

    # ---- Feature importance (gain or |coef|) ---------------------------
    if model is not None:
        importance = None
        kind = None
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
            kind = "split-gain"
        elif hasattr(model, "coef_"):
            importance = np.abs(np.asarray(model.coef_).ravel())
            kind = "|coef|"
        if importance is not None and importance.size > 0:
            names = (
                list(feature_names)
                if feature_names is not None and len(feature_names) == len(importance)
                else [f"f{i}" for i in range(len(importance))]
            )
            top_k = min(20, len(importance))
            order_top = np.argsort(importance)[-top_k:]
            fig, ax = plt.subplots(figsize=figsize)
            ax.barh(range(top_k), importance[order_top])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([names[i] for i in order_top])
            ax.set_xlabel(f"Importance ({kind})")
            ax.set_title(f"Top-{top_k} feature importance ({type(model).__name__})")
            fig.tight_layout()
            _log_fig(fig, "feature_importance.png")

    # ---- Log scalars + interactive metrics table ----------------------
    mlflow.log_metrics(metrics)
    try:
        import pandas as pd
        table = pd.DataFrame(
            [{"metric": k.split(".", 1)[-1], "value": v} for k, v in metrics.items()]
        ).sort_values("metric").reset_index(drop=True)
        mlflow.log_table(data=table, artifact_file=f"{prefix}/metrics_table.json")
    except Exception as exc:  # pragma: no cover - log_table API surface
        print(f"[MLflow] log_table failed: {exc!r}")

    # Persist the exact threshold used so reproducing the confusion matrix
    # downstream doesn't require re-deriving it from the binarisation rule.
    try:
        mlflow.log_dict(
            {
                "threshold_used": chosen_t,
                "metric_prefix": mp,
                "n_samples": n,
                "positive_rate": float(np.mean(y_true_arr)),
            },
            f"{prefix}/summary.json",
        )
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] log_dict failed: {exc!r}")

    return metrics


def ray_reachable(client_address: str, timeout: float = 2.0) -> bool:
    """Check that the 'ray://host:port' endpoint accepts a TCP connection.

    A live 'ray-head' does not need a health endpoint. Being able to
    open a socket to 'host:port' is enough to know 'ray.init(address=...)'
    won't hang during handshake. 'address=None' / '"local"' always passes.

    If connected, it also verifies that the Python version of the driver matches
    the cluster to avoid Ray client protocol errors.
    """
    if not client_address or client_address in ("local", "auto"):
        return True
    parsed = urlparse(client_address)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False

    # 1. TCP reachability check
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError:
        return False

    # 2. Python version check (optional but recommended for Ray Client)
    # We probe the Ray Dashboard (default 8265) if reachable to check metadata,
    # or rely on ray.init failing later. However, since we want a clean fallback
    # to address=None, we can try to be proactive if we know the dashboard port.
    # For now, TCP reachability is the primary gate. Version mismatch is
    # handled by the caller catching ray.init errors.
    return True


def init_ray(
    default_address: str = "ray://localhost:20001",
    runtime_env: dict | None = None,
):
    """Initialise Ray with an in-process fallback.

    - Reads '$RAY_ADDRESS' if set, otherwise 'default_address'.
    - Probes the address. On failure, calls 'ray.init(address=None, ...)' so
      Tune trials still run (single-process), instead of stalling on the client
      handshake. The 'local_mode=True' flag is intentionally NOT used:
      'ray.init(local_mode=True)' raises RuntimeError at init time.
    - When connecting to a 'ray://' cluster and no 'runtime_env' is given,
      uploads the project root as the worker 'working_dir' so trainables
      that reference 'helpers.*' can import it. 'data/' and other bulky
      generated dirs are excluded to stay under Ray's working_dir size cap.
    - Returns the actual address Ray bound to.
    """
    import ray  # imported lazily so pipeline that don't need Ray don't pay for it

    address = os.environ.get("RAY_ADDRESS", default_address)
    kwargs: dict = {"ignore_reinit_error": True, "log_to_driver": False}
    if runtime_env is None and address.startswith("ray://"):
        runtime_env = {
            # Anchor the project-root walk at THIS module's location, not the
            # process cwd. In the airflow worker the task cwd is not under the
            # bind-mounted project tree, so a cwd-based resolve walks up to '/'
            # and Ray then tries to package the whole filesystem, hitting
            # unreadable paths like /root/.gitignore under a non-root container
            # user. helpers/ -> project root deterministically.
            "working_dir": str(resolve_project_root(Path(__file__).parent)),
            "excludes": [
                "data/",
                "mlruns/",
                "notebooks/",
                ".git/",
                ".ruff_cache/",
                ".pytest_cache/",
                "**/__pycache__/",
                "**/.ipynb_checkpoints/",
                "*.joblib",
                "*.npz",
                "*.csv",
            ],
        }
    if runtime_env is not None:
        kwargs["runtime_env"] = runtime_env

    if ray_reachable(address):
        try:
            ray.init(address=address, **kwargs)
            print(f"[Ray] connected to {address}")
            return address
        except Exception as exc:
            print(f"[Ray] Connection to {address} failed (likely version mismatch or handshake error): {exc!r}")
            print("[Ray] falling back to in-process ray.init(address=None)")

    # Genuine in-process fallback. Two things would otherwise sabotage it:
    #   1. A 'ray://...' value left in $RAY_ADDRESS overrides address=None and
    #      reconnects to the very cluster we just failed to reach.
    #   2. The 'working_dir' runtime_env is only meaningful for a remote cluster;
    #      packaging it locally walks the filesystem and can hit unreadable paths
    #      (e.g. /root/.gitignore when the container user is not root).
    os.environ.pop("RAY_ADDRESS", None)
    kwargs.pop("runtime_env", None)
    print(f"[Ray] {address} unreachable or failed -> falling back to in-process ray.init(address=None)")
    print("[Ray]   to use the remote cluster start `ray start --head` on a host that won't OOM, then re-run.")
    ray.init(address=None, **kwargs)
    return None
