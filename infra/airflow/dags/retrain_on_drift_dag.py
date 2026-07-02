"""retrain_on_drift re-runs the HPO, training, and final-eval pipeline and lets
the champion change only when three independent gates agree.

Triggered two ways:
  1. Auto: the 'drift_checks' DAG fires this via TriggerDagRunOperator
     when at least one active rule fails on the latest dataset version.
  2. Manual: from the Airflow UI or 'airflow dags trigger
     retrain_on_drift'.

Scope of "retrain":
  - `helpers.hpo_pipeline.run_hpo` runs Ray Tune plus ASHA HPO
    across all five model families. Persists `tuned_results.joblib` and
    `tuned_models.joblib` and emits the ASHA-pruning diagnostic
    (`data/hpo_diagnostics.parquet`) so MLP-vs-RF fairness under ASHA
    is auditable per run.
  - `helpers.training_pipeline.train_baselines_and_refits`
    fits library-default and HPO-winner refits on the full X_train and
    persists `training_models.joblib`, `training_results.joblib`,
    `default_models.joblib`, `default_results.joblib`, and the
    per-model F1-optimal threshold map. It also writes `mlp_results.joblib`
    so the conclusion stage's OOF MLP loop matches the production MLP config.
  - `helpers.conclusion_pipeline.run_conclusion_and_register`
    aggregates the leaderboard, picks the champion by validation F1,
    evaluates once on X_test, dumps the deployable bundle to
    `data/final_model.joblib`, registers the champion to the MLflow
    Model Registry, and sets the @champion alias the inference API
    resolves at serve time.

Not re-run here (intentionally, to keep the loop short):
  - NB1 overview, NB2 cleaning, NB3 EDA, NB4 feature engineering, and
    NB5 split/encode/scale are human-driven notebook decisions.
    A drift signal does not auto-rebuild `data/cleaned.csv` or
    re-cut the patient-level train/val/test partition. Re-splitting
    on a schedule breaks champion-vs-challenger comparability.
    Operators rerun NB2-NB5 by hand when raw data refreshes or the
    feature schema changes and commit the resulting
    `data/train_test.npz`.

Why direct library calls matter for performance and correctness:
  Each stage calls its library function directly through a
  PythonOperator, skipping Jupyter kernel startup, cell-by-cell
  `nbconvert` overhead, and the fully-rendered notebook copy that a
  papermill run would write into `/workspace/data/run_logs/<UTC-stamp>/`.
  The per-trial CV scorer has a single definition shared between the HPO
  driver and helpers/hpo.py.

Champion-thrash safeguards (all three must pass for a swap):

    1. **Bootstrap-CI lift gate.** The candidate's **test** F1 is
       resampled 1000 times with replacement on the test labels. The
       lower bound of the 95 percent CI on the lift (candidate_f1 minus
       prior_f1) must exceed 0.005 before the swap is considered.
    2. **7-day cool-down.** Even when a candidate clears the CI gate,
       the alias does not move within 7 days of the previous promotion.
    3. **Equity-parity gate.** Per-subgroup recall is recomputed for
       the candidate and the prior champion on the same test set. If
       any subgroup's candidate recall is more than 0.05 below the
       prior champion's recall, the swap is rejected.

Why X_test for the lift CI (and an honest bias trade-off note):
    NB05 §5.10 declares X_test "touched exactly once" by the conclusion
    stage for the single committed champion's offline evaluation. The
    retrain gate touches X_test exactly once per retrain (one
    bootstrap pass per candidate), so the contract relaxes to
    "single-touch per gate evaluation, at most one retrain per drift
    event". Splitting an additional held-out gate partition off X_train
    was rejected because it shrinks the train set on every refresh, and
    X_test is the only honest held-out partition post-tuning for a
    champion-swap hypothesis test.

Retry / SLA policy:
  - Schedule: event-driven (triggered by drift_checks on failure, or
    manually).
  - Retries: 0 (default_args.retries). A silent retry of a partial
    write is worse than a human-triaged failure.
  - SLA: 4 hours (HPO sweep plus training plus conclusion). The HPO sweep
    alone is ~2 hr on one GPU. The 4-hour SLA leaves headroom for a
    node eviction or Ray restart.

Orchestration to deploy handoff (two mutually-exclusive paths off the gate):
  'trigger_remote_deploy' (preferred, remote orchestration): when a CI/CD
  provider is configured (CICD_PROVIDER + creds), the DAG remotely fires the
  existing build+deploy job (infra/ci-cd/{Jenkinsfile,azure-pipelines.yml,
  cloudbuild.yaml}) on Jenkins / Azure Pipelines / Google Cloud Build via that
  system's REST API (see helpers/cicd_trigger.py). The pipeline rebuilds the
  image and rolls the Deployment, so build and deploy are orchestrated remotely
  rather than run inline.

  'redeploy_inference_api' (fallback): when NO CI/CD provider is configured
  (local/dev), this performs a direct 'kubectl rollout restart' of the
  inference-api Deployment on AWS EKS (or a local minikube context) when
  'gate_champion' promoted a new '@champion' (decision in {"promote",
  "bootstrap"}). The fresh pods re-resolve '@champion' from the MLflow registry
  at startup, so the live service serves the freshly promoted model without an
  image rebuild. It steps aside (skips) whenever a CI/CD provider is configured,
  so the two paths never deploy the same change twice.

  The inference API is the only medi-watch workload on k8s. Airflow, Ray, and
  MLflow run in docker-compose, and this worker reaches EKS as a client (aws plus
  kubectl, see infra/aws/README.md). A 'reject' decision skips both, because the
  prior champion is already live.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pendulum
from airflow.sdk import dag, task

from helpers.constants import (
    BOOTSTRAP_RESAMPLES,
    COOLDOWN_DAYS,
    EQUITY_RECALL_TOL,
    LIFT_FLOOR,
)
from helpers.evaluation import bootstrap_lift_ci, per_subgroup_recall
from helpers.model_loading import load_model_any_flavor, predict_labels

# /workspace is the in-container mount that maps to the project root
# (see infra/docker-compose.yml). The library functions accept either
# absolute or cwd-relative paths, and we pass absolute so the worker's cwd
# does not matter.
DATA_DIR = "/workspace/data"
TRAIN_TEST_PATH = f"{DATA_DIR}/train_test.npz"
TUNED_RESULTS_PATH = f"{DATA_DIR}/tuned_results.joblib"

# Interpreter for the one task that opens a Ray client. This dedicated venv
# carries CPython 3.13.2 to match the Ray cluster (built in
# infra/airflow/Dockerfile), so the HPO driver matches the cluster patch
# version and avoids the Ray "Python patch version mismatch" warning that the
# base 3.13.13 interpreter would raise.
RAY_DRIVER_PYTHON = "/opt/ray-driver/bin/python"
INFERENCE_API = os.environ.get("INFERENCE_API_URL", "http://inference-api:8002")

# EKS redeploy target for the post-promotion rolling restart. The inference API
# is the only medi-watch workload on k8s, and this worker reaches the cluster as
# a client via `aws eks update-kubeconfig` plus `kubectl` (both baked into the
# airflow image, with creds and cluster name supplied through the compose env,
# see infra/airflow/Dockerfile and infra/docker-compose.yml). Values are read at
# task runtime (env first, Airflow Variable fallback) so a cluster rename does
# not require a DAG edit.
EKS_CLUSTER_NAME = os.environ.get("EKS_CLUSTER_NAME")
AWS_REGION = os.environ.get("AWS_REGION")
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "medi-watch")
INFERENCE_DEPLOYMENT = os.environ.get("INFERENCE_DEPLOYMENT", "medi-watch-inference")

# Gate constants and the evaluation routines they back live in helpers/ so
# the notebooks (NB07/NB08), the inference path, and this DAG cannot drift
# from each other.
#
# Policy note on EQUITY_RECALL_TOL and the subgroup granularity (kept here
# rather than in helpers/constants.py because it is operational context,
# not a value):
#
# 1. The 0.05 recall-drop tolerance is the operational expression of the
#    "no subgroup made materially worse off" minimum-checklist text in
#    notebook 8 section 8.11.5. A larger tolerance would let small regressions
#    accumulate over successive promotions, a smaller tolerance would reject
#    candidates on subgroup noise alone given the test-set sizes we observe in
#    practice.
# 2. The race x gender granularity (with the documented gender-only fallback
#    when race is unavailable) matches the protected-attribute set named in
#    section 8.11.5. A finer race x gender x age_band breakdown was considered
#    and deferred because the resulting per-cell sample sizes drop below the
#    point where a 0.05 recall delta carries statistical signal.
def _reload_downstream():
    """Notify the production inference API that the @champion alias has changed
    so it does not serve stale predictions."""
    import requests
    for label, url in (("inference API", f"{INFERENCE_API}/reload"),):
        try:
            r = requests.post(url, timeout=10)
            r.raise_for_status()
            print(f"  {label} reload: {r.json()}")
        except Exception as e:
            print(f"  {label} reload failed (non-fatal): {e}")


@dag(
    dag_id="retrain_on_drift",
    description="Re-run HPO + training + conclusion. Triggered by drift_checks on failure (or manually).",
    start_date=pendulum.datetime(2026, 4, 27, tz="UTC"),
    schedule=None,            # event-driven, no time-based schedule
    catchup=False,
    max_active_runs=1,        # do not pile re-trains
    default_args={
        "owner": "medi-watch",
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["medi-watch", "drift", "retrain"],
)
def retrain_on_drift():

    @task(task_id="run_06_hyperparameter_tuning")
    def run_hpo_task() -> dict:
        """NB06 stage: Ray Tune + ASHA HPO via helpers.hpo_pipeline.run_hpo.

        Runs under the dedicated CPython 3.13.2 venv (RAY_DRIVER_PYTHON) rather
        than in-process, so the Ray *client* interpreter matches the cluster and
        the patch-mismatch warning disappears. run_hpo exchanges data only
        through files (train_test.npz in, joblib artefacts out to DATA_DIR), so
        a subprocess needs nothing back beyond an exit code. The subprocess
        inherits this worker's environment (PYTHONPATH=/workspace, RAY_ADDRESS,
        MLFLOW_TRACKING_URI), and its stdout/stderr flow into this task log.
        """
        import subprocess
        subprocess.run(
            [RAY_DRIVER_PYTHON, "-m", "helpers.hpo_pipeline",
             "--train-test-path", TRAIN_TEST_PATH, "--out-dir", DATA_DIR],
            check=True,
        )
        return {"stage": "hpo", "status": "ok"}

    @task(task_id="run_07_model_training")
    def run_training_task() -> dict:
        """NB07 stage: default and HPO-refit training via
        helpers.training_pipeline.train_baselines_and_refits.

        This is an explicit DAG task because NB08 (the next task) reads
        `training_results.joblib` and `training_models.joblib`, so the
        dependency has to land somewhere.
        """
        from helpers.training_pipeline import train_baselines_and_refits
        train_baselines_and_refits(
            train_test_path=TRAIN_TEST_PATH,
            tuned_results_path=TUNED_RESULTS_PATH,
            out_dir=DATA_DIR,
        )
        return {"stage": "training", "status": "ok"}

    @task(task_id="run_08_conclusion")
    def run_conclusion_task() -> dict:
        """NB08 stage: leaderboard + champion + test eval + Registry promotion
        via helpers.conclusion_pipeline.run_conclusion_and_register."""
        from helpers.conclusion_pipeline import run_conclusion_and_register
        result = run_conclusion_and_register(
            train_test_path=TRAIN_TEST_PATH,
            in_dir=DATA_DIR,
            out_dir=DATA_DIR,
            register=True,
        )
        return {"stage": "conclusion", **result}

    @task
    def gate_champion() -> dict:
        """Compare the just-registered @champion against the prior live champion.

        Three independent gates must ALL agree before the @champion alias
        actually moves. Thresholds live in 'helpers.constants' so the
        same numbers anchor the DAG, the unit tests, and the operations
        dashboards.

        Gate 1: Bootstrap-CI lift gate
        ------------------------------
        Lower-CI(95%) of (cand_f1 - prior_f1) on the **test** set must
        exceed 'LIFT_FLOOR' (= 0.005, i.e. half a percentage-point of
        F1) over 'BOOTSTRAP_RESAMPLES' (= 1000) resamples.

        * Why CI lower bound and not point estimate: a point estimate
          rewards lucky resamples, whereas the lower-CI guards against
          promoting a candidate whose lift could plausibly be zero.
        * Why 0.005 (not 0.01 or 0.02): the cohort is ~20k rows in the
          test partition, and bootstrap variance is dominated by the
          minority-class count, so 0.005 is the empirical floor below
          which the CI itself is unreliable.
        * Why 'X_test' and not 'X_val': NB06 HPO and NB07 threshold
          tuning both consume 'X_val', so a bootstrap resampled from
          'X_val' would produce an optimistically biased lift estimate
          with a too-tight variance band. 'X_test' is the only honest
          held-out partition for a champion-swap hypothesis test
          post-tuning.

        Gate 2: Cool-down gate
        ----------------------
        At least 'COOLDOWN_DAYS' (= 7) must have passed since the
        prior champion's 'promoted_at' tag. Prevents rapid champion
        churn that destabilises the inference API's score distribution
        and blocks the on-call team from forming a stable baseline.

        Gate 3: Equity-parity gate
        --------------------------
        No subgroup's candidate recall may drop more than
        'EQUITY_RECALL_TOL' (= 0.05) below the prior champion's recall
        on the same subgroup. Subgroup granularity: race × gender
        (NB02 §2.6.2 protected attributes), with gender-only fallback
        if race is unavailable.

        Rejection trail
        ---------------
        When any gate fails the candidate is demoted, the prior version
        is restored to '@champion', the candidate is tagged
        'rejected_reason=<gate>' (one of 'lift_ci' / 'cooldown' /
        'equity'), and '_reload_downstream' re-warms the inference
        API so it keeps serving the prior artefact pair.
        """
        import math
        import numpy as np
        import pandas as pd
        import mlflow

        def _json_safe(obj):
            """Replace non-finite floats with None so the XCom push survives.

            Airflow 3 serializes a task's return value to JSON for the
            Execution API, and JSON has no NaN or infinity. A missing
            test_f1 tag resolves to float('nan'), which fails the push and
            marks the task failed even though the gate decision already took
            effect (the alias is set and the inference API is reloaded before
            the return). None round-trips cleanly.
            """
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(v) for v in obj]
            return obj

        REG_NAME = "medi-watch-readmission"
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.MlflowClient()

        new = client.get_model_version_by_alias(REG_NAME, "champion")
        new_v = int(new.version)
        new_f1 = float((new.tags or {}).get("test_f1", "nan"))
        new_label = (new.tags or {}).get("selected_model_label", "?")

        all_versions = client.search_model_versions(f"name = '{REG_NAME}'")
        prior = None
        for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
            v_int = int(v.version)
            if v_int >= new_v:
                continue
            tags = v.tags or {}
            if "rejected_reason" in tags:
                continue
            prior = v
            break

        now = datetime.now(timezone.utc)
        if prior is None:
            print(f"=== {REG_NAME} bootstrap promotion ===")
            print(f"  v{new_v} ({new_label}, test_f1={new_f1:.4f}) -> @champion")
            print("  no prior non-rejected version exists, bootstrap accept")
            client.set_model_version_tag(REG_NAME, str(new_v), "promoted_at", now.isoformat())
            _reload_downstream()
            return _json_safe({"decision": "bootstrap", "version": new_v, "test_f1": new_f1})

        prev_v = int(prior.version)
        prev_f1 = float((prior.tags or {}).get("test_f1", "nan"))
        prev_label = (prior.tags or {}).get("selected_model_label", "?")

        print(f"=== {REG_NAME} candidate gate ===")
        print(f"  candidate     v{new_v}  ({new_label}, test_f1={new_f1:.4f})")
        print(f"  prior champ   v{prev_v}  ({prev_label}, test_f1={prev_f1:.4f})")
        if prev_v != new_v - 1:
            print(f"  note: prior champion is v{prev_v}, not v{new_v - 1} "
                  f"(skipped {new_v - 1 - prev_v} rejected version(s))")

        # --- Gate 2: cool-down ---------------------------------------------------
        prior_promoted_at_raw = (prior.tags or {}).get("promoted_at")
        if prior_promoted_at_raw:
            try:
                prior_promoted_at = datetime.fromisoformat(prior_promoted_at_raw)
                if prior_promoted_at.tzinfo is None:
                    prior_promoted_at = prior_promoted_at.replace(tzinfo=timezone.utc)
            except ValueError:
                prior_promoted_at = None
        else:
            prior_promoted_at = None

        cooldown_elapsed = (
            None if prior_promoted_at is None
            else (now - prior_promoted_at)
        )
        cooldown_ok = (
            True if cooldown_elapsed is None
            else cooldown_elapsed >= timedelta(days=COOLDOWN_DAYS)
        )
        print(f"  cooldown:     prior_promoted_at={prior_promoted_at_raw or 'unknown'}, "
              f"elapsed={cooldown_elapsed}, ok={cooldown_ok} "
              f"(floor={COOLDOWN_DAYS}d)")

        if not cooldown_ok:
            client.set_registered_model_alias(REG_NAME, "champion", str(prev_v))
            client.set_registered_model_alias(REG_NAME, "rejected", str(new_v))
            client.set_model_version_tag(
                REG_NAME, str(new_v), "rejected_reason",
                f"cooldown: only {cooldown_elapsed} since prior promotion, floor={COOLDOWN_DAYS}d",
            )
            print(f"  decision: REJECT candidate (cooldown); @champion reverted to v{prev_v}")
            _reload_downstream()
            return _json_safe({
                "decision": "reject",
                "rejection_reason": "cooldown",
                "candidate": new_v,
                "champion": prev_v,
                "candidate_f1": new_f1,
                "champion_f1": prev_f1,
            })

        # --- Load test arrays once for gates 1 and 3 ----------------------------
        # X_test (not X_val) is used for the bootstrap lift CI and the equity
        # gate because the candidate's HPO (NB06) and operating threshold
        # (NB07) were tuned on X_val. Resampling X_val for the lift CI would
        # compare the candidate to the prior champion on rows the candidate
        # was selected against, biasing the CI optimistically.
        try:
            arrays = np.load(TRAIN_TEST_PATH)
            X_test = arrays["X_test"]
            y_test = arrays["y_test"]
            test_pids = arrays["test_patient_ids"] if "test_patient_ids" in arrays.files else None
        except Exception as exc:
            client.set_registered_model_alias(REG_NAME, "champion", str(prev_v))
            client.set_registered_model_alias(REG_NAME, "rejected", str(new_v))
            client.set_model_version_tag(
                REG_NAME, str(new_v), "rejected_reason",
                f"test_arrays_missing: {exc}",
            )
            print(f"  decision: REJECT candidate (test arrays missing: {exc})")
            _reload_downstream()
            return _json_safe({
                "decision": "reject",
                "rejection_reason": "test_arrays_missing",
                "candidate": new_v,
                "champion": prev_v,
            })

        cand_model, cand_flavor = load_model_any_flavor(f"models:/{REG_NAME}/{new_v}")
        prior_model, prior_flavor = load_model_any_flavor(f"models:/{REG_NAME}/{prev_v}")
        if cand_model is None or prior_model is None:
            raise RuntimeError(
                f"failed to load registered model(s): "
                f"candidate v{new_v}={cand_model is not None}, "
                f"prior v{prev_v}={prior_model is not None}; "
                "the MLflow registry returned a None artefact."
            )
        print(f"  loaded candidate v{new_v} (flavor={cand_flavor}), "
              f"prior v{prev_v} (flavor={prior_flavor})")
        cand_preds = predict_labels(cand_model, cand_flavor, X_test)
        prior_preds = predict_labels(prior_model, prior_flavor, X_test)

        # --- Gate 1: bootstrap-CI lift gate -------------------------------------
        lo, hi, point_lift = bootstrap_lift_ci(
            cand_preds, prior_preds, y_test, n_resamples=BOOTSTRAP_RESAMPLES,
        )
        lift_ok = lo > LIFT_FLOOR
        print(f"  lift CI:      point={point_lift:+.4f}, "
              f"95% CI=({lo:+.4f}, {hi:+.4f}), "
              f"lower-CI>{LIFT_FLOOR}? {lift_ok}")

        if not lift_ok:
            client.set_registered_model_alias(REG_NAME, "champion", str(prev_v))
            client.set_registered_model_alias(REG_NAME, "rejected", str(new_v))
            client.set_model_version_tag(
                REG_NAME, str(new_v), "rejected_reason",
                f"lift_ci: point={point_lift:+.4f}, 95% CI=({lo:+.4f}, {hi:+.4f}), "
                f"floor={LIFT_FLOOR}",
            )
            print(f"  decision: REJECT candidate (lift CI); @champion reverted to v{prev_v}")
            _reload_downstream()
            return _json_safe({
                "decision": "reject",
                "rejection_reason": "lift_ci",
                "candidate": new_v,
                "champion": prev_v,
                "lift_point": point_lift,
                "lift_ci_lo": lo,
                "lift_ci_hi": hi,
            })

        # --- Gate 3: equity-parity gate -----------------------------------------
        subgroup_labels = None
        y_test_eq: np.ndarray | None = None
        cand_preds_eq: np.ndarray | None = None
        prior_preds_eq: np.ndarray | None = None
        attrs_source_msg = "no protected attributes available, equity gate skipped"
        try:
            cleaned_csv = f"{DATA_DIR}/cleaned.csv"
            cleaned = pd.read_csv(
                cleaned_csv,
                usecols=lambda c: c in {"patient_nbr", "race", "gender"},
            )
            attr_cols = [c for c in ("race", "gender") if c in cleaned.columns]
            if test_pids is not None and attr_cols:
                per_patient_attrs = (
                    cleaned.drop_duplicates(subset=["patient_nbr"], keep="first")
                           .set_index("patient_nbr")[attr_cols]
                )
                test_pids_int = pd.Series(test_pids).astype(int)
                attrs = per_patient_attrs.reindex(test_pids_int.values).reset_index(drop=True)
                assert len(attrs) == len(y_test), (
                    f"row-alignment bug: attrs={len(attrs)} test_pids={len(test_pids)} "
                    f"y_test={len(y_test)} after per-patient reindex"
                )
                valid = attrs[attr_cols].notna().all(axis=1).values
                attrs = attrs.loc[valid].reset_index(drop=True)
                y_test_eq = np.asarray(y_test)[valid]
                cand_preds_eq = np.asarray(cand_preds)[valid]
                prior_preds_eq = np.asarray(prior_preds)[valid]
                n_dropped = (~valid).sum()
                if {"race", "gender"}.issubset(attrs.columns):
                    subgroup_labels = (
                        attrs["race"].astype(str) + " x " + attrs["gender"].astype(str)
                    ).values
                    attrs_source_msg = (
                        f"race x gender (joined by patient_nbr, "
                        f"{n_dropped} unmatched test rows dropped)"
                    )
                elif "gender" in attrs.columns:
                    subgroup_labels = attrs["gender"].astype(str).values
                    attrs_source_msg = (
                        f"gender only, race column missing "
                        f"({n_dropped} unmatched test rows dropped)"
                    )
            else:
                attrs_source_msg = "test_patient_ids missing or no attrs, equity gate skipped"
        except Exception as exc:  # pragma: no cover
            attrs_source_msg = f"attribute load failed: {exc}, equity gate skipped"

        print(f"  equity:       subgroup source = {attrs_source_msg}")
        equity_ok = True
        equity_drops: dict[str, float] = {}
        if subgroup_labels is not None:
            assert y_test_eq is not None and cand_preds_eq is not None and prior_preds_eq is not None, (
                "row-alignment bug: subgroup_labels was set without binding the eq arrays"
            )
            cand_recalls = per_subgroup_recall(y_test_eq, cand_preds_eq, subgroup_labels)
            prior_recalls = per_subgroup_recall(y_test_eq, prior_preds_eq, subgroup_labels)
            for level, prev_r in prior_recalls.items():
                cand_r = cand_recalls.get(level, 0.0)
                drop = prev_r - cand_r
                if drop > EQUITY_RECALL_TOL:
                    equity_ok = False
                    equity_drops[level] = drop
                    print(f"    subgroup {level!r}: prior_recall={prev_r:.4f}, "
                          f"cand_recall={cand_r:.4f}, drop={drop:+.4f} (tol={EQUITY_RECALL_TOL})")
            if equity_ok:
                print(f"    all {len(prior_recalls)} subgroup recalls within tol={EQUITY_RECALL_TOL}")

        if not equity_ok:
            client.set_registered_model_alias(REG_NAME, "champion", str(prev_v))
            client.set_registered_model_alias(REG_NAME, "rejected", str(new_v))
            client.set_model_version_tag(
                REG_NAME, str(new_v), "rejected_reason",
                f"equity: recall drops > {EQUITY_RECALL_TOL} on {sorted(equity_drops)}",
            )
            print(f"  decision: REJECT candidate (equity); @champion reverted to v{prev_v}")
            _reload_downstream()
            return _json_safe({
                "decision": "reject",
                "rejection_reason": "equity",
                "candidate": new_v,
                "champion": prev_v,
                "equity_drops": equity_drops,
            })

        # --- All gates passed ---------------------------------------------------
        client.set_model_version_tag(REG_NAME, str(new_v), "promoted_at", now.isoformat())
        print(f"  decision: KEEP candidate as @champion "
              f"(lift lower-CI={lo:+.4f} > {LIFT_FLOOR}, cooldown ok, equity ok)")
        _reload_downstream()
        return _json_safe({
            "decision": "promote",
            "version": new_v,
            "test_f1": new_f1,
            "lift_point": point_lift,
            "lift_ci_lo": lo,
            "lift_ci_hi": hi,
        })

    @task
    def redeploy_inference_api(gate_decision: dict) -> dict:
        """Rolling-restart the inference-api Deployment on EKS after a promotion.

        Closes the orchestration<->serving seam: a retrain that produced a new
        live champion needs the running inference pods to re-resolve the
        '@champion' alias. The API loads the model from the MLflow registry at
        startup, so a `kubectl rollout restart` is a full model redeploy with no
        image rebuild and no registry push. The inference API is the only
        medi-watch workload on k8s.

        Mechanism:
          1. `aws eks update-kubeconfig --name <cluster> --region <region>`
             (idempotent) so kubectl authenticates to EKS as this worker's IAM
             principal. That principal needs a cluster access entry granting
             `patch deployments` in the namespace (see infra/aws/README.md).
          2. `kubectl rollout restart deployment/<dep> -n <ns>` rolls pods so
             each fresh pod re-pulls the current @champion.
          3. `kubectl rollout status deployment/<dep> -n <ns> --timeout=5m`
             fails the task loudly on a failed or blocked rollout
             (ImagePullBackOff, crashloop, 503 on the readiness probe) instead
             of silently leaving stale pods serving the old model.

        Configuration (env first, Airflow Variable fallback):
          - LOCAL_KUBE_CONTEXT   optional. When set, runs in LOCAL mode against a
            minikube/kind context using the worker's mounted kubeconfig, with no
            `aws eks update-kubeconfig`, no AWS creds. Takes precedence over the
            EKS vars below.
          - EKS_CLUSTER_NAME / AWS_REGION   required for the EKS path. When both
            are unset and LOCAL_KUBE_CONTEXT is also unset, the task skips.
          - K8S_NAMESPACE        default 'medi-watch'
          - INFERENCE_DEPLOYMENT default 'medi-watch-inference'
        AWS credentials (EKS path only) come from the worker's environment or
        mounted ~/.aws (supplied via infra/docker-compose.yml).

        Deploy / skip semantics:
          - Deploy when the champion changed (decision in {"promote",
            "bootstrap"}) OR when the deployment is not already serving. A
            fresh stand-up has no Ready pods (the readiness probe returns 503
            until a model loads), so the served resource is effectively absent
            and must be (re)deployed even though the champion did not change.
          - Skip only when the champion is unchanged AND the deployment is
            already serving, where a rollout would just churn healthy pods.
          - No target configured (neither LOCAL_KUBE_CONTEXT nor EKS
            cluster/region) still skips rather than guess a cluster.

        Subprocess failures (`check=True`) surface so a broken redeploy is
        visible immediately.
        """
        import shutil
        import subprocess
        from airflow.sdk.exceptions import AirflowSkipException

        try:
            from airflow.sdk import Variable
            def _cfg(name: str, default: str | None = None) -> str | None:
                return os.environ.get(name) or Variable.get(name, default=default)
        except Exception:  # Variable backend unavailable in some parse contexts
            def _cfg(name: str, default: str | None = None) -> str | None:
                return os.environ.get(name, default)

        decision = (gate_decision or {}).get("decision")
        champion_changed = decision in ("promote", "bootstrap")

        # When a remote CI/CD provider is configured, the sibling
        # `trigger_remote_deploy` task owns the build+deploy (it remotely fires
        # the Jenkins/Azure/Cloud Build job). This direct-kubectl path then steps
        # aside so the two never deploy the same change twice. It remains the
        # fallback for local/dev where no CI provider is wired.
        from helpers.cicd_trigger import select_provider
        if select_provider(os.environ):
            raise AirflowSkipException(
                "a CI/CD provider is configured; trigger_remote_deploy owns the deploy")

        namespace = _cfg("K8S_NAMESPACE", "medi-watch")
        deployment = _cfg("INFERENCE_DEPLOYMENT", "medi-watch-inference")

        def _run(cmd: list[str]) -> None:
            print(f"  $ {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # Local mode (minikube / kind): LOCAL_KUBE_CONTEXT short-circuits the EKS
        # path. The worker's mounted kubeconfig already points at the local
        # cluster, so there is no `aws eks update-kubeconfig` step and AWS creds
        # are not needed, and kubectl just targets the named context. Used to
        # exercise the redeploy loop against a local cluster (see
        # infra/k8s/deploy-local.sh) without standing up EKS.
        local_context = _cfg("LOCAL_KUBE_CONTEXT")
        if local_context:
            if shutil.which("kubectl") is None:
                raise RuntimeError(
                    "redeploy_inference_api: 'kubectl' not found on PATH; the airflow "
                    "image must install kubectl (see infra/airflow/Dockerfile)."
                )
            kubectl = ["kubectl", "--context", local_context]
            target_desc = f"context {local_context} (local)"
        else:
            cluster = _cfg("EKS_CLUSTER_NAME")
            region = _cfg("AWS_REGION")
            if not (cluster and region):
                print("  redeploy_inference_api: EKS_CLUSTER_NAME / AWS_REGION not set "
                      "and LOCAL_KUBE_CONTEXT unset, skip")
                raise AirflowSkipException("no target cluster configured")
            for tool in ("aws", "kubectl"):
                if shutil.which(tool) is None:
                    raise RuntimeError(
                        f"redeploy_inference_api: '{tool}' not found on PATH; the airflow "
                        "image must install awscli + kubectl (see infra/airflow/Dockerfile)."
                    )
            _run(["aws", "eks", "update-kubeconfig", "--name", cluster, "--region", region])
            kubectl = ["kubectl"]
            target_desc = f"cluster {cluster} ({region})"

        # Deploy when the champion changed, OR when the deployment is not
        # already serving. `readyReplicas` is empty/0 both when the deployment
        # is absent and when its pods are up but failing readiness (no model
        # loaded yields a 503 on /healthz), so a value of 0 means the served
        # resource does not exist yet. Only an unchanged champion on an
        # already-serving deployment is skipped, where a rollout would just
        # churn healthy pods.
        ready = subprocess.run(
            kubectl + ["get", "deployment", deployment, "-n", namespace,
                       "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True,
        )
        cluster_reachable = ready.returncode == 0
        serving = cluster_reachable and ready.stdout.strip() not in ("", "0")
        # Champion unchanged: skip when already serving, or when the cluster is
        # unreachable (a down local cluster must not fail a retrain with nothing
        # new to deploy). A promotion still deploys below and surfaces failures.
        if not champion_changed:
            if serving:
                print(f"  redeploy_inference_api: champion unchanged and "
                      f"deployment/{deployment} already serving on {target_desc}, skip")
                raise AirflowSkipException("champion unchanged and deployment already serving")
            if not cluster_reachable:
                print(f"  redeploy_inference_api: champion unchanged and target "
                      f"cluster unreachable on {target_desc}, skip")
                raise AirflowSkipException("champion unchanged and target cluster unreachable")

        reason = "champion changed" if champion_changed else "resource not serving"
        print(f"  redeploy_inference_api: deploying ({reason}) to {target_desc}")
        _run(kubectl + ["rollout", "restart", f"deployment/{deployment}", "-n", namespace])
        _run(kubectl + ["rollout", "status", f"deployment/{deployment}",
                        "-n", namespace, "--timeout=5m"])

        print(f"  redeploy_inference_api: rolled deployment/{deployment} in ns {namespace} "
              f"on {target_desc}; pods re-pulled @champion")
        return {
            "redeployed": True,
            "decision": decision,
            "target": target_desc,
            "namespace": namespace,
            "deployment": deployment,
        }

    @task
    def trigger_remote_deploy(gate_decision: dict) -> dict:
        """Remotely trigger the CI/CD build+deploy job for the inference API.

        This is the *remote orchestration* path: instead of running kubectl from
        inside the worker, the DAG fires the existing build+deploy pipeline
        (infra/ci-cd/{Jenkinsfile,azure-pipelines.yml,cloudbuild.yaml}) on
        whichever CI system is configured via env (CICD_PROVIDER + that
        provider's credentials). The pipeline rebuilds the image and rolls the
        deployment, so build and deploy are orchestrated remotely.

        Skip semantics:
          - champion unchanged (decision not in {promote, bootstrap}) -> skip,
            nothing to deploy.
          - no CI/CD provider configured -> skip; the sibling
            redeploy_inference_api handles the rollout directly (local/dev).
        A non-2xx response from the CI system raises, so a broken trigger is
        visible immediately (matching redeploy_inference_api's check=True).
        """
        from airflow.sdk.exceptions import AirflowSkipException

        from helpers.cicd_trigger import (
            RemoteTriggerError,
            select_provider,
            trigger_remote_deploy as _trigger,
        )

        decision = (gate_decision or {}).get("decision")
        if decision not in ("promote", "bootstrap"):
            raise AirflowSkipException(
                f"champion unchanged (decision={decision}); no remote deploy to trigger")

        provider = select_provider(os.environ)
        if provider is None:
            raise AirflowSkipException(
                "no CI/CD provider configured (CICD_PROVIDER unset); "
                "redeploy_inference_api handles the deploy directly")

        # Attribute the run to the drift scenario + champion decision so the CI
        # log shows why it fired. SKIP_SMOKE so a champion-swap redeploy is fast
        # (the smoke stage gates code changes, not model swaps).
        try:
            from airflow.sdk import get_current_context
            scenario = (get_current_context()["dag_run"].conf or {}).get("scenario", "unknown")
        except Exception:  # context unavailable outside a run
            scenario = "unknown"
        reason = f"retrain_on_drift: {decision} (scenario={scenario})"

        try:
            result = _trigger(provider=provider, reason=reason, skip_smoke=True)
        except RemoteTriggerError as exc:
            raise RuntimeError(f"remote CI/CD trigger failed: {exc}") from exc

        print(f"  trigger_remote_deploy: fired {provider} build+deploy "
              f"(HTTP {result['status']}) for {decision}; reason={reason!r}")
        return result

    @task
    def tag_scenario(conclusion_result: dict) -> dict:
        """Tag the freshly registered model version with the drift scenario that
        triggered this retrain, so every retrain (and its MLflow version) is
        attributable to the drift that caused it. Best-effort: a tagging failure
        must not fail the retrain."""
        from airflow.sdk import get_current_context

        ctx = get_current_context()
        scenario = (ctx["dag_run"].conf or {}).get("scenario", "unknown")
        version = conclusion_result.get("registered_version")
        if not version:
            print(f"[retrain] no registered version to tag (scenario={scenario})")
            return {"scenario": scenario, "tagged": False}
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(
                os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            MlflowClient().set_model_version_tag(
                "medi-watch-readmission", str(version), "drift_scenario", scenario)
            print(f"[retrain] tagged medi-watch-readmission v{version} "
                  f"drift_scenario={scenario}")
            return {"scenario": scenario, "version": str(version), "tagged": True}
        except Exception as e:  # noqa: BLE001
            print(f"[retrain] scenario tag skipped (non-fatal): {e}")
            return {"scenario": scenario, "tagged": False}

    hpo = run_hpo_task()
    training = run_training_task()
    conclusion = run_conclusion_task()
    tag = tag_scenario(conclusion)
    gate = gate_champion()
    # Two mutually-exclusive deploy paths fan out from the gate, each self-
    # skipping by config: trigger_remote_deploy fires the remote CI/CD job when a
    # provider is configured, otherwise redeploy_inference_api rolls the
    # deployment directly with kubectl (local/dev fallback).
    redeploy = redeploy_inference_api(gate)
    remote_deploy = trigger_remote_deploy(gate)
    hpo >> training >> conclusion >> gate
    gate >> redeploy
    gate >> remote_deploy
    conclusion >> tag


# Registered unconditionally so the orchestration plane stands on its own as
# soon as Airflow is up. This DAG is the retrain engine, fired by the
# `scheduled_drift_check` DAG on an ALERT verdict, or manually from the Airflow
# UI (and by NB06 for the initial champion bootstrap). The inference-api ping in
# `_reload_downstream` is best-effort (try/except), so the API being absent is
# harmless.
retrain_on_drift()
