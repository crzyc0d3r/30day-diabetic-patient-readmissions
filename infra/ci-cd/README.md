# CI/CD

The platform ships four CI/CD definitions and a remote-trigger mechanism that
lets the orchestration layer drive a build and deploy with no human in the path.

## Pipeline definitions

Three controller pipelines and a GitHub-native workflow run the same stages:

| Definition | Stages |
|---|---|
| `Jenkinsfile` | validate, build the four images, smoke test, publish, deploy via `kubectl` rollout |
| `azure-pipelines.yml` | validate, build, deploy via `kubectl apply` + rollout |
| `cloudbuild.yaml` | validate, build, deploy to GKE |
| `.github/workflows/ci.yml` | a Validate job (requirements pin lockstep, advisory Ruff, structural pytest, notebook nbformat parse, the remote-trigger handoff) and a Build job (compose-config validation and the inference-api image) |

The GitHub Actions workflow runs on every push and pull request to `main`. Its
run history is visible in the repository's Actions tab.

## Remote triggering from the orchestration

On a champion promotion the `retrain_on_drift` DAG's `trigger_remote_deploy` task
calls [`helpers/cicd_trigger.py`](../../helpers/cicd_trigger.py), a
provider-agnostic REST client that fires the configured CI job: Jenkins
`buildWithParameters`, Azure Pipelines `runs`, or Cloud Build `triggers:run`.
When a provider is configured, the direct-`kubectl` `redeploy_inference_api` task
self-skips, so the two deploy paths are mutually exclusive. `test/test_cicd_trigger.py`
covers the request shaping and the self-skip for each provider.

Inspect or fire the handoff:

```bash
python -m helpers.cicd_trigger --provider jenkins --dry-run   # print the exact request
python infra/ci-cd/demo_remote_trigger.py                     # fire at a local mock CI server
```

## Local runner

`run_ci_local.sh` runs the Validate and smoke stages the controller pipelines
define: requirements pin lockstep, structural pytest, notebook parse, the
remote-trigger handoff, and an inference-API `/healthz` smoke check. It exercises
the gating logic without a CI controller.

## Files

| File | Purpose |
|---|---|
| `helpers/cicd_trigger.py` | provider-agnostic remote trigger (+ `python -m helpers.cicd_trigger` CLI) |
| `demo_remote_trigger.py` | Airflow-to-CI handoff demo against a mock CI server |
| `run_ci_local.sh` | local runner for the Validate and smoke stages |
| `.github/workflows/ci.yml` | GitHub Actions workflow, runs on push and pull request |
