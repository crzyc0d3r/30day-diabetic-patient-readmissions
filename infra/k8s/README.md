# Kubernetes Topology

> **Scope (current).** The **inference API is the only workload medi-watch
> deploys to Kubernetes**: `inference-api.yaml`. Airflow, Ray, and MLflow run
> via `infra/docker-compose.yml` (off k8s); the Airflow `retrain_on_drift` DAG
> reaches the cluster as a *client* and `kubectl rollout restart`s the inference
> Deployment whenever a new `@champion` is promoted. The AWS path lives in
> `infra/aws/` (EKS + ECR + runbook); a local minikube/kind path lives in
> `deploy-local.sh` (below).
>
> The sibling `mlflow.yaml` / `ray.yaml` / `airflow.yaml` manifests are **retained
> only as optional, self-contained local-cluster reference** (Postgres-backed
> MLflow, Ray head+workers, LocalExecutor Airflow). They are NOT part of the AWS
> deploy path and the drift DAG never touches them. The legacy four-workload
> topology is documented further down for that optional use.

> **Reproducibility: loose spec vs lockfile.** `requirements.txt` at the
> project root is the **loose dependency spec**, floated against the
> Python 3.13 wheel matrix (Ray, MLflow, Torch, and
> friends all bump frequently on aarch64 + Blackwell). The exact pin set
> the pipeline was last green on lives in `requirements.lock.txt`,
> regenerated with `pip freeze > requirements.lock.txt` from inside the
> working venv after every successful end-to-end run. Use the lockfile
> when reproducing a published champion or chasing a "works on my
> machine" Ray gRPC crash. Use the loose spec when bringing up a fresh
> environment that's allowed to track upstream. If `requirements.lock.txt`
> is missing in your checkout, regenerate it after your first green
> training run. The file is intentionally not consumed by Docker builds,
> so it stays human-curated.

## Components

| Manifest | What lands in the cluster | Why |
|---|---|---|
| `mlflow.yaml`         | In-cluster Postgres `StatefulSet` (`mlflow-postgres`) + headless `Service` + 5 Gi PVC for the DB. MLflow `Deployment` (1 replica) + `Service` + 5 Gi PVC for the artifact store. `ConfigMap` pointing `MLFLOW_BACKEND_STORE_URI` at `mlflow-postgres.medi-watch.svc.cluster.local`. | Experiment tracking and Model Registry on the same Postgres-only contract `init_mlflow` enforces. `MLFLOW_TRACKING_URI` for every other Pod resolves to the MLflow Service. |
| `ray.yaml`            | `ray-head` Deployment + `Service` (ports 6379 GCS, 10001 client, 8265 dashboard) + `ray-worker` Deployment (2 replicas). | Large-scale HPO fan-out for nb06 (`hyperparameter_tuning`). Workers join the head via GCS once the head's `getent hosts` lookup succeeds. |
| `airflow.yaml`        | LocalExecutor `Deployment` (webserver + scheduler containers) + `Service` (port 8080) + `PVC` (5 Gi). | Orchestration of `prepare_data_dag`, `evidently_drift_dag`, and `retrain_on_drift_dag`. |
| `inference-api.yaml`  | `Deployment` (2 replicas) + `Service` (port 80) + `HorizontalPodAutoscaler`. | Serves predictions from the `@champion` MLflow registry alias. |

All Pods live in the `medi-watch` namespace, run as non-root, and resolve
each other via in-cluster DNS (`<service>.medi-watch.svc.cluster.local`).
State that must survive a Pod restart sits on a `PersistentVolumeClaim`. That
state is the `mlflow-postgres` StatefulSet's PVC for tracking + registry
metadata, MLflow's artifact PVC for run artifacts, and Airflow's metadata
DB plus logs. Everything else is stateless.

```
   ┌──────────┐         ┌──────────────────┐
   │ Airflow  │─DAG run→│  Ray (head+2w)   │  HPO trials
   │ scheduler│         └──────────────────┘
   └────┬─────┘                  │
        │ logs / metrics         │ trial metrics
        ▼                        ▼
   ┌─────────────────────────────────────────┐
   │            MLflow tracker               │  registry: medi-watch-readmission @champion
   │  (backend: mlflow-postgres StatefulSet, │
   │   artifacts: 5Gi PVC)                   │
   └─────────────────┬───────────────────────┘
                     │ resolves @champion
                     ▼
              ┌─────────────────┐
              │ inference API   │  Service: 80/TCP
              │ (HPA 2..6)      │  HealthcheckProbe -> /healthz
              └─────────────────┘
```

## Deploy the inference API

**AWS / EKS:** see `infra/aws/README.md` → `infra/aws/deploy.sh` (build → ECR push
→ `envsubst` → `kubectl apply` → rollout status).

**Local (minikube / kind):** `infra/k8s/deploy-local.sh`. It builds the image,
loads it into the local cluster, and applies the same `inference-api.yaml`. It
needs a reachable MLflow with a registered `medi-watch-readmission@champion`
(from a minikube pod the host is `host.minikube.internal`; start MLflow with
`--allowed-hosts host.minikube.internal:5000`).

```bash
minikube start --driver=docker
# ... ensure MLflow is up and a @champion is registered ...
bash infra/k8s/deploy-local.sh
kubectl get pods -n medi-watch          # medi-watch-inference 2/2 Running

# Simulate the drift DAG's redeploy (what redeploy_inference_api runs):
kubectl rollout restart deployment/medi-watch-inference -n medi-watch
kubectl rollout status  deployment/medi-watch-inference -n medi-watch --timeout=4m
```

`maxUnavailable: 0 / maxSurge: 1` makes the restart a zero-downtime model swap:
fresh pods re-resolve `@champion` from the registry on startup, so no image
rebuild is needed for a model change.

Confirm the pod is serving the champion:

```bash
kubectl port-forward svc/medi-watch-inference 8002:80 -n medi-watch &
curl -fsS http://localhost:8002/healthz
```

---

## Legacy: optional four-workload local cluster

> Only for running the full stack (MLflow + Ray + Airflow + inference) **inside**
> one local cluster as a self-contained demo. Not the production path.

`mlflow → ray → airflow → inference-api` keeps every Pod's dependencies up when it starts. The probes
tolerate a single-shot `kubectl apply -f .`, though a fresh cluster will see the
inference Pod restart once while `mlflow.medi-watch.svc.cluster.local` becomes routable.

```bash
# from infra/k8s/
kubectl apply -f mlflow.yaml
kubectl apply -f ray.yaml
kubectl apply -f airflow.yaml
kubectl apply -f inference-api.yaml
kubectl get pods -n medi-watch
```

Bootstrap a local cluster for that legacy path (pick whichever you have):

```bash
# kind
kind create cluster --name medi-watch
docker compose -f ../docker-compose.yml build mlflow ray airflow inference-api
for img in mlops-mlflow:latest mlops-ray:latest mlops-airflow:latest mlops-inference-api:latest; do
  kind load docker-image "$img" --name medi-watch
done

# k3d
k3d cluster create medi-watch
# … same build step …
for img in mlops-mlflow:latest mlops-ray:latest mlops-airflow:latest mlops-inference-api:latest; do
  k3d image import "$img" -c medi-watch
done
```
