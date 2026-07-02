# AWS deploy: medi-watch inference API on EKS

The inference API is the **only** workload medi-watch runs on Kubernetes. Airflow,
Ray, and MLflow stay in `docker-compose` (see `infra/docker-compose.yml`). The
Airflow `retrain_on_drift` DAG is a *client* of this EKS cluster: when a retrain
promotes a new `@champion`, its `redeploy_inference_api` task runs
`kubectl rollout restart` so fresh pods re-pull the promoted model from the MLflow
registry.

```
 docker-compose (local / VM)                    AWS
   Airflow  ──(aws + kubectl rollout restart)──▶ EKS: Deployment medi-watch-inference
   Ray, MLflow*, Postgres                         pulls image from ECR
        │ @champion                                   │ resolves @champion
        ▼                                             ▼
   MLflow registry  ◀── reachable URL ─────────  inference pods
```
`*` MLflow is an existing endpoint that **must be routable from the EKS VPC**.

## Prerequisites

- `awscli` v2, `eksctl`, `kubectl`, `docker`, `envsubst` (from `gettext`).
- AWS credentials with rights to create EKS + ECR + the IAM/VPC resources eksctl
  provisions (`aws sts get-caller-identity` should succeed).

## Required configuration

| Variable | Example | Used by |
|---|---|---|
| `AWS_REGION` | `us-east-1` | deploy.sh, DAG, eksctl config (`metadata.region`) |
| `EKS_CLUSTER_NAME` | `medi-watch` | deploy.sh, DAG, eksctl config (`metadata.name`) |
| `ECR_REGISTRY` | `123456789012.dkr.ecr.us-east-1.amazonaws.com` | deploy.sh, manifest image ref |
| `MLFLOW_TRACKING_URI` | `http://mlflow.internal.example.com:5000` | manifest ConfigMap (pod resolves `@champion`) |
| `INFERENCE_API_TOKEN` | `$(openssl rand -hex 32)` | manifest Secret, clients send `Authorization: Bearer` |

> `AWS_REGION`, `EKS_CLUSTER_NAME`, and `metadata.{region,name}` in
> `eksctl-cluster.yaml` must all agree, or `update-kubeconfig` points at the wrong
> (or a nonexistent) cluster.

## One-time provision

```bash
# 1. cluster (~15 min). Edit eksctl-cluster.yaml first if you changed name/region.
eksctl create cluster -f infra/aws/eksctl-cluster.yaml

# 2. config
export AWS_REGION=us-east-1 EKS_CLUSTER_NAME=medi-watch
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
export MLFLOW_TRACKING_URI=http://mlflow.internal.example.com:5000
export INFERENCE_API_TOKEN=$(openssl rand -hex 32)   # save this in your secrets manager

# 3. build → push → apply → wait for rollout
bash infra/aws/deploy.sh
```

## Let the Airflow worker redeploy (EKS access entry)

The DAG runs `kubectl rollout restart` from inside the Airflow container, which
authenticates to EKS as a *separate* IAM principal. IAM proves identity. The
cluster's **access entries** grant in-cluster rights. Grant the worker's principal
permission to patch deployments in the `medi-watch` namespace:

```bash
WORKER_PRINCIPAL_ARN=arn:aws:iam::123456789012:user/airflow-deployer  # the creds the worker uses

# Cluster access entry + a namespaced RBAC policy (EKS access-entry API):
eksctl create accessentry \
  --cluster medi-watch --region us-east-1 \
  --principal-arn "$WORKER_PRINCIPAL_ARN" \
  --kubernetes-groups medi-watch-deployers

# Bind that group to rollout-restart rights (kubectl as a cluster admin):
kubectl -n medi-watch create role inference-redeployer \
  --verb=get,list,watch,patch,update --resource=deployments,deployments/scale
kubectl -n medi-watch create rolebinding inference-redeployer \
  --role=inference-redeployer --group=medi-watch-deployers
```

Then give the Airflow worker AWS creds and cluster info. See `infra/.env` keys
`AWS_*`, `EKS_CLUSTER_NAME`, `K8S_NAMESPACE`, `INFERENCE_DEPLOYMENT`, and the
`x-airflow-common` env block / `~/.aws` mount in `infra/docker-compose.yml`.

## Verify

```bash
kubectl get pods -n medi-watch                       # medi-watch-inference 2/2 Running
kubectl port-forward svc/medi-watch-inference 8002:80 -n medi-watch &
curl -fsS http://localhost:8002/healthz              # {"status":"ok","model":{"version":...}}
```

A 503 on `/healthz` almost always means the pod cannot reach `MLFLOW_TRACKING_URI`
or there is no `@champion` registered yet, so check `kubectl logs` and VPC routing
to the MLflow endpoint before anything else.

## Redeploy loop (what the DAG does)

On a promotion the DAG runs, against this cluster:

```bash
kubectl rollout restart deployment/medi-watch-inference -n medi-watch
kubectl rollout status  deployment/medi-watch-inference -n medi-watch --timeout=5m
```

No image rebuild. The model is pulled from MLflow at pod startup. Re-run
`deploy.sh` only when the serving **code** (image) changes.

## Teardown (stop billing)

```bash
eksctl delete cluster -f infra/aws/eksctl-cluster.yaml
aws ecr delete-repository --repository-name mlops-inference-api --region us-east-1 --force
```
