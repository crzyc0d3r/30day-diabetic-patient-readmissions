#!/usr/bin/env bash
# Build the medi-watch inference image, push it to ECR, and deploy/refresh the
# inference-api Deployment on EKS. Idempotent and safe to re-run for every release.
#
# This is the ONLY workload deployed to k8s. It does NOT touch airflow/ray/mlflow
# (those run in docker-compose). The Airflow `retrain_on_drift` DAG performs the
# subsequent model-swap redeploys via `kubectl rollout restart`. This script is
# for the initial deploy and for serving-CODE releases.
#
# Required environment (the script fails fast if any is unset/empty):
#   AWS_REGION            e.g. us-east-1
#   EKS_CLUSTER_NAME      e.g. medi-watch  (matches eksctl-cluster.yaml metadata.name)
#   ECR_REGISTRY          e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com
#   MLFLOW_TRACKING_URI   endpoint reachable from the EKS VPC (registry source of @champion)
#   INFERENCE_API_TOKEN   bearer token clients must present, stamped into the Secret
# Optional:
#   IMAGE_TAG            defaults to the current git SHA
#   ECR_REPO             defaults to mlops-inference-api
#   K8S_NAMESPACE        defaults to medi-watch
#
# Usage (from anywhere in the repo):
#   export AWS_REGION=us-east-1 EKS_CLUSTER_NAME=medi-watch
#   export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
#   export MLFLOW_TRACKING_URI=http://mlflow.internal.example.com:5000
#   export INFERENCE_API_TOKEN=$(openssl rand -hex 32)
#   bash infra/aws/deploy.sh

set -euo pipefail

# --- resolve repo root so the docker build context + manifest path are stable ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ECR_REPO="${ECR_REPO:-mlops-inference-api}"
K8S_NAMESPACE="${K8S_NAMESPACE:-medi-watch}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"

# --- fail fast on missing config (envsubst silently leaves $VAR literal otherwise) ---
missing=()
for var in AWS_REGION EKS_CLUSTER_NAME ECR_REGISTRY MLFLOW_TRACKING_URI INFERENCE_API_TOKEN; do
  if [[ -z "${!var:-}" ]]; then missing+=("${var}"); fi
done
if (( ${#missing[@]} )); then
  echo "ERROR: required environment variables unset: ${missing[*]}" >&2
  echo "       see the header of $0 for the full list." >&2
  exit 1
fi

export ECR_REGISTRY IMAGE_TAG MLFLOW_TRACKING_URI INFERENCE_API_TOKEN
IMAGE_REF="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo "==> Target"
echo "    cluster   : ${EKS_CLUSTER_NAME} (${AWS_REGION})"
echo "    image     : ${IMAGE_REF}"
echo "    namespace : ${K8S_NAMESPACE}"
echo "    mlflow    : ${MLFLOW_TRACKING_URI}"

# --- 1. ensure the ECR repo exists (ignore "already exists") ---
echo "==> Ensuring ECR repository ${ECR_REPO} exists"
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
       --image-scanning-configuration scanOnPush=true >/dev/null

# --- 2. docker login to ECR ---
echo "==> Logging docker in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# --- 3. build + push. Build context is the REPO ROOT so the image can ship
#        helpers/model_loading.py (see infra/inference-api/Dockerfile). ---
echo "==> Building ${IMAGE_REF}"
docker build \
  -f "${REPO_ROOT}/infra/inference-api/Dockerfile" \
  -t "${IMAGE_REF}" \
  -t "${ECR_REGISTRY}/${ECR_REPO}:latest" \
  "${REPO_ROOT}"

echo "==> Pushing image (sha tag + latest)"
docker push "${IMAGE_REF}"
docker push "${ECR_REGISTRY}/${ECR_REPO}:latest"

# --- 4. point kubectl at the cluster (idempotent) ---
echo "==> Updating kubeconfig for ${EKS_CLUSTER_NAME}"
aws eks update-kubeconfig --name "${EKS_CLUSTER_NAME}" --region "${AWS_REGION}"

# --- 5. render + apply the inference manifest (the ONLY k8s workload) ---
echo "==> Applying inference-api manifest"
envsubst '${ECR_REGISTRY} ${IMAGE_TAG} ${MLFLOW_TRACKING_URI} ${INFERENCE_API_TOKEN}' \
  < "${REPO_ROOT}/infra/k8s/inference-api.yaml" \
  | kubectl apply -f -

# --- 6. wait for the rollout so an ImagePullBackOff / crashloop fails loudly ---
echo "==> Waiting for rollout"
kubectl rollout status deployment/medi-watch-inference -n "${K8S_NAMESPACE}" --timeout=5m

echo "==> Done. Smoke test:"
echo "    kubectl port-forward svc/medi-watch-inference 8002:80 -n ${K8S_NAMESPACE}"
echo "    curl -fsS http://localhost:8002/healthz          # no auth — liveness + loaded model meta"
echo "    curl -fsS -H \"Authorization: Bearer \$INFERENCE_API_TOKEN\" \\"
echo "         -H 'Content-Type: application/json' -d '{\"instances\":[{}]}' \\"
echo "         http://localhost:8002/invocations            # auth required"
