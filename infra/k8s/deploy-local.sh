#!/usr/bin/env bash
# Local mirror of infra/aws/deploy.sh for a minikube/kind cluster — build the
# inference image, load it into the local cluster, and apply the SAME
# inference-api.yaml manifest (the only medi-watch workload on k8s). Use this to
# validate the inference-on-k8s + rollout-restart loop without any AWS spend.
#
# Prereqs:
#   • a running local cluster:  minikube start --driver=docker   (or kind/k3d)
#   • a reachable MLflow with a registered medi-watch-readmission@champion.
#     From a minikube pod the host is `host.minikube.internal`; start MLflow with
#     --allowed-hosts including "host.minikube.internal:5000". See the repo's
#     local smoke notes for a stub-champion registration helper.
#
# Env (all optional — sensible local defaults):
#   CLUSTER_TOOL          minikube | kind   (default: minikube)
#   IMAGE_TAG             default: latest
#   MLFLOW_TRACKING_URI   default: http://host.minikube.internal:5000
#   INFERENCE_API_TOKEN   default: local-smoke-token
#   K8S_NAMESPACE         default: medi-watch
#
# Usage:
#   bash infra/k8s/deploy-local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLUSTER_TOOL="${CLUSTER_TOOL:-minikube}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
K8S_NAMESPACE="${K8S_NAMESPACE:-medi-watch}"
# ECR_REGISTRY is a placeholder locally — imagePullPolicy IfNotPresent + a
# preloaded image named `local/...` means the registry host is never contacted.
export ECR_REGISTRY="local"
export IMAGE_TAG
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://host.minikube.internal:5000}"
export INFERENCE_API_TOKEN="${INFERENCE_API_TOKEN:-local-smoke-token}"
IMAGE_REF="local/mlops-inference-api:${IMAGE_TAG}"

echo "==> Target: ${CLUSTER_TOOL} cluster, image ${IMAGE_REF}, mlflow ${MLFLOW_TRACKING_URI}"

echo "==> Building ${IMAGE_REF} (context = repo root)"
docker build -f "${REPO_ROOT}/infra/inference-api/Dockerfile" -t "${IMAGE_REF}" "${REPO_ROOT}"

echo "==> Loading image into ${CLUSTER_TOOL}"
case "${CLUSTER_TOOL}" in
  minikube) minikube image load "${IMAGE_REF}" ;;
  kind)     kind load docker-image "${IMAGE_REF}" ;;
  *) echo "unsupported CLUSTER_TOOL: ${CLUSTER_TOOL}" >&2; exit 1 ;;
esac

echo "==> Rendering + applying inference-api.yaml"
envsubst '${ECR_REGISTRY} ${IMAGE_TAG} ${MLFLOW_TRACKING_URI} ${INFERENCE_API_TOKEN}' \
  < "${REPO_ROOT}/infra/k8s/inference-api.yaml" \
  | kubectl apply -f -

# ROLLOUT_WAIT=false skips the blocking wait — used by `mediwatch.sh init --k8s`,
# where a fresh stack has no @champion yet so the pods cannot reach Ready until a
# training/retrain run registers one. The Deployment is still applied; it goes
# green automatically once a champion exists.
if [ "${ROLLOUT_WAIT:-true}" = "false" ]; then
  echo "==> Applied (ROLLOUT_WAIT=false) — not blocking on readiness."
  echo "    Pods reach Ready once medi-watch-readmission@champion is registered."
  kubectl get pods -n "${K8S_NAMESPACE}" 2>/dev/null || true
else
  echo "==> Waiting for rollout"
  kubectl rollout status deployment/medi-watch-inference -n "${K8S_NAMESPACE}" --timeout=4m
fi

echo "==> Done. Smoke test:"
echo "    kubectl port-forward svc/medi-watch-inference 8002:80 -n ${K8S_NAMESPACE} &"
echo "    curl -fsS http://localhost:8002/healthz"
echo "    curl -fsS -H 'Authorization: Bearer ${INFERENCE_API_TOKEN}' -H 'Content-Type: application/json' \\"
echo "      -d '{\"instances\":[{\"race\":\"Caucasian\",\"gender\":\"Male\",\"time_in_hospital\":5,\"num_medications\":12}]}' \\"
echo "      http://localhost:8002/invocations"
echo
echo "    # Simulate the drift DAG's redeploy on a new @champion:"
echo "    kubectl rollout restart deployment/medi-watch-inference -n ${K8S_NAMESPACE}"
echo "    kubectl rollout status  deployment/medi-watch-inference -n ${K8S_NAMESPACE} --timeout=4m"
