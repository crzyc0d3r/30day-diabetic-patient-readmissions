#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_HOST:=postgres}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_USER:?POSTGRES_USER required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD required}"
: "${MLFLOW_DB:=mlflow}"
: "${ARTIFACT_ROOT:=/mlflow/artifacts}"

echo "mlflow: waiting for postgres at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
until nc -z "${POSTGRES_HOST}" "${POSTGRES_PORT}"; do
    sleep 1
done
echo "mlflow: postgres is reachable"

mkdir -p "${ARTIFACT_ROOT}"

BACKEND_URI="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${MLFLOW_DB}"

exec mlflow server \
    --backend-store-uri "${BACKEND_URI}" \
    --default-artifact-root "mlflow-artifacts:/" \
    --artifacts-destination "${ARTIFACT_ROOT}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-*}" \
    --cors-allowed-origins "${MLFLOW_CORS_ORIGINS:-*}"
