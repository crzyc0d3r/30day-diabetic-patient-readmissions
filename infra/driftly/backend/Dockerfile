# Driftly API. Pinned to the stack interpreter (3.13.2) for reproducible builds.
# Build context is the REPO ROOT (see infra/docker-compose.yml) so the image can
# ship the slice of `helpers/` it imports (drift_sim + constants), exactly like
# the inference-api image ships helpers/model_loading.py.
FROM python:3.13.2-slim

# libgomp1: scipy/numpy OpenMP runtime dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY infra/driftly/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Backend modules (flat: main imports config/drift_metrics/history/simulator/
# airflow_client by name).
COPY infra/driftly/backend/config.py infra/driftly/backend/drift_metrics.py \
     infra/driftly/backend/history.py infra/driftly/backend/main.py \
     infra/driftly/backend/simulator.py infra/driftly/backend/airflow_client.py /app/
# Minimal helpers slice: the package marker + the two modules drift_sim needs.
COPY helpers/__init__.py /app/helpers/__init__.py
COPY helpers/constants.py /app/helpers/constants.py
COPY helpers/drift_sim.py /app/helpers/drift_sim.py

EXPOSE 8003
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=4 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/api/health').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
