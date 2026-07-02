# Patch-pinned to match the Ray cluster / driver interpreter (3.13.2). This
# service is HTTP-only (it does not pickle across the Ray boundary), so the
# patch need not match for correctness. It is pinned for reproducible builds.
FROM python:3.13.2-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build context is the REPO ROOT (see infra/docker-compose.yml) so this image
# can ship helpers/model_loading.py, the flavor-aware loader shared with the
# retrain DAG. A repo-root .dockerignore keeps the context lean (no data/, .venv).
WORKDIR /app
COPY infra/inference-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY infra/inference-api/main.py /app/
# Minimal slice of the helpers package: just the package marker + the shared
# loader. main.py imports `helpers.model_loading`, and /app is on sys.path.
COPY helpers/__init__.py /app/helpers/__init__.py
COPY helpers/model_loading.py /app/helpers/model_loading.py

EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=4 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/healthz').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
