# Patch-pinned for reproducible builds. MLflow is an HTTP tracking server and
# never shares Python objects with the Ray cluster, so staying on 3.12 is fine.
# 3.12.13 is the current 3.12 slim patch.
FROM python:3.12.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        netcat-openbsd \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        mlflow==3.11.1 \
        psycopg2-binary==2.9.12

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /mlflow
EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
