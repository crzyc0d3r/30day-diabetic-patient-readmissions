# Digest-pinned, not just tag-pinned. The `3.2.1-python3.13` tag pins the Airflow
# release but not the CPython patch (Apache bakes 3.13.13 into it). The @sha256
# lock freezes that patch so the base interpreter is reproducible across builds.
#
# This base interpreter runs Airflow itself and the four non-Ray retrain tasks
# (training, conclusion, gate, trigger). It is NOT the Ray driver: the retrain
# DAG's run_06_hyperparameter_tuning task shells out to the separate
# /opt/ray-driver venv built below, which carries CPython 3.13.2 to match the
# Ray cluster (infra/ray/Dockerfile). The cluster/driver cloudpickle boundary
# therefore lives entirely on that 3.13.2 venv, so the Apache patch here need
# not match the cluster, and no "Python patch version mismatch" warning fires.
#
# To refresh the digest after an intentional Airflow bump:
#   docker buildx imagetools inspect apache/airflow:3.2.1-python3.13 \
#     --format '{{.Manifest.Digest}}'
FROM apache/airflow:3.2.1-python3.13@sha256:069285a766069fc32c6c3a6d6dafc3838936d8b9e71e9e2184126d96b2b8452a

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
 && python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (airflow)"

# awscli v2 + kubectl for the retrain DAG's `redeploy_inference_api` task: on a
# new @champion it runs `aws eks update-kubeconfig` + `kubectl rollout restart`
# against the EKS cluster that serves the inference API (the only medi-watch
# workload on k8s). The base Apache image blocks `pip` as root and the awscli v2
# bundle is not on PyPI anyway, so both come from their official release
# archives. Pinned for reproducible builds, and bumped deliberately.
USER root
ARG AWSCLI_VERSION=2.17.20
ARG KUBECTL_VERSION=1.31.0
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) aws_arch=x86_64; kube_arch=amd64 ;; \
      arm64) aws_arch=aarch64; kube_arch=arm64 ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    apt-get update && apt-get install -y --no-install-recommends curl unzip; \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${aws_arch}-${AWSCLI_VERSION}.zip" -o /tmp/awscliv2.zip; \
    unzip -q /tmp/awscliv2.zip -d /tmp; \
    /tmp/aws/install; \
    curl -fsSL "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/${kube_arch}/kubectl" -o /usr/local/bin/kubectl; \
    chmod +x /usr/local/bin/kubectl; \
    rm -rf /tmp/awscliv2.zip /tmp/aws /var/lib/apt/lists/*; \
    aws --version; kubectl version --client
USER airflow

# Second, isolated interpreter for the Ray *driver*. Apache bakes CPython
# 3.13.13 into the base image (above), but the Ray cluster runs 3.13.2
# (infra/ray/Dockerfile). The retrain DAG's run_06_hyperparameter_tuning task
# shells out to /opt/ray-driver/bin/python so the Ray client interpreter matches
# the cluster, removing the "Python patch version mismatch" warning at the
# source. uv downloads the exact 3.13.2 standalone build and resolves the
# cluster-matching wheels. First-party helpers/ stay on the /workspace mount
# (PYTHONPATH=/workspace), so only third-party libs live in this venv.
COPY ray-driver-requirements.txt /ray-driver-requirements.txt
# uv comes from its own image, not pip: the Apache base blocks `pip` as root.
# UV_PYTHON_INSTALL_DIR puts the managed 3.13.2 interpreter in a shared, readable
# path (the default is under root's home, which the airflow user cannot read at
# runtime). Both that dir and the venv are chowned to airflow so the task can run
# /opt/ray-driver/bin/python.
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
USER root
RUN uv python install 3.13.2 \
 && uv venv --python 3.13.2 /opt/ray-driver \
 && uv pip install --no-cache --python /opt/ray-driver/bin/python \
        -r /ray-driver-requirements.txt \
 && chown -R airflow /opt/ray-driver /opt/uv-python
USER airflow
