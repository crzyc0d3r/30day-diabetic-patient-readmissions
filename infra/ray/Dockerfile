# The `-gpu` tag is a multi-arch manifest (linux/amd64 + linux/arm64), so Docker
# auto-selects the right variant for the host: amd64 on Intel/AMD boxes, arm64 on
# the GB10 dev box this project was first built on. Override with --build-arg
# RAY_IMAGE=rayproject/ray:2.55.1-py313-cpu for a CPU-only build.
# Digest-pinned (@sha256), not just tag-pinned: the `2.55.1-py313-gpu` tag locks
# the Ray release but NOT the CPython patch baked inside (3.13.2). That floating
# patch is the root cause of the "Python patch version mismatch" warning between
# the Ray client (driver) and the cluster: a re-push of the tag with a newer
# patch would silently move the cluster's interpreter. The digest freezes it.
# The digest points at the manifest list, so multi-arch auto-selection (above) is
# preserved. When overriding RAY_IMAGE to the -cpu tag, drop the digest (a stale
# digest would win over the tag). Refresh after an intentional Ray bump with:
#   docker buildx imagetools inspect rayproject/ray:2.55.1-py313-gpu --format '{{.Manifest.Digest}}'
ARG RAY_IMAGE=rayproject/ray:2.55.1-py313-gpu@sha256:644794b391a69fbbac91246d7b6d735c3aaad2e382d2b0ea842bce6960ea174e
FROM ${RAY_IMAGE}

USER root
RUN pip install --no-cache-dir py-spy \
 && chown root:root /home/ray/anaconda3/bin/py-spy \
 && chmod u+s /home/ray/anaconda3/bin/py-spy \
 && mkdir -p /tmp/ray \
 && chown -R ray:users /tmp/ray
USER ray

RUN pip install --no-cache-dir \
        mlflow==3.11.1 \
        scikit-learn==1.8.0 \
        pandas==2.3.3 \
        numpy==2.4.4 \
        xgboost==3.2.0 \
        catboost==1.2.10 \
        joblib==1.5.3 \
        ipython==9.7.0

# torch matches the host medi-watch env so the MLP path in HPO trials runs
# the same kernel on workers as on the driver. ARM+CUDA-13 wheels live on
# the dedicated PyTorch index, not stock PyPI, hence its own RUN.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130

EXPOSE 6379 8265 10001
