# NVIDIA Triton Inference Server. The `-py3` image ships the python backend
# (Python 3.12, CUDA 13.x) that hosts every forecasting/clustering model. GPU is
# required at runtime (Chronos + XGBoost on CUDA, clustering on cuML). No vLLM /
# OpenAI frontend — document OCR is offloaded to a cloud endpoint by the docling
# service, so this image no longer serves any LLM/VLM.
FROM nvcr.io/nvidia/tritonserver:25.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/umd \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

# Build toolchain for pmdarima (cython) + prophet (cmdstan) source builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Inference deps installed into the container's Python (the python-backend
# stub interpreter). RAPIDS wheels come from the NVIDIA index in requirements.
# `--ignore-installed pip`: the base image's pip 24.0 is a Debian apt package
# (no RECORD file), so a plain `--upgrade` fails to uninstall it — install the
# new pip alongside it into /usr/local, which wins on PATH.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --ignore-installed pip \
    && pip install -r /tmp/requirements.txt

# The base image ships numpy 1.26.4 without a RECORD/METADATA file (apt-style
# install), so pip cannot uninstall it when RAPIDS upgrades numpy — the stale,
# metadata-less `numpy-1.26.4.dist-info` is left behind and shadows the real
# one, making `importlib.metadata.version("numpy")` return None. That breaks
# transformers' import-time version guard (`numpy>=1.17`, need=1.17 found=None),
# which crashes the Chronos model's `initialize()` and takes the whole server
# down (Triton exits when any model fails to load). Drop any numpy dist-info
# that lacks a METADATA file so a single, resolvable version remains.
RUN for d in /usr/local/lib/python3.12/dist-packages/numpy-*.dist-info; do \
        [ -f "$d/METADATA" ] || rm -rf "$d"; \
    done

# Shared inference code, importable by every model.py through PYTHONPATH.
COPY common/ /opt/umd/

# The model repository: python-backend forecasting/clustering models.
COPY model_repository/ /models/

# HTTP :8000, gRPC :8001, metrics :8002 (KServe) — all internal.
EXPOSE 8000 8001 8002

# Standard Triton launch — serves HTTP/gRPC/metrics for the forecaster/
# clustering adapters and the /v2/health/ready probe.
CMD ["tritonserver", "--model-repository=/models"]
