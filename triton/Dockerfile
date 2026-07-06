# NVIDIA Triton Inference Server. The `-py3` image ships the python backend
# (Python 3.12, CUDA 12.x) that every model in model_repository/ uses. GPU is
# required at runtime (Chronos/XGBoost on CUDA, cuML for clustering).
FROM nvcr.io/nvidia/tritonserver:25.02-py3

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

# Shared inference code, importable by every model.py through PYTHONPATH.
COPY common/ /opt/umd/

# The python-backend model repository (one dir per algorithm).
COPY model_repository/ /models/

EXPOSE 8000 8001 8002

# HTTP :8000, gRPC :8001, metrics :8002 (internal to the compose network).
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=0"]
