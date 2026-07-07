# NVIDIA Triton Inference Server. The `-vllm-python-py3` image ships BOTH the
# python backend (Python 3.12, CUDA 13.x — every forecasting/clustering model)
# AND the vLLM backend + the OpenAI-compatible frontend used to serve the
# docling VLM (granite-docling) over `/v1/chat/completions`. GPU is required at
# runtime. 25.10 bundles vLLM 0.10.2, whose newer multimodal processor can run
# granite-docling (25.02's vLLM 0.7.0 could not); the tied-weights bug is worked
# around by serving the model's `untied` revision (see granite_docling/model.json).
FROM nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3

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

# Extra Python deps required by Triton's OpenAI-compatible frontend (fastapi /
# uvicorn / sse-starlette / openai + the HF tokenizer loader). Install the
# frontend's own requirements file if the image ships one.
RUN if [ -f /opt/tritonserver/python/openai/requirements.txt ]; then \
        pip install -r /opt/tritonserver/python/openai/requirements.txt; \
    fi

# Shared inference code, importable by every model.py through PYTHONPATH.
COPY common/ /opt/umd/

# The model repository: python-backend forecasting/clustering models + the
# `granite_docling` vLLM-backend model for docling PDF conversion.
COPY model_repository/ /models/

# HTTP :8000, gRPC :8001, metrics :8002 (KServe) + OpenAI :9000 — all internal.
EXPOSE 8000 8001 8002 9000

# Launch the OpenAI-compatible frontend (serves granite_docling on its default
# port :9000) while `--enable-kserve-frontends` keeps the standard HTTP/gRPC/
# metrics endpoints up so forecaster/clustering gRPC + the `/v2/health/ready`
# probe still work.
CMD ["python3", "/opt/tritonserver/python/openai/openai_frontend/main.py", \
     "--model-repository", "/models", \
     "--tokenizer", "ibm-granite/granite-docling-258M", \
     "--enable-kserve-frontends"]
