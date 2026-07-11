# Ultimate Macroeconomics Dashboard

Technical stack (not exhaustive):

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Google Gemini](https://img.shields.io/badge/google%20gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white)
![GitHub Copilot](https://img.shields.io/badge/github_copilot-8957E5?style=for-the-badge&logo=github-copilot&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Pydantic](https://img.shields.io/badge/pydantic-%23E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)
![Polars](https://img.shields.io/badge/polars-0075ff?style=for-the-badge&logo=polars&logoColor=white)
![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)
![NVIDIA Triton](https://img.shields.io/badge/nvidia%20triton-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Qdrant](https://img.shields.io/badge/qdrant-%23dc2626.svg?style=for-the-badge&logo=qdrant&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![DuckDuckGo](https://img.shields.io/badge/duckduckgo-de5833?style=for-the-badge&logo=duckduckgo&logoColor=white)

[`Ultimate Macroeconomics Dashboard`](https://github.com/alexveider1/Ultimate-Macroeconomics-Dashboard) is an AI-powered macroeconomic analytics tool: a **multi-page Streamlit dashboard** talking directly to Postgres + Qdrant and to FastAPI micro-services for the AI analyst, forecasting, clustering, on-demand data ingestion, document conversion, and a sandboxed Python executor, plus an NVIDIA Triton Inference Server hosting all model inference. It covers World Bank, Yahoo Finance, Binance crypto, FRED US-state and Eurostat EU-regional (NUTS-2) data, plus a **30 000+** article news RAG corpus — **70+** World Bank indicators, **50+** Yahoo Finance tickers, and **150+** prebuilt charts.

## Architecture

The stack is a set of `Docker` containers following a strict micro-service design, each container responsible for one capability. The dashboard is at **`http://localhost:8501`**.

**Application services**

* `app` — the multi-page Streamlit dashboard (this is what the user opens in the browser). It talks **directly** to the backend: the read-only Postgres role for page reads, Qdrant for the news/RAG pages, and the agent / forecaster / clustering / docling HTTP APIs. The AI chat supports **multimodal input** — attach text/image/audio/document files or record a voice message and it normalizes them (transcribes audio via Whisper, converts documents via `docling`, forwards images to the vision model) before streaming to the agent.
* `agent` — `FastAPI` backend hosting the multi-agent AI analyst (LangGraph supervisor + specialised workers).
* `forecaster` — `FastAPI` adapter that forwards time-series forecasting requests (ARIMA family, Prophet, Chronos, moving-average, XGBoost) to `triton` over gRPC.
* `clustering` — `FastAPI` adapter that forwards unsupervised clustering requests (KMeans, DBSCAN, …) to `triton` over gRPC.
* `downloader_extra` — `FastAPI` micro-service that ingests additional data on demand from five sources (World Bank indicator, Yahoo ticker, Binance pair, FRED state indicator, Eurostat NUTS-2 dataset), called by the agent.
* `python_sandbox` — `FastAPI` sandbox that executes LLM-generated Plotly/Polars code in an isolated environment.
* `docling` — `FastAPI` micro-service that converts uploaded documents (`.pdf` / `.docx` / `.pptx` / `.xlsx`) to Markdown for the chat's file input. PDFs use docling's VLM pipeline, offloading OCR inference to a **cloud OpenAI-compatible endpoint** (configured under `docling.vlm` in `config.yaml`); Office formats parse locally.
* `triton` — NVIDIA Triton Inference Server: hosts every forecasting + clustering model (python backend, CUDA/cuML where supported). No vLLM/VLM. Internal-only ports.

**Data services**

* `db` — relational database (`PostgreSQL 18`) for tabular data: World Bank, Yahoo Finance, Binance crypto, FRED US-state, and Eurostat EU-NUTS2.
* `vector_db` — vector database (`Qdrant`) for news + curated-news + World Bank document embeddings.
* `downloader_general` — fetches the initial dataset into both databases, then stays running as an incremental update scheduler that keeps each source fresh.

**Observability & operations**

* Langfuse (`langfuse_web` + `langfuse_worker` + backing ClickHouse / Redis / MinIO) — self-hosted LLM tracing for the AI analyst; UI at `http://localhost:3000`.
* Grafana + Prometheus + OpenTelemetry (`grafana`, `prometheus`, `otel-collector`, `blackbox_exporter`) — external container/service resource + health monitoring; Grafana at `http://localhost:3001`, Prometheus at `http://localhost:9092`.
* `backup` — optional scheduled cloud backups of Postgres + Qdrant via `rclone` (off by default).

## Quick start

Prerequisites: Docker (with the Compose plugin) and a working **NVIDIA GPU + the NVIDIA Container Toolkit** — the `triton` service reserves the GPU for all model inference.

```bash
# 1. Clone repo
git clone https://github.com/alexveider1/Ultimate-Macroeconomics-Dashboard
cd Ultimate-Macroeconomics-Dashboard/

# 2. Create the `.env` file (fill in your secrets)
cp .env.example .env
$EDITOR .env

# 3. Set the shared.openai_* keys in `_container_data/config.yaml`
$EDITOR _container_data/config.yaml

# 4. Build and run
docker compose up --build
```

On first boot, the stack downloads the datasets and inserts them into both databases (relational and vector). How long this takes depends heavily on your network speed, but it usually takes 1–2 hours. The dashboard is not available while the data is downloading; once the download completes, it becomes available at <http://localhost:8501>.

### Required `.env` variables

| Variable                   | Purpose                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `POSTGRES_USER`            | Postgres superuser created natively by the `postgres:18` image on first boot. |
| `POSTGRES_PASSWORD`        | Password for the superuser.                                            |
| `POSTGRES_DB`              | Default database created on first boot.                                |
| `POSTGRES_LLM_USER`        | Read-only role used by the AI analyst and the Streamlit dashboard to query the database. |
| `POSTGRES_LLM_PASSWORD`    | Password for the read-only role (rotatable; takes effect on next boot). |
| `QDRANT__SERVICE__API_KEY` | Bearer token protecting the Qdrant HTTP API.                           |
| `OPENAI_API_KEY`           | API key for the LLM provider in `config.yaml`; also used for embeddings and Whisper audio transcription. |
| `FRED_API_KEY`             | API key for FRED (GeoFRED) US-state indicator ingestion.               |
| `LANGFUSE_*`               | Langfuse project keys (`PUBLIC`/`SECRET`), first-boot login (`INIT_USER_*`), and self-host infra secrets (`ENCRYPTION_KEY`, `SALT`, `NEXTAUTH_SECRET`, and the backing-store passwords). |
| `GRAFANA_ADMIN_PASSWORD`   | Password for the Grafana `admin` login.                                |

> Never commit `.env`

### Required `config.yaml` keys

Set these under `shared:` to point at your LLM provider (any OpenAI-compatible API works):

```yaml
shared:
  openai_base_url: https://api.openai.com/v1
  openai_llm_model: gpt-5.4
  openai_llm_model_fast: gpt-5.4-mini
  openai_embedding_model: openai/text-embedding-3-small
```

The chat's multimodal inputs are configured the same way: voice/audio transcription under `whisper:` (defaults to the same OpenAI-compatible endpoint) and document OCR under `docling.vlm:` (`base_url` + `model` of a cloud OpenAI-compatible vision endpoint) — both authenticate with the shared `OPENAI_API_KEY`. Everything else has working defaults. See [`_container_data/config.yaml`](_container_data/config.yaml) for the full schema.

## LLM requirements

The agent needs a model with reasoning, tool/function calling, vision, and ≥256k context. Any recent flagship from OpenAI, Google, Anthropic, Qwen, or DeepSeek works. Local models served via [vLLM](https://github.com/vllm-project/vllm) on a powerful GPU also work.

## Custom theming

The active colour palette is controlled by `_container_data/themes.yaml`, read directly by the Streamlit `app`. Each theme defines a token tree that the app exposes as colour tokens across the pages and registers as a Plotly template. To change the palette, set the `active` key (or add a theme that covers every token — no rebuild needed):

```yaml
active: dark
themes:
  dark:
    ...
```

## Adding extra indicators

To add more World Bank indicators to the dashboard, append them to `_container_data/_configs/world_bank_download_config.json`. Each top-level key is one dashboard page:

```json
{
    "General Economics Indicators": [
        {
            "name": "GDP",
            "id": "NY.GDP.MKTP.CD",
            "db": 2
        },
        {
            "name": "GDP_PPP",
            "id": "NY.GDP.MKTP.PP.CD",
            "db": 2
        },
        ...
    ],
    ...
}
```

`downloader_general` will pick the new entries up on the next clean boot. Already-running stacks can fetch new data on demand via the AI analyst (which delegates to `downloader_extra`) — this covers World Bank indicators, Yahoo tickers, Binance pairs, FRED state indicators, and Eurostat NUTS-2 datasets.

## Cloud backups

The optional `backup` service periodically dumps Postgres and snapshots Qdrant, then uploads both to any [rclone](https://rclone.org/)-supported cloud (S3, Backblaze B2, Google Drive, …). It is **off by default** and turned on entirely through `config.yaml`.

**Enable it:**

1. Create the rclone remote. Copy the example and edit it, or generate one interactively:

   ```bash
   cp _container_data/backup/rclone.conf.example _container_data/backup/rclone.conf
   # or, interactively (writes the same file):
   rclone config --config _container_data/backup/rclone.conf
   ```

   `rclone.conf` holds cloud credentials and is gitignored — never commit it.

2. Edit the `backup:` block in `_container_data/config.yaml`:

   ```yaml
   backup:
     enabled: true                # master on/off switch
     interval_minutes: 60         # how often to back up
     run_on_start: true           # also back up immediately on container start
     rclone_remote: "s3remote"    # must match a [remote] name in rclone.conf
     rclone_path: macro-backups   # destination dir/bucket under that remote
     retention_days: 7            # prune remote backups older than N days (negative = keep forever)
   ```

3. Start (or restart) the service:

   ```bash
   docker compose up -d --build backup
   docker compose logs -f backup
   ```

Each run writes `postgres/<db>_<timestamp>.dump` (a `pg_dump` custom-format archive) and `qdrant/<name>.snapshot` (a full-storage Qdrant snapshot) under the remote path, then prunes anything older than `retention_days`. To **turn backups off**, set `enabled: false` and restart — the container stays up but idles, taking no backups.

**Restore** (run inside the service so it has `rclone` + `pg_restore` + the mounted config):

```bash
# list what's on the remote
docker compose run --rm backup python restore.py --list

# restore a Postgres dump (overwrites the target DB)
docker compose run --rm backup python restore.py --postgres macro_2026-07-04T12-00-00Z.dump

# download a Qdrant snapshot and print its (restart-based) recovery steps
docker compose run --rm backup python restore.py --qdrant full-snapshot-....snapshot
```

Postgres restore is automated (`pg_restore --clean --if-exists`); a full Qdrant snapshot is recovered into storage at Qdrant startup, so `restore.py` downloads it and prints the exact steps.

## Monitoring

Container and service health is monitored **externally** by a fully open-source **Grafana + Prometheus + OpenTelemetry** stack — deliberately separate from the dashboard so it keeps reporting even when the app is down. Every container is treated as an external service: both its resources and its health are tracked from the outside, with no per-service instrumentation. Open **Grafana at `http://localhost:3001`** (log in as `admin` with `GRAFANA_ADMIN_PASSWORD` from your `.env`). It gives you, out of the box:

- **Per-container CPU / RAM / disk / network** for every container in the stack, plus host-level metrics — collected by an **OpenTelemetry Collector** (`docker_stats` + `hostmetrics` receivers reading the Docker socket + host `/proc`/`/sys`), with history, not just a live snapshot.
- **Per-service health checks** — a **blackbox exporter** runs HTTP probes against each service's health endpoint (`agent`, `forecaster`, `clustering`, `downloader_extra`, `python_sandbox`, `docling`, `app`, `triton`, `vector_db`, `langfuse_web`) and TCP probes for the databases (Postgres + the Langfuse backing stores).
- Three provisioned dashboards — **Containers**, **Host**, and **Service health** — backed by **Prometheus** (`http://localhost:9092`), which also scrapes Triton's native inference/GPU metrics.

To add or change a health probe, edit the target list in `_container_data/prometheus/prometheus.yml` (the `blackbox-http` / `blackbox-tcp` jobs) and restart the `prometheus` service. The stack runs fully independently of the app services (no `depends_on` either way), so a monitor outage can never affect the stack. Set `GRAFANA_ADMIN_PASSWORD` in `.env` before first boot (see `.env.example`).

## Disclaimer

All data is sourced from third-party providers and presented as-is. The author makes no representations about its accuracy or completeness.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
