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
![PyPi](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Pydantic](https://img.shields.io/badge/pydantic-%23E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)
![Polars](https://img.shields.io/badge/polars-0075ff?style=for-the-badge&logo=polars&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Qdrant](https://img.shields.io/badge/qdrant-%23dc2626.svg?style=for-the-badge&logo=qdrant&logoColor=white)
![DuckDuckGo](https://img.shields.io/badge/duckduckgo-de5833?style=for-the-badge&logo=duckduckgo&logoColor=white)

[`Ultimate Macroeconomics Dashboard`](https://github.com/alexveider1/Ultimate-Macroeconomics-Dashboard) is an AI-powered macroeconomic analytics tool: a Streamlit dashboard backed by Postgres + Qdrant, plus FastAPI services for an AI analyst, forecasting, clustering, on-demand data ingestion, and a Python sandbox. **70+** World Bank indicators, **30 000+** news articles, **50+** Yahoo Finance tickers, **150+** prebuilt charts.

The full stack consists of 10 `Docker` containers — `db`, `vector_db`, `downloader_general`, `app`, `agent`, `forecaster`, `clustering`, `downloader_extra`, `python_sandbox`, and `bff` (plus the optional `backup` service described below). The project follows a strict micro-service design, with each container responsible for one capability:

* `db` — relational database (`PostgreSQL`) for tabular data from `World Bank Data API` and `Yahoo Finance`.
* `vector_db` — vector database (`Qdrant`) for news article embeddings sourced from the `Webz.io` open dataset.
* `downloader_general` — one-shot job that fetches the initial dataset (World Bank, Yahoo Finance, news) into the databases.
* `app` — the `Streamlit` dashboard itself (this is what the user opens in the browser).
* `agent` — `FastAPI` backend hosting the multi-agent AI analyst (LangGraph supervisor + specialised workers).
* `forecaster` — `FastAPI` micro-service for time-series forecasting (`pmdarima`, `prophet`, `chronos`).
* `clustering` — `FastAPI` micro-service for unsupervised clustering (KMeans, DBSCAN).
* `downloader_extra` — `FastAPI` micro-service that downloads additional World Bank indicators on demand.
* `python_sandbox` — `FastAPI` sandbox that executes LLM-generated code in an isolated environment.
* `bff` — read-only `FastAPI` backend-for-frontend: typed JSON reads of the macro data, Qdrant news search, and proxies to the forecaster / clustering / agent services (for a future JS frontend; the Streamlit app does not use it yet).

## Quick start

Prerequisites: Docker (with the Compose plugin). An NVIDIA GPU is optional, though it significantly improves the forecaster's performance.

```bash
# 1. Clone repo
git clone https://github.com/alexveider1/Ultimate-Macroeconomics-Dashboard
cd Ultimate-Macroeconomics-Dashboard/

# 2. Create the `.env` file (fill in your secrets)
cp _container_data/.env.example _container_data/.env
$EDITOR _container_data/.env

# 3. Set the shared.openai_* keys in `_container_data/config.yaml`
$EDITOR _container_data/config.yaml

# 4. If no CUDA-capable GPU is available, remove the `deploy` block under `forecaster:` in docker-compose.yaml
$EDITOR docker-compose.yaml

# 5. Build and run
docker compose up --build
```

On first boot, the stack downloads the datasets and inserts them into both databases (relational and vector). How long this takes depends heavily on your network speed, but it usually takes around 30 minutes. The dashboard is not available while the data is downloading; once the download completes, it becomes available at <http://localhost:8501>.

### Required `.env` variables

| Variable                   | Purpose                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `POSTGRES_USER`            | Postgres superuser created natively by the `postgres:18` image on first boot. |
| `POSTGRES_PASSWORD`        | Password for the superuser.                                            |
| `POSTGRES_DB`              | Default database created on first boot (typically `postgres`).         |
| `POSTGRES_LLM_USER`        | Read-only role used by the AI analyst and the dashboard's bulk reads to query the database. |
| `POSTGRES_LLM_PASSWORD`    | Password for the read-only role (rotatable; takes effect on next boot). |
| `QDRANT__SERVICE__API_KEY` | Bearer token protecting the Qdrant HTTP API.                           |
| `OPENAI_API_KEY`           | API key for the LLM/embedding provider in `config.yaml`.               |

> Never commit `_container_data/.env`

### Required `config.yaml` keys

Set these under `shared:` to point at your LLM provider (any OpenAI-compatible API works):

```yaml
shared:
  openai_base_url: https://api.openai.com/v1
  openai_llm_model: gpt-5.4
  openai_embedding_model: openai/text-embedding-3-small
```

Everything else has working defaults. See [`_container_data/config.yaml`](_container_data/config.yaml) for the full schema.

## LLM requirements

The agent needs a model with reasoning, tool/function calling, vision, and ≥256k context. Any recent flagship from OpenAI, Google, Anthropic, Qwen, or DeepSeek works. Local models served via [vLLM](https://github.com/vllm-project/vllm) on a powerful GPU also work.

## Illustrations

|                               |                                  |
| ----------------------------- | -------------------------------- |
| ![](app/assets/structure.png) | ![](app/assets/ai_structure.png) |
| ![](app/assets/1.png)         | ![](app/assets/2.png)            |
| ![](app/assets/3.png)         | ![](app/assets/4.png)            |
| ![](app/assets/5.png)         | ![](app/assets/6.png)            |
| ![](app/assets/7.png)         | ![](app/assets/8.png)            |
| ![](app/assets/9.png)         | ![](app/assets/10.png)            |
| ![](app/assets/11.png)         | ![](app/assets/12.png)            |
| ![](app/assets/13.png)         | ![](app/assets/14.png)            |

## Custom theming

The active colour palette is controlled by `_container_data/themes.yaml`. The bundled themes (`dark`, `dark-blue`, `light-green`) drive both the Plotly chart template and the Streamlit page colours. To change the palette, set the `active` key:

```yaml
active: dark-blue
themes:
  dark:
    ...
  dark-blue:
    ...
  light-green:
    ...
```

Streamlit also reads its theme from `app/.streamlit/config.toml`. If you want to customise the dashboard chrome directly, edit it there:

```toml
[theme]
primaryColor = "#10c8f1"
backgroundColor = "#000000"
secondaryBackgroundColor = "#073642"
textColor = "#ffffff"
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

`downloader_general` will pick the new entries up on the next clean boot. Already-running stacks can fetch new indicators on demand via the AI analyst (which delegates to `downloader_extra`).

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

## Disclaimer

All data is sourced from third-party providers and presented as-is. The author makes no representations about its accuracy or completeness.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
