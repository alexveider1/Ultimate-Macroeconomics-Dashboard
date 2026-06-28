# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`Ultimate Macroeconomics Dashboard` is a 9-container Docker stack: a Streamlit multi-page dashboard backed by Postgres + Qdrant, with FastAPI micro-services for the AI analyst, forecasting, clustering, on-demand data ingestion, and a sandboxed Python executor. Read `README.md` for the full description; the sections below cover only what isn't obvious from the code.

## Running the stack

Linting/formatting is done with `ruff` and type-checking with `ty` (both Astral); each service ships a `[dependency-groups] dev` block with both plus `pytest` and `pytest-cov`. Tests live under `<service>/tests/` and run via `uv run pytest` — `[tool.pytest.ini_options].addopts` enables coverage by default (`--cov --cov-report=term-missing`), with per-service `[tool.coverage.run].source` pointing at that service's package(s). Everything runs inside containers via Docker Compose. Every service is Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for dependency management — each service has its own `pyproject.toml` + `uv.lock`, and the Dockerfile runs `uv sync --frozen` into `/opt/venv`.

```bash
# Full stack (build + run, foreground)
docker compose up --build

# Single service rebuild
docker compose build agent
docker compose up -d agent

# Logs
docker compose logs -f app
docker compose logs -f agent
```

For local iteration without rebuilding the image, work in any service directory:

```bash
cd app           # or agent, forecaster, etc.
uv sync          # creates .venv from pyproject.toml + uv.lock
uv run python -m streamlit run app.py   # or uvicorn main:app for FastAPI services
uv add <package>      # add a dependency (updates pyproject.toml + uv.lock)
uv lock --upgrade     # refresh the lockfile
```

First boot requires `_container_data/.env` (copy from `_container_data/.env.example`) and a populated LLM section in `_container_data/config.yaml`. The `postgres:18` image creates the superuser (`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`) natively on first volume init; `downloader_general` then upserts the read-only LLM role via `src/utils/db_bootstrap.py` (also grants `SELECT` on `public` plus default privileges, so future tables are readable automatically) and runs the ingestion (~1–2h) for World Bank + Yahoo Finance + Webz.io news. The dashboard is at `http://localhost:8501`.

The bootstrap step runs on **every** `downloader_general` container start (cheap, idempotent), so rotating `POSTGRES_LLM_PASSWORD` or adding new tables in a future release takes effect on the next `docker compose up -d downloader_general` without wiping volumes. Only the downloads themselves are one-shot, gated by `_container_data/downloader_general/.download_completed`.

If the host has no NVIDIA GPU, remove the `deploy:` block from the `forecaster` service in `docker-compose.yaml` (the `chronos` model will be skipped; every other model — `auto_arima`, manual `arima`, `sarima`, `prophet`, `moving_average`, `xgboost` — runs CPU-only).

## Architecture

### Service map and ports

| Service              | Port | Role                                                                                  |
| -------------------- | ---- | ------------------------------------------------------------------------------------- |
| `db`                 | 5432 | Postgres 18 — World Bank + Yahoo Finance + Binance crypto tabular data                |
| `vector_db`          | 6333 | Qdrant — news article embeddings                                                      |
| `downloader_general` | —    | One-shot: bootstraps LLM role, ingests WB + Yahoo + Binance + news, populates both DBs |
| `app`                | 8501 | Streamlit dashboard (entry point: `app/app.py`)                                       |
| `agent`              | 8000 | FastAPI — LangGraph multi-agent AI analyst                                            |
| `forecaster`         | 8001 | FastAPI — forecasting (ARIMA family, Prophet, Chronos, MA, XGBoost)                   |
| `clustering`         | 8002 | FastAPI — KMeans / DBSCAN                                                             |
| `downloader_extra`   | 8003 | FastAPI — on-demand World Bank indicator ingestion (called by the agent)              |
| `python_sandbox`     | 8004 | FastAPI — isolated executor for LLM-generated Plotly/Polars code                      |

Inside the Compose network, services address each other by container name and the port from `config.yaml` (e.g. `http://agent:8000`, `http://forecaster:8001`). The `app` resolves these via `app/core/api_client.py`, which also honours `*_BASE_URL` env vars as overrides.

### Configuration

`_container_data/config.yaml` is the **single source of truth** for ports, hostnames, LLM/embedding settings, forecaster toggles, etc. It is bind-mounted read-only into every service.

Important: `docker-compose.yaml` duplicates the ports and bind-mount paths declared in `config.yaml`. Changing a port or path in one file requires changing it in the other. The two files are not auto-synced.

Each service parses the slice of `config.yaml` it needs through a small Pydantic model in its own `config.py` (`load_config(path)` → typed object; access is attribute-based — `CONFIG.postgres.host`, not `CONFIG.get("postgres", {}).get("host")`). Secrets are likewise typed per service via `pydantic-settings` `BaseSettings` in a `settings.py`, and each service declares and receives **only the secrets it actually uses** (least privilege): `clustering` / `forecaster` / `python_sandbox` get none, `agent` gets the read-only LLM role but not the superuser password, `app` gets both Postgres roles + the Qdrant key but no `OPENAI_API_KEY`. `docker-compose.yaml` enforces this at the container boundary — the old blanket `env_file: ./_container_data/.env` is replaced by a per-service `environment:` block that injects only that service's variables via `${VAR}` interpolation. A gitignored root-level `.env` symlink → `_container_data/.env` lets `docker compose up` resolve those `${VAR}`s with the documented command unchanged (alternatively: `docker compose --env-file _container_data/.env up`). There is no shared config/secrets package — each service is its own container with its own `pyproject.toml`, so the tiny models are duplicated per service by design.

Other config files in `_container_data/`:

- `.env` — secrets (Postgres creds, Qdrant API key, `OPENAI_API_KEY`). Never commit; gitignored. No longer blanket-mounted into every container — each service receives only the secrets it uses (see the per-service scoping note above), and a gitignored root `.env` symlink points here so Compose `${VAR}` interpolation resolves.
- `database_schema.yaml` — column-level documentation of Postgres tables; mounted into `agent` so the SQL worker can ground its queries.
- `_configs/world_bank_download_config.json` — list of WB indicators grouped by dashboard page. Append here to add indicators on next clean boot; or add at runtime via the AI analyst (it calls `downloader_extra`).
- `_configs/news_download_config.json` — news topics for the RAG corpus.
- `_configs/yahoo_download_config.json` — Yahoo Finance tickers.
- `_configs/binance_download_config.json` — Binance crypto ingestion tunables (`base_url`, `quote_asset`, `top_n`, `kline_interval`, `max_parallel_symbols`, `exclude_base_assets`). No curated coin list — the top-N coins are chosen at runtime by 24h quote volume.
- `themes.yaml` — colour palettes. `active:` key selects one; bundled themes are `dark`, `dark-blue`, `light-green`. Drives both the registered Plotly template (`"app"`) and Streamlit chrome. **Deploy-time only** — the runtime theme picker was removed in v0.6. Adding a custom theme means covering every semantic token used in code (`positive`, `negative`, `reference_line`, `map_coastline`, `sector_*`, `diverging_*`, `sequential_*`, `card_title_color`, `confidence_band_alpha`, `selected_marker`, `wordcloud_background`, `wordcloud_colormap`) — `get_color` raises `KeyError` on a missing token, no silent fallback.
- `app/.streamlit/config.toml` — Streamlit's own theme/server config. Mirror of `themes.yaml` for the chrome side; edit `server.address = "0.0.0.0"` to expose the local dev build on the LAN.

### `app` (Streamlit)

Entry point `app/app.py` registers the Plotly template, sets up `st.session_state` (chat history, per-service health flags), declares the multi-page navigation, and shows a one-time data disclaimer dialog. Pages live under `app/pages/`, numbered `01_…` through `18_…` for ordering (the numbering also encodes the v0.3 navigation renormalisation). The indicator pages (`01`–`10`) are **config-driven** — they call `render_page_from_config` (`app/pages/page_utils.py`) with section keys from `world_bank_download_config.json` rather than hand-rolling charts; `14_yahoo_finance.py`, `15_news.py` and `16_crypto.py` are the bespoke "Other data" pages; `17_token_usage.py` and `18_monitoring.py` are the two ops pages. `16_crypto.py` mirrors the Yahoo page (market overview table, top-coin log-scale trend, BTC candlestick, all-coin return-correlation heatmap) and reads the `binance_*` tables via `get_all_binance_historical_prices` / `get_all_binance_metadata`; its candlestick + heatmap come from the **generic** `build_candlestick_plot` / `build_correlation_heatmap` in `core/plotting.py` (the Yahoo page keeps its own `build_yahoo_*` variants). Shared infrastructure is in `app/core/`:

- `api_client.py` — typed wrappers around every backend HTTP endpoint (forecaster, agent SSE stream, clustering, plot interpretation, downloader_extra). Always use these wrappers rather than `requests.post` directly — they handle the base-URL resolution and request logging.
- `postgres_client.py` / `qdrant_client.py` — connection helpers with retries (hardened in v0.5).
- `plotting.py` — Plotly helpers; pages call `get_color` / `get_colorway` rather than hard-coding hex values, so palette swaps work via `themes.yaml`.
- `theming.py` — registers the `"app"` Plotly template from the active theme.
- `token_usage.py` — in-memory aggregator shown on the **Token Usage page** (`17_token_usage.py`); cleared on session end. `token_usage_store.py` is the separate Postgres-backed persistence layer (writes via the superuser role).
- `monitoring.py` — service-health probing + container stats behind the **Monitoring page** (`18_monitoring.py`). It hits each service's `/health` (`/_stcore/health` for Streamlit, a TCP/SQL probe for bare Postgres) and reads per-container CPU/memory/network from the Docker Engine API over a **read-only `/var/run/docker.sock`** bind-mounted into `app`. That mount is required for the Monitoring page and is declared only in `docker-compose.yaml`, not `config.yaml`.
- `app_logging.py` — centralised page-render and HTTP-request logging.
- `page_helpers.py` / `pages/page_utils.py` — shared dashboard-page rendering helpers (config-driven indicator renderer + common indicator-slice cleaning) extracted from per-page duplication; new indicator pages should reuse these rather than re-implementing the slice/clean/plot flow.

### `agent` (LangGraph supervisor)

`agent/agent/graph.py` is the heart of the AI analyst. The flow is:

1. **`GuardrailAgent`** — heuristic-first screen. Three regexes (auto-allow for short greetings + in-scope keywords, auto-block for clear red flags) decide most messages without an LLM call; only ambiguous ones escalate to the structured-output LLM.
2. **`MacroSupervisorAgent`** — plans, picks the next worker, and decides FINISH. Branches off `last_worker_status` (a `Literal["SUCCESS","EMPTY","ERROR","NEEDS_DOWNLOAD","BLOCKED","UNKNOWN"]` returned by every worker) rather than regex-matching prose. Static preamble + macro context block stay constant across turns so the prefix is provider-cacheable.
3. **Workers** (one of `WORKER_NAMES` in `graph.py`) — every worker also receives the last ~3 chat turns so follow-ups disambiguate:
   - `sql_agent` — up-to-5-step exploration. Defaults to WDI (`db_id = 2`) and skips the database-lookup step for typical macro queries; carries 3 worked few-shot examples.
   - `plotly_agent` — generates Plotly code, runs it in `python_sandbox`, returns the figure as an artifact.
   - `table_agent` — Polars transformations on prior worker output.
   - `rag_agent` — Qdrant semantic search over the news corpus.
   - `web_search` — DuckDuckGo fallback.
   - `downloader_agent` — calls `downloader_extra` to ingest WB indicators on demand. Triggered when `sql_agent` returns `last_worker_status = NEEDS_DOWNLOAD`.
   - `chat_agent` — conversational synthesis / general-knowledge answers.

The graph builds **two `ChatOpenAI` instances** (`MacroAgentGraph.__init__`): a strong model (`shared.openai_llm_model`) for the reasoning-heavy roles — `supervisor` (planning + the final answer), `sql_agent`, `plotly_agent`, `chat_agent` — and a fast model (`shared.openai_llm_model_fast`) for the cheap `GuardrailAgent` screen and the lightweight `table_agent` / `rag_agent` / `web_search` / `downloader_agent`. Both share the base URL + API key and differ only by model name; when `openai_llm_model_fast` is unset it falls back to the strong model (the previous single-model behaviour). Vision (`/plots/interpret`) stays on the strong model.

When the supervisor picks FINISH, it writes the **complete polished markdown answer** into `isolated_worker_task` and that draft is streamed to the user verbatim in ~24-char chunks (`MacroAgentGraph._stream_supervisor_draft`). There is no second synthesis LLM call — the supervisor's draft is the answer, with only a small line-level leak filter (`_sanitize_draft`) stripping any line that accidentally contains worker names / sandbox / traceback tokens.

Streaming protocol is SSE on `POST /chat/stream` with `step` / `token` / `final` / `error` events; the `final` event carries the answer plus an `artifacts` dict and a `usage` block. `POST /plots/interpret` is a separate vision endpoint that reads a base64 Plotly screenshot with two modes (`no_hallucinations` strict description vs. analyst interpretation).

Per-LLM-call token accounting is attached via `UsageTracker` (`agent/agent/usage.py`) on every LangChain LLM in the graph (guardrail when it escalates, supervisor, each worker) — it's a graph-level callback, so it aggregates across both the strong and fast models automatically and labels each call with the model that served it. Worker output schemas live in `agent/agent/schemas.py`; prompt text lives in `agent/agent/prompts.py` and follows a stable-prefix / dynamic-tail layout so provider-side automatic prefix caching can match across requests.

External backend calls (sandbox, downloader_extra) use one shared `httpx.AsyncClient` from `agent.tools._get_httpx_client()` (closed in the FastAPI shutdown hook); the rendered database-schema text is `functools.lru_cache`d so it isn't re-serialised on every SQL step.

### `forecaster`

`forecaster/main.py` exposes a single `POST /predict` endpoint backed by seven model wrappers under `forecaster/forecasters/`:

| model id         | wrapper                          | notes                                                            |
| ---------------- | -------------------------------- | ---------------------------------------------------------------- |
| `auto_arima`     | `AutoArimaForecaster` (pmdarima) | non-seasonal `pmdarima.auto_arima`; refits per request           |
| `arima`          | `ArimaForecaster` (statsmodels)  | manual `(p, d, q)` from `model_params`                           |
| `sarima`         | `SarimaForecaster` (SARIMAX)     | manual `(p, d, q)` + `(P, D, Q, s)`; relaxed stationarity checks |
| `prophet`        | `ProphetForecaster`              | Facebook Prophet; `interval_width = 1 - alpha`                   |
| `chronos`        | `ChronosForecaster`              | Amazon Chronos T5 sample-based; preloaded at startup             |
| `moving_average` | `MovingAverageForecaster`        | flat-mean baseline; residual sd × √horizon CI                    |
| `xgboost`        | `XgboostForecaster`              | lag + rolling features, recursive multi-step                     |

Per-model hyperparameters travel in a single `model_params: dict[str, Any]` field on `ForecastRequest`; each wrapper's `predict` signature pulls the keys it needs by name (`p`/`d`/`q`, `window`, `lags`, …) and `**kwargs` swallows the rest. Adding a new model means: new file under `forecaster/forecasters/`, lazy-import branch in `_get_forecaster`, the id added to the `ModelType` literal in `forecaster/schemas.py`, the GraphBox UI in `app/core/plotting.py` (model dropdown option + a branch in `_render_model_param_inputs`), and a smoke test in `forecaster/tests/test_arima_smoke.py`.

`ARIMA_AVAILABLE` / `PROPHET_AVAILABLE` / `CHRONOS_AVAILABLE` toggles in `config.yaml` gate the three heavy-dep families. `auto_arima` / `arima` / `sarima` all ride on `ARIMA_AVAILABLE`. `moving_average` and `xgboost` are always available.

### Data ingestion

`downloader_general/src/` is split into `core/` (orchestration, schema validation in `utils/schema.py`), `extractors/` (one module per source: `world_bank`, `yahoo`, `binance`, `github` for the news repo), and `utils/`. It's a one-shot job — its container exits after success. Re-running it from scratch requires removing the `_container_data/downloader_general/.download_completed` marker (gitignored) and the persistent volumes (`postgres_data`, `qdrant_data`).

Ingestion progress is reported **through the logs, not `tqdm`** (terminal progress bars don't render in container logs). Long loops wrap their iterable in `log_progress(iterable, label=..., total=...)` (`src/utils/downloads.py`), which emits a throttled INFO line (≤ one per 5 s, plus a final 100% line); git clone progress goes through the log-emitting `CloneProgress` in the same module. `tqdm` is no longer a declared dependency anywhere — reuse `log_progress` for any new progress reporting rather than reintroducing it.

World Bank access goes through a hand-rolled **async `httpx` client** (`downloader_general/src/utils/wb_client.py`), which replaced the old `wbgapi` dependency — it pages the documented `https://api.worldbank.org/v2/...` REST endpoints directly (`/source`, `/country`, `/indicator`, `/country/all/indicator/{id}`, `/sources/{db}/series/{id}/metadata` with a `/indicator/{id}` fallback) and returns plain dicts shaped exactly as the schema cast expects. Aggregate economies (`region.id == "NA"`) are dropped (old `skipAggs=True` parity) and null observations kept; the indicator phase runs concurrently under an `asyncio.Semaphore(max_parallel_indicators)`. `downloader_extra` ships its own trimmed copy of the same client (`downloader_extra/wb_client.py`, data-fetch only) — duplicated per service by design, like the other tiny per-service models.

For incremental WB indicator additions during a live stack, the agent's `downloader_agent` worker calls `downloader_extra` (port 8003), which writes directly into the running Postgres without touching the marker.

Binance crypto ingestion (`downloader_general/src/extractors/binance_download.py`) goes through its own async `httpx` client (`src/utils/binance_client.py`), hitting only the documented public spot endpoints (`/api/v3/exchangeInfo`, `/api/v3/ticker/24hr`, `/api/v3/klines`) — no API key. It selects the `top_n` USDT spot pairs by trailing-24h quote volume (dropping stablecoins via `exclude_base_assets` and leveraged UP/DOWN/BULL/BEAR tokens in code), writes the ranked master data to `binance_metadata`, then pages each pair's full daily candle history into `binance_historical_prices` (PK `[date, symbol]`, FK → `binance_metadata.symbol`) concurrently under an `asyncio.Semaphore(max_parallel_symbols)`. The "description" is synthesized from documented fields (the REST API exposes no prose coin descriptions). Like Yahoo/WB it's marker-gated and bootstraps **only** its own `binance` schema group, so adding crypto to an existing volume needs a clean boot (there is no `downloader_extra` path for crypto). It reuses the shared retry helper `wb_client.call_with_retries`.

## Conventions worth knowing

- The codebase uses **Polars**, not Pandas. Don't introduce `pandas` in new code.
- Charts are always **Plotly** going through the `"app"` registered template; pull colours from `core/theming` helpers rather than hard-coding.
- Agent worker outputs are **Pydantic models** (`agent/agent/schemas.py`); structured-output LLM calls use `with_structured_output(...)`. Adding a worker means: schema in `schemas.py`, tool wrappers in `tools.py`, node + supervisor routing in `graph.py`, and the worker name in `WORKER_NAMES`.
- Every backend HTTP endpoint should have a typed wrapper in `app/core/api_client.py` — don't bypass it from pages.
- Two Postgres roles: the superuser `POSTGRES_USER` (created natively by the `postgres:18` image at first volume init) is used by `downloader_general` / `downloader_extra` for ingestion writes and by `app/core/token_usage_store.py` for `token_usage` inserts; the read-only `POSTGRES_LLM_USER` is used by the `agent` (SQL worker) AND by the `app` for all `connectorx` page reads. `db_bootstrap.ensure_llm_role` grants the read-only role `USAGE` on `public` plus `SELECT` on all existing and future tables (via `ALTER DEFAULT PRIVILEGES`) — don't grant it anything more.
- DB access: the `app` keeps `connectorx` for bulk Polars reads; the `agent`'s `sql_agent` worker keeps raw `text()` because the LLM generates dynamic SQL. Everywhere else uses SQLAlchemy ORM (`select`/`delete`/`Session`) on `Mapped` models — don't sprinkle new `text()` calls.
- Postgres database name resolution: every client (`app/core/postgres_client.py`, `agent/main.py`, `downloader_extra/main.py`, `downloader_general/main.py`) reads `os.getenv("POSTGRES_DB") or config["postgres"]["database"]`. `POSTGRES_DB` in `.env` is the source of truth; `config.yaml`'s `postgres.database` is the fallback when the env var is unset. The postgres image itself only honours `POSTGRES_DB` at first volume init — changing the value on a populated volume requires re-creating the DB or wiping the volume.
- News page embedding panels (`app/pages/15_news.py: _render_embedding_map` and `_render_distance_histogram`) run only when their `st.form_submit_button` is clicked and cache the result in `st.session_state` so picking a different article re-draws the highlight on the cached projection without re-clustering. The scatter highlights the selected article using the `selected_marker` theme token; add that token to any custom theme before shipping it.

## Rules of repo

- Use only `uv` and `pnpm` for package management
- Use `ruff` and `es-lint` for linting/formatting code
- Use `ty` for static type checking, most code of should be strictly typed
- Use pydantic for all data types validation
- Use `SQLAlchemy` as ORM, all tables in postgresql should be descriped in ORM
- Never push to github
- In commits messages use short and simple messages
- Tests for each service should be written
- For frontend testing use playwright, don't finish until web ui looks clean and smooth, without bugs and visual problems
- Work only in `dev` branch
- If something is not even slightly understood by you - always ask user
- On each step of development, update `CLAUDE.md`, `TODO.md`, `CHANGELOG.md` (only for v0.x commits with tags) and `PLAN.md`
