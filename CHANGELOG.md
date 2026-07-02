# Changelog

All notable changes to **Ultimate Macroeconomics Dashboard** are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v0.14]

Extended on-demand ingestion beyond the World Bank: the agent can now download a single Yahoo Finance ticker or Binance crypto pair that isn't yet in the database, the same way it already pulls missing WB indicators.

### Added

- **`downloader_extra` is now multi-source.** Its single `POST /ingest` endpoint dispatches on a new `source` field (`worldbank` / `yahoo` / `binance`); the old WB-only body (`{indicator_id, db_id}`) still works because `source` defaults to `worldbank`. Two new per-source clients sit beside `client_wb.py`: `client_yahoo.py` (`fetch_and_store_yahoo`, reusing the single-ticker `yfinance` metadata+history logic from `downloader_general`) and `client_binance.py` + a trimmed `binance_client.py` transport copy (validates the full pair symbol via `/api/v3/exchangeInfo?symbol=`, pulls the 24h ticker + full daily klines). Each path short-circuits on `already_downloaded`, else delete-then-inserts the fresh rows with the metadata row written first to satisfy the FK. New ORM models (`YahooMetadata`/`YahooHistoricalPrice`, `BinanceMetadata`/`BinanceHistoricalPrice`) mirror `database_schema.yaml`; an on-demand Binance metadata row leaves `rank` NULL (no batch-wide ranking). Adds `yfinance==1.2.0`. Covered by offline `test_binance_client.py` / `test_ingest_request.py` plus testcontainer `test_market_orm_roundtrip.py` / `test_store_integration.py`.
- **Agent: on-demand Yahoo + Binance downloads (no master catalogue).** The `downloader_agent` worker is now source-aware — it extracts a unified `DownloadPlan` (`source` + the relevant id) and POSTs the unified `/ingest` body via the generalised `tools.ingest_data`. Unlike World Bank (which has the `database_indicators` master table), Yahoo/Binance have no catalogue, so the worker infers the ticker / full USDT pair symbol from world knowledge (Apple→`AAPL`, Solana→`SOLUSDT`) and `downloader_extra` validates it against the live API. `sql_agent` gained Binance querying (Plan C: `binance_metadata` + `binance_historical_prices`) and a new `NEEDS_DOWNLOAD (source=yahoo|binance)` signal when a `*_metadata` lookup comes back empty (suppressed when metadata exists but a price filter is just empty). Supervisor + sql_agent prompts updated to route untracked tickers/coins to `downloader_agent` instead of FINISH-ing with "not tracked". New `agent/tests/test_downloader_routing.py` covers the two pure routing helpers (`DownloaderAgent._build_payload`, `_detect_market_needs_download`).

### Changed

- **`DownloadIndicatorPlan` → `DownloadPlan`** (agent schemas): WB-only `{indicator_id, db_id}` replaced by a source-tagged model with optional `indicator_id`/`db_id`/`ticker`/`symbol`. **`tools.download_indicator` → `tools.ingest_data(payload)`** (posts the unified body; timeout raised 120s→300s for full-history pulls).

## [v0.13]

Added end-to-end **cryptocurrency** coverage: Binance ingestion in `downloader_general` plus a new Crypto dashboard page.

### Added

- **Binance crypto ingestion in `downloader_general`.** A new async `httpx` client (`src/utils/binance_client.py`) and extractor (`src/extractors/binance_download.py`, `BinanceDownloader` + `BaseBinanceDownloader` ABC) hit only the documented public spot endpoints (`/api/v3/exchangeInfo`, `/api/v3/ticker/24hr`, `/api/v3/klines`) — no API key. They select the top-30 USDT spot pairs ranked by trailing-24h quote volume (dropping stablecoins via `exclude_base_assets` and leveraged UP/DOWN/BULL/BEAR tokens in code), write ranked master data to a new **`binance_metadata`** table, and page each pair's full daily candle history into **`binance_historical_prices`** (PK `[date, symbol]`, FK → `binance_metadata.symbol`) concurrently under an `asyncio.Semaphore`. Both tables are declared in `database_schema.yaml` (so the agent's SQL worker sees them); the master-data "description" is synthesized from documented fields since the REST API exposes no prose. Wired into `main.py` after the Yahoo step, marker-gated like the other one-shot sources, and tunable via the new `_configs/binance_download_config.json` (`base_url`, `quote_asset`, `top_n`, `kline_interval`, `max_parallel_symbols`, `exclude_base_assets`). Reuses the existing `wb_client.call_with_retries` retry helper and `schema.py` write/bootstrap utilities. Covered by offline `tests/test_binance_client.py` (mocked `httpx` transport).
- **Crypto dashboard page (`app/pages/16_crypto.py`, "Crypto" under "Other data").** Mirrors the Yahoo page: a market-overview table, a top-coin price-dynamics line chart on a log axis (so coins at very different price levels stay comparable), a Bitcoin candlestick, and an all-coin daily-return correlation heatmap. Reads via two new error-resilient `core/postgres_client.py` helpers (`get_all_binance_historical_prices`, `get_all_binance_metadata`, returning an empty frame when the tables predate this release).
- **Generic `build_candlestick_plot` / `build_correlation_heatmap` in `core/plotting.py`**, lifted from the Yahoo-page builders so the Crypto page reuses them (the Yahoo page keeps its own `build_yahoo_*` variants).

## [v0.12]

Dropped the `wbgapi` dependency in favour of a hand-rolled async `httpx` World Bank client, and broadened the Yahoo Finance company universe.

### Changed

- **World Bank ingestion now uses a plain async `httpx` client instead of `wbgapi`.** New `downloader_general/src/utils/wb_client.py` pages the documented `https://api.worldbank.org/v2/...` REST endpoints directly — `/source` (databases), `/country` (countries), `/indicator?source={db}` (per-source indicator catalogue), `/country/all/indicator/{id}?source={db}` (observations), and the advanced `/sources/{db}/series/{id}/metadata` (rich series metadata) with a `/indicator/{id}` fallback. The extractor (`world_bank_download.py`) was rewritten async: per-indicator metadata+data fetches run concurrently under an `asyncio.Semaphore(max_parallel_indicators)` (replacing the old `ThreadPoolExecutor`), and blocking Postgres writes are offloaded with `asyncio.to_thread`. Output schema is **identical** — aggregate economies (`region.id == "NA"`) dropped (old `skipAggs=True` parity), null observations kept, country/metadata column shapes unchanged.
- **`downloader_extra` (on-demand ingestion) likewise dropped `wbgapi`.** It ships a trimmed copy of the client (`downloader_extra/wb_client.py`, data-fetch only); `fetch_and_store_indicator` is now `async` and awaited directly from `POST /ingest`, fetching over async `httpx` and offloading the DB write to a worker thread (instead of running the whole sync path in a threadpool).
- **`wbgapi` (and its now-orphaned `tabulate`) removed** from both services' `pyproject.toml` + `uv.lock`.
- **`tqdm` dropped as a declared dependency; ingestion progress now goes through the logs.** `downloader_general`'s progress bars are replaced by a logging-based `log_progress` helper (and a log-emitting `CloneProgress`) in `src/utils/downloads.py`, so progress is visible in `docker compose logs` instead of a terminal bar that doesn't render there; the bespoke `_TqdmHandler` in `main.py` is gone (plain `StreamHandler`). The explicit `tqdm==4.67.3` pin was removed from `downloader_general` / `agent` / `forecaster` `pyproject.toml` + `uv.lock` (it had no code use in `agent` / `forecaster`); it now resolves only as a transitive dep of `openai` / `transformers` / `prophet` / `cmdstanpy` / `huggingface-hub`.

### Added

- **34 more companies in `yahoo_download_config.json`** (50 → 84) — broader sector and geography coverage: more US tech/financials/healthcare/consumer (NFLX, ADBE, CRM, CSCO, INTC, QCOM, IBM, WFC, MS, GS, BLK, PFE, MRK, DIS, CMCSA, NKE, SBUX) plus international names across Europe/Asia/energy (SAP, SONY, BABA, NESN.SW, UL, MC.PA, HSBC, AZN, NVS, SHEL, BP, TTE, VOW3.DE, 1211.HK, RELIANCE.NS, AIR.PA). All tickers verified live against Yahoo Finance.
- **`tests/test_wb_client.py`** in both `downloader_general` and `downloader_extra` — offline unit tests (faked HTTP via `httpx.MockTransport`) covering pagination, aggregate skipping, null retention, metadata parsing + XML/fallback handling, and the retry wrapper.

## [v0.11]

Typed configuration and least-privilege secrets across every service, plus a two-model agent for cheaper, faster inference.

### Added

- **Typed config per service.** Each service now parses the slice of `config.yaml` it reads through a Pydantic model in its own `config.py` (`load_config(path)` → validated object, attribute access). Replaces the ~10× duplicated `yaml.safe_load(...) + .get("x", {}).get("y")` boilerplate (and the bespoke `_require()` nested-key walker in `downloader_general`); a malformed config now fails at startup with a precise `ValidationError`.
- **Typed secrets per service (`pydantic-settings`).** Services that use secrets gained a `settings.py` `BaseSettings` model declaring only the variables they read; the Qdrant key's three legacy env-var names collapse into one `AliasChoices` field. Config/settings validation tests added for every service.
- **`shared.openai_llm_model_fast`** config key naming the agent's fast model (falls back to `openai_llm_model` when unset).

### Changed

- **The agent runs two models.** `MacroAgentGraph` now builds a strong `ChatOpenAI` (`openai_llm_model`) for the reasoning-heavy roles — supervisor (planning + the final answer), `sql_agent`, `plotly_agent`, `chat_agent` — and a fast `ChatOpenAI` (`openai_llm_model_fast`) for the cheap `GuardrailAgent` screen and the lightweight `table_agent` / `rag_agent` / `web_search` / `downloader_agent`. Lower latency and cost on the cheap steps, quality preserved where it matters. `UsageTracker` is a graph-level callback, so per-call token accounting spans both models and labels each call with the model that served it. Vision (`/plots/interpret`) stays on the strong model.
- **Least-privilege secret injection.** `docker-compose.yaml` no longer mounts the whole `_container_data/.env` into every container via `env_file`. Each service gets a per-service `environment:` block injecting only the secrets it uses via `${VAR}` interpolation: `clustering` / `forecaster` / `python_sandbox` get none, `agent` gets the read-only LLM role (not the superuser password), `app` gets both Postgres roles + the Qdrant key (no `OPENAI_API_KEY`); `db` / `vector_db` / `downloader_extra` get only their own. A gitignored root `.env` symlink → `_container_data/.env` keeps `docker compose up` working unchanged (or run `docker compose --env-file _container_data/.env up`).
- **Dropped dead `load_dotenv` / `python-dotenv`** from `clustering`, `forecaster`, and `python_sandbox` (they read no secrets); added `pydantic` to `app` / `python_sandbox` and `pydantic-settings` to `app` / `agent` / `downloader_extra` / `downloader_general`.

### Operator notes

- After pulling, create the gitignored root `.env` symlink once — `ln -sfn _container_data/.env .env` — then `docker compose up --build` as before. Set `shared.openai_llm_model_fast` in `config.yaml` to a fast model available on your endpoint (ships as a `gpt-5.4-mini` placeholder; remove the key to fall back to the strong model).

## [v0.10]

Forecasting expansion (four new models + per-model hyperparameter inputs), forecast-chart polish (dashed history→forecast connector + optional point markers), tighter and cheaper LLM plot descriptions, two embedding-visualisation panels on the News page that ride on the existing clustering service, a `selected_marker` semantic theme token, and a `POSTGRES_DB`-resolution fix that finally lets the env var pick the database name everywhere instead of only at first volume init.

### Added — forecasting

- **Four new models in `forecaster`** alongside the existing `prophet` / `chronos`:
  - `arima` — manual `(p, d, q)` via `statsmodels.tsa.arima.model.ARIMA`. The previous auto-tuned `pmdarima.auto_arima` wrapper has been renamed to **`auto_arima`** (the old `arima` model id is no longer accepted; switch saved presets accordingly).
  - `sarima` — manual seasonal ARIMA via `statsmodels.tsa.statespace.SARIMAX` with stationarity/invertibility checks relaxed so user-supplied orders outside the strict region still produce a forecast.
  - `moving_average` — naive flat-mean baseline; CI from in-sample residual sd × √horizon.
  - `xgboost` — recursive XGBoost on lag + rolling-mean/std + time-index features; CI uses the same residual×√h heuristic as MA.
- **`ForecastRequest.model_params`** — flat dict carrying per-model hyperparameters (`p`/`d`/`q`, `P`/`D`/`Q`/`s`, `window`, `lags`/`n_estimators`/`max_depth`/`learning_rate`). Wrappers `**kwargs` through what they need and ignore the rest, so callers can keep one shape regardless of model.
- **GraphBox forecast popover** now exposes the model dropdown outside the form (so the per-model hyperparam widgets swap immediately on change) and gained a **Highlight points** toggle that flips every historical and forecast trace from `lines` to `lines+markers`. Off by default.
- **Smoke tests** for every concrete forecaster in `forecaster/tests/test_arima_smoke.py` — same shape/CI bracketing assertion across `auto_arima`, manual `arima`, `sarima`, `moving_average`, and `xgboost` (with `pytest.importorskip("xgboost")` so the suite survives a missing wheel).

### Added — news page embedding visualisations

- **Embedding Map** (`_render_embedding_map`) — 2D/3D scatter of every article in the selected topic, projected via the clustering service (`POST /cluster`) with user-chosen reducer (`tsne` / `umap` / `pca`), clustering method (`kmeans` / `dbscan` / `hdbscan` / `hierarchical`), and `k`. The currently selected article is rendered as a star (2D) / diamond (3D) using the new `selected_marker` theme token; other points are coloured by cluster from `get_colorway()`. Capped at 1000 sampled points with a deterministic seed.
- **Distance Histogram** (`_render_distance_histogram`) — cosine-distance distribution from the selected article to every other article in the topic, plus min / median / max metrics below the chart.
- Both panels live behind an explicit `st.form_submit_button("Run …")` so nothing fires until the user asks; the result is cached in `st.session_state` keyed by collection so switching to a different article in the news selector only re-draws the highlight on the already-projected map (no re-clustering) and leaves the histogram as-is until the user reruns it.
- **`core.qdrant_client.scroll_collection`** gained a `with_vectors: bool = False` argument used by the embedding panels' vector-load helper.

### Added — theming

- **`selected_marker`** semantic colour token in `_container_data/themes.yaml` for all three bundled themes (`dark`, `dark-blue`, `light-green`). Used by the News page embedding scatter to highlight the active article; pick a contrasting colour if you ship a custom theme.

### Changed

- **Forecast trace shape.** `build_line_plot` now draws a dashed segment from the last actual point to the first forecast point (no CI band on that segment) and keeps the existing dashed line + shaded CI starting at the first predicted point. The visual gap between history and forecast is gone; the band still only carries the model's uncertainty over the forecast window, not over the join. The connector and marker behaviour are unified across the grouped and ungrouped code paths.
- **Plot interpretation prompts shortened.** `agent /plots/interpret` system prompts were rewritten to two short sentences (strict mode: "3 bullets, ~40 words"; creative mode: "max 60 words, mark hypotheses with 'likely'"), and the OpenAI call now caps `max_tokens` (200 / 260) so the model can't blow past the budget. Same endpoint, same modes, smaller and faster responses.
- **`api_client.forecast_timeseries`** signature gained an optional `model_params: dict[str, Any] | None` argument that is forwarded to the forecaster service.

### Fixed

- **`POSTGRES_DB` env var was ignored by every Python client.** The postgres image creates the database named by `POSTGRES_DB` on first volume init, but `app/core/postgres_client.py`, `agent/main.py`, `downloader_extra/main.py`, and `downloader_general/main.py` all read the database name from `_container_data/config.yaml` (`postgres.database: postgres`, hardcoded). Setting `POSTGRES_DB=macroeconomics` therefore created the right database but every connection still pointed at `postgres`. Every call site now resolves `os.getenv("POSTGRES_DB") or config["postgres"]["database"]`. On existing deployments the volume already carries the originally-named database — either `docker compose down -v` and re-ingest, or `docker compose exec db createdb -U $POSTGRES_USER $POSTGRES_DB` and run the downloader against it.
- **`forecaster/uv.lock` didn't contain `xgboost`.** The previous lockfile pre-dated the new dep, and the Dockerfile uses `uv sync --frozen`, so the container booted without xgboost and `forecasters/xgboost_model.py` failed at import time. The lockfile has been regenerated with `xgboost==2.1.4` pinned (numpy-2.x compatible, manylinux wheels). `cd forecaster && uv lock && docker compose build forecaster` is needed once.

### Operator notes

- Forecaster image rebuild required: `docker compose build forecaster && docker compose up -d forecaster`. The new XGBoost wheel adds ~80 MB to the image.
- App image rebuild required for the News page additions: `docker compose build app && docker compose up -d app`.
- `themes.yaml` carries a new key (`selected_marker`); existing custom themes need to add it or the News page embedding scatter will raise `KeyError`.
- No `config.yaml` changes required; the existing `postgres.database` line stays as a fallback.

## [v0.9]

AI-analyst quality and latency pass plus a critical fix to the read-only LLM Postgres role's permissions, on top of a broader maintenance pass: new `tests/` suites under every service (`agent`, `app`, `clustering`, `downloader_extra`, `downloader_general`, `forecaster`, `python_sandbox`), two new app pages (`17_token_usage`, `18_monitoring`) replacing the legacy `16_settings`, retirement of the standalone `_container_data/db_init.sh` (its job is now done by `downloader_general`'s bootstrap), and assorted refactors across every service. Same architecture, same overall UI; the multi-agent graph is shorter (one fewer LLM call per turn) and the supervisor / SQL / chat / RAG / web-search workers are now grounded in chat history, a centralised macro context block, and a deterministic worker-status channel.

The notes below detail the AI-analyst + DB-permission work — the broader file-by-file diffs across the rest of the stack are visible in `git log` for the v0.9 commit.

### Fixed
- **`downloader_general`: read-only LLM role had no table privileges.** `ensure_llm_role` only created/altered the role but never granted any `SELECT`, so the AI analyst's `sql_agent` failed with `permission denied for table databases / database_indicators / indicators / …` on every query. The bootstrap now also issues `GRANT USAGE ON SCHEMA public`, `GRANT SELECT ON ALL TABLES IN SCHEMA public`, and `ALTER DEFAULT PRIVILEGES … GRANT SELECT ON TABLES` (run as the superuser so future tables created by `downloader_general` / `downloader_extra` are readable automatically). The new test `test_ensure_llm_role_grants_select_on_existing_and_future_tables` pins the contract.
- **`downloader_general` entrypoint never re-ran the bootstrap on upgrade.** The shell entrypoint exited early on the `.download_completed` marker, so any change to `ensure_llm_role` (such as the grant fix above) would never take effect on an existing volume. The marker check now lives inside `main.py`: bootstrap runs every container start, downloads still run only once.

### Added — agent quality
- **Pass chat history to workers.** `SQLAgent`, `RAGAgent`, `WebSearchAgent`, and `ChatAgent` now receive the last 3 user/assistant turns from `AgentState.messages` in their prompts so follow-ups like "now Germany too" don't require the supervisor to re-state context. Plot / table workers don't get it — they operate on the already-fetched artifact.
- **Worked SQL examples (few-shot).** The `SQLAgent` preamble carries 3 canonical examples (single-country WDI lookup, ticker-known Yahoo fetch, multi-country WDI with `IN (...)`) so first-shot SQL quality improves on smaller models.
- **`last_worker_status` channel.** New `AgentState` field carrying a `Literal["SUCCESS","EMPTY","ERROR","NEEDS_DOWNLOAD","BLOCKED","UNKNOWN"]` tag returned by every worker. The supervisor's branching is now driven by this enum instead of regex-matching `"SQL_AGENT INDICATOR_NOT_DOWNLOADED:"` prose, and the prompt explicitly references the tag (e.g. "route to `downloader_agent` when `last_worker_status` is `NEEDS_DOWNLOAD`").
- **Centralised macro context.** The supervisor preamble now states the always-on assumptions (WDI is `db_id = 2`; `indicators.economy` holds ISO-3 codes; Yahoo is the only stock/index source; today's date) instead of expecting the planner to re-derive them.
- **Short-circuit WDI lookup in `sql_agent`.** The SQL prompt now defaults `database_id = 2` (WDI) and skips the `databases`-table lookup step unless the user explicitly names another World Bank database — drops one LLM call from most macro queries.

### Changed — agent speed
- **Dropped the FINAL_SYNTHESIS LLM call.** The supervisor already writes the final markdown answer to `isolated_worker_task` when it picks `FINISH`; `MacroAgentGraph._stream_final_synthesis` re-fed it through an extra streaming LLM call. Replaced with a chunked character-streamer (`_stream_supervisor_draft`) that emits the draft in ~24-char bursts directly — no model call, no risk of the synthesis pass altering numbers. A small leak filter (`_sanitize_draft`) strips any line containing worker names / sandbox / traceback tokens as a last-mile guard.
- **Heuristic-first guardrail.** The previous unconditional LLM screening step is gone; `GuardrailAgent` now uses three regexes (auto-allow for short greetings + in-scope keywords, auto-block for obvious red flags) and only escalates ambiguous messages to the structured-output LLM. Same safety profile, no LLM call for the typical user message.
- **Shared `httpx.AsyncClient`.** `execute_code_in_sandbox` and `download_indicator` used to spin up `async with httpx.AsyncClient()` per call (full TLS handshake every time). One pool is now built lazily in `agent.tools._get_httpx_client()` and closed via a FastAPI `shutdown` hook.
- **Cached `get_database_schema_text()`.** The YAML-to-text rendering used to run on every SQL step; it's now wrapped in `functools.lru_cache(maxsize=1)` (and invalidated by `configure_runtime` for tests).
- **Prefix-cacheable prompt layout.** Supervisor / SQL / Plotly / Polars / RAG / web-search / chat prompts have been restructured so the static role + scope + rules + (where applicable) database schema and few-shot examples form the prompt prefix, and only the per-call dynamic data lives in the trailing suffix. Providers that auto-cache identical prefixes (OpenAI, Anthropic) can now reuse them across turns.
- **Trimmed supervisor `worker_results`.** `worker_results` uses `operator.add` and grew unboundedly across turns; the supervisor prompt now keeps the last two verbatim and summarises older entries to a single line each.
- **Healthcheck noise reduced.** Compose healthchecks for the HTTP services (`agent` / `app` / `forecaster` / `clustering` / `downloader_extra` / `python_sandbox` / `vector_db`) bumped from `interval: 10s` to `30s` so `/health` polling no longer drowns real errors in `docker compose logs`. `db` stays at 5s (it gates `depends_on: condition: service_healthy` on stack startup).

### Removed
- **`MacroAgentGraph._stream_final_synthesis`** and its `FINAL_SYNTHESIS_SYSTEM_PROMPT` — replaced by direct chunked emission of the supervisor's draft.

### Operator notes
- **Existing deployments** need a one-time rebuild of `downloader_general` so the new bootstrap runs against their already-populated DB: `docker compose build downloader_general && docker compose up -d downloader_general`. The container will re-apply role + grants in seconds, find the download marker, and exit without re-running the multi-hour ingestion.
- **No `.env` / `config.yaml` changes** required.

## [v0.8]

Codebase-wide refactoring pass across all ten containers. No new user-facing features — the goals were dedup, dead-code removal, async-safety, error-handling hardening, and modular structure. Same architecture, same UI, cleaner internals.

### Added
- **`app/core/page_helpers.py`** — shared `prepare_indicator_slice` + `fetch_indicator_slice` helpers that replace 10 verbatim copies of `_prepare_indicator_slice` previously inlined in `app/pages/01..10_*.py`. Dashboard pages now import these instead of re-implementing the same World Bank normalization recipe each time.
- **`agent/agent/prompts.py`** — central location for the LangGraph system prompts (147-line supervisor prompt and the guardrail prompt), pulled out of `graph.py` for readability.
- **`python_sandbox`: `RLIMIT_AS` (2 GB) and `RLIMIT_CPU`** applied to the subprocess running LLM-generated code (via `preexec_fn`). Previously only a wall-clock timeout existed; misbehaving code now gets killed by the kernel instead of starving the container.
- **`agent /chat/stream`: 5-minute SSE stream timeout.** An `asyncio.timeout(...)` guard wraps `astream_events`; on timeout the client receives a final `error` event instead of an open-ended stream.
- **`forecaster`: `FORECASTER_CONFIG_PATH` env var** to override the default `config.yaml` location.

### Changed
- **`forecaster /predict` and `clustering /cluster` are now `async def`** with the CPU-bound work dispatched via `fastapi.concurrency.run_in_threadpool(...)`. Previously the sync handlers blocked the FastAPI event loop for the entire duration of `Prophet.fit()` / `auto_arima()` / `TSNE.fit_transform()`.
- **`forecaster` model cache is now thread-safe.** Moved from a module-level dict to `app.state.model_cache` + `asyncio.Lock` with double-checked locking; concurrent first-time requests for the same model no longer race on heavy ML imports.
- **`agent /plots/interpret` no longer blocks the event loop.** The blocking `openai.OpenAI.chat.completions.create(...)` call is now wrapped with `asyncio.to_thread(...)`.
- **`app/core/api_client.py` error messages.** All `RuntimeError`s wrapping HTTP failures now include the actual status code and a body excerpt instead of the misleading "No available base URL candidates" string.
- **`app/core/plotting.py: GraphBox.render_streamlit_ui`** decomposed: the settings popover and the dropped-log-points caption were extracted into helper methods, reducing the 435-line monolith.
- **`agent/agent/graph.py`** worker `except Exception` blocks now call `logger.exception(...)` before returning the fallback result, so failures show up in container logs instead of vanishing.
- **`python_sandbox/main.py`** logs subprocess start, finish, and temp-file cleanup outcomes.
- **`downloader_general`**: post-download `sleep(10)` is now a configurable class attribute (`between_download_sleep_seconds`).
- **`forecaster /models`, `agent /models`, and assorted page settings** narrowed bare `except Exception:` to specific exceptions with logging.

### Removed
- **10 duplicated copies** of `_prepare_indicator_slice` across `app/pages/01..10_*.py` (~250 lines of dead duplication).
- **Unused `pandas==3.0.1`** from `downloader_extra/requirements.txt` (~50 MB image bloat).
- **`successfull_connections` field** from `downloader_general/src/extractors/world_bank_download.py` (set but never read; also fixed the typo by deletion).
- **Duplicated `all_records = []` / `offset = None` initialization** in `app/core/qdrant_client.py` (copy-paste artifact).

### Deferred (called out, not done in this pass)
- Splitting `agent/agent/graph.py` (~1.5k lines) into per-worker modules under `agent/agent/workers/`.
- Rewriting `plotly.express` chart calls in dashboard pages to consume polars natively (removing the remaining `.to_pandas()` conversions). The remaining sites rely on pandas-specific boolean masking / `.empty` / `.fillna` patterns and would need a careful go.Figure rewrite.
- Shared `wb_api.py` between `downloader_general` and `downloader_extra` (requires multi-service file sharing).
- Replacing pandas with polars internally in the `forecaster` models.
- Adding the missing PostgreSQL indexes via a `db_init` migration.

## [v0.7]

Improvements of descriptions for the dashboard

### Added
- `.drawio` and `.png` diagrams with description of dashboard architecture and agentic system architecture
- Renamed `TODO` into `TODO.md`
- updated `.gitignore`
- presentation of the dashboard
- text of the dashboard introduction in $\LaTeX$ and its compiled version in `.pptx`

## [v0.6]

The "hosting-ready" release. Adds an alternate deployment topology for public VPS hosting (on the `hosting` branch), introduces in-session LLM token-usage accounting, and removes the runtime theme picker.

### Added
- **Session token usage panel.** The Settings page now shows a per-model breakdown of prompt / completion / total tokens consumed by the AI analyst during the current Streamlit session, with a "Reset token counter" button. Tracking is in-memory only and clears on session end.
- **Token usage in the agent API.** The `agent` service attaches a per-request `UsageTracker` callback to every LangChain LLM call (guardrail, supervisor, every worker, plus the final synthesis stream) and surfaces aggregated token counts in the `final` SSE event of `/chat/stream`. `/plots/interpret` now also returns a `usage` block.
- **`CHANGELOG.md`** (this file).
- **Hardened production deployment topology** on the new `hosting` branch (added in a follow-up commit on that branch): `nginx` reverse proxy as the sole public entry point, internal-only Docker network for backend services, per-session user-supplied LLM credentials forwarded to the agent via request headers, embedding/RAG credentials kept server-side. See `README.md` § "Deployment (hosting)".

### Changed
- **README.md** — full rewrite: clearer prose, fixed spelling, dedicated `.env` variable table, separated "local development" and "Deployment (hosting)" flows, refreshed configuration section.
- `themes.yaml` is now the only place to switch themes; this is a deploy-time concern, not a runtime UI option.

### Removed
- **Runtime theme picker** from the Settings page. The dropdown, "Apply theme" button and the code that rewrote `app/.streamlit/config.toml` are gone. `core/theming.set_active_theme`, `list_theme_names`, `get_active_theme_name`, and the `_sync_streamlit_config` helpers were removed (they had no remaining callers).

## [v0.5]

### Added
- `_container_data/.env.example` to make the required environment variables explicit.

### Changed
- Sturdier external-service clients: `app/core/postgres_client.py` and `app/core/qdrant_client.py` got better error handling and retries.
- `python_sandbox/main.py` — significant rework of sandbox execution and timeouts.
- `clustering` service: new dependencies, more robust input handling.
- `downloader_general`, `downloader_extra` and `forecaster` received reliability fixes around file handling and configuration loading.
- Minor adjustments in `agent/main.py`, `app/core/plotting.py` and `app/pages/16_settings.py`.

## [v0.4]

### Added
- **Theming system** under `_container_data/themes.yaml` plus `app/core/theming.py`:
  - Three bundled themes (`dark`, `dark-blue`, `light-green`).
  - A registered Plotly template (`"app"`) derived from the active theme.
  - A Streamlit-config sync that mirrored the active theme into `app/.streamlit/config.toml`.
- A theme picker on the Settings page (later removed in v0.6).

### Changed
- All dashboard pages migrated off hard-coded chart colours and onto the new theme tokens (`get_color`, `get_colorway`).
- `app/.streamlit/config.toml` now derives from the active theme rather than holding fixed colours.

## [v0.3]

### Added
- **Multi-agent architecture** for the AI analyst built on LangGraph. `agent/agent/graph.py` was rewritten end-to-end (~1k lines) to introduce:
  - A `GuardrailAgent` that screens incoming user messages.
  - A `MacroSupervisorAgent` that plans, delegates and finishes.
  - Specialised workers: `sql_agent` (3-step World Bank exploration + Yahoo Finance lookup), `plotly_agent` (Plotly code generation in a sandboxed runtime), `table_agent` (Polars transformations), `rag_agent` (Qdrant news search), `web_search` (DuckDuckGo), `downloader_agent` (calls `downloader_extra` to ingest new World Bank indicators), `chat_agent` (conversational synthesis).
  - Streaming SSE protocol from `/chat/stream` (`step` / `token` / `final` / `error` events).
- New `agent/agent/schemas.py` Pydantic models for every structured worker output.

### Changed
- AI analyst page rebuilt around the new streaming protocol with execution log, plot artifact rendering and a dedicated data-table expander.
- Page numbering renormalised under the new navigation structure (pages `06`–`16`).

### Removed
- The legacy single-prompt agent path in `app/core/api_client.py` and the older `10_ai_agent_chat.py` page.

## [v0.2]

### Added
- Six new dashboard pages: Governance & Institutions, Technology & Innovation, Health & Wellbeing, Education & Human Capital, Environment, **AI Analyst**, **Custom Plot Constructor**, **Clustering Sandbox**, **Yahoo Finance**, **News Explorer**, and the first **Settings** page.
- World Bank download configuration extended with additional indicators across the new domains.
- Centralised page-render and HTTP-request logging in `app/core/app_logging.py`.
- `db_init` container that creates the application's PostgreSQL roles and schema (added shortly before this tag, alongside `_container_data/db_init.sh` and `init-user.sh`).
- `downloader_general` got a structured `src/utils/schema.py` validator and improved retry handling for World Bank fetches.

### Changed
- Database schema (`_container_data/database_schema.yaml`) refactored — column names and types tightened.
- `downloader_extra/client_wb.py`, `downloader_general/main.py` and the `world_bank` / `yahoo` / `github` extractors hardened against API timeouts and partial pages.
- README polished and a `TODO` file added.

## [Initial commit]

### Added
- Initial scaffolding of the project: ten-container `docker-compose.yaml`, the Streamlit `app` skeleton with the first World Bank indicator pages, the `agent` FastAPI service, `forecaster`, `clustering`, `downloader_general`, `downloader_extra`, `python_sandbox`, plus `db` (PostgreSQL) and `vector_db` (Qdrant).
- World Bank, Yahoo Finance and Webz.io news ingestion pipelines.
- Core dashboard infrastructure: postgres / qdrant clients, plotting helpers, asset templates, base config files (`config.yaml`, `database_schema.yaml`, download configs).
- README with installation instructions; MIT license.
