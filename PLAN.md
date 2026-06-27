# Development Plan

> Detailed roadmap for the next major iteration of **Ultimate Macroeconomics Dashboard**.
> This supersedes the earlier 12-line wishlist; every original item is tagged `(was #N)` for traceability.
> Architectural choices below were locked in a planning discussion — see **Guiding decisions**.

---

## Guiding decisions (locked)

| Topic | Decision |
| ----- | -------- |
| Frontend framework | **React + Next.js + Apache ECharts** |
| Frontend cutover | **Strangler** — run Next.js alongside Streamlit, migrate page-by-page, retire Streamlit last |
| Data/API layer | **New Python FastAPI BFF** ("backend-for-frontend") reusing the existing SQLAlchemy `Mapped` models; all reads wrapped in ORM |
| Agent inference | **Two models** — *light* for guardrail + routine workers (`sql`/`table`/`chat`), *heavy* for supervisor planning, final synthesis, `plotly`/vision |
| New data sources | **Extend the existing downloaders** (`downloader_general` one-shot + `downloader_extra` on-demand) with new extractor modules |
| Sub-national data | FRED (US states), Eurostat (EU NUTS regions), crypto → **new dedicated pages with sub-national choropleths**; existing country pages unchanged |
| Monitoring | **Prometheus + Grafana + cAdvisor** — replaces `app/core/monitoring.py` and the in-dashboard Monitoring page |
| ML serving | **Full NVIDIA Triton consolidation, GPU required** — forecaster + clustering models unified, RAPIDS (cuDF/cuML) for data/clustering |
| Dev workflow | **Tests + coverage gates + agent eval harness** — built *first* as the correctness safety net |
| Backups | **Nightly `pg_dump` + Qdrant snapshots** pushed to an `rclone` remote, with retention + restore script |

---

## Roadmap at a glance

Sequencing principle (decided): **build the safety net before the big rewrites.** Each phase is a verifiable milestone; later phases assume the BFF/test harness from earlier ones.

| Phase | Theme | Original items | Depends on |
| ----- | ----- | -------------- | ---------- |
| **0** | Claude-Code dev loop: tests, coverage, CI, agent eval | #10 | — |
| **1** | FastAPI BFF + shared ORM data layer | #1 (backend), #9 | 0 |
| **1b** | Agent two-model routing *(parallelizable quick win)* | #2 | 0 |
| **2** | Next.js + ECharts frontend (strangler) + dynamic theming | #1 (frontend) | 1 |
| **3** | Data expansion: WB-httpx, Yahoo, FRED, Eurostat, crypto, GDELT | #9, #5, #3, #4, #6, #7 | 1 (ingestion), 2 (new pages) |
| **4** | Ops: Prometheus/Grafana monitoring + rclone backups | #11, #8 | 1 |
| **5** | GPU/ML consolidation: RAPIDS + NVIDIA Triton | #12 | 1, 0 |

Phases 1b, 3 (ingestion), and 4 can proceed in parallel with the frontend (Phase 2) because they touch backend/ops only.

---

## Phase 0 — Claude-Code-driven dev loop *(was #10)*

**Goal:** a deterministic "make a change → prove it correct → iterate" loop the agent can run unattended, so every later phase lands behind a gate.

**Approach**
- **Per-service test expansion** — raise `pytest` coverage on each service; today only smoke tests exist (e.g. `forecaster/tests/test_arima_smoke.py`). Set a `--cov-fail-under` threshold in each `[tool.pytest.ini_options].addopts` so coverage regressions fail the run.
- **Agent eval harness** — implement the "table of tasks + expected results" from `TODO.md` (medium-term). A fixture set of prompts → asserted worker path / `last_worker_status` / answer-contains checks, run against the LangGraph in `agent/agent/graph.py`. This is the regression net for prompt and routing changes.
- **CI** — GitHub Actions matrix over the services: `ruff check`, `ty`, `uv run pytest` per service. Green CI is the merge gate.
- **Pre-commit** — `ruff format` + `ruff check --fix` + `ty` hooks for fast local feedback.
- **Verify command** — a top-level script (`make verify` or `uv run` task) that runs lint+type+test for a single service, so the agent has one command to loop on.

**Touchpoints:** every `<service>/pyproject.toml`, new `.github/workflows/ci.yml`, new `.pre-commit-config.yaml`, new `agent/tests/eval/` suite.
**Risks:** agent eval is non-deterministic (LLM outputs) — assert on structure (worker path, status, presence of artifacts) rather than exact prose; pin a cheap model for eval runs.

---

## Phase 1 — FastAPI BFF + shared ORM data layer *(was #1 backend, #9)*

**Goal:** move every in-process Streamlit DB read behind an HTTP API so the JS frontend has a typed data source, with **all SQL wrapped in ORM**.

**Approach**
- **New `bff` service** (FastAPI, port `8005`) — typed JSON endpoints mirroring what the dashboard pages need (indicator slices, Yahoo series, clustering inputs, news search proxy). Frontend calls only the BFF.
- **Shared models package** — extract the SQLAlchemy `Mapped` models into a small installable package (e.g. `packages/db_models/`) imported by `bff`, `downloader_*`, and the `agent`, so the schema lives in one place. Today reads are split: `app` uses `connectorx` for bulk Polars; the `agent` SQL worker uses raw `text()`. The BFF replaces the `app`'s `connectorx`/`postgres_client.py` read path with ORM `select()` queries returning Polars/JSON; the agent's dynamic `text()` stays (LLM-generated SQL).
- **Endpoint parity** — port the logic in `app/core/api_client.py` (forecaster/clustering/agent proxies) and the per-page data prep (`app/pages/page_utils.py`, `app/core/page_helpers.py`) into BFF routers. Reuse the existing forecaster/clustering/agent services unchanged behind the BFF where it just proxies.
- **#9 — drop `wbgapi`** — rewrite the World Bank extractor (`downloader_general/src/extractors/world_bank.py`) and `downloader_extra/client_wb.py` to call `https://api.worldbank.org/v2/...` directly with `httpx`; remove `wbgapi` from both `pyproject.toml`s. Do this here because it de-risks ingestion before Phase 3 piles on new sources. Keep the output schema identical (validated by Phase 0 tests).

**Touchpoints:** new `bff/`, new `packages/db_models/`, `docker-compose.yaml` + `config.yaml` (new service block + port — remember these two are **not** auto-synced), `downloader_general/src/extractors/world_bank.py`, `downloader_extra/client_wb.py`.
**New deps:** `httpx` already present in downloaders; BFF gets `fastapi`/`uvicorn`/`sqlalchemy`/`polars`.
**Risks:** behavioural parity with the current Streamlit reads — snapshot-test BFF JSON against current page outputs before cutover.

---

## Phase 1b — Agent two-model routing *(was #2)*

**Goal:** cut cost/latency by running cheap calls on a light model and reserving the flagship for hard reasoning.

**Approach**
- **config.yaml** — add `shared.openai_llm_model_heavy` and `shared.openai_llm_model_light` (keep `openai_llm_model` as an alias/default for back-compat).
- **graph.py** — instantiate two `ChatOpenAI` clients. Assign:
  - *Light:* `GuardrailAgent` (when it escalates), `sql_agent`, `table_agent`, `chat_agent`.
  - *Heavy:* `MacroSupervisorAgent` (planning + the final streamed draft), `plotly_agent`, and the `/plots/interpret` vision endpoint.
- **usage.py** — `UsageTracker` already attaches per-call; ensure it records which model served each call so the Token Usage view can break down spend by tier.

**Touchpoints:** `_container_data/config.yaml`, `agent/agent/graph.py`, `agent/agent/usage.py`, `agent/main.py` (vision endpoint).
**Risks:** light model underperforming on `sql_agent` — gate behind the Phase 0 agent-eval harness; make the per-node assignment overridable in config so it's tunable without code changes.

---

## Phase 2 — Next.js + ECharts frontend *(was #1 frontend)*

**Goal:** replace the Streamlit `app` with a modern, responsive, high-performance Next.js frontend, **preserving the existing page structure**, with **runtime dynamic theming** (re-enabling what v0.6 removed).

**Approach**
- **New `frontend` service** (Next.js, port `3000`) consuming the Phase-1 BFF. Charts via Apache ECharts (`echarts-for-react` or native). SSR/RSC for fast first paint.
- **Strangler cutover** — a reverse proxy (nginx or Next rewrites) routes already-migrated routes to Next.js and the rest to the still-running Streamlit `app`. Migrate page-by-page in the existing order (`01_basic_indicators` → … → `15_news`), retiring each Streamlit page as its Next.js equivalent reaches parity. Streamlit is deleted only when the last page is ported.
- **Dynamic theming** — expose the `themes.yaml` palette via a BFF `/theme` endpoint; map semantic tokens (`positive`, `negative`, `sector_*`, `diverging_*`, `sequential_*`, `selected_marker`, …) to CSS variables + a registered ECharts theme. Add a runtime theme switcher in the UI. Port the `KeyError`-on-missing-token strictness so custom themes still fail loud.
- **Componentise the config-driven pages** — the indicator pages (`01`–`10`) are already config-driven (`render_page_from_config` over `world_bank_download_config.json`); reproduce that as a generic React `<IndicatorPage>` driven by the same config served from the BFF.

**Touchpoints:** new `frontend/`, reverse-proxy config, `docker-compose.yaml` (+ Node service), BFF `/theme` + config endpoints.
**New deps:** Node toolchain in a new Dockerfile; `next`, `react`, `echarts`, `echarts-for-react`.
**Risks:** dual-maintenance window (two frontends live); choropleth/wordcloud/embedding-map parity (the News page's projection panels and Plotly maps are the hardest to port). Tackle simple indicator pages first, complex viz pages last.

---

## Phase 3 — Data expansion *(was #9, #5, #3, #4, #6, #7)*

All sources **extend the existing downloaders**: new modules under `downloader_general/src/extractors/`, new on-demand routes in `downloader_extra`, new `_configs/*.json` files, new rows in `database_schema.yaml`, and new tables granted automatically to the read-only role by the idempotent bootstrap.

- **#9 World Bank → httpx** — *(done in Phase 1; listed here for completeness)*.
- **#5 More Yahoo Finance tickers** — append to `_configs/yahoo_download_config.json` (with sector grouping for the choropleth/sector tokens). Pure config + re-ingest; lowest effort.
- **#3 FRED (US states)** — new `extractors/fred.py` (FRED API, **needs `FRED_API_KEY` in `.env`**); new `_configs/fred_download_config.json` (state-level series: GDP, unemployment, etc.). New Next.js page **US States** with a US-state choropleth (Plotly/ECharts built-in US GeoJSON).
- **#4 Eurostat (EU NUTS regions)** — new `extractors/eurostat.py` (Eurostat REST / JSON-stat, **keyless**); new `_configs/eurostat_download_config.json`. New page **EU Regions** with a NUTS-2/3 choropleth (GISCO GeoJSON boundaries).
- **#6 Crypto (Binance)** — new `extractors/binance.py` (public klines REST, **keyless** for historical OHLCV); new `_configs/binance_download_config.json`. New page **Crypto** (price history, mirrors the Yahoo Finance page).
- **#7 News RAG → GDELT** — replace the `github` news extractor with `extractors/gdelt.py`. Query GDELT (DOC 2.0 / GKG) filtered by the topics in `news_download_config.json` rather than the full firehose; embed with the same model into Qdrant. Remove the GitHub-repo dependency.

**Touchpoints:** `downloader_general/src/extractors/`, `downloader_extra/` (on-demand variants), `_configs/`, `database_schema.yaml`, new frontend pages, `.env`/`config.yaml` for `FRED_API_KEY`.
**Risks:** GDELT volume + dedup/rate-limits; sub-national GeoJSON sourcing and join keys (FIPS for US states, NUTS codes for EU). Country-level pages are unaffected.

---

## Phase 4 — Ops: monitoring + backups *(was #11, #8)*

**#11 — Monitoring out of the dashboard**
- Add **Prometheus + Grafana + cAdvisor** services to the stack. Each FastAPI service exposes `/metrics` (`prometheus-fastapi-instrumentator`); add `postgres_exporter`; Qdrant exposes `/metrics` natively; cAdvisor covers per-container CPU/mem/net.
- Provision Grafana dashboards (service health, latency, container stats) as code.
- **Remove** `app/core/monitoring.py`, `app/pages/18_monitoring.py`, and the read-only `/var/run/docker.sock` bind-mount from the `app` service — the bespoke health/Docker-socket probing is replaced by the standard stack.

**#8 — rclone backups**
- New scheduled `backup` service (cron-style). Each run: `pg_dump` the Postgres DB + call the Qdrant snapshot API (`POST /collections/{c}/snapshots`), then `rclone copy` both artifacts to a remote configured in `.env` (`rclone.conf` mounted). Retention prune (keep N daily / M weekly).
- Commit a **restore script** + documented restore procedure.

**Touchpoints:** `docker-compose.yaml` (prometheus/grafana/cadvisor/postgres_exporter/backup services), each service's `main.py` (`/metrics`), new `monitoring/` (Grafana provisioning) and `backup/` dirs, `.env` (rclone remote), `config.yaml`.
**Risks:** secrets handling for the rclone remote; ensuring snapshots are consistent (snapshot Qdrant and dump Postgres close in time).

---

## Phase 5 — GPU/ML consolidation *(was #12)*

**Goal:** GPU-driven ML unified under one **NVIDIA Triton** runtime; **GPU is a hard requirement** for this deployment profile.

**Approach**
- **RAPIDS** — port clustering (`clustering`: KMeans/DBSCAN) to **cuML**, and DataFrame ops to **cuDF** where they help. GPU-native XGBoost in the forecaster.
- **Triton** — stand up a `triton` service backed by a model repository. Migrate model serving out of the per-request Python wrappers into Triton (ONNX where exportable, Python backend otherwise). Chronos (already GPU) and XGBoost are natural first movers.
- **Consolidation** — `forecaster` and `clustering` become thin FastAPI fronts that call Triton, or are folded into a single inference gateway.

**Touchpoints:** `forecaster/`, `clustering/`, new `triton/` model repo, `docker-compose.yaml` (`deploy.resources.reservations.devices` GPU block, already used by `forecaster`).
**⚠ Open technical issue (see below):** the classical statistical models (`auto_arima`, `arima`, `sarima`, `prophet`, `moving_average`) have **no GPU implementation**. "GPU required, drop CPU paths" needs a carve-out decision — keep them as Triton Python-backend (CPU) models, or drop them in favour of GPU-capable forecasters (Chronos/XGBoost/neural).

---

## Cross-cutting concerns

- **`config.yaml` ↔ `docker-compose.yaml` drift** — every new service (bff, frontend, prometheus, grafana, cadvisor, backup, triton, postgres_exporter) means a port + bind-mount declared in **both** files; they are not auto-synced. Proposed ports: `bff 8005`, `frontend 3000`, `grafana 3001`, `prometheus 9090`, `cadvisor 8080`, `triton 8100/8101/8102` — reconcile against the existing `8000`–`8004` to avoid collisions (Triton's defaults `8000/8001/8002` clash with `agent`/`forecaster`/`clustering`).
- **Container count** — grows from 9 to ~16. Consider Compose `profiles` (e.g. `monitoring`, `gpu`) so GPU-less / minimal runs stay lean.
- **New secrets/env** — `FRED_API_KEY`, rclone remote creds, the heavy/light model names. Extend `.env.example` and the README's required-vars table.
- **Schema source of truth** — the shared `db_models` package (Phase 1) plus `database_schema.yaml` must stay in sync; the agent's SQL worker grounds on the YAML.

---

## Assumptions & open questions

These were defaulted while writing this plan — flag any to revisit:

1. **GDELT scope (#7):** filtered topic-based subset (DOC 2.0/GKG queried by existing news topics), **not** the full firehose. Confirm acceptable corpus size/freshness.
2. **API keys:** FRED needs a key; Eurostat and Binance historical market data are assumed keyless. Confirm no Binance auth is needed for the chosen endpoints.
3. **Sub-national maps:** US states via built-in FIPS GeoJSON; EU via Eurostat GISCO NUTS GeoJSON. Confirm the NUTS level (2 vs 3).
4. **Triton CPU carve-out (#12):** decision needed on the classical CPU-only forecasters (see Phase 5 ⚠).
5. **Two-model assignment (#2):** the light/heavy split above is the starting point; final per-node mapping is config-tunable.
6. **"Structure left the same" (#1):** interpreted as *preserve existing pages/navigation during migration*; new data sources (#3/#4/#6) legitimately **add** new pages on top.
7. **Frontend auth:** none assumed (same as today's open dashboard). Add if the BFF will be internet-exposed.
