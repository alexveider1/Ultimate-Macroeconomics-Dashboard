# Roadmap (archived detailed plan)

> Moved here from the old root-level `PLAN.md`. The flat, actionable backlog now lives in
> [`../TODO.md`](../TODO.md); this file keeps the rationale, phase sequencing, and per-phase
> touchpoints. Kept under `.claude/` so it stays available to Claude Code without loading into
> every session's base context.

---

# Development Plan

> Detailed roadmap for the next major iteration of **Ultimate Macroeconomics Dashboard**.
> This supersedes the earlier 12-line wishlist; every original item is tagged `(was #N)` for traceability.
> Architectural choices below were locked in two planning discussions — see **Guiding decisions**.

---

## Guiding decisions (locked)

| Topic | Decision |
| ----- | -------- |
| Frontend framework | **React + Next.js + Apache ECharts** |
| Frontend cutover | **Strangler, built LAST** — run Next.js alongside Streamlit, migrate page-by-page, retire Streamlit last; the whole frontend phase is sequenced after all backend/data/ops/ML work so it is built once against a final backend |
| Data/API layer | **New Python FastAPI BFF** ("backend-for-frontend") reusing the existing SQLAlchemy `Mapped` models; all reads wrapped in ORM. Built **early but additive** — Streamlit is *not* rewired to it and keeps `connectorx` until retired |
| Config management | **`pydantic-settings`** — one shared typed `Settings` model loads `config.yaml` + `.env` + env (precedence `env > .env > yaml`), validated at boot; replaces the ~10× duplicated `yaml.safe_load + load_dotenv` boilerplate. Hydra/Dynaconf rejected (overkill / less type-safe) |
| Secrets | **`SecretStr` + Docker Compose `secrets:`** — pydantic-settings `SecretStr` (typed, masked in logs) for every secret; Compose `secrets:` file-mounts (`/run/secrets/...`) for the crown jewels (DB password, `OPENAI_API_KEY`). No new infra; same workstream as config |
| Agent inference | **Two models** — *light* for guardrail + routine workers (`sql`/`table`/`chat`), *heavy* for supervisor planning, final synthesis, `plotly`/vision |
| New data sources | **Extend the existing downloaders** (`downloader_general` one-shot + `downloader_extra` on-demand) with new extractor modules |
| Sub-national data | FRED (US states), Eurostat (EU NUTS regions), crypto → **new dedicated pages with sub-national choropleths**; existing country pages unchanged |
| Observability | **Self-hosted Grafana LGTM** — instrument every service with OpenTelemetry (structured logs + traces); OTel Collector → **Loki** (logs) + **Tempo** (traces) + **Prometheus** (metrics) + cAdvisor, visualised in Grafana. Replaces the bespoke plaintext `app/core/app_logging.py` and the in-dashboard Monitoring page. No SaaS |
| ML serving | **Full NVIDIA Triton consolidation, GPU required** — forecaster + clustering models unified, RAPIDS (cuDF/cuML) for data/clustering. ⚠ Classical CPU-only models need a carve-out decision at Phase 5 start (see Phase 5) |
| Dev workflow | **Tests + coverage gates + agent eval harness** — built *first* as the correctness safety net |
| Backups | **Nightly `pg_dump` + Qdrant snapshots** pushed to an `rclone` remote, with retention + restore script |

> **Status (v0.14):** Phase 1 *config* + *secrets* and Phase 2b *agent two-model routing* are implemented — pragmatically. Because each service is its own container with its own `pyproject.toml` (no shared package), config/secrets use **per-service** Pydantic `config.py` + `pydantic-settings` `settings.py` rather than one shared `Settings`, and least privilege is enforced by **per-service `environment:` scoping in `docker-compose.yaml`** (each container gets only the secrets it uses) instead of Docker `secrets:` file-mounts. `SecretStr` log-masking and a BFF-wide shared model remain open follow-ups. From **Phase 3**, the Yahoo Finance universe (#5), Binance crypto (#6 — ingestion + a Crypto dashboard page), FRED US-state indicators (#3 — ingestion + a "Regional Statistics" dashboard page), and the WB→httpx swap (#9) have shipped; on-demand ingestion (`downloader_extra` + the agent's `downloader_agent`) is now **multi-source** (WB indicator / Yahoo ticker / Binance pair / FRED state indicator / Eurostat NUTS-2 dataset). Eurostat (#4 — ingestion + a second "Regional Statistics" page) has now shipped too. News-RAG expansion (#7) has shipped — GDELT was dropped (no full article text) and superseded by two new Qdrant sources, **Actually Relevant** (curated-news API) and **World Bank documents** (WDS `txturl` → chunked). **Phase 2 (BFF)** has now shipped too — a read-only `bff` service (port 8005) with ORM read routers + Qdrant news search + forecaster/clustering/agent proxies (the shared `packages/db_models/` package is the one deferred carve-out; see Phase 2 below). Phase 4's **backups** half shipped (the `backup` service). **Phase 5 (GPU/Triton) has now shipped** — a new `triton` service (NVIDIA Triton Inference Server) hosts **all** forecasting + clustering inference as python-backend models; the CPU-only carve-out was resolved by keeping the classical models (ARIMA family, Prophet, moving-average) on Triton's python CPU backend while Chronos + XGBoost run on CUDA and clustering/dim-reduction use RAPIDS cuML (`forecaster`/`clustering` are now GPU-free adapters forwarding over gRPC; GPU is required on `triton`). Phase 4's **observability** half plus Phase 6 (frontend) remain open.

---

## Roadmap at a glance

Sequencing principle (decided): **build the safety net and foundations before the big rewrites, and build the frontend last.** Each phase is a verifiable milestone; later phases assume the foundations/BFF/test harness from earlier ones.

| Phase | Theme | Original items | Depends on |
| ----- | ----- | -------------- | ---------- |
| **0** | Claude-Code dev loop: tests, coverage, CI, agent eval | #10 | — |
| **1** | Foundations: pydantic-settings config + SecretStr/Docker secrets + OTel structured-logging instrumentation | #2 (partial) | 0 |
| **2** | FastAPI BFF + shared ORM data layer (additive) + WB→httpx | #1 (backend), #9 | 0, 1 |
| **2b** | Agent two-model routing *(parallelizable quick win)* | #2 | 1 |
| **3** | Data expansion: Yahoo, FRED, Eurostat, crypto, GDELT | #5, #3, #4, #6, #7 | 2 (ingestion + BFF) |
| **4** | Ops: Prometheus/Loki/Tempo/Grafana monitoring + rclone backups | #11, #8 | 1 (instrumentation), 2 |
| **5** | GPU/ML consolidation: RAPIDS + NVIDIA Triton | #12 | 0, 2 |
| **6** | Next.js + ECharts frontend (strangler) + dynamic theming — **LAST** | #1 (frontend) | 2, 3 |

Phases 2b, 3, and 4 can proceed in parallel with each other (backend/ops only). The frontend (Phase 6) is gated on the BFF (Phase 2) and the new data sources/pages (Phase 3) being stable, so it is built once against the final surface.

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

## Phase 1 — Foundations: config + secrets + logging *(was #2 partial)*

**Goal:** kill the duplicated, untyped config/secrets/logging boilerplate before the big rewrites — every later phase inherits typed config, masked secrets, and OpenTelemetry instrumentation. Cross-cutting; touches every service.

**Approach**
- **Config (`pydantic-settings`)** — add a shared `Settings` model (per-service subclass) in a minimal `packages/config/` package (later co-located with the `db_models` package from Phase 2). Sources via `pydantic_settings.YamlConfigSettingsSource` (`config.yaml`) + `DotEnvSettingsSource` (`.env`) + env, precedence `env > .env > yaml`. One `get_settings()` accessor per service, validated at import/boot (fail fast). Replace every `yaml.safe_load(...) + load_dotenv(...)` call site (`app/core/postgres_client.py`, `qdrant_client.py`, `plotting.py`, `theming.py`, `monitoring.py`, `token_usage_store.py`; `agent/main.py`; `forecaster/main.py`; `clustering/main.py`; `downloader_general/main.py`; `downloader_extra/main.py`; `python_sandbox/main.py`) with typed `settings.*` access. Add a boot-time assertion that `config.yaml` ports match `docker-compose.yaml` where feasible (mitigates the documented compose↔yaml drift; not a full auto-sync).
- **Secrets (`SecretStr` + Docker secrets)** — all secret fields typed `pydantic.SecretStr` so they never render in the new structured logs. Move DB password and `OPENAI_API_KEY` to Compose `secrets:` (file-mounted `/run/secrets/...`); `Settings` reads the `*_FILE` convention with `.env` fallback for local dev. Update `docker-compose.yaml` + `.env.example`.
- **Logging (OTel structured logs + traces)** — replace the `app/core/app_logging.py` plaintext formatter with structured/OTLP logging; add an equivalent shared helper for the other services. Auto-instrument FastAPI, HTTPX, SQLAlchemy. Highest-value target: `agent/agent/graph.py` — a span per worker + per LLM call carrying token/latency attributes (complements `agent/agent/usage.py`'s `UsageTracker`; can back the Token Usage page). In this phase ship OTLP to stdout / a local collector so instrumentation is verifiable before the Phase-4 backend exists.
  - **LLM-tracing slice — DONE (self-hosted Langfuse).** The agent/LLM half of this landed early via a self-hosted **Langfuse** stack (`langfuse_web` + worker + ClickHouse/Redis/MinIO/Postgres in `docker-compose.yaml`; SDK `langfuse==4.13.0`). The graph-level `langfuse.langchain.CallbackHandler` in `MacroAgentGraph.astream_events` already yields a span per worker + per LLM call with token/latency/cost — exactly the "highest-value target" above — plus a scoped manual generation for the vision endpoint and `langfuse.openai`-patched embedding traces in `downloader_general` / `bff`. Gated by `config.yaml` `langfuse.enabled` + `.env` keys; each service carries its own `tracing.py` (no shared package yet). The **broader OTel logs + non-LLM traces + Grafana LGTM backend (#11) remain**; Langfuse can either stay the dedicated LLM-observability plane or later export OTLP into Tempo.

**Touchpoints:** new `packages/config/`, every service `main.py` / `app/core/*.py` config + logging call sites, `agent/agent/graph.py`, `docker-compose.yaml` (`secrets:`), `_container_data/.env.example`.
**New deps:** `pydantic-settings`, `opentelemetry-*` SDK + instrumentation packages per service.
**Risks:** instrumentation overhead in the hot path (sandbox/agent) — keep exporters async/batched; assert no secret leaks via the verification grep below.

---

## Phase 2 — FastAPI BFF + shared ORM data layer *(was #1 backend, #9)*

> **Status:** ✅ **shipped** (the BFF service; `#9 wbgapi→httpx` was pulled forward earlier). The one carve-out is the **shared `packages/db_models/` package**, which was **deliberately deferred** — per the repo's per-service convention (each container owns its `pyproject.toml`/`Dockerfile`/build context, no shared package) the BFF carries its own read-subset `schema.py` copy. Extracting the shared package (rewiring `agent` + `downloader_*` + three Dockerfiles) plus the pre-cutover JSON snapshot tests remain open follow-ups (tracked in `TODO.md`).

**Goal:** give the future JS frontend a typed HTTP data source, with **all SQL wrapped in ORM** — built **additively** so the live Streamlit app is untouched.

**What shipped** — new read-only `bff/` service (port `8005`, flat layout): ORM `select()` read routers mirroring `app/core/postgres_client.py` (`worldbank` / `yahoo` / `crypto` / `fred` / `eurostat`), a Qdrant `news` router (collections + browse + `POST /news/search` semantic search reusing the agent's RAG merge logic + `ALWAYS_SEARCH_COLLECTION_PREFIXES`), and thin proxies to the **existing** forecaster (`/predict`) / clustering (`/cluster`) / agent (`/models`, `/plots/interpret`, `/chat/stream` SSE) services via one shared `httpx.AsyncClient`. Typed JSON contract in `models.py` (decoupled from the ORM models); least-privilege secrets (read-only `POSTGRES_LLM` role + Qdrant key + `OPENAI_API_KEY` for embeddings only). Tests: ORM routers over a `testcontainers` Postgres, news over fake async clients, proxies over `httpx.MockTransport`. Streamlit is **not** rewired.

**Approach**
- **New `bff` service** (FastAPI, port `8005`) — typed JSON endpoints mirroring what the dashboard pages need (indicator slices, Yahoo series, clustering inputs, news search proxy). Frontend (Phase 6) calls only the BFF.
- **Shared models package** — extract the SQLAlchemy `Mapped` models into a small installable package (e.g. `packages/db_models/`) imported by `bff`, `downloader_*`, and the `agent`, so the schema lives in one place. The BFF uses ORM `select()` queries returning Polars/JSON. **Streamlit is NOT rewired** — it keeps `connectorx`/`postgres_client.py` until retired in Phase 6 (avoids throwaway rewiring). The agent's dynamic `text()` stays (LLM-generated SQL).
- **Endpoint parity** — port the logic in `app/core/api_client.py` (forecaster/clustering/agent proxies) and the per-page data prep (`app/pages/page_utils.py`, `app/core/page_helpers.py`) into BFF routers. Reuse the existing forecaster/clustering/agent services unchanged behind the BFF where it just proxies.
- **#9 — drop `wbgapi`** — ✅ **done** (pulled forward, ahead of the rest of Phase 2). The World Bank extractor (`downloader_general/src/extractors/world_bank_download.py`) now drives a new async `httpx` client `downloader_general/src/utils/wb_client.py`, and `downloader_extra/client_wb.py` drives its own trimmed copy `downloader_extra/wb_client.py`; both call `https://api.worldbank.org/v2/...` directly. `wbgapi` (and its orphaned `tabulate`) removed from both `pyproject.toml` + `uv.lock`. Output schema kept identical — aggregates dropped (`skipAggs` parity), nulls kept, rich series metadata preserved via the advanced `/metadata` endpoint with a `/indicator/{id}` fallback. Validated by new `tests/test_wb_client.py` suites in both services.

**Touchpoints:** new `bff/`, new `packages/db_models/`, `docker-compose.yaml` + `config.yaml` (new service block + port — these two are **not** auto-synced), `downloader_general/src/extractors/world_bank.py`, `downloader_extra/client_wb.py`.
**Risks:** behavioural parity — snapshot-test BFF JSON against current Streamlit page outputs before the Phase-6 cutover.

---

## Phase 2b — Agent two-model routing *(was #2)*

**Goal:** cut cost/latency by running cheap calls on a light model and reserving the flagship for hard reasoning.

**Approach**
- **config.yaml** — add `shared.openai_llm_model_heavy` and `shared.openai_llm_model_light` (keep `openai_llm_model` as an alias/default for back-compat). Exposed through the Phase-1 `Settings` model.
- **graph.py** — instantiate two `ChatOpenAI` clients. Assign:
  - *Light:* `GuardrailAgent` (when it escalates), `sql_agent`, `table_agent`, `chat_agent`.
  - *Heavy:* `MacroSupervisorAgent` (planning + the final streamed draft), `plotly_agent`, and the `/plots/interpret` vision endpoint.
- **usage.py** — `UsageTracker` already attaches per-call; ensure it records which model served each call (now also surfaced as an OTel span attribute from Phase 1) so the Token Usage view can break down spend by tier.

**Touchpoints:** `_container_data/config.yaml`, `agent/agent/graph.py`, `agent/agent/usage.py`, `agent/main.py` (vision endpoint).
**Risks:** light model underperforming on `sql_agent` — gate behind the Phase 0 agent-eval harness; make the per-node assignment overridable in config.

---

## Phase 3 — Data expansion *(was #5, #3, #4, #6, #7)*

All sources **extend the existing downloaders**: new modules under `downloader_general/src/extractors/`, new on-demand routes in `downloader_extra`, new `_configs/*.json` files, new rows in `database_schema.yaml`, new BFF endpoints (Phase 2), and new tables granted automatically to the read-only role by the idempotent bootstrap. New pages are added to Streamlit in the interim and ported to the frontend in Phase 6.

- **#9 World Bank → httpx** — ✅ **done** (Phase 2; listed here for completeness).
- **#5 More Yahoo Finance tickers** — ✅ **done** (v0.12, 50→84 tickers in `_configs/yahoo_download_config.json`).
- **#6 Crypto (Binance)** — ✅ **done** (v0.13): `extractors/binance_download.py` (public klines REST, keyless), `_configs/binance_download_config.json`, `binance_metadata` / `binance_historical_prices`, and a **Crypto** dashboard page mirroring the Yahoo page.
- **On-demand ingestion → multi-source** — ✅ **done** (v0.14, extended for FRED): `downloader_extra`'s unified `POST /ingest` (`source=worldbank|yahoo|binance|fred`) + the agent's source-aware `downloader_agent`. WB uses the `database_indicators` master catalogue; Yahoo/Binance/FRED have **no catalogue**, so the agent infers the ticker / full pair symbol / representative single-state series id and `downloader_extra` validates it live.
- **#3 FRED (US states)** — ✅ **done**: `extractors/fred_download.py` + async `utils/fred_client.py` (GeoFRED `series/group` + `regional/data`, **needs `FRED_API_KEY`**); `_configs/fred_download_config.json` (36 state-level series: GDP, unemployment, income, housing, sector employment, …). New `fred` schema group — `states` / `state_indicators` / `state_indicator_values` — and a **"Regional Statistics" → United States (FRED)** page (`19_fred_regional.py`) with a US-state choropleth, top/bottom rankings, and multi-state trends. Observations map to states by FIPS `code` (not series-id prefix). Also wired into the AI Analyst (sql_agent grounding + on-demand `source=fred`).
- **#4 Eurostat (EU NUTS regions)** — **SHIPPED.** `extractors/eurostat_download.py` + `utils/eurostat_client.py` (Eurostat dissemination JSON-stat, **keyless**, row-major stride decoding); `_configs/eurostat_download_config.json` (25 NUTS-2 indicators across Economy / Labour / Demography / Innovation); `eurostat` schema group (`eurostat_regions` / `eurostat_indicators` / `eurostat_indicator_values`). New `20_eurostat_regional.py` **EU (Eurostat)** page — the second page under the **"Regional Statistics"** nav group — with a **NUTS-2** choropleth (`build_nuts_choropleth` over a bundled GISCO `NUTS_RG_60M_2021_4326_LEVL_2` GeoJSON in `_configs/`, shared read-only with the downloader); reuses the region-agnostic `build_region_ranking_bar` / `build_region_trend_lines` helpers. On-demand (`source=eurostat`) + `sql_agent` awareness wired into the AI Analyst.
- **#7 News RAG expansion** — ✅ **shipped (GDELT dropped).** GDELT was investigated and rejected: its DOC 2.0 / GKG feeds expose no full article text, only metadata. Instead **two** new sources were added alongside the `github` (Webhose) extractor, both embedding with the same model into Qdrant via the shared `src/core/qdrant_uploader.py` mixin and keeping the news payload shape (so `15_news.py` + the agent read them unchanged): (a) **Actually Relevant** (`actually_relevant_download.py` + `actually_relevant_client.py`) — a keyless curated-news JSON API whose items are AI analysis (not source bodies); each point embeds a composed analysis document and stories bucket into 5 `actually_relevant_<macro>` collections via the `/api/issues` taxonomy. (b) **World Bank documents** (`world_bank_articles_download.py` + `wds_client.py`) — per ~30 configured macro queries, the top-N WDS docs' `txturl` plain text (no docling) is token-chunked into one point per chunk, one collection per query (`world_bank_<slug>`). The agent RAG search always folds in the new-source collections (`ALWAYS_SEARCH_COLLECTION_PREFIXES`).

**Touchpoints:** `downloader_general/src/extractors/`, `downloader_extra/`, `_configs/`, `database_schema.yaml`, BFF routers, `Settings`/secrets for `FRED_API_KEY`.
**Risks:** GDELT volume + dedup/rate-limits; sub-national GeoJSON sourcing and join keys (FIPS for US states, NUTS codes for EU). Country-level pages are unaffected.

---

## Phase 4 — Ops: monitoring + backups *(was #11, #8)*

**#11 — Container/service monitoring (metrics + health)** — ✅ **SHIPPED (via Netdata, scope narrowed).**
- Instead of the full Grafana **LGTM** stack, the container-metrics + service-health slice landed as a single self-hosted **Netdata** container (`netdata`, `netdata/netdata:v2.10.3`, UI `:19999`). It auto-discovers every Compose container's CPU/RAM/mem/net + host metrics from the host cgroups/`proc`/`sys` + docker socket (bind-mounted read-only), and probes each service's health with `go.d` **httpcheck** (HTTP `/health` endpoints) + **portcheck** (Postgres + Langfuse backing stores) jobs, configured under `_container_data/netdata/go.d/`.
- Independent observability (mirrors the Langfuse precedent): on the default bridge so it reaches internal-only services by name, **no `depends_on`** in either direction, telemetry off (`DO_NOT_TRACK=1`, no Cloud claim), no new `.env` secrets; non-secret knobs in `config.yaml` `monitoring:`; history persisted in `netdata_lib` / `netdata_cache` volumes.
- **Removed** `app/core/monitoring.py`, `app/pages/18_monitoring.py`, and the read-only `/var/run/docker.sock` bind-mount from the `app` service — the bespoke health/Docker-socket probing is replaced by Netdata (which now owns the socket mount).
- **Still open (deferred):** the log/trace-aggregation half — **OTel Collector + Loki + Tempo (+ Prometheus/Grafana)** for OTLP structured logs + non-LLM distributed traces. Netdata covers metrics + health but not centralized log/trace search; that remains future work (see TODO "Observability — logs & traces").

**#8 — rclone backups** — ✅ **SHIPPED.**
- New `backup/` service — a long-running **Python** scheduler (simple `interval_minutes` loop, `run_on_start`, SIGTERM-aware; no OS cron). Each run: `pg_dump -Fc` the Postgres DB + take a **full-storage** Qdrant snapshot (`POST /snapshots` → download → `DELETE`), then `rclone copy` both artifacts to the configured remote and prune the remote `--min-age <retention_days>d`.
- **Off by default** (`backup.enabled: false` in `config.yaml`); when disabled the container logs + idles (no restart-loop under `restart: unless-stopped`). Cloud creds stay out of `config.yaml` — they live in a gitignored `_container_data/backup/rclone.conf` (`.example` committed); the container reuses the existing `POSTGRES_*` + `QDRANT__SERVICE__API_KEY` env, so **no new `.env` vars**.
- Committed **`restore.py`**: automates `pg_restore --clean --if-exists` and downloads + documents the Qdrant full-snapshot recovery (a live-instance full-snapshot restore needs a boot-time `--storage-snapshot`, so it's a documented manual step).
- Image adds `postgresql-client-18` (PGDG apt, matches the server major) + `rclone` (multi-stage `COPY --from=rclone/rclone`). Chose **full-storage** over per-collection snapshots (single artifact) and **interval** over cron (matches the Python-everywhere convention).

**Touchpoints:** *(shipped)* `docker-compose.yaml` (`netdata` + `backup` services), `_container_data/netdata/go.d/` (httpcheck/portcheck), `config.yaml` (`monitoring:` + `backup:`), removal of `app/core/monitoring.py` / `app/pages/18_monitoring.py` / the `app` docker.sock mount, `backup/` dir + rclone secret. *(still open, logs/traces half)* OTel-collector/loki/tempo services, each service's `main.py` (OTLP export), new `monitoring/` provisioning dir.
**Risks:** consistent snapshots (snapshot Qdrant and dump Postgres close in time); rclone remote secret handling; Netdata cgroup access on WSL2 (host `/proc`+`/sys` mounts must populate per-container charts).

---

## Phase 5 — GPU/ML consolidation *(was #12)* — **SHIPPED**

**Goal:** GPU-driven ML unified under one **NVIDIA Triton** runtime; **GPU is a hard requirement** for this deployment profile.

**Status (shipped):** the `triton` service (from `triton/`, `FROM nvcr.io/nvidia/tritonserver:*-py3`) hosts every forecasting + clustering model as a **python-backend** model. The carve-out below was resolved by **keeping** the classical CPU-only models on Triton's python CPU backend (`KIND_CPU`: ARIMA family, Prophet, moving-average) while Chronos (Torch/T5, preloaded) + XGBoost (`device="cuda"`) run on `KIND_GPU`, and clustering/dim-reduction use **cuML** (KMeans/DBSCAN/PCA/t-SNE/UMAP, `KIND_GPU`) with a scikit-learn CPU fallback for the algorithms cuML lacks. `forecaster`/`clustering` became **thin FastAPI adapters** forwarding to Triton over gRPC (a single `TYPE_STRING` JSON in/out contract; `max_batch_size 0` since every model fits-per-request). Native ONNX/TensorRT export was **deferred** (fragile for Chronos' enc/dec + sampling, marginal at short horizons) — python backend everywhere. Ported maths lives in `triton/common/umd_common` (CPU-unit-tested under `triton/tests/`). Triton's ports stay **internal** (no host publish), sidestepping the documented `8000/8001/8002` clash with `agent`/`forecaster`/`clustering`.

**Approach**
- **RAPIDS** — port clustering (`clustering`: KMeans/DBSCAN) to **cuML**, and DataFrame ops to **cuDF** where they help. GPU-native XGBoost in the forecaster. Keep **Polars** as the I/O / data-prep layer; convert Polars→Arrow→GPU array only at the cuML boundary (cuDF/cuGraph not required by current data sizes).
- **Triton** — stand up a `triton` service backed by a model repository. Migrate model serving out of the per-request Python wrappers into Triton (ONNX where exportable, Python backend otherwise). Chronos (already GPU) and XGBoost are natural first movers.
- **Consolidation** — `forecaster` and `clustering` become thin FastAPI fronts that call Triton, or are folded into a single inference gateway.

**⚠ Open carve-out (resolve at phase start):** the project's "inference" is mostly **online fit-per-request** (`auto_arima` refits per call; `arima`/`sarima`/`prophet`/`moving_average`/KMeans/DBSCAN train on the request's data), which is *not* Triton's serve-pretrained sweet spot — the genuine Triton wins are Chronos and any future self-hosted embedding model. Decide: keep the classical CPU-only models as Triton Python-backend (CPU) models, or drop them in favour of GPU-capable forecasters (Chronos/XGBoost/neural).

**Touchpoints:** `forecaster/`, `clustering/`, new `triton/` model repo, `docker-compose.yaml` (`deploy.resources.reservations.devices` GPU block, already used by `forecaster`).

---

## Phase 6 — Next.js + ECharts frontend *(was #1 frontend)* — **LAST**

**Goal:** replace the Streamlit `app` with a modern, responsive Next.js frontend, **preserving the existing page structure**, with **runtime dynamic theming** (re-enabling what v0.6 removed). Built last so it targets the final BFF surface (all Phase 2–3 endpoints) in one pass.

**Approach**
- **New `frontend` service** (Next.js, port `3000`) consuming the Phase-2 BFF. Charts via Apache ECharts (`echarts-for-react` or native). SSR/RSC for fast first paint.
- **Strangler cutover** — a reverse proxy (nginx or Next rewrites) routes already-migrated routes to Next.js and the rest to the still-running Streamlit `app`. Migrate page-by-page in the existing order (`01_basic_indicators` → … → `15_news`, plus the Phase-3 pages), retiring each Streamlit page as its Next.js equivalent reaches parity. Streamlit is deleted only when the last page is ported — at which point Streamlit's `connectorx` read path retires too.
- **Dynamic theming** — expose the `themes.yaml` palette via a BFF `/theme` endpoint; map semantic tokens (`positive`, `negative`, `sector_*`, `diverging_*`, `sequential_*`, `selected_marker`, …) to CSS variables + a registered ECharts theme. Add a runtime theme switcher. Port the `KeyError`-on-missing-token strictness so custom themes still fail loud.
- **Componentise the config-driven pages** — reproduce `render_page_from_config` as a generic React `<IndicatorPage>` driven by the same config served from the BFF.

**Touchpoints:** new `frontend/`, reverse-proxy config, `docker-compose.yaml` (+ Node service), BFF `/theme` + config endpoints.
**New deps:** Node toolchain in a new Dockerfile; `next`, `react`, `echarts`, `echarts-for-react`.
**Risks:** dual-maintenance window (two frontends live); choropleth/wordcloud/embedding-map parity (the News page's projection panels and the Plotly maps are the hardest to port). Tackle simple indicator pages first, complex viz pages last.

---

## Cross-cutting concerns

- **`config.yaml` ↔ `docker-compose.yaml` drift** — every new service (bff, frontend, otel-collector, loki, tempo, prometheus, grafana, cadvisor, backup, triton, postgres_exporter) means a port + bind-mount declared in **both** files; they are not auto-synced (the Phase-1 boot assertion catches port mismatches). Proposed ports: `bff 8005`, `frontend 3000`, `grafana 3001`, `prometheus 9090`, `cadvisor 8080`, `loki 3100`, `tempo 3200`, `otel-collector 4317/4318`, `triton 8100/8101/8102` — reconcile against the existing `8000`–`8004` to avoid collisions (Triton's defaults `8000/8001/8002` clash with `agent`/`forecaster`/`clustering`).
- **Container count** — grows from 9 to ~18. Use Compose `profiles` (e.g. `monitoring`, `gpu`) so GPU-less / minimal runs stay lean.
- **New secrets/env** — `FRED_API_KEY`, rclone remote creds, the heavy/light model names. All flow through the Phase-1 `Settings` + Docker secrets; extend `.env.example` and the README's required-vars table.
- **Schema source of truth** — the shared `db_models` package (Phase 2) plus `database_schema.yaml` must stay in sync; the agent's SQL worker grounds on the YAML.

---

## Assumptions & open questions

These were defaulted while writing this plan — flag any to revisit:

1. **News-RAG expansion (#7):** ~~filtered GDELT subset~~ — **decided against GDELT** (no full article text). Shipped instead: **Actually Relevant** (all ~1.5k curated stories → 5 macro collections, embedding the curated analysis, since the API serves no source body) and **World Bank documents** (~30 macro queries × top-100 docs, WDS `txturl` plain text → token-chunked, one collection per query). `txturl` chosen over docling to keep the downloader image light.
2. **API keys:** FRED needs a key; Eurostat and Binance historical market data are assumed keyless. Confirm no Binance auth is needed for the chosen endpoints.
3. **Sub-national maps:** US states via built-in FIPS GeoJSON; EU via Eurostat GISCO NUTS GeoJSON. **Resolved:** NUTS level **2** (~309 EU+EFTA+candidate regions), bundled `NUTS_RG_60M_2021_4326_LEVL_2`.
4. **Triton CPU carve-out (#12):** decision needed on the classical CPU-only forecasters at Phase 5 start (see Phase 5 ⚠).
5. **Two-model assignment (#2):** the light/heavy split above is the starting point; final per-node mapping is config-tunable.
6. **"Structure left the same" (#1):** interpreted as *preserve existing pages/navigation during migration*; new data sources (#3/#4/#6) legitimately **add** new pages on top.
7. **Frontend auth:** none assumed (same as today's open dashboard). Add if the BFF will be internet-exposed.
8. **Observability backend (#4):** self-hosted LGTM only; no external SaaS. Confirm retention/storage budget for Loki/Tempo on the host.
