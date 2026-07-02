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

> **Status (v0.14):** Phase 1 *config* + *secrets* and Phase 2b *agent two-model routing* are implemented — pragmatically. Because each service is its own container with its own `pyproject.toml` (no shared package), config/secrets use **per-service** Pydantic `config.py` + `pydantic-settings` `settings.py` rather than one shared `Settings`, and least privilege is enforced by **per-service `environment:` scoping in `docker-compose.yaml`** (each container gets only the secrets it uses) instead of Docker `secrets:` file-mounts. `SecretStr` log-masking and a BFF-wide shared model remain open follow-ups. From **Phase 3**, the Yahoo Finance universe (#5), Binance crypto (#6 — ingestion + a Crypto dashboard page), and the WB→httpx swap (#9) have shipped; on-demand ingestion (`downloader_extra` + the agent's `downloader_agent`) is now **multi-source** (WB indicator / Yahoo ticker / Binance pair). FRED (#3), Eurostat (#4), and GDELT (#7) remain open, as do all of Phases 2 (BFF), 4 (observability/backups), 5 (GPU/Triton), and 6 (frontend).

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

**Touchpoints:** new `packages/config/`, every service `main.py` / `app/core/*.py` config + logging call sites, `agent/agent/graph.py`, `docker-compose.yaml` (`secrets:`), `_container_data/.env.example`.
**New deps:** `pydantic-settings`, `opentelemetry-*` SDK + instrumentation packages per service.
**Risks:** instrumentation overhead in the hot path (sandbox/agent) — keep exporters async/batched; assert no secret leaks via the verification grep below.

---

## Phase 2 — FastAPI BFF + shared ORM data layer *(was #1 backend, #9)*

**Goal:** give the future JS frontend a typed HTTP data source, with **all SQL wrapped in ORM** — built **additively** so the live Streamlit app is untouched.

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
- **On-demand ingestion → multi-source** — ✅ **done** (v0.14): `downloader_extra`'s unified `POST /ingest` (`source=worldbank|yahoo|binance`) + the agent's source-aware `downloader_agent`. WB uses the `database_indicators` master catalogue; Yahoo/Binance have **no catalogue**, so the agent infers the ticker / full pair symbol and `downloader_extra` validates it live.
- **#3 FRED (US states)** — new `extractors/fred.py` (FRED API, **needs `FRED_API_KEY`**); new `_configs/fred_download_config.json` (state-level series: GDP, unemployment, etc.). New **US States** page with a US-state choropleth.
- **#4 Eurostat (EU NUTS regions)** — new `extractors/eurostat.py` (Eurostat REST / JSON-stat, **keyless**); new `_configs/eurostat_download_config.json`. New **EU Regions** page with a NUTS-2/3 choropleth (GISCO GeoJSON boundaries).
- **#7 News RAG → GDELT** — replace the `github` news extractor with `extractors/gdelt.py`. Query GDELT (DOC 2.0 / GKG) filtered by the topics in `news_download_config.json` rather than the full firehose; embed with the same model into Qdrant. Remove the GitHub-repo dependency.

**Touchpoints:** `downloader_general/src/extractors/`, `downloader_extra/`, `_configs/`, `database_schema.yaml`, BFF routers, `Settings`/secrets for `FRED_API_KEY`.
**Risks:** GDELT volume + dedup/rate-limits; sub-national GeoJSON sourcing and join keys (FIPS for US states, NUTS codes for EU). Country-level pages are unaffected.

---

## Phase 4 — Ops: monitoring + backups *(was #11, #8)*

**#11 — Self-hosted Grafana LGTM observability**
- Add **OTel Collector + Loki + Tempo + Prometheus + Grafana + cAdvisor** services. The OTel Collector receives the OTLP logs/traces emitted by every service since Phase 1 and fans out to Loki (logs) and Tempo (traces); FastAPI services also expose `/metrics` (`prometheus-fastapi-instrumentator`) scraped by Prometheus, plus `postgres_exporter`, Qdrant's native `/metrics`, and cAdvisor for per-container CPU/mem/net.
- Provision Grafana dashboards + datasources (service health, latency, traces, logs, container stats) as code.
- **Remove** `app/core/monitoring.py`, `app/pages/18_monitoring.py`, and the read-only `/var/run/docker.sock` bind-mount from the `app` service — the bespoke health/Docker-socket probing is replaced by the standard stack.

**#8 — rclone backups**
- New scheduled `backup` service (cron-style). Each run: `pg_dump` the Postgres DB + call the Qdrant snapshot API (`POST /collections/{c}/snapshots`), then `rclone copy` both artifacts to a remote configured via secrets (`rclone.conf` mounted). Retention prune (keep N daily / M weekly).
- Commit a **restore script** + documented restore procedure.

**Touchpoints:** `docker-compose.yaml` (otel-collector/loki/tempo/prometheus/grafana/cadvisor/postgres_exporter/backup services), each service's `main.py` (`/metrics`), new `monitoring/` (Grafana + collector provisioning) and `backup/` dirs, secrets for the rclone remote, `config.yaml`.
**Risks:** consistent snapshots (snapshot Qdrant and dump Postgres close in time); rclone remote secret handling (via Phase-1 Docker secrets).

---

## Phase 5 — GPU/ML consolidation *(was #12)*

**Goal:** GPU-driven ML unified under one **NVIDIA Triton** runtime; **GPU is a hard requirement** for this deployment profile.

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

1. **GDELT scope (#7):** filtered topic-based subset (DOC 2.0/GKG queried by existing news topics), **not** the full firehose. Confirm acceptable corpus size/freshness.
2. **API keys:** FRED needs a key; Eurostat and Binance historical market data are assumed keyless. Confirm no Binance auth is needed for the chosen endpoints.
3. **Sub-national maps:** US states via built-in FIPS GeoJSON; EU via Eurostat GISCO NUTS GeoJSON. Confirm the NUTS level (2 vs 3).
4. **Triton CPU carve-out (#12):** decision needed on the classical CPU-only forecasters at Phase 5 start (see Phase 5 ⚠).
5. **Two-model assignment (#2):** the light/heavy split above is the starting point; final per-node mapping is config-tunable.
6. **"Structure left the same" (#1):** interpreted as *preserve existing pages/navigation during migration*; new data sources (#3/#4/#6) legitimately **add** new pages on top.
7. **Frontend auth:** none assumed (same as today's open dashboard). Add if the BFF will be internet-exposed.
8. **Observability backend (#4):** self-hosted LGTM only; no external SaaS. Confirm retention/storage budget for Loki/Tempo on the host.
