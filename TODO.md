# TODO

> Flat backlog of features to implement and bugs to fix.
> The detailed, phased roadmap (rationale, sequencing, touchpoints, "Guiding decisions")
> lives in [`.claude/roadmap.md`](.claude/roadmap.md); this file is the actionable checklist.
> Phase tags below point back into that roadmap.

## Short-term

* **Dev loop (Phase 0):** raise per-service `pytest` coverage and add a `--cov-fail-under` gate to each service's `[tool.pytest.ini_options].addopts`, so coverage regressions fail the run.
* **Dev loop (Phase 0):** add GitHub Actions CI — a matrix over the services running `ruff check`, `ty`, and `uv run pytest`. Green CI as the merge gate.
* **Foundations (Phase 1):** finish secrets hardening — type every secret as `pydantic.SecretStr` (log-masking) and move the DB password + `OPENAI_API_KEY` to Docker Compose `secrets:` file-mounts (`/run/secrets/...`) with a `.env` fallback for local dev.
* **Foundations (Phase 1):** add a boot-time assertion that `config.yaml` ports match `docker-compose.yaml` (mitigates the documented compose ↔ yaml drift).

## Medium-term

* **Agent eval harness (Phase 0):** a fixture table of prompts → asserted worker path / `last_worker_status` / answer-contains checks, run against the LangGraph in `agent/agent/graph.py`, as the regression net for prompt and routing changes.
* Add hierarchical clustering to the `clustering` container.
* **Observability (Phase 1 → 4):** OpenTelemetry structured logs + traces per service (a span per worker / per LLM call in the agent, carrying token + latency attributes), shipped to a self-hosted Grafana LGTM stack (OTel Collector → Loki / Tempo / Prometheus + cAdvisor + `postgres_exporter`). Retire `app/core/monitoring.py`, `app/pages/18_monitoring.py`, and the read-only `docker.sock` mount once it lands.
* **FastAPI BFF (Phase 2):** new `bff` service (port 8005) exposing typed, ORM-only reads, plus a shared `db_models` package imported by `bff` / `downloader_*` / `agent`. Additive — Streamlit keeps `connectorx` until the frontend cutover.
* ~~**Data expansion (Phase 3):** FRED US-state series (needs `FRED_API_KEY`) + a US-states (FIPS) choropleth page.~~ **DONE** — 36 state indicators via the GeoFRED API (`fred` schema group: `states` / `state_indicators` / `state_indicator_values`), a new "Regional Statistics" nav group with the `19_fred_regional.py` choropleth/ranking/trend page, and on-demand + `sql_agent` wiring into the AI Analyst.
* ~~**Data expansion (Phase 3):** Eurostat EU NUTS regions (keyless, JSON-stat) + a NUTS choropleth page (GISCO GeoJSON boundaries) — slots in as the second page under the existing "Regional Statistics" nav group; reuse the region-agnostic `build_region_ranking_bar` / `build_region_trend_lines` helpers.~~ **DONE** — 25 NUTS-2 indicators via the keyless Eurostat dissemination (JSON-stat) API (`eurostat` schema group: `eurostat_regions` / `eurostat_indicators` / `eurostat_indicator_values`), the `20_eurostat_regional.py` choropleth/ranking/trend page (new `build_nuts_choropleth` over a bundled GISCO NUTS-2 GeoJSON), and on-demand (`source=eurostat`) + `sql_agent` wiring into the AI Analyst.
* **Data expansion (Phase 3):** replace the GitHub news extractor with a GDELT (DOC 2.0 / GKG) extractor, topic-filtered from `news_download_config.json` into Qdrant.
* **Backups (Phase 4):** a scheduled `backup` service — nightly `pg_dump` + Qdrant snapshots pushed to an `rclone` remote, with retention prune and a committed restore script.

## Long-term

* **Frontend (Phase 6, LAST):** migrate the Streamlit app to Next.js + React + Apache ECharts via a strangler cutover, preserving the existing page structure; re-add runtime dynamic theming (a BFF `/theme` endpoint → CSS variables + a registered ECharts theme, keeping the `KeyError`-on-missing-token strictness).
* **GPU/ML consolidation (Phase 5):** RAPIDS (cuML / cuDF) + NVIDIA Triton unifying the `forecaster` + `clustering` serving path (resolve the classical CPU-only-model carve-out at phase start).
* Interactive graph networks visualizing connections between countries in the global economy; dedicated graph-based analysis pages.
* Option for RAG over a graph-based knowledge database.
* Voice and file input for the AI agent; voice output from the agent.
* Dashboard page with educational videos explaining how the ML/DL models used in the project work.
* Option to store data in cloud services.

## Backlog

* New data source: Maddison Project Database.
* Dynamic, LLM-based macroeconomics reports rendered as documents.
