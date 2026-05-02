# Changelog

All notable changes to **Ultimate Macroeconomics Dashboard** are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
