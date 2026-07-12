# TODO

> Flat backlog of the features still to implement and bugs still to fix.
> Completed work is reflected in `CLAUDE.md` (delivered architecture) and the git
> history — this file tracks only what remains open.

## Short-term

* **Dev loop:** raise per-service `pytest` coverage and add a `--cov-fail-under` gate to each service's `[tool.pytest.ini_options].addopts`, so coverage regressions fail the run.
* **Dev loop:** add GitHub Actions CI — a matrix over the services running `ruff check`, `ty`, and `uv run pytest`. Green CI as the merge gate.
* **Secrets hardening:** type every secret as `pydantic.SecretStr` (log-masking) and move the DB password + `OPENAI_API_KEY` to Docker Compose `secrets:` file-mounts (`/run/secrets/...`) with a `.env` fallback for local dev.
* **Config-drift guard:** add a boot-time assertion that `config.yaml` ports match `docker-compose.yaml` (the two files are not auto-synced).

## Medium-term

* **Agent eval harness:** a fixture table of prompts → asserted worker path / `last_worker_status` / answer-contains checks, run against the LangGraph in `agent/agent/graph.py`, as the regression net for prompt and routing changes.
* **Observability — logs & traces:** OpenTelemetry structured logs + non-LLM distributed traces aggregated with self-hosted **Loki / Tempo** alongside the existing Prometheus/Grafana. The OTel Collector already runs with an **idle `otlp` receiver** (4317/4318) ready to accept them; still to do is standing up Loki/Tempo and instrumenting each service with the OTel SDK. Complements the shipped resource/health monitor (which covers metrics + health but not log/trace aggregation).
* **Observability — Langfuse follow-ups:** define custom **model prices** for `gpt-5.4` / `gpt-5.4-mini` in the Langfuse UI (Settings → Models) so cost populates; optionally set `langfuse.sample_rate` < 1.0 for the high-volume initial ingest.
* **Shared DB models package:** extract the duplicated ORM `Mapped` models into a shared `packages/db_models/` imported by `downloader_general` / `downloader_extra` / `agent` (each carries its own `schema.py` copy today).

## Long-term

* **Triton follow-ups:** freeze `triton/requirements.txt` to the exact known-good resolution after the first successful GPU build (RAPIDS pins numpy/pandas — currently light on pins). A pinned/frozen numpy would also let the Dockerfile drop its post-install `numpy-*.dist-info` cleanup workaround (the base image's apt-installed numpy 1.26.4 has no `METADATA`, so pip can't remove it when RAPIDS upgrades numpy, and the leftover metadata-less dir crashes transformers' import guard). Also consider Triton **explicit model-control** so disabled families (`*_AVAILABLE: false`) aren't loaded at all; optionally revisit a true ONNX/TensorRT Chronos export.
* **Voice output:** the Streamlit chat already accepts file + voice **input** (documents via `docling`, audio via Whisper, images as vision parts — `app/core/multimodal.py`). Still to do: **voice output** — synthesize the agent's answer to speech (TTS) for a hands-free reply.
* Interactive graph networks visualizing connections between countries in the global economy; dedicated graph-based analysis pages.
* Option for RAG over a graph-based knowledge database.
* Dashboard page with educational videos explaining how the ML/DL models used in the project work.
* Option to store data in cloud services.

## Backlog

* New data source: Maddison Project Database.
* Dynamic, LLM-based macroeconomics reports rendered as documents.
