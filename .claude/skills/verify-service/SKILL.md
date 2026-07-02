---
name: verify-service
description: Run the lint + type-check + test loop for one service (ruff format --check, ruff check, ty check, uv run pytest) and report failures. Use to verify a change to a single service before committing, or as the Phase-0 dev-loop "one command to iterate on".
---

# verify-service

Verify a single service in this repo. Each service is isolated with its own `.venv` (no root venv), so run from inside the service directory — `uv run` uses that service's environment while `ruff`/`ty` walk up to the root `ruff.toml` / `ty.toml`.

## Usage

The argument is the service name (one of: `agent`, `app`, `clustering`, `downloader_general`, `downloader_extra`, `forecaster`, `python_sandbox`). If none is given, ask which service, or run the whole loop for each in turn.

## Steps

Run these from the service directory and report the outcome of each step (don't stop at the first failure — collect them all):

```bash
cd <service>
uv run ruff format --check .     # formatting (drop --check to auto-fix)
uv run ruff check .              # lint (add --fix to auto-fix)
uv run ty check .                # static types
uv run pytest                    # tests (coverage is on by default via addopts)
```

## Reporting

- Summarise pass/fail per step. For failures, quote the specific ruff rule / ty diagnostic / failing test id and the file:line — do not paste the whole output.
- If coverage dropped or a `--cov-fail-under` gate fired, call that out separately from test failures.
- Suggest the minimal fix (or apply it if the change is trivial and clearly correct), then re-run. Do NOT commit.
