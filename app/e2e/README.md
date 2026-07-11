# app e2e (Playwright)

Node-based Playwright end-to-end tests for the Streamlit dashboard. Scaffolded
ahead of the frontend switch — specs under `tests/` are placeholders for now.

## Setup

```bash
cd app/e2e
pnpm install                       # install @playwright/test
pnpm exec playwright install chromium   # browser binary (add firefox webkit later)
```

On a fresh Ubuntu box you may also need the browser system libraries:

```bash
pnpm exec playwright install-deps chromium   # requires sudo
```

## Run

The dashboard must be reachable at `http://localhost:8501` (or set `BASE_URL`).

```bash
pnpm test            # headless
pnpm run test:headed # headed
pnpm run test:ui     # interactive UI mode
pnpm run report      # open the last HTML report
```
