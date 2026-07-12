# app e2e (Playwright)

Node-based Playwright end-to-end tests for the Streamlit dashboard (`app`).
They drive the real UI against a running stack (live Postgres/Qdrant data).

## Specs (`tests/`)

- `helpers.ts` — shared setup: dismiss the disclaimer modal, wait for Streamlit
  to boot, the page ↔ heading table, and the no-traceback assertion.
- `smoke.spec.ts` — the app shell boots, the disclaimer gate works, every
  sidebar group is listed.
- `navigation.spec.ts` — walks all 17 sidebar pages on one session and asserts
  each renders its heading with no Python traceback (the main regression net).
- `dashboard.spec.ts` — a World Bank page: charts render, the cross-page country
  selector persists, and the per-graph settings popover opens.
- `data-pages.spec.ts` — Yahoo / Crypto / FRED / Eurostat each draw a chart.
- `interactive.spec.ts` — the AI chat's multimodal inputs, the clustering
  sandbox, and the news explorer render and are driveable.

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
