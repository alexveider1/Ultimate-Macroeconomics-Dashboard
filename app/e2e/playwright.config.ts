import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for the Streamlit dashboard.
 *
 * The dashboard listens on http://localhost:8501 (see config.yaml / docker-compose).
 * Override with the BASE_URL env var when pointing at another host/port. The
 * stack must be up (`docker compose up -d app`) before running — these specs
 * drive the real UI against live Postgres/Qdrant data.
 */
const baseURL = process.env.BASE_URL ?? "http://localhost:8501";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 1,
  // Streamlit is a single server; a handful of concurrent sessions is plenty
  // and keeps heavy, data-backed pages from starving each other.
  workers: process.env.CI ? 2 : 3,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  reporter: [["html", { open: "never" }], ["list"]],
  use: {
    baseURL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    // Enable once the corresponding browsers are installed
    // (`pnpm exec playwright install firefox webkit`):
    // { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    // { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
});
