import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for the Streamlit dashboard.
 *
 * The dashboard listens on http://localhost:8501 (see config.yaml / docker-compose).
 * Override with the BASE_URL env var when pointing at another host/port.
 *
 * These tests are a placeholder baseline — real specs land after the frontend
 * switch. Run with `pnpm test` from this folder once the stack is up.
 */
const baseURL = process.env.BASE_URL ?? "http://localhost:8501";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
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
