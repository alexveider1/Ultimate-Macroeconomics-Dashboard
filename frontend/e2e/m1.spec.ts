import { expect, test } from "@playwright/test";

import { registerApiMocks } from "./mocks";

// Self-contained: the dashboard reads (`/config/dashboard`, `/worldbank/*`,
// `/geo/world`, `/forecast/models`) are all mocked, so no BFF is required.
test.beforeEach(async ({ page }) => {
  await registerApiMocks(page);
});

test.describe("M1 config-driven dashboards", () => {
  test("dashboard page renders GraphBox cards with real charts", async ({ page }) => {
    await page.goto("/dashboard/economy-structure");
    await expect(page.getByRole("heading", { name: "Economy Structure" })).toBeVisible();
    await expect(page.getByText(/Countries for time trends/)).toBeVisible();

    // Each GraphBox renders a choropleth + a right chart (ECharts canvases).
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    expect(await page.locator("canvas").count()).toBeGreaterThanOrEqual(2);

    await page.screenshot({ path: "test-results/m1-dashboard.png", fullPage: true });
  });

  test("settings popover switches the right chart to distribution", async ({ page }) => {
    await page.goto("/dashboard/economy-structure");
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });

    await page.getByRole("button", { name: "Chart settings" }).first().click();
    await page.getByLabel("Right-side chart").selectOption("distribution");
    await expect(page.getByLabel("Distribution type")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.locator("canvas").first()).toBeVisible();
  });

  test("log transform toggles without error", async ({ page }) => {
    await page.goto("/dashboard/economy-structure");
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    await page.getByText("Apply log transformation").first().click();
    await expect(page.locator("canvas").first()).toBeVisible();
  });

  test("custom plot builder renders a chart", async ({ page }) => {
    await page.goto("/constructors/custom-plot");
    await expect(page.getByRole("heading", { name: "Custom Plot Constructor" })).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
  });
});
