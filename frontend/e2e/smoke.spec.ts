import { expect, test } from "@playwright/test";

import { registerApiMocks } from "./mocks";

// Self-contained: every `/api/**` call (theme config, world-bank reads, …) is
// mocked, so the shell/nav/theme flow runs without a BFF.
test.beforeEach(async ({ page }) => {
  await registerApiMocks(page);
});

test.describe("M0 skeleton", () => {
  test("renders shell + nav and loads the theme from config", async ({ page }) => {
    await page.goto("/");

    // Sidebar nav groups (mirroring the Streamlit navigation) are present.
    await expect(page.getByRole("link", { name: "Yahoo Finance" })).toBeVisible();
    await expect(page.getByRole("link", { name: "AI Analyst" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();

    // The active theme was applied to the document root from the config.
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });

  test("runtime theme switch re-applies colours from config (no reload)", async ({ page }) => {
    await page.goto("/");
    const body = page.locator("body");
    const before = await body.evaluate((el) => getComputedStyle(el).backgroundColor);

    await page.getByLabel("Theme").selectOption("light-green");

    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await expect
      .poll(() => body.evaluate((el) => getComputedStyle(el).backgroundColor))
      .not.toBe(before);

    await page.screenshot({ path: "test-results/overview-light-green.png", fullPage: true });
  });

  test("client-side routing to a nav page works", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("link", { name: "Crypto" }).click();
    await expect(page.getByRole("heading", { name: "Crypto" })).toBeVisible();
    await expect(page).toHaveURL(/\/crypto$/);
  });
});
