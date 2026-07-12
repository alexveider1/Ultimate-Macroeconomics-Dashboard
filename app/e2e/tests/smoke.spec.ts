import { expect, test } from "@playwright/test";

import { dismissDisclaimer, expectNoException, openApp, sidebarNav } from "./helpers";

/**
 * Smoke tests: the app shell boots, the disclaimer gate works, and the sidebar
 * navigation renders every section. These are the fast "is the dashboard alive"
 * checks; per-page coverage lives in navigation.spec.ts.
 */
test.describe("app shell", () => {
  test("boots and shows the disclaimer, then the dashboard", async ({ page }) => {
    await page.goto("/", { waitUntil: "domcontentloaded" });

    // The Streamlit root mounts.
    await expect(page.getByTestId("stApp")).toBeVisible({ timeout: 60_000 });

    // First visit is gated by the modal data disclaimer.
    const accept = page.getByRole("button", { name: "I understand" });
    await expect(accept).toBeVisible({ timeout: 20_000 });
    await accept.click();
    await expect(accept).toBeHidden({ timeout: 20_000 });

    // A landing dashboard page renders without a Python traceback.
    await expect(sidebarNav(page)).toBeVisible({ timeout: 60_000 });
    await expectNoException(page);
  });

  test("has a non-empty document title", async ({ page }) => {
    await openApp(page);
    await expect(page).toHaveTitle(/.+/);
  });

  test("sidebar lists every navigation group", async ({ page }) => {
    await openApp(page);
    const nav = sidebarNav(page);
    for (const group of ["Dashboard", "Other data", "Regional Statistics", "AI"]) {
      await expect(nav.getByText(group, { exact: true })).toBeVisible();
    }
  });

  test("landing page renders the first dashboard heading", async ({ page }) => {
    await openApp(page);
    await expect(
      page.getByRole("heading", { name: "General Economics Indicators", level: 1 }),
    ).toBeVisible({ timeout: 60_000 });
  });
});
