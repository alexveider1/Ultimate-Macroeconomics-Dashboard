import { expect, test } from "@playwright/test";

import { fulfillChatError, registerApiMocks } from "./mocks";

/**
 * Cross-cutting app flows not covered by the milestone suites: the Overview
 * landing page, navigating across dashboard sections, the shared country
 * selection surviving client-side navigation *and* a reload (Zustand persist),
 * a downstream agent error surfacing in the chat, and an empty-data state.
 * Fully self-contained via the shared `/api/**` mocks (no BFF).
 */
test.beforeEach(async ({ page }) => {
  await registerApiMocks(page);
});

test.describe("app shell + landing", () => {
  test("overview landing renders the sample indicator chart for the selected countries", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
    // Default selection from the store.
    await expect(page.getByText(/Series for .*USA/)).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
  });

  test("navigates across dashboard sections and each renders charts", async ({ page }) => {
    await page.goto("/");
    for (const [navLabel, heading] of [
      ["General Economics", "General Economics Indicators"],
      ["Trade & External", "Trade and External Sector"],
      ["Health & Wellbeing", "Health and Wellbeing"],
    ] as const) {
      await page.getByRole("link", { name: navLabel }).click();
      await expect(page.getByRole("heading", { name: heading })).toBeVisible();
      await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    }
  });
});

test.describe("shared country selection (Zustand + persist)", () => {
  test("selection carries across pages and survives a reload", async ({ page }) => {
    // Seed once (guarded) so the reload below rehydrates the *persisted* change,
    // not the seed — addInitScript otherwise re-runs on every navigation.
    await page.addInitScript(() => {
      if (!localStorage.getItem("umd.ui")) {
        localStorage.setItem(
          "umd.ui",
          JSON.stringify({ state: { selectedCountries: ["USA", "CHN", "DEU"] }, version: 0 }),
        );
      }
    });

    await page.goto("/dashboard/economy-structure");
    // Seeded countries render as removable chips (outside any popover).
    await expect(page.getByRole("button", { name: "Remove USA" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Remove CHN" })).toBeVisible();

    // Drop one — the store (and its persisted copy) updates.
    await page.getByRole("button", { name: "Remove CHN" }).click();
    await expect(page.getByRole("button", { name: "Remove CHN" })).toHaveCount(0);

    // Client-side nav to another dashboard page: the in-memory store is shared.
    await page.getByRole("link", { name: "Demography" }).click();
    await expect(page.getByRole("heading", { name: "Demography" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Remove USA" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Remove CHN" })).toHaveCount(0);

    // Full reload rehydrates from localStorage: the change persisted.
    await page.reload();
    await expect(page.getByRole("button", { name: "Remove USA" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Remove CHN" })).toHaveCount(0);
  });
});

test.describe("error + empty states", () => {
  test("AI chat surfaces a downstream agent error frame", async ({ page }) => {
    // Override just the chat stream with an error-terminated stream (wins over
    // the beforeEach mock — Playwright runs the most-recent matching route first).
    await page.route((url) => url.pathname === "/api/agent/chat/stream", fulfillChatError);

    await page.goto("/ai");
    await expect(page.getByRole("heading", { name: "AI Analyst" })).toBeVisible();

    await page.getByPlaceholder("Ask the AI analyst…").fill("Break something");
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(
      page.getByText("The analyst service is temporarily unavailable."),
    ).toBeVisible({ timeout: 15000 });
  });

  test("overview shows an empty state when the indicator has no data", async ({ page }) => {
    // Return an empty series for the indicator values (wins over the shared mock).
    await page.route(
      (url) => /\/api\/worldbank\/indicators\/[^/]+\/values$/.test(url.pathname),
      (route) => {
        const id = new URL(route.request().url()).pathname.split("/").at(-2) ?? "x";
        return route.fulfill({ json: { indicator_id: id, name: id, points: [] } });
      },
    );

    await page.goto("/");
    await expect(page.getByText("No data available for this indicator.")).toBeVisible({
      timeout: 20000,
    });
  });
});
