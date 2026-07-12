import { expect, test } from "@playwright/test";

import { expectNoException, gotoNav, openApp, RENDER_TIMEOUT } from "./helpers";

/**
 * World Bank dashboard pages: the shared indicator panel (map + trend/dist),
 * the cross-page country selector, and the per-graph settings popover. Data is
 * read straight from Postgres, so charts actually render here.
 */
test.describe("World Bank dashboard page", () => {
  test.beforeEach(async ({ page }) => {
    await openApp(page); // lands on "General Economics Indicators"
  });

  test("renders indicator charts and the country selector", async ({ page }) => {
    // GraphBox draws real Plotly figures (choropleth + trend) from live data.
    await expect(page.getByTestId("stPlotlyChart").first()).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    await expect(page.getByText("Countries for time trends")).toBeVisible();
    await expectNoException(page);
  });

  test("pre-selects default countries as multiselect chips", async ({ page }) => {
    const selector = page.getByTestId("stMultiSelect").first();
    await expect(selector).toBeVisible({ timeout: RENDER_TIMEOUT });
    // USA/CHN/DEU are seeded by default; the chip label ends with "(USA)".
    await expect(selector.getByText(/\(USA\)/)).toBeVisible();
  });

  test("keeps the country selection when switching dashboard pages", async ({ page }) => {
    await expect(page.getByTestId("stMultiSelect").first().getByText(/\(USA\)/)).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });

    await gotoNav(page, "Demography");
    await expect(page.getByRole("heading", { name: "Demography", level: 1 })).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    // The shared selection carries across pages (SHARED_COUNTRIES_STATE_KEY).
    await expect(page.getByTestId("stMultiSelect").first().getByText(/\(USA\)/)).toBeVisible();
    await expectNoException(page);
  });

  test("settings popover reveals the layout and forecasting controls", async ({ page }) => {
    await expect(page.getByTestId("stPlotlyChart").first()).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });

    // Open the first graph's settings popover (⚙️). Every GraphBox keeps its
    // settings in the DOM, so scope to the one popover that is actually visible.
    await page.getByRole("button", { name: "⚙️" }).first().click();

    // The popover exposes the right-panel selector, the forecasting form, and
    // its "Run model" submit — all rendered without a traceback.
    await expect(page.getByText("Right-side chart").filter({ visible: true })).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    await expect(page.getByText("Time Series Forecasting").filter({ visible: true })).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Run model" }).filter({ visible: true }),
    ).toBeVisible();
    await expectNoException(page);
  });
});
