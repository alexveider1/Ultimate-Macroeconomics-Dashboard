import { expect, test } from "@playwright/test";

import { expectHeading, expectNoException, gotoNav, openApp, RENDER_TIMEOUT } from "./helpers";

/**
 * Market + regional data pages (Yahoo Finance, Crypto, FRED states, Eurostat
 * NUTS-2). Each reads its own Postgres tables and draws Plotly charts on load,
 * so the assertions confirm a chart actually renders — not just the heading.
 */
const CHART_PAGES: { nav: string; heading: string | RegExp }[] = [
  { nav: "Yahoo Finance", heading: "Yahoo Finance Dashboard" },
  { nav: "Crypto", heading: "Crypto (Binance) Dashboard" },
  { nav: "United States (FRED)", heading: /Regional Statistics \(FRED\)/ },
  { nav: "European Union (Eurostat)", heading: /Regional Statistics \(Eurostat\)/ },
];

for (const { nav, heading } of CHART_PAGES) {
  test(`"${nav}" renders its heading and a chart`, async ({ page }) => {
    await openApp(page);
    await gotoNav(page, nav);
    await expectHeading(page, heading);
    await expect(page.getByTestId("stPlotlyChart").first()).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    await expectNoException(page);
  });
}
