import { expect, type Locator, type Page } from "@playwright/test";

/**
 * Shared helpers for the Streamlit dashboard e2e specs.
 *
 * Streamlit boots over a websocket and re-runs the whole script on every
 * interaction, so selectors target stable data-testids / accessible names
 * rather than DOM structure, and waits are generous (charts pull real data
 * from Postgres/Qdrant on render).
 */

/** Long timeout for Streamlit reruns that fetch data and redraw charts. */
export const RENDER_TIMEOUT = 60_000;

/**
 * The sidebar navigation entries, paired with the `<h1>` each page renders.
 *
 * `nav` is the exact `st.Page` title shown as the sidebar link; `heading`
 * is a substring/regex of the page's `st.title(...)` (they intentionally
 * differ on a few pages, e.g. "Yahoo Finance" → "Yahoo Finance Dashboard").
 */
export const PAGES: { nav: string; heading: string | RegExp; group: string }[] = [
  { nav: "General Economics Indicators", heading: "General Economics Indicators", group: "Dashboard" },
  { nav: "Economy Structure", heading: "Economy Structure", group: "Dashboard" },
  { nav: "Finance and Monetary", heading: "Finance and Monetary", group: "Dashboard" },
  { nav: "Trade and External sector", heading: "Trade and External sector", group: "Dashboard" },
  { nav: "Demography", heading: "Demography", group: "Dashboard" },
  { nav: "Governance and Institutions", heading: "Governance and Institutions", group: "Dashboard" },
  { nav: "Technology and Innovations", heading: "Technology and Innovations", group: "Dashboard" },
  { nav: "Health and wellbeing", heading: "Health and wellbeing", group: "Dashboard" },
  { nav: "Education and Human Capital", heading: "Education and Human Capital", group: "Dashboard" },
  { nav: "Environment and Sustainability", heading: "Environment and Sustainability", group: "Dashboard" },
  { nav: "Yahoo Finance", heading: "Yahoo Finance Dashboard", group: "Other data" },
  { nav: "Crypto", heading: "Crypto (Binance) Dashboard", group: "Other data" },
  { nav: "News Explorer", heading: "News Explorer", group: "Other data" },
  { nav: "United States (FRED)", heading: /Regional Statistics \(FRED\)/, group: "Regional Statistics" },
  { nav: "European Union (Eurostat)", heading: /Regional Statistics \(Eurostat\)/, group: "Regional Statistics" },
  { nav: "Custom Plot Constructor", heading: "Custom Plot Constructor", group: "Constructors" },
  { nav: "Clustering Sandbox", heading: "Clustering Sandbox", group: "Constructors" },
  { nav: "AI Analyst", heading: "AI Analyst", group: "AI" },
];

/** Dismiss the one-per-session "Data Disclaimer" modal if it is showing. */
export async function dismissDisclaimer(page: Page): Promise<void> {
  const accept = page.getByRole("button", { name: "I understand" });
  try {
    await accept.waitFor({ state: "visible", timeout: 20_000 });
    await accept.click();
    await accept.waitFor({ state: "hidden", timeout: 20_000 });
  } catch {
    // Already accepted this session, or the dialog never appeared — carry on.
  }
}

/**
 * Navigate to `path`, wait for Streamlit to finish its first render, and clear
 * the disclaimer modal. Returns once the sidebar navigation is interactive.
 */
export async function openApp(page: Page, path = "/"): Promise<void> {
  await page.goto(path, { waitUntil: "domcontentloaded" });
  await page.getByTestId("stApp").waitFor({ state: "visible", timeout: RENDER_TIMEOUT });
  await dismissDisclaimer(page);
  await expect(sidebarNav(page)).toBeVisible({ timeout: RENDER_TIMEOUT });
}

/** The sidebar navigation container (`st.navigation`). */
export function sidebarNav(page: Page): Locator {
  return page.getByTestId("stSidebarNav");
}

/** Click a sidebar link by its exact `st.Page` title. */
export async function gotoNav(page: Page, navLabel: string): Promise<void> {
  await sidebarNav(page).getByRole("link", { name: navLabel, exact: true }).click();
}

/**
 * Assert the page rendered no uncaught Python exception. Streamlit surfaces a
 * traceback in a `stException` box, so its absence is a strong "page rendered
 * cleanly" signal independent of the underlying data.
 */
export async function expectNoException(page: Page): Promise<void> {
  await expect(page.getByTestId("stException")).toHaveCount(0);
}

/** Assert the page's `st.title(...)` heading is visible. */
export async function expectHeading(page: Page, heading: string | RegExp): Promise<void> {
  await expect(page.getByRole("heading", { name: heading, level: 1 })).toBeVisible({
    timeout: RENDER_TIMEOUT,
  });
}
