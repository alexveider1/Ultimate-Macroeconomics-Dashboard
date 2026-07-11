import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, type Route, test } from "@playwright/test";

/**
 * M2 pages (regional / market / news / AI chat) driven entirely by mocked
 * `/api/**` responses, so the suite is self-contained (no BFF needed). The two
 * regional maps are served from the real bundled GeoJSON on disk; everything
 * else is synthetic but shaped exactly like the BFF contract.
 */

const CONFIGS = resolve(process.cwd(), "..", "_container_data", "_configs");
const geo = (name: string) => readFileSync(resolve(CONFIGS, name), "utf-8");

const THEME = {
  label: "Dark",
  mode: "dark",
  fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
  chrome: {
    background: "#0e1117",
    surface: "#1e1e1e",
    border: "#2a2f3a",
    text: "#fafafa",
    textMuted: "#94a3b8",
    primary: "#10c8f1",
    primaryText: "#0e1117",
  },
  series: {
    colorway: ["#10c8f1", "#ff6b6b", "#ffd93d", "#6bcf7f", "#a685e2"],
    sequential: ["#0e1117", "#10c8f1"],
    diverging: ["#60a5fa", "#1e293b", "#f87171"],
  },
  semantic: {
    positive: "#22c55e",
    negative: "#ef4444",
    referenceLine: "#94a3b8",
    selectedMarker: "#fde047",
    mapCoastline: "#7dd3fc",
    mapLand: "#1e293b",
    sectors: { agriculture: "#34d399", manufacturing: "#f59e0b", services: "#60a5fa" },
  },
  charts: {
    confidenceBandAlpha: 0.18,
    gridLine: "#2a2f3a",
    axisLine: "#3a4150",
    tooltipBackground: "#1e1e1e",
    tooltipText: "#fafafa",
  },
  wordcloud: { background: "#0e1117", colors: ["#440154", "#3b528b", "#21918c", "#5ec962"] },
};

/** Build monthly OHLCV points for 2022–2023 (24 points) for one series key. */
function monthlySeries(base: number): { date: string; o: number; h: number; l: number; c: number; v: number }[] {
  const out = [];
  let price = base;
  for (let y = 2022; y <= 2023; y += 1) {
    for (let m = 1; m <= 12; m += 1) {
      const open = price;
      const close = price * (1 + (((y + m) % 5) - 2) / 25);
      out.push({
        date: `${y}-${String(m).padStart(2, "0")}-01T00:00:00`,
        o: open,
        h: Math.max(open, close) * 1.03,
        l: Math.min(open, close) * 0.97,
        c: close,
        v: 1_000_000 + m * 1000,
      });
      price = close;
    }
  }
  return out;
}

const FRED_STATES = [
  { id: "CA", name: "California", fips: "06", region: "West", division: "Pacific" },
  { id: "TX", name: "Texas", fips: "48", region: "South", division: "West South Central" },
  { id: "NY", name: "New York", fips: "36", region: "Northeast", division: "Mid-Atlantic" },
  { id: "FL", name: "Florida", fips: "12", region: "South", division: "South Atlantic" },
  { id: "WA", name: "Washington", fips: "53", region: "West", division: "Pacific" },
];

const EU_REGIONS = [
  { id: "DE21", name: "Oberbayern", country_code: "DE", country_name: "Germany", nuts1_id: "DE2", level: 2 },
  { id: "FR10", name: "Île-de-France", country_code: "FR", country_name: "France", nuts1_id: "FR1", level: 2 },
  { id: "ES30", name: "Comunidad de Madrid", country_code: "ES", country_name: "Spain", nuts1_id: "ES3", level: 2 },
  { id: "ITC4", name: "Lombardia", country_code: "IT", country_name: "Italy", nuts1_id: "ITC", level: 2 },
  { id: "PL91", name: "Warszawski", country_code: "PL", country_name: "Poland", nuts1_id: "PL9", level: 2 },
];

function regionValues(codes: string[]) {
  const rows = [];
  for (const code of codes) {
    for (let year = 2018; year <= 2023; year += 1) {
      rows.push({ region: code, year, value: 20 + code.charCodeAt(0) % 30 + (year - 2018) * 2 });
    }
  }
  return rows;
}

async function fulfillApi(route: Route) {
  const url = new URL(route.request().url());
  const path = url.pathname.replace(/^\/api/, "");
  const json = (data: unknown) => route.fulfill({ json: data });

  if (path === "/config/themes") return json({ active: "dark", themes: { dark: THEME } });
  if (path === "/geo/us-states")
    return route.fulfill({ contentType: "application/geo+json", body: geo("us_states.geojson") });
  if (path === "/geo/nuts2")
    return route.fulfill({
      contentType: "application/geo+json",
      body: geo("nuts_level2_2021.geojson"),
    });
  if (path === "/geo/world")
    return route.fulfill({ contentType: "application/geo+json", body: geo("world_countries.geojson") });

  // FRED
  if (path === "/fred/states") return json(FRED_STATES);
  if (path === "/fred/indicators")
    return json([
      {
        indicator_id: "unemployment_rate",
        name: "Unemployment Rate",
        category: "Labor",
        units: "%",
        frequency: "Monthly",
        seasonal_adjustment: "SA",
        region_type: "state",
        series_group: "1224",
        example_series_id: "CAUR",
        min_date: "1990",
        max_date: "2024",
        notes: "State unemployment rate.",
      },
    ]);
  if (path.startsWith("/fred/indicators/") && path.endsWith("/values"))
    return json(regionValues(FRED_STATES.map((s) => s.id)));

  // Eurostat
  if (path === "/eurostat/regions") return json(EU_REGIONS);
  if (path === "/eurostat/indicators")
    return json([
      {
        indicator_id: "gdp_per_capita_pps",
        name: "GDP per capita (PPS)",
        category: "Economy",
        dataset: "nama_10r_2gdp",
        filters: '{"unit":"EUR_HAB"}',
        units: "EUR per inhabitant",
        frequency: "Annual",
        nuts_level: 2,
        min_year: 2018,
        max_year: 2023,
        source_label: "Eurostat",
        notes: null,
      },
    ]);
  if (path.startsWith("/eurostat/indicators/") && path.endsWith("/values"))
    return json(regionValues(EU_REGIONS.map((r) => r.id)));

  // Yahoo
  if (path === "/yahoo/metadata")
    return json([
      { ticker: "AAPL", asset_name: "Apple Inc", category: "Companies", sector: "Technology" },
      { ticker: "MSFT", asset_name: "Microsoft", category: "Companies", sector: "Technology" },
      { ticker: "NVDA", asset_name: "NVIDIA", category: "Companies", sector: "Technology" },
      { ticker: "XOM", asset_name: "Exxon", category: "Companies", sector: "Energy" },
      { ticker: "^GSPC", asset_name: "S&P 500", category: "Indices", sector: null },
    ]);
  if (path === "/yahoo/prices") {
    const rows = [];
    for (const ticker of ["AAPL", "MSFT", "NVDA", "XOM", "^GSPC"]) {
      for (const p of monthlySeries(100 + ticker.length * 10)) {
        rows.push({ date: p.date, open: p.o, high: p.h, low: p.l, close: p.c, volume: p.v, ticker });
      }
    }
    return json(rows);
  }

  // Crypto
  if (path === "/crypto/metadata")
    return json(
      ["BTC", "ETH", "SOL", "BNB", "XRP"].map((base, i) => ({
        symbol: `${base}USDT`,
        base_asset: base,
        quote_asset: "USDT",
        status: "TRADING",
        rank: i + 1,
        description: null,
        last_price: 1000 * (5 - i),
        price_change_percent_24h: i % 2 ? -1.5 : 2.3,
        high_24h: 1,
        low_24h: 1,
        quote_volume_24h: 1e9 / (i + 1),
        trade_count_24h: 100000,
      })),
    );
  if (path === "/crypto/prices") {
    const rows = [];
    for (const base of ["BTC", "ETH", "SOL", "BNB", "XRP"]) {
      for (const p of monthlySeries(base === "BTC" ? 20000 : 100)) {
        rows.push({
          date: p.date,
          open: p.o,
          high: p.h,
          low: p.l,
          close: p.c,
          volume: p.v,
          quote_volume: p.v * p.c,
          symbol: `${base}USDT`,
          base_asset: base,
        });
      }
    }
    return json(rows);
  }

  // News
  if (path === "/news/collections") return json({ collections: ["actually_relevant", "world_bank"] });
  if (path.includes("/news/collections/") && path.endsWith("/articles"))
    return json(
      Array.from({ length: 6 }, (_, i) => ({
        id: `a${i}`,
        title: `Inflation and growth outlook ${i}`,
        text: "The economy shows resilient growth as inflation moderates and markets rally across sectors.",
        url: "https://example.com/story",
        published: `2023-0${(i % 9) + 1}-01`,
        source: "Example",
        topic: "economy",
        sentiment: "neutral",
        collection: "actually_relevant",
      })),
    );
  if (path === "/news/search")
    return json({
      articles: [
        {
          id: "s1",
          title: "Central banks signal rate cuts",
          text: "Policymakers hint at easing as inflation cools.",
          url: "https://example.com/s1",
          published: "2023-06-01",
          source: "Example",
          topic: "economy",
          sentiment: "positive",
          collection: "world_bank",
          score: 0.87,
        },
      ],
      message: null,
    });

  return json([]);
}

async function fulfillChat(route: Route) {
  const frames = [
    { type: "step", node: "supervisor" },
    { type: "step", node: "sql_agent" },
    { type: "token", delta: "GDP grew " },
    { type: "token", delta: "steadily." },
    { type: "final", answer: "GDP grew steadily over the period.", artifacts: {}, usage: {} },
  ];
  const body = frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("");
  await route.fulfill({ contentType: "text/event-stream", body });
}

// Match only real BFF calls (pathname starting with `/api/`). A bare `**/api/**`
// glob would also swallow Vite's own module URLs for the app's `src/api/*.ts`
// files (served at `/src/api/…`), returning JSON for a JS module and white-screening.
const isApi = (url: string) => new URL(url).pathname.startsWith("/api/");

test.beforeEach(async ({ page }) => {
  await page.route(
    (url) => url.pathname === "/api/agent/chat/stream",
    fulfillChat,
  );
  await page.route(
    (url) => isApi(url.href) && url.pathname !== "/api/agent/chat/stream",
    fulfillApi,
  );
});

test.describe("M2 pages", () => {
  test("FRED regional page renders map + rankings + trend", async ({ page }) => {
    await page.goto("/regional/fred");
    await expect(
      page.getByRole("heading", { name: /United States — Regional Statistics/ }),
    ).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    expect(await page.locator("canvas").count()).toBeGreaterThanOrEqual(3);
    await page.screenshot({ path: "test-results/m2-fred.png", fullPage: true });
  });

  test("Eurostat regional page renders", async ({ page }) => {
    await page.goto("/regional/eurostat");
    await expect(
      page.getByRole("heading", { name: /European Union — Regional Statistics/ }),
    ).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    await page.screenshot({ path: "test-results/m2-eurostat.png", fullPage: true });
  });

  test("Yahoo page renders trend, candlestick, treemap, heatmap", async ({ page }) => {
    await page.goto("/yahoo");
    await expect(page.getByRole("heading", { name: "Yahoo Finance Dashboard" })).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    expect(await page.locator("canvas").count()).toBeGreaterThanOrEqual(3);
    await page.screenshot({ path: "test-results/m2-yahoo.png", fullPage: true });
  });

  test("Crypto page renders overview table + charts", async ({ page }) => {
    await page.goto("/crypto");
    await expect(page.getByRole("heading", { name: "Crypto (Binance) Dashboard" })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Market overview/ })).toBeVisible();
    await expect(page.getByText("BTCUSDT").first()).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });
    await page.screenshot({ path: "test-results/m2-crypto.png", fullPage: true });
  });

  test("News page renders word cloud and returns search results", async ({ page }) => {
    await page.goto("/news");
    await expect(page.getByRole("heading", { name: "News Explorer" })).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });

    await page.getByPlaceholder("Search the news corpus…").fill("rate cuts");
    await page.getByRole("button", { name: "Search" }).click();
    await expect(page.getByText("Central banks signal rate cuts")).toBeVisible();
    await page.screenshot({ path: "test-results/m2-news.png", fullPage: true });
  });

  test("AI chat streams steps + final answer", async ({ page }) => {
    await page.goto("/ai");
    await expect(page.getByRole("heading", { name: "AI Analyst" })).toBeVisible();

    await page.getByPlaceholder("Ask the AI analyst…").fill("How did GDP grow?");
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(page.getByText("GDP grew steadily over the period.")).toBeVisible({
      timeout: 15000,
    });
    await expect(page.getByText(/router → sql_agent/)).toBeVisible();
    await page.screenshot({ path: "test-results/m2-ai.png", fullPage: true });
  });
});
