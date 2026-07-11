import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Page, Route } from "@playwright/test";

/**
 * Shared, self-contained `/api/**` mocks for the e2e suite. Every endpoint the
 * dashboard reads is stubbed here so the whole Playwright run needs no BFF (nor
 * any downstream forecaster/clustering/agent) — the two GeoJSON maps are served
 * from the real bundled files on disk, everything else is synthetic but shaped
 * exactly like the BFF JSON contract (`bff/models.py` ↔ `src/api/types.ts`).
 *
 * `registerApiMocks(page)` wires it into a page; individual tests can still layer
 * a more specific `page.route(...)` on top (Playwright runs the most-recently
 * registered handler first) to exercise an error/empty path.
 */

const CONFIGS = resolve(process.cwd(), "..", "_container_data", "_configs");
export const geo = (name: string): string => readFileSync(resolve(CONFIGS, name), "utf-8");

/** A full dark theme satisfying `assertValidTheme` (mode: dark). */
export const DARK_THEME = {
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

/** A full light theme (mode: light) — used to prove a runtime theme switch. */
export const LIGHT_GREEN_THEME = {
  label: "Light Green",
  mode: "light",
  fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
  chrome: {
    background: "#f5faf3",
    surface: "#ffffff",
    border: "#d6e4d0",
    text: "#14251a",
    textMuted: "#4b6b57",
    primary: "#2e8b57",
    primaryText: "#ffffff",
  },
  series: {
    colorway: ["#2e8b57", "#e07a5f", "#e9c46a", "#457b9d", "#8367c7"],
    sequential: ["#f5faf3", "#2e8b57"],
    diverging: ["#2e8b57", "#f5faf3", "#e07a5f"],
  },
  semantic: {
    positive: "#16a34a",
    negative: "#dc2626",
    referenceLine: "#4b6b57",
    selectedMarker: "#ca8a04",
    mapCoastline: "#457b9d",
    mapLand: "#e6efe2",
    sectors: { agriculture: "#16a34a", manufacturing: "#d97706", services: "#2563eb" },
  },
  charts: {
    confidenceBandAlpha: 0.18,
    gridLine: "#d6e4d0",
    axisLine: "#b7ccae",
    tooltipBackground: "#ffffff",
    tooltipText: "#14251a",
  },
  wordcloud: { background: "#f5faf3", colors: ["#14532d", "#166534", "#22c55e", "#86efac"] },
};

export const ECONOMIES = ["USA", "CHN", "DEU", "FRA", "JPN", "GBR"];
const INDICATORS = [
  { id: "NV.AGR.TOTL.ZS", name: "Agriculture (% GDP)" },
  { id: "NV.IND.MANF.ZS", name: "Manufacturing (% GDP)" },
  { id: "NV.SRV.TOTL.ZS", name: "Services (% GDP)" },
];

/**
 * Every `world_bank_download_config.json` section a dashboard page maps to (see
 * `config/dashboardPages.ts`). The mock returns the same indicator triplet under
 * each so *any* dashboard route renders GraphBoxes, not just economy-structure.
 */
const DASHBOARD_SECTIONS = [
  "General Economics Indicators",
  "Structure",
  "Finance and Monetary",
  "Fiscal",
  "Trade and External sector",
  "Demography",
  "Governance and Institutions",
  "Technology and Innovations",
  "Health and wellbeing",
  "Education and Human Capital",
  "Environment and ecology",
];

function dashboardConfig(): Record<string, { id: string; name: string }[]> {
  const config: Record<string, { id: string; name: string }[]> = {};
  for (const section of DASHBOARD_SECTIONS) {
    config[section] = INDICATORS.map((it) => ({ id: it.id, name: it.name }));
  }
  return config;
}

/** WB values: every economy over 2015–2020 with a deterministic value. */
function indicatorValues(id: string) {
  const points = [];
  for (const [i, economy] of ECONOMIES.entries()) {
    for (let year = 2015; year <= 2020; year += 1) {
      points.push({ economy, year, value: 10 + i * 5 + (year - 2015) * 2 + id.length });
    }
  }
  return { indicator_id: id, name: id, points };
}

/** Monthly OHLCV points for 2022–2023 (24 points) for one series. */
function monthlySeries(base: number) {
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
      rows.push({ region: code, year, value: 20 + (code.charCodeAt(0) % 30) + (year - 2018) * 2 });
    }
  }
  return rows;
}

/** The comprehensive `/api/**` route handler (everything except the chat stream). */
export async function fulfillApi(route: Route): Promise<void> {
  const url = new URL(route.request().url());
  const path = url.pathname.replace(/^\/api/, "");
  const json = (data: unknown) => route.fulfill({ json: data });
  const geojson = (name: string) =>
    route.fulfill({ contentType: "application/geo+json", body: geo(name) });

  // Config / theming
  if (path === "/config/themes")
    return json({ active: "dark", themes: { dark: DARK_THEME, "light-green": LIGHT_GREEN_THEME } });
  if (path === "/config/theme") return json(DARK_THEME);
  if (path === "/config/dashboard") return json(dashboardConfig());

  // GeoJSON maps (real bundled files)
  if (path === "/geo/world") return geojson("world_countries.geojson");
  if (path === "/geo/us-states") return geojson("us_states.geojson");
  if (path === "/geo/nuts2") return geojson("nuts_level2_2021.geojson");

  // World Bank
  if (path === "/worldbank/countries")
    return json(
      ECONOMIES.map((id) => ({ id, name: id, region: null, income_level: null, aggregate: false })),
    );
  const wbInfo = path.match(/^\/worldbank\/indicators\/([^/]+)$/);
  if (wbInfo) return json({ indicator_id: wbInfo[1], name: wbInfo[1], units: "% of GDP" });
  const wbValues = path.match(/^\/worldbank\/indicators\/([^/]+)\/values$/);
  if (wbValues) return json(indicatorValues(wbValues[1]));

  // Forecast / cluster proxies
  if (path === "/forecast/models") return json({ models: ["prophet", "arima", "moving_average"] });
  if (path === "/forecast")
    return json({
      model_used: "prophet",
      forecast: [2021, 2022, 2023].map((year) => ({
        ds: `${year}-01-01 00:00:00`,
        yhat: 30 + (year - 2021) * 2,
        yhat_lower: 28 + (year - 2021) * 2,
        yhat_upper: 32 + (year - 2021) * 2,
      })),
    });
  if (path === "/cluster/methods")
    return json({
      available_methods: ["kmeans", "dbscan", "hdbscan", "hierarchical"],
      available_reductions: ["tsne", "pca", "umap"],
    });
  if (path === "/cluster") {
    const body = route.request().postDataJSON() as { dataframe: Record<string, unknown>[] };
    const dataframe = body.dataframe.map((row, index) => ({
      ...row,
      cluster: index % 2,
      __viz_x: index,
      __viz_y: -index,
    }));
    return json({
      method_used: "kmeans",
      dataframe,
      visualization_mode: "tsne",
      visualization_columns: ["__viz_x", "__viz_y"],
      visualization_labels: ["dim 1", "dim 2"],
    });
  }

  // FRED / Eurostat regional
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
  if (path === "/news/collections")
    return json({ collections: ["actually_relevant", "world_bank"] });
  if (path.includes("/news/collections/") && path.endsWith("/articles"))
    return json(
      Array.from({ length: 8 }, (_, i) => ({
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
  if (path.endsWith("/projection")) {
    const points = Array.from({ length: 8 }, (_, i) => ({
      id: `a${i}`,
      title: `Story ${i}`,
      cluster: String(i % 3),
      x: Math.cos(i),
      y: Math.sin(i),
      z: null,
    }));
    return json({
      points,
      output_dim: 2,
      mode: "tsne",
      distances: [0.1, 0.2, 0.25, 0.3, 0.5, 0.6, 0.8],
      query_id: "a0",
      query_title: "Story 0",
      message: null,
    });
  }

  // Agent (non-stream)
  if (path === "/agent/plots/interpret")
    return json({ description: "The chart shows a steady upward trend over the period." });

  return json([]);
}

/** A well-formed SSE chat stream: two steps, two tokens, a final answer. */
export async function fulfillChat(route: Route): Promise<void> {
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

/** A chat stream that ends in an `error` frame (downstream agent failure). */
export async function fulfillChatError(route: Route): Promise<void> {
  const frames = [
    { type: "step", node: "supervisor" },
    { type: "error", answer: "The analyst service is temporarily unavailable." },
  ];
  const body = frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("");
  await route.fulfill({ contentType: "text/event-stream", body });
}

const isApi = (url: URL) => url.pathname.startsWith("/api/");

/**
 * Register the standard mocks on a page: the SSE chat stream + the comprehensive
 * `/api/**` handler. Call from `test.beforeEach`. A test may register a more
 * specific `page.route` afterwards to override one endpoint (e.g. an error path).
 */
export async function registerApiMocks(page: Page): Promise<void> {
  await page.route((url) => url.pathname === "/api/agent/chat/stream", fulfillChat);
  await page.route(
    (url) => isApi(url) && url.pathname !== "/api/agent/chat/stream",
    fulfillApi,
  );
}
