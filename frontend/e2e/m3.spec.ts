import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, type Route, test } from "@playwright/test";

/**
 * M3 hard features — GraphBox forecasting + LLM plot description, the clustering
 * sandbox, and the News embedding map — all driven by mocked `/api/**` responses
 * so the suite is self-contained (no BFF/forecaster/clustering/agent needed).
 */

const CONFIGS = resolve(process.cwd(), "..", "_container_data", "_configs");
const geo = (name: string) => readFileSync(resolve(CONFIGS, name), "utf-8");

const THEME = {
  label: "Dark",
  mode: "dark",
  fontFamily: "Inter, sans-serif",
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

const ECONOMIES = ["USA", "DEU", "FRA", "JPN", "GBR"];
const INDICATORS = [
  { id: "NV.AGR.TOTL.ZS", name: "Agriculture (% GDP)" },
  { id: "NV.IND.MANF.ZS", name: "Manufacturing (% GDP)" },
  { id: "NV.SRV.TOTL.ZS", name: "Services (% GDP)" },
];

/** Build a WB values response: every economy over 2015–2020 with a deterministic value. */
function indicatorValues(id: string) {
  const points = [];
  for (const [i, economy] of ECONOMIES.entries()) {
    for (let year = 2015; year <= 2020; year += 1) {
      points.push({ economy, year, value: 10 + i * 5 + (year - 2015) * 2 + id.length });
    }
  }
  return { indicator_id: id, name: id, points };
}

async function fulfillApi(route: Route) {
  const url = new URL(route.request().url());
  const path = url.pathname.replace(/^\/api/, "");
  const json = (data: unknown) => route.fulfill({ json: data });

  if (path === "/config/themes") return json({ active: "dark", themes: { dark: THEME } });
  if (path === "/geo/world")
    return route.fulfill({ contentType: "application/geo+json", body: geo("world_countries.geojson") });

  if (path === "/config/dashboard")
    return json({ Structure: INDICATORS.map((it) => ({ id: it.id, name: it.name })) });
  if (path === "/worldbank/countries")
    return json(
      ECONOMIES.map((id) => ({ id, name: id, region: null, income_level: null, aggregate: false })),
    );
  const infoMatch = path.match(/^\/worldbank\/indicators\/([^/]+)$/);
  if (infoMatch) return json({ indicator_id: infoMatch[1], name: infoMatch[1], units: "% of GDP" });
  const valuesMatch = path.match(/^\/worldbank\/indicators\/([^/]+)\/values$/);
  if (valuesMatch) return json(indicatorValues(valuesMatch[1]));

  if (path === "/forecast/models") return json({ models: ["prophet", "arima", "moving_average"] });
  if (path === "/forecast") {
    return json({
      model_used: "prophet",
      forecast: [2021, 2022, 2023].map((year) => ({
        ds: `${year}-01-01 00:00:00`,
        yhat: 30 + (year - 2021) * 2,
        yhat_lower: 28 + (year - 2021) * 2,
        yhat_upper: 32 + (year - 2021) * 2,
      })),
    });
  }

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

  if (path === "/news/collections") return json({ collections: ["actually_relevant"] });
  if (path.includes("/news/collections/") && path.endsWith("/articles"))
    return json(
      Array.from({ length: 8 }, (_, i) => ({
        id: `a${i}`,
        title: `Story ${i}`,
        text: "The economy shows resilient growth as inflation moderates across sectors.",
        url: "https://example.com",
        published: "2023-01-01",
        source: "Example",
        topic: "economy",
        sentiment: "neutral",
        collection: "actually_relevant",
      })),
    );
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

  if (path === "/agent/plots/interpret")
    return json({ description: "The chart shows a steady upward trend over the period." });

  return json([]);
}

const isApi = (url: URL) => url.pathname.startsWith("/api/");

test.beforeEach(async ({ page }) => {
  await page.route((url) => isApi(url), fulfillApi);
});

/**
 * NOTE: Radix popovers (`MultiSelect`, GraphBox settings gear) don't position
 * correctly under headless Chromium — Floating UI leaves the content off-screen
 * — so popover-gated interactions can't be driven here. Those flows are covered by
 * unit tests (`clusterMatrix.test.ts`, `charts.m3.test.ts`) and by the News
 * embedding map below (same cluster-scatter builder, end-to-end). Popovers work in
 * real/headed browsers. The tests here exercise everything reachable without a popover.
 */
test.describe("M3 features", () => {
  test("clustering sandbox renders its controls", async ({ page }) => {
    await page.goto("/constructors/clustering");
    await expect(page.getByRole("heading", { name: "Clustering Sandbox" })).toBeVisible();
    await expect(page.getByText("Category")).toBeVisible();
    await expect(page.getByRole("button", { name: /Pick indicators/ })).toBeVisible();
    await expect(
      page.getByText("Select at least two indicators to continue."),
    ).toBeVisible();
    await page.screenshot({ path: "test-results/m3-clustering.png", fullPage: true });
  });

  test("news embedding map projects points + distance histogram", async ({ page }) => {
    await page.goto("/news");
    await expect(page.getByRole("heading", { name: "Embedding map" })).toBeVisible();

    await page.getByRole("button", { name: "Run embedding map" }).click();
    // Scatter + histogram canvases appear once the projection resolves.
    await expect
      .poll(async () => page.locator("canvas").count(), { timeout: 20000 })
      .toBeGreaterThanOrEqual(2);
    await page.screenshot({ path: "test-results/m3-embedding.png", fullPage: true });
  });

  test("GraphBox describes the chart via the LLM (inline, no popover)", async ({ page }) => {
    await page.goto("/dashboard/economy-structure");
    await expect(page.locator("canvas").first()).toBeVisible({ timeout: 20000 });

    // The describe controls are inline below each chart (not popover-gated).
    await page.getByRole("button", { name: "Factual reading" }).first().click();
    await expect(page.getByText("steady upward trend")).toBeVisible({ timeout: 20000 });
    await page.screenshot({ path: "test-results/m3-graphbox.png", fullPage: true });
  });
});
