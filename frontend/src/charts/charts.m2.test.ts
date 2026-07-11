import { describe, expect, it } from "vitest";

import { buildCandlestickOption } from "./candlestick";
import { buildCorrelationHeatmapOption, computeReturnsCorrelation } from "./correlationHeatmap";
import { buildChoroplethOption } from "./mapChoropleth";
import { buildRankingBarOption } from "./rankingBar";
import { computeWordFrequencies } from "./wordcloud";

const seq: [string, string] = ["#001", "#eee"];

describe("buildChoroplethOption", () => {
  it("keys data by the feature nameProperty and reverses the ramp when asked", () => {
    const option = buildChoroplethOption([{ key: "AL", name: "Alabama", value: 5 }], {
      mapName: "us-states",
      nameProperty: "postal",
      valueLabel: "Rate",
      reverse: true,
      tokens: { sequential: seq, mapLand: "#111", mapBorder: "#222" },
    });
    const series = (option.series as { nameProperty: string; data: { name: string }[] }[])[0];
    expect(series.nameProperty).toBe("postal");
    expect(series.data[0].name).toBe("AL");
    expect((option.visualMap as { inRange: { color: string[] } }).inRange.color).toEqual([
      "#eee",
      "#001",
    ]);
  });
});

describe("buildRankingBarOption", () => {
  it("picks the top-N and puts the most-extreme bar on top of the axis", () => {
    const option = buildRankingBarOption(
      [
        { label: "a", value: 1 },
        { label: "b", value: 3 },
        { label: "c", value: 2 },
      ],
      { valueLabel: "x", color: "#f00", ascending: false, topN: 2 },
    );
    // Category axis renders index 0 at the bottom → largest must be last.
    expect((option.yAxis as { data: string[] }).data).toEqual(["c", "b"]);
    expect((option.series as { data: number[] }[])[0].data).toEqual([2, 3]);
  });
});

describe("buildCandlestickOption", () => {
  it("emits [open, close, low, high] per candle", () => {
    const option = buildCandlestickOption(
      [{ date: "2020-01-01", open: 1, high: 4, low: 0, close: 2 }],
      { upColor: "#0f0", downColor: "#f00" },
    );
    const series = (option.series as { data: number[][] }[])[0];
    expect(series.data[0]).toEqual([1, 2, 0, 4]);
  });
});

describe("computeReturnsCorrelation", () => {
  it("gives a unit diagonal and detects perfect anti-correlation", () => {
    const rows = [
      { date: "2020-01-01", key: "A", close: 100 },
      { date: "2020-01-02", key: "A", close: 110 },
      { date: "2020-01-03", key: "A", close: 99 },
      { date: "2020-01-01", key: "B", close: 100 },
      { date: "2020-01-02", key: "B", close: 90 },
      { date: "2020-01-03", key: "B", close: 99 },
    ];
    const { labels, matrix } = computeReturnsCorrelation(rows);
    expect(labels).toEqual(["A", "B"]);
    expect(matrix[0][0]).toBeCloseTo(1, 5);
    // A returns [+0.1, -0.1] vs B returns [-0.1, +0.1] → correlation -1.
    expect(matrix[0][1]).toBeCloseTo(-1, 5);
  });

  it("produces heatmap cells for every pair", () => {
    const corr = computeReturnsCorrelation([
      { date: "d1", key: "A", close: 1 },
      { date: "d2", key: "A", close: 2 },
      { date: "d1", key: "B", close: 2 },
      { date: "d2", key: "B", close: 4 },
    ]);
    const option = buildCorrelationHeatmapOption(corr, { diverging: ["#00f", "#fff", "#f00"] });
    expect((option.series as { data: unknown[] }[])[0].data).toHaveLength(4);
  });
});

describe("computeWordFrequencies", () => {
  it("drops stop-words and ranks by frequency", () => {
    const words = computeWordFrequencies("the economy economy grows and the market", 10);
    expect(words[0]).toEqual({ name: "economy", value: 2 });
    expect(words.some((w) => w.name === "the")).toBe(false);
  });
});
