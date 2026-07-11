import { describe, expect, it } from "vitest";

import { buildDistributionOption } from "./distribution";
import { buildTimeTrendOption, type TrendPoint } from "./timeTrend";
import { buildWorldChoroplethOption } from "./worldChoropleth";

const distTokens = { bar: "#10c8f1", reference: "#94a3b8", selected: "#fde047" };

describe("buildTimeTrendOption", () => {
  it("emits one line series per economy over the merged year axis", () => {
    const series = new Map<string, TrendPoint[]>([
      ["USA", [
        { year: 2000, value: 1 },
        { year: 2001, value: 2 },
      ]],
      ["DEU", [{ year: 2001, value: 5 }]],
    ]);
    const option = buildTimeTrendOption(series, {
      showMarkers: false,
      valueLabel: "USD",
      labelFor: (code) => code,
    });
    expect(option.xAxis).toMatchObject({ data: ["2000", "2001"] });
    expect(Array.isArray(option.series)).toBe(true);
    expect((option.series as unknown[]).length).toBe(2);
  });
});

describe("buildDistributionOption", () => {
  it("bins values into a bar histogram", () => {
    const values = Array.from({ length: 50 }, (_, i) => i);
    const option = buildDistributionOption(values, {
      plotType: "histogram",
      orientation: "vertical",
      valueLabel: "x",
      tokens: distTokens,
    });
    const series = (option.series as { type: string; data: number[] }[])[0];
    expect(series.type).toBe("bar");
    expect(series.data.reduce((a, b) => a + b, 0)).toBe(50);
  });

  it("produces a 5-number box for a box plot", () => {
    const option = buildDistributionOption([1, 2, 3, 4, 5], {
      plotType: "box",
      orientation: "vertical",
      valueLabel: "x",
      tokens: distTokens,
    });
    const series = (option.series as { type: string; data: number[][] }[])[0];
    expect(series.type).toBe("boxplot");
    expect(series.data[0]).toHaveLength(5);
    expect(series.data[0][0]).toBe(1);
    expect(series.data[0][4]).toBe(5);
  });

  it("returns an empty option for no values", () => {
    expect(buildDistributionOption([], {
      plotType: "histogram",
      orientation: "vertical",
      valueLabel: "x",
      tokens: distTokens,
    })).toEqual({});
  });
});

describe("buildWorldChoroplethOption", () => {
  it("maps rows to a keyed map series with a visualMap range", () => {
    const option = buildWorldChoroplethOption(
      [
        { code: "USA", name: "United States", value: 10 },
        { code: "DEU", name: "Germany", value: 20 },
      ],
      {
        mapName: "world",
        valueLabel: "USD",
        tokens: { sequential: ["#000", "#fff"], mapLand: "#111", mapBorder: "#222" },
      },
    );
    const series = (option.series as { type: string; map: string; data: unknown[] }[])[0];
    expect(series.type).toBe("map");
    expect(series.map).toBe("world");
    expect(series.data).toHaveLength(2);
    expect(option.visualMap).toMatchObject({ min: 10, max: 20 });
  });
});
