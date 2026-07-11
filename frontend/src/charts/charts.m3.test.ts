import { describe, expect, it } from "vitest";

import { buildClusterScatterOption } from "./clusterScatter";
import { buildForecastTrendOption, type ForecastPoint } from "./forecastTrend";
import { buildHistogramOption } from "./histogram";
import type { TrendPoint } from "./timeTrend";

const COLORWAY = ["#111", "#222", "#333"];

describe("buildForecastTrendOption", () => {
  it("emits history + dashed forecast + a two-series confidence band per code", () => {
    const history = new Map<string, TrendPoint[]>([
      ["USA", [
        { year: 2000, value: 10 },
        { year: 2001, value: 12 },
      ]],
    ]);
    const forecast = new Map<string, ForecastPoint[]>([
      ["USA", [{ ds: "2002-01-01 00:00:00", yhat: 14, yhat_lower: 13, yhat_upper: 15 }]],
    ]);
    const option = buildForecastTrendOption(history, forecast, {
      showMarkers: false,
      valueLabel: "GDP",
      labelFor: (c) => c,
      colorway: COLORWAY,
      confidenceBandAlpha: 0.2,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series = option.series as any[];
    // history line + forecast line + 2 band series = 4.
    expect(series).toHaveLength(4);
    expect(series[0].data[0]).toEqual([Date.UTC(2000, 0, 1), 10]);
    expect(series[1].lineStyle.type).toBe("dashed");
    // Forecast line is prefixed with the connector to the last historical point.
    expect(series[1].data[0]).toEqual([Date.UTC(2001, 0, 1), 12]);
    // Band range = upper − lower = 2.
    expect(series[3].areaStyle.opacity).toBe(0.2);
    expect(series[3].data[0][1]).toBe(2);
    // Every series shares the code's colour + legend name → one toggle hides all.
    expect(series.every((s) => s.color === "#111" && s.name === "USA")).toBe(true);
    expect((option.legend as { data: string[] }).data).toEqual(["USA"]);
  });
});

describe("buildClusterScatterOption", () => {
  it("makes one 2D series per cluster + a highlight series", () => {
    const option = buildClusterScatterOption(
      [
        { x: 1, y: 2, cluster: "0", label: "A", detail: "a" },
        { x: 3, y: 4, cluster: "1", label: "B", detail: "b" },
      ],
      { is3d: false, xLabel: "x", yLabel: "y", colorway: COLORWAY, highlightLabel: "b", highlightColor: "#ff0" },
    );
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series = option.series as any[];
    // 2 clusters + 1 highlight.
    expect(series).toHaveLength(3);
    expect(series[0].type).toBe("scatter");
    expect(series[2].name).toBe("Selected");
    expect(series[2].color).toBe("#ff0");
    expect(series[2].data[0].value).toEqual([3, 4]);
  });

  it("switches to scatter3D and 3-tuple data in 3D mode", () => {
    const option = buildClusterScatterOption([{ x: 1, y: 2, z: 3, cluster: "0", label: "A" }], {
      is3d: true,
      xLabel: "x",
      yLabel: "y",
      zLabel: "z",
      colorway: COLORWAY,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series = option.series as any[];
    expect(series[0].type).toBe("scatter3D");
    expect(series[0].data[0].value).toEqual([1, 2, 3]);
  });
});

describe("buildHistogramOption", () => {
  it("bins values and draws the mean marker line", () => {
    const option = buildHistogramOption([0, 0, 1, 1, 1, 2], {
      binCount: 3,
      color: "#111",
      valueLabel: "d",
      markerValue: 1,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series = option.series as any[];
    const counts = series[0].data.map((d: [number, number]) => d[1]);
    expect(counts.reduce((a: number, b: number) => a + b, 0)).toBe(6);
    expect(series[0].markLine.data[0].xAxis).toBe(1);
  });
});
