import type { EChartsOption } from "echarts";

import type { TrendPoint } from "./timeTrend";

/** One predicted point with its confidence interval (mirrors the forecaster). */
export interface ForecastPoint {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

export interface ForecastTrendOptions {
  showMarkers: boolean;
  valueLabel: string;
  /** Map an economy code to its display label (e.g. "Germany"). */
  labelFor: (code: string) => string;
  /** Explicit series colour ramp (theme `series.colorway`) so bands match lines. */
  colorway: string[];
  /** Opacity of the shaded confidence band (theme `charts.confidenceBandAlpha`). */
  confidenceBandAlpha: number;
}

/** `[timestampMs, value]` datum for a time-axis series. */
type TimeDatum = [number, number];

const yearToMs = (year: number): number => Date.UTC(year, 0, 1);

/** Parse the forecaster's `"%Y-%m-%d %H:%M:%S"` (or ISO) timestamp to epoch ms. */
function dsToMs(ds: string): number {
  const iso = ds.includes("T") ? ds : ds.replace(" ", "T") + "Z";
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? Date.parse(ds) : ms;
}

/**
 * Historical lines (solid) plus, per series, a dashed forecast continuation and
 * a shaded confidence band — all rendered on a shared **time** axis so future
 * forecast periods extend naturally past the history. Every series for one code
 * shares the code's colour and legend name, so a single legend toggle hides the
 * line, its forecast, and its band together. Colours come from the theme ramp
 * (never hard-coded); the band opacity is the `confidenceBandAlpha` token.
 */
export function buildForecastTrendOption(
  historyByCode: Map<string, TrendPoint[]>,
  forecastByCode: Map<string, ForecastPoint[]>,
  { showMarkers, valueLabel, labelFor, colorway, confidenceBandAlpha }: ForecastTrendOptions,
): EChartsOption {
  const codes = [...historyByCode.keys()];
  const legendNames: string[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any[] = [];

  codes.forEach((code, index) => {
    const color = colorway[index % colorway.length];
    const name = labelFor(code);
    legendNames.push(name);

    const history = [...(historyByCode.get(code) ?? [])]
      .filter((p) => Number.isFinite(p.value))
      .sort((a, b) => a.year - b.year);
    const histData: TimeDatum[] = history.map((p) => [yearToMs(p.year), p.value]);

    series.push({
      name,
      type: "line",
      color,
      showSymbol: showMarkers,
      symbolSize: 6,
      connectNulls: true,
      emphasis: { focus: "series" },
      data: histData,
    });

    const forecast = forecastByCode.get(code);
    if (!forecast || forecast.length === 0) return;

    // Connect the forecast to the last historical point so the join reads cleanly.
    const last = history[history.length - 1];
    const connector: TimeDatum[] = last ? [[yearToMs(last.year), last.value]] : [];
    const fcLine: TimeDatum[] = forecast.map((p) => [dsToMs(p.ds), p.yhat]);
    series.push({
      name,
      type: "line",
      color,
      showSymbol: showMarkers,
      symbolSize: 6,
      lineStyle: { type: "dashed" },
      data: [...connector, ...fcLine],
    });

    // Confidence band = transparent lower baseline + stacked (upper − lower) area.
    const stack = `conf-${code}`;
    const lowerData: TimeDatum[] = forecast.map((p) => [dsToMs(p.ds), p.yhat_lower]);
    const rangeData: TimeDatum[] = forecast.map((p) => [
      dsToMs(p.ds),
      p.yhat_upper - p.yhat_lower,
    ]);
    series.push({
      name,
      type: "line",
      stack,
      color,
      symbol: "none",
      silent: true,
      lineStyle: { opacity: 0 },
      data: lowerData,
    });
    series.push({
      name,
      type: "line",
      stack,
      color,
      symbol: "none",
      silent: true,
      lineStyle: { opacity: 0 },
      areaStyle: { color, opacity: confidenceBandAlpha },
      data: rangeData,
    });
  });

  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, type: "scroll", data: legendNames },
    grid: { left: 64, right: 24, top: 40, bottom: 40 },
    xAxis: { type: "time" },
    yAxis: { type: "value", scale: true, name: valueLabel },
    series,
  };
}
