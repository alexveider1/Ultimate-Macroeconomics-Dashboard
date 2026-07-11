import type { EChartsOption } from "echarts";

export interface HistogramOptions {
  binCount?: number;
  color: string;
  valueLabel: string;
  /** Optional vertical reference line (e.g. the mean distance). */
  markerValue?: number;
  markerColor?: string;
  markerLabel?: string;
}

/**
 * A fixed-width-bin histogram of `values`. Colours come from the caller (theme
 * tokens), never hard-coded. An optional `markerValue` draws a labelled vertical
 * reference line via `markLine`.
 */
export function buildHistogramOption(
  values: number[],
  { binCount = 30, color, valueLabel, markerValue, markerColor, markerLabel }: HistogramOptions,
): EChartsOption {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) {
    return { xAxis: { type: "value", name: valueLabel }, yAxis: { type: "value" }, series: [] };
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const width = (max - min) / binCount || 1;
  const counts = new Array(binCount).fill(0) as number[];
  for (const v of finite) {
    const idx = Math.min(binCount - 1, Math.floor((v - min) / width));
    counts[idx] += 1;
  }
  const data = counts.map((count, i) => [min + (i + 0.5) * width, count]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any = {
    type: "bar",
    color,
    barWidth: "99%",
    data,
  };
  if (markerValue !== undefined && Number.isFinite(markerValue)) {
    series.markLine = {
      silent: true,
      symbol: "none",
      lineStyle: { color: markerColor ?? color, type: "dashed" },
      label: { formatter: markerLabel ?? "" },
      data: [{ xAxis: markerValue }],
    };
  }

  return {
    tooltip: { trigger: "axis" },
    grid: { left: 56, right: 24, top: 24, bottom: 48 },
    xAxis: { type: "value", name: valueLabel, scale: true },
    yAxis: { type: "value", name: "Count" },
    series: [series],
  };
}
