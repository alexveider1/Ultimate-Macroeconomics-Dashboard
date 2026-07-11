import type { EChartsOption } from "echarts";

export interface PricePoint {
  date: string;
  value: number;
}

export interface PriceTrendOptions {
  valueLabel: string;
  /** Logarithmic y-axis (crypto coins span very different price levels). */
  logY?: boolean;
  /** Fill under each line (used for the indices overview). */
  area?: boolean;
}

/**
 * Multi-series line over a real time axis (dates). One line per label; colours
 * come from the registered theme colorway. Optional log y-axis and area fill.
 */
export function buildPriceTrendOption(
  seriesByLabel: Map<string, PricePoint[]>,
  { valueLabel, logY = false, area = false }: PriceTrendOptions,
): EChartsOption {
  const series = [...seriesByLabel.entries()].map(([label, points]) => {
    const clean = points
      .filter((p) => Number.isFinite(p.value) && (!logY || p.value > 0))
      .sort((a, b) => a.date.localeCompare(b.date));
    return {
      name: label,
      type: "line" as const,
      showSymbol: false,
      connectNulls: true,
      emphasis: { focus: "series" as const },
      ...(area ? { areaStyle: { opacity: 0.25 } } : {}),
      data: clean.map((p) => [p.date.slice(0, 10), p.value] as [string, number]),
    };
  });

  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, type: "scroll" },
    grid: { left: 8, right: 24, top: 40, bottom: 48, containLabel: true },
    xAxis: { type: "time" },
    yAxis: logY
      ? { type: "log", name: valueLabel }
      : { type: "value", scale: true, name: valueLabel },
    dataZoom: [
      { type: "inside", start: 0, end: 100 },
      { type: "slider", start: 0, end: 100, bottom: 12 },
    ],
    series,
  };
}
