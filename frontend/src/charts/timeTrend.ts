import type { EChartsOption } from "echarts";

export interface TrendPoint {
  year: number;
  value: number;
}

export interface TrendOptions {
  showMarkers: boolean;
  valueLabel: string;
  /** Map an economy code to its display label (e.g. "Germany"). */
  labelFor: (code: string) => string;
}

/** One line series per economy over the shared year axis. Colours from the registered theme. */
export function buildTimeTrendOption(
  seriesByCode: Map<string, TrendPoint[]>,
  { showMarkers, valueLabel, labelFor }: TrendOptions,
): EChartsOption {
  const years = new Set<number>();
  const valueByYearByCode = new Map<string, Map<number, number>>();
  for (const [code, points] of seriesByCode) {
    const m = new Map<number, number>();
    for (const p of points) {
      if (!Number.isFinite(p.value)) continue;
      years.add(p.year);
      m.set(p.year, p.value);
    }
    valueByYearByCode.set(code, m);
  }
  const sortedYears = [...years].sort((a, b) => a - b);

  const series = [...valueByYearByCode.entries()].map(([code, byYear]) => ({
    name: labelFor(code),
    type: "line" as const,
    showSymbol: showMarkers,
    symbolSize: 6,
    connectNulls: true,
    emphasis: { focus: "series" as const },
    data: sortedYears.map((year) => byYear.get(year) ?? null),
  }));

  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, type: "scroll" },
    grid: { left: 64, right: 24, top: 40, bottom: 40 },
    xAxis: { type: "category", data: sortedYears.map(String), boundaryGap: false },
    yAxis: { type: "value", scale: true, name: valueLabel },
    series,
  };
}
