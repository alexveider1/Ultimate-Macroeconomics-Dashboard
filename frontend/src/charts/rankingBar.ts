import type { EChartsOption } from "echarts";

export interface RankingRow {
  label: string;
  value: number;
}

export interface RankingOptions {
  valueLabel: string;
  /** Bar colour (theme colorway[0]). */
  color: string;
  /** When true, pick the smallest values (bottom-N); otherwise the largest. */
  ascending?: boolean;
  topN?: number;
}

/**
 * Horizontal bar of the top/bottom-N rows by value, most-extreme bar on top.
 * Region-agnostic (used by both the FRED and Eurostat pages).
 */
export function buildRankingBarOption(
  rows: RankingRow[],
  { valueLabel, color, ascending = false, topN = 10 }: RankingOptions,
): EChartsOption {
  const clean = rows.filter((r) => Number.isFinite(r.value));
  // Pick the extreme N, then order so the most-extreme sits at the top of the axis.
  const picked = [...clean]
    .sort((a, b) => (ascending ? a.value - b.value : b.value - a.value))
    .slice(0, topN)
    .sort((a, b) => (ascending ? b.value - a.value : a.value - b.value));

  return {
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: unknown) => {
        const items = params as { name: string; value: number }[];
        const p = items[0];
        return p ? `${p.name}<br/>${valueLabel}: ${p.value.toLocaleString()}` : "";
      },
    },
    grid: { left: 8, right: 24, top: 16, bottom: 32, containLabel: true },
    xAxis: { type: "value", name: valueLabel },
    yAxis: { type: "category", data: picked.map((r) => r.label) },
    series: [
      {
        type: "bar",
        data: picked.map((r) => r.value),
        itemStyle: { color },
        barMaxWidth: 22,
      },
    ],
  };
}
