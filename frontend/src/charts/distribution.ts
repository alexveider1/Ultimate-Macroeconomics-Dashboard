import type { EChartsOption } from "echarts";

export type DistributionType = "histogram" | "density" | "box";
export type Orientation = "vertical" | "horizontal";

export interface ReferenceLine {
  label: string;
  value: number;
}

export interface DistributionTokens {
  bar: string;
  reference: string;
  selected: string;
}

export interface DistributionOptions {
  plotType: DistributionType;
  orientation: Orientation;
  valueLabel: string;
  referenceLines?: ReferenceLine[];
  tokens: DistributionTokens;
}

function niceBinCount(n: number): number {
  return Math.min(40, Math.max(8, Math.ceil(Math.sqrt(n))));
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[base + 1];
  return next !== undefined ? sorted[base] + rest * (next - sorted[base]) : sorted[base];
}

/**
 * Distribution of the cross-section values as a histogram, normalized density,
 * or box plot, with optional reference lines for selected economies. Histogram
 * bins use a category axis (reference lines snap to the nearest bin); the box
 * plot uses a value axis (exact reference lines).
 */
export function buildDistributionOption(
  values: number[],
  { plotType, orientation, valueLabel, referenceLines, tokens }: DistributionOptions,
): EChartsOption {
  const clean = values.filter((v) => Number.isFinite(v));
  if (clean.length === 0) return {};
  const horizontal = orientation === "horizontal";

  if (plotType === "box") {
    const sorted = [...clean].sort((a, b) => a - b);
    const box = [
      sorted[0],
      quantile(sorted, 0.25),
      quantile(sorted, 0.5),
      quantile(sorted, 0.75),
      sorted[sorted.length - 1],
    ];
    const catAxis = { type: "category" as const, data: [valueLabel] };
    const valAxis = { type: "value" as const, scale: true, name: valueLabel };
    const markLine = referenceLines?.length
      ? {
          symbol: "none" as const,
          data: referenceLines.map((r) => ({
            [horizontal ? "xAxis" : "yAxis"]: r.value,
            label: { formatter: r.label, color: tokens.selected },
            lineStyle: { color: tokens.reference, type: "dashed" as const },
          })),
        }
      : undefined;
    return {
      tooltip: { trigger: "item" },
      grid: { left: 64, right: 24, top: 24, bottom: 40 },
      xAxis: horizontal ? valAxis : catAxis,
      yAxis: horizontal ? catAxis : valAxis,
      series: [
        {
          type: "boxplot",
          data: [box],
          itemStyle: { color: tokens.bar, borderColor: tokens.reference },
          markLine,
        },
      ],
    };
  }

  // Histogram / density.
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const bins = niceBinCount(clean.length);
  const width = (max - min) / bins || 1;
  const counts = new Array<number>(bins).fill(0);
  for (const v of clean) {
    let idx = Math.floor((v - min) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  }
  const centers = counts.map((_, i) => min + width * (i + 0.5));
  const labels = centers.map((c) => c.toLocaleString(undefined, { maximumSignificantDigits: 3 }));
  const total = clean.length;
  const barData = plotType === "density" ? counts.map((c) => c / total / width) : counts;
  const countLabel = plotType === "density" ? "Density" : "Count";

  const nearestBin = (value: number): number => {
    let idx = Math.round((value - min) / width - 0.5);
    if (idx < 0) idx = 0;
    if (idx >= bins) idx = bins - 1;
    return idx;
  };
  const markLine = referenceLines?.length
    ? {
        symbol: "none" as const,
        data: referenceLines.map((r) => ({
          [horizontal ? "yAxis" : "xAxis"]: nearestBin(r.value),
          label: { formatter: r.label, color: tokens.selected },
          lineStyle: { color: tokens.reference, type: "dashed" as const },
        })),
      }
    : undefined;

  const catAxis = { type: "category" as const, data: labels, name: valueLabel };
  const valAxis = { type: "value" as const, name: countLabel };

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 64, right: 24, top: 24, bottom: 48 },
    xAxis: horizontal ? valAxis : catAxis,
    yAxis: horizontal ? catAxis : valAxis,
    series: [
      {
        type: "bar",
        data: barData,
        itemStyle: { color: tokens.bar },
        barCategoryGap: "10%",
        markLine,
      },
    ],
  };
}
