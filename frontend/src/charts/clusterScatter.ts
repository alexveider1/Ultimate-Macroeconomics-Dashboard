import type { EChartsOption } from "echarts";

/** One plotted point: projected coords + cluster label + display metadata. */
export interface ScatterPoint {
  x: number;
  y: number;
  z?: number | null;
  cluster: string;
  label: string;
  /** Secondary hover line (e.g. economy code or article id). */
  detail?: string;
}

export interface ClusterScatterOptions {
  is3d: boolean;
  xLabel: string;
  yLabel: string;
  zLabel?: string;
  /** Categorical colour ramp per cluster (theme `series.colorway`). */
  colorway: string[];
  /** Point id to highlight (e.g. the selected article) + its highlight colour. */
  highlightLabel?: string;
  highlightColor?: string;
}

/**
 * A 2D (`scatter`) or 3D (`scatter3D`, needs echarts-gl loaded) scatter, one
 * series per cluster coloured from the theme ramp. An optional highlighted point
 * is drawn as a larger marker in the `selectedMarker` colour. The option is built
 * loosely and cast because `scatter3D`/`grid3D` aren't in echarts' core typings.
 */
export function buildClusterScatterOption(
  points: ScatterPoint[],
  { is3d, xLabel, yLabel, zLabel, colorway, highlightLabel, highlightColor }: ClusterScatterOptions,
): EChartsOption {
  const byCluster = new Map<string, ScatterPoint[]>();
  for (const point of points) {
    if (!byCluster.has(point.cluster)) byCluster.set(point.cluster, []);
    byCluster.get(point.cluster)!.push(point);
  }
  const clusters = [...byCluster.keys()].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  const seriesType = is3d ? "scatter3D" : "scatter";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any[] = clusters.map((cluster, index) => {
    const color = colorway[index % colorway.length];
    const rows = byCluster.get(cluster)!;
    return {
      name: `Cluster ${cluster}`,
      type: seriesType,
      color,
      symbolSize: is3d ? 8 : 10,
      data: rows.map((point) => ({
        value: is3d ? [point.x, point.y, point.z ?? 0] : [point.x, point.y],
        name: point.label,
        detail: point.detail ?? "",
      })),
    };
  });

  const highlight = highlightLabel
    ? points.find((point) => point.label === highlightLabel || point.detail === highlightLabel)
    : undefined;
  if (highlight) {
    series.push({
      name: "Selected",
      type: seriesType,
      color: highlightColor,
      symbolSize: is3d ? 16 : 20,
      z: 10,
      data: [
        {
          value: is3d ? [highlight.x, highlight.y, highlight.z ?? 0] : [highlight.x, highlight.y],
          name: highlight.label,
          detail: highlight.detail ?? "",
        },
      ],
    });
  }

  const tooltip = {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    formatter: (p: any) => {
      const d = p.data ?? {};
      const detail = d.detail ? `<br/>${d.detail}` : "";
      return `<b>${d.name ?? ""}</b>${detail}<br/>${p.seriesName}`;
    },
  };

  if (is3d) {
    return {
      tooltip,
      legend: { top: 0, type: "scroll" },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      grid3D: {} as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xAxis3D: { name: xLabel } as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yAxis3D: { name: yLabel } as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      zAxis3D: { name: zLabel ?? "" } as any,
      series,
    } as unknown as EChartsOption;
  }

  return {
    tooltip,
    legend: { top: 0, type: "scroll" },
    grid: { left: 56, right: 24, top: 40, bottom: 48 },
    xAxis: { type: "value", scale: true, name: xLabel },
    yAxis: { type: "value", scale: true, name: yLabel },
    series,
  };
}
