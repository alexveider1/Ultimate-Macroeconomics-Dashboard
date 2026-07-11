import type { EChartsOption } from "echarts";

/** One country's cluster assignment for the cluster-shaded world map. */
export interface ClusterMapDatum {
  /** ISO-3 economy code (matches the world map `ADM0_A3` feature property). */
  code: string;
  name: string;
  cluster: string;
}

export interface ClusterMapOptions {
  mapName: string;
  /** Categorical colour ramp keyed by cluster index (theme `series.colorway`). */
  colorway: string[];
  /** Fill for countries not in the clustered set. */
  mapLand: string;
  mapBorder: string;
}

/**
 * A world choropleth coloured by categorical **cluster** rather than a value
 * ramp: each country's fill is its cluster's colour from the theme ramp (no
 * `visualMap`). Countries outside the clustered set keep the `mapLand` fill.
 */
export function buildClusterMapOption(
  rows: ClusterMapDatum[],
  { mapName, colorway, mapLand, mapBorder }: ClusterMapOptions,
): EChartsOption {
  const clusters = [...new Set(rows.map((r) => r.cluster))].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
  const colorFor = new Map(clusters.map((c, i) => [c, colorway[i % colorway.length]]));

  return {
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const datum = params as { data?: { displayName?: string; cluster?: string }; name?: string };
        const label = datum.data?.displayName ?? datum.name ?? "";
        if (!datum.data) return `${label}: not clustered`;
        return `${label}<br/>Cluster: ${datum.data.cluster}`;
      },
    },
    series: [
      {
        type: "map",
        map: mapName,
        nameProperty: "ADM0_A3",
        roam: false,
        itemStyle: { areaColor: mapLand, borderColor: mapBorder, borderWidth: 0.5 },
        emphasis: { label: { show: false } },
        data: rows.map((r) => ({
          name: r.code,
          value: 1,
          displayName: r.name,
          cluster: r.cluster,
          itemStyle: { areaColor: colorFor.get(r.cluster) },
        })),
      },
    ],
  };
}
