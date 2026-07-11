import type { EChartsOption } from "echarts";

/** One region's value on a registered map (key = the feature's `nameProperty` value). */
export interface ChoroplethDatum {
  /** Matches the map feature's `nameProperty` (ISO-3, USPS postal, NUTS code…). */
  key: string;
  /** Human label shown in the tooltip. */
  name: string;
  value: number;
}

export interface ChoroplethTokens {
  /** Two-stop sequential ramp `[low, high]`. */
  sequential: [string, string];
  /** Fill for regions with no data. */
  mapLand: string;
  /** Region border colour. */
  mapBorder: string;
}

export interface ChoroplethConfig {
  /** Registered ECharts map name (see `useRegisteredMap`). */
  mapName: string;
  /** GeoJSON property the series data `name` is matched against. */
  nameProperty: string;
  valueLabel: string;
  tokens: ChoroplethTokens;
  /** Allow pan/zoom (off for the small national maps, handy for NUTS). */
  roam?: boolean;
  /** Clamp the layout to a lng/lat box `[[leftLng, topLat], [rightLng, bottomLat]]`. */
  boundingCoords?: [[number, number], [number, number]];
  /** Flip the ramp when lower values are "better" (e.g. unemployment). */
  reverse?: boolean;
}

/**
 * Generic choropleth on a registered ECharts map. Series data is keyed by the
 * feature's `nameProperty`; every colour comes from the passed theme tokens so
 * nothing is hard-coded. Backs the world, US-state and NUTS-2 maps.
 */
export function buildChoroplethOption(
  rows: ChoroplethDatum[],
  config: ChoroplethConfig,
): EChartsOption {
  const { mapName, nameProperty, valueLabel, tokens, roam, boundingCoords, reverse } = config;
  const values = rows.map((r) => r.value).filter((v) => Number.isFinite(v));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const ramp: [string, string] = reverse
    ? [tokens.sequential[1], tokens.sequential[0]]
    : tokens.sequential;

  return {
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => {
        const datum = params as { data?: { displayName?: string; value?: number }; name?: string };
        const label = datum.data?.displayName ?? datum.name ?? "";
        const value = datum.data?.value;
        if (value === undefined || !Number.isFinite(value)) return `${label}: no data`;
        return `${label}<br/>${valueLabel}: ${value.toLocaleString()}`;
      },
    },
    visualMap: {
      min,
      max: max === min ? min + 1 : max,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 4,
      inRange: { color: ramp },
    },
    series: [
      {
        type: "map",
        map: mapName,
        nameProperty,
        roam: roam ?? false,
        ...(boundingCoords ? { boundingCoords } : {}),
        itemStyle: { areaColor: tokens.mapLand, borderColor: tokens.mapBorder, borderWidth: 0.5 },
        emphasis: { label: { show: false }, itemStyle: { areaColor: undefined } },
        data: rows.map((r) => ({ name: r.key, value: r.value, displayName: r.name })),
      },
    ],
  };
}
