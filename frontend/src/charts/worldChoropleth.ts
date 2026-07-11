import type { EChartsOption } from "echarts";

import { buildChoroplethOption, type ChoroplethTokens } from "./mapChoropleth";

/** One economy's value for the choropleth (code = WB ISO-3 = GeoJSON ADM0_A3). */
export interface ChoroplethRow {
  code: string;
  name: string;
  value: number;
}

export type { ChoroplethTokens } from "./mapChoropleth";

export interface ChoroplethOptions {
  /** Registered ECharts map name (see `useWorldMap`). */
  mapName: string;
  valueLabel: string;
  tokens: ChoroplethTokens;
}

/**
 * World choropleth of an indicator's value by economy. A thin wrapper over the
 * generic {@link buildChoroplethOption} pinned to `nameProperty: ADM0_A3`
 * (ISO-3). Kept as its own export because the dashboard `GraphBox` builds
 * `ChoroplethRow`s keyed by ISO-3.
 */
export function buildWorldChoroplethOption(
  rows: ChoroplethRow[],
  { mapName, valueLabel, tokens }: ChoroplethOptions,
): EChartsOption {
  return buildChoroplethOption(
    rows.map((r) => ({ key: r.code, name: r.name, value: r.value })),
    { mapName, nameProperty: "ADM0_A3", valueLabel, tokens },
  );
}
