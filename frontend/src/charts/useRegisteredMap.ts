import { useQuery } from "@tanstack/react-query";
import * as echarts from "echarts";

import { getJson } from "@/api/http";

/** ECharts registers maps in a process-wide registry — track which we've done. */
const registered = new Set<string>();

/**
 * Fetch a bundled GeoJSON from the BFF (`/geo/{geoName}`) and register it with
 * ECharts once under `mapName`. Choropleth series then reference `mapName` and
 * match their data to features via a `nameProperty`. Registration is global +
 * idempotent, so many chart instances share the one registered map.
 */
export function useRegisteredMap(mapName: string, geoName: string) {
  const query = useQuery({
    queryKey: ["geo", geoName],
    queryFn: async () => {
      const geojson = await getJson<Parameters<typeof echarts.registerMap>[1]>(`/geo/${geoName}`);
      if (!registered.has(mapName)) {
        echarts.registerMap(mapName, geojson);
        registered.add(mapName);
      }
      return mapName;
    },
    staleTime: Infinity,
  });
  return {
    mapName,
    ready: query.isSuccess,
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
