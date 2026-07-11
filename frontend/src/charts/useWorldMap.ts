import { useRegisteredMap } from "./useRegisteredMap";

export const WORLD_MAP_NAME = "world";
export const US_STATES_MAP_NAME = "us-states";
export const NUTS_MAP_NAME = "nuts2";

/** World map (WB choropleth) — series ISO-3 matched via `nameProperty: ADM0_A3`. */
export function useWorldMap() {
  return useRegisteredMap(WORLD_MAP_NAME, "world");
}

/** US-states map (FRED page) — series USPS code matched via `nameProperty: postal`. */
export function useUsStatesMap() {
  return useRegisteredMap(US_STATES_MAP_NAME, "us-states");
}

/** EU NUTS-2 map (Eurostat page) — series NUTS code matched via `nameProperty: NUTS_ID`. */
export function useNutsMap() {
  return useRegisteredMap(NUTS_MAP_NAME, "nuts2");
}
