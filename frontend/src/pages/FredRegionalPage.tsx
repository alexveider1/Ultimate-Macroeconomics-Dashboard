import { useMemo } from "react";

import { useFredIndicators, useFredIndicatorValues, useFredStates } from "@/api/hooks";
import type { FredIndicatorOut } from "@/api/types";
import { US_STATES_MAP_NAME, useUsStatesMap } from "@/charts/useWorldMap";
import { RegionalExplorer, type RegionEntry } from "@/components/RegionalExplorer";

const CAPTION =
  "State-level indicators from the Federal Reserve (FRED) regional data: unemployment, GDP, " +
  "income, housing, sector employment and more, across the 50 states and DC. Annual values.";

function AboutFred({ indicator }: { indicator: FredIndicatorOut }) {
  return (
    <div className="space-y-1 text-sm">
      <p className="font-semibold">
        {indicator.name} · <span className="italic text-muted-foreground">{indicator.category}</span>
      </p>
      <ul className="ml-4 list-disc space-y-0.5 text-muted-foreground">
        <li>Units: {indicator.units || "n/a"}</li>
        <li>
          Native frequency: {indicator.frequency || "n/a"} ({indicator.seasonal_adjustment || "NSA"});
          shown as annual values
        </li>
        <li>
          Coverage: {indicator.min_date || "?"} → {indicator.max_date || "?"}
        </li>
        <li>
          FRED series group: {indicator.series_group || "n/a"} (example series{" "}
          <code>{indicator.example_series_id || "n/a"}</code>)
        </li>
      </ul>
      {indicator.notes && <p className="text-xs text-muted-foreground">{indicator.notes}</p>}
    </div>
  );
}

/** United States regional statistics (FRED) — the first Regional page. */
export function FredRegionalPage() {
  const indicatorsQuery = useFredIndicators();
  const statesQuery = useFredStates();
  const { ready: mapReady } = useUsStatesMap();

  const regions = useMemo<RegionEntry[]>(
    () => (statesQuery.data ?? []).map((s) => ({ id: s.id, name: s.name ?? s.id })),
    [statesQuery.data],
  );

  if (indicatorsQuery.isLoading || statesQuery.isLoading) {
    return <p className="text-muted-foreground">Loading FRED catalogue…</p>;
  }
  if (indicatorsQuery.isError || !indicatorsQuery.data?.length) {
    return (
      <p className="text-muted-foreground">
        No FRED regional data available. It is ingested on a clean boot of the
        <code> downloader_general </code> container (needs a <code>FRED_API_KEY</code>).
      </p>
    );
  }

  return (
    <RegionalExplorer<FredIndicatorOut>
      title="United States — Regional Statistics (FRED)"
      caption={CAPTION}
      mapName={US_STATES_MAP_NAME}
      nameProperty="postal"
      mapReady={mapReady}
      indicators={indicatorsQuery.data}
      regions={regions}
      defaultIndicatorId="unemployment_rate"
      defaultTrendRegions={["CA", "TX", "NY", "FL"]}
      nounSingular="state"
      nounPlural="states"
      useValues={useFredIndicatorValues}
      renderAbout={(indicator) => <AboutFred indicator={indicator} />}
    />
  );
}
