import { useMemo } from "react";

import {
  useEurostatIndicators,
  useEurostatIndicatorValues,
  useEurostatRegions,
} from "@/api/hooks";
import type { EurostatIndicatorOut } from "@/api/types";
import { NUTS_MAP_NAME, useNutsMap } from "@/charts/useWorldMap";
import { RegionalExplorer, type RegionEntry } from "@/components/RegionalExplorer";

const CAPTION =
  "NUTS-2 region-level indicators from Eurostat: GDP, unemployment, population, life expectancy, " +
  "R&D and more, across the EU, EFTA and candidate-country regions. Annual values. " +
  "Boundaries © EuroGeographics (GISCO); data © Eurostat.";

// Clamp the map layout to continental Europe so overseas NUTS-2 regions
// (Canaries, French DOM, Azores) don't blow up the bounds.
const EUROPE_BOUNDS: [[number, number], [number, number]] = [
  [-25, 72],
  [45, 34],
];

function AboutEurostat({ indicator }: { indicator: EurostatIndicatorOut }) {
  return (
    <div className="space-y-1 text-sm">
      <p className="font-semibold">
        {indicator.name} · <span className="italic text-muted-foreground">{indicator.category}</span>
      </p>
      <ul className="ml-4 list-disc space-y-0.5 text-muted-foreground">
        <li>Units: {indicator.units || "n/a"}</li>
        <li>Frequency: {indicator.frequency || "Annual"}</li>
        <li>
          Coverage: {indicator.min_year ?? "?"} → {indicator.max_year ?? "?"}
        </li>
        <li>
          Eurostat dataset: <code>{indicator.dataset || "n/a"}</code> (filters:{" "}
          <code>{indicator.filters || "{}"}</code>)
        </li>
        {indicator.source_label && <li>Source: {indicator.source_label}</li>}
      </ul>
      {indicator.notes && <p className="text-xs text-muted-foreground">{indicator.notes}</p>}
    </div>
  );
}

/** European Union regional statistics (Eurostat) — the second Regional page. */
export function EurostatRegionalPage() {
  const indicatorsQuery = useEurostatIndicators();
  const regionsQuery = useEurostatRegions();
  const { ready: mapReady } = useNutsMap();

  const regions = useMemo<RegionEntry[]>(
    () =>
      (regionsQuery.data ?? []).map((r) => ({
        id: r.id,
        name: r.name ?? r.id,
        caption: r.country_name ?? undefined,
      })),
    [regionsQuery.data],
  );

  if (indicatorsQuery.isLoading || regionsQuery.isLoading) {
    return <p className="text-muted-foreground">Loading Eurostat catalogue…</p>;
  }
  if (indicatorsQuery.isError || !indicatorsQuery.data?.length) {
    return (
      <p className="text-muted-foreground">
        No Eurostat regional data available. It is ingested on a clean boot of the
        <code> downloader_general </code> container (keyless — no API key required).
      </p>
    );
  }

  return (
    <RegionalExplorer<EurostatIndicatorOut>
      title="European Union — Regional Statistics (Eurostat)"
      caption={CAPTION}
      mapName={NUTS_MAP_NAME}
      nameProperty="NUTS_ID"
      mapReady={mapReady}
      boundingCoords={EUROPE_BOUNDS}
      indicators={indicatorsQuery.data}
      regions={regions}
      defaultIndicatorId="gdp_per_capita_pps"
      defaultTrendRegions={["DE21", "FR10", "ES30", "ITC4", "PL91"]}
      nounSingular="region"
      nounPlural="regions"
      useValues={useEurostatIndicatorValues}
      renderAbout={(indicator) => <AboutEurostat indicator={indicator} />}
    />
  );
}
