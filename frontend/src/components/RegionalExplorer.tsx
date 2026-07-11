import type { UseQueryResult } from "@tanstack/react-query";
import { type ReactNode, useMemo, useState } from "react";

import type { RegionValuePoint } from "@/api/types";
import { buildChoroplethOption } from "@/charts/mapChoropleth";
import { buildRankingBarOption } from "@/charts/rankingBar";
import { buildTimeTrendOption, type TrendPoint } from "@/charts/timeTrend";
import { EChart } from "@/components/charts/EChart";
import { Metric } from "@/components/Metric";
import { MultiSelect } from "@/components/MultiSelect";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { useTheme } from "@/theme/useTheme";

/** One entry of the region catalogue (state or NUTS-2). */
export interface RegionEntry {
  id: string;
  name: string;
  /** Secondary label shown in parentheses (e.g. the country for a NUTS region). */
  caption?: string;
}

/** Minimal indicator description shared by the FRED and Eurostat catalogues. */
export interface RegionalIndicator {
  indicator_id: string;
  name: string | null;
  category: string | null;
  units: string | null;
}

export interface RegionalExplorerProps<T extends RegionalIndicator> {
  title: string;
  caption: string;
  mapName: string;
  nameProperty: string;
  mapReady: boolean;
  boundingCoords?: [[number, number], [number, number]];
  indicators: T[];
  regions: RegionEntry[];
  defaultIndicatorId: string;
  defaultTrendRegions: string[];
  maxTrendRegions?: number;
  /** "state" / "region" — used in the snapshot tile and multiselect labels. */
  nounSingular: string;
  nounPlural: string;
  useValues: (indicatorId: string | undefined) => UseQueryResult<RegionValuePoint[]>;
  renderAbout: (indicator: T) => ReactNode;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function fmt(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/**
 * Catalogue-driven regional explorer shared by the FRED (US-state) and Eurostat
 * (NUTS-2) pages: indicator picker, year slider, snapshot tiles, choropleth,
 * top/bottom ranking bars, and a multi-region trend chart. Region-agnostic — the
 * only differences (map, catalogue shape, "about" copy) come in via props.
 */
export function RegionalExplorer<T extends RegionalIndicator>({
  title,
  caption,
  mapName,
  nameProperty,
  mapReady,
  boundingCoords,
  indicators,
  regions,
  defaultIndicatorId,
  defaultTrendRegions,
  maxTrendRegions = 10,
  nounSingular,
  nounPlural,
  useValues,
  renderAbout,
}: RegionalExplorerProps<T>) {
  const { config } = useTheme();

  const ordered = useMemo(
    () =>
      [...indicators].sort((a, b) =>
        `${a.category ?? ""} ${a.name ?? ""}`.localeCompare(`${b.category ?? ""} ${b.name ?? ""}`),
      ),
    [indicators],
  );

  const [selectedId, setSelectedId] = useState<string>(() =>
    indicators.some((i) => i.indicator_id === defaultIndicatorId)
      ? defaultIndicatorId
      : (ordered[0]?.indicator_id ?? ""),
  );
  const [yearOverride, setYearOverride] = useState<number | null>(null);
  const [trendRegions, setTrendRegions] = useState<string[]>(defaultTrendRegions);

  const indicator = useMemo(
    () => indicators.find((i) => i.indicator_id === selectedId),
    [indicators, selectedId],
  );
  const units = indicator?.units || "Value";

  const regionById = useMemo(() => new Map(regions.map((r) => [r.id, r])), [regions]);

  const valuesQuery = useValues(selectedId || undefined);
  const points = useMemo(() => valuesQuery.data ?? [], [valuesQuery.data]);

  const availableYears = useMemo(() => {
    const years = new Set<number>();
    for (const p of points) if (p.value !== null) years.add(p.year);
    return [...years].sort((a, b) => a - b);
  }, [points]);

  const selectedYear = useMemo(() => {
    if (availableYears.length === 0) return null;
    const hi = availableYears[availableYears.length - 1];
    const lo = availableYears[0];
    const year = yearOverride ?? hi;
    return Math.min(Math.max(year, lo), hi);
  }, [availableYears, yearOverride]);

  const cross = useMemo(() => {
    if (selectedYear === null) return [] as { id: string; name: string; caption?: string; value: number }[];
    const rows: { id: string; name: string; caption?: string; value: number }[] = [];
    for (const p of points) {
      if (p.year !== selectedYear || p.value === null) continue;
      const meta = regionById.get(p.region);
      rows.push({ id: p.region, name: meta?.name ?? p.region, caption: meta?.caption, value: p.value });
    }
    return rows;
  }, [points, selectedYear, regionById]);

  const snapshot = useMemo(() => {
    if (cross.length === 0) return null;
    const sorted = [...cross].sort((a, b) => b.value - a.value);
    return {
      median: median(cross.map((r) => r.value)),
      hi: sorted[0],
      lo: sorted[sorted.length - 1],
    };
  }, [cross]);

  const choroplethOption = useMemo(
    () =>
      buildChoroplethOption(
        cross.map((r) => ({ key: r.id, name: r.name, value: r.value })),
        {
          mapName,
          nameProperty,
          valueLabel: units,
          boundingCoords,
          roam: Boolean(boundingCoords),
          tokens: {
            sequential: config.series.sequential,
            mapLand: config.semantic.mapLand,
            mapBorder: config.semantic.mapCoastline,
          },
        },
      ),
    [cross, mapName, nameProperty, units, boundingCoords, config],
  );

  const topOption = useMemo(
    () =>
      buildRankingBarOption(
        cross.map((r) => ({ label: r.name, value: r.value })),
        { valueLabel: units, color: config.series.colorway[0], ascending: false, topN: 10 },
      ),
    [cross, units, config],
  );
  const bottomOption = useMemo(
    () =>
      buildRankingBarOption(
        cross.map((r) => ({ label: r.name, value: r.value })),
        { valueLabel: units, color: config.series.colorway[0], ascending: true, topN: 10 },
      ),
    [cross, units, config],
  );

  const availableRegions = useMemo(() => {
    const ids = new Set(points.map((p) => p.region));
    return [...ids]
      .map((id) => {
        const meta = regionById.get(id);
        const label = meta ? `${meta.name} (${meta.caption ?? id})` : id;
        return { value: id, label };
      })
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [points, regionById]);

  const trendSeries = useMemo<Map<string, TrendPoint[]>>(() => {
    const wanted = new Set(trendRegions);
    const series = new Map<string, TrendPoint[]>();
    for (const p of points) {
      if (!wanted.has(p.region) || p.value === null) continue;
      if (!series.has(p.region)) series.set(p.region, []);
      series.get(p.region)!.push({ year: p.year, value: p.value });
    }
    return series;
  }, [points, trendRegions]);

  const trendOption = useMemo(
    () =>
      buildTimeTrendOption(trendSeries, {
        showMarkers: false,
        valueLabel: units,
        labelFor: (id) => regionById.get(id)?.name ?? id,
      }),
    [trendSeries, units, regionById],
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">{title}</h2>
        <p className="max-w-3xl text-muted-foreground">{caption}</p>
      </div>

      <div className="max-w-xl space-y-1">
        <label className="text-sm text-muted-foreground">Indicator</label>
        <Select
          aria-label="Indicator"
          value={selectedId}
          onChange={(event) => {
            setSelectedId(event.target.value);
            setYearOverride(null);
          }}
        >
          {ordered.map((item) => (
            <option key={item.indicator_id} value={item.indicator_id}>
              {`${item.category ?? "Other"} — ${item.name ?? item.indicator_id}`}
            </option>
          ))}
        </Select>
      </div>

      {valuesQuery.isLoading ? (
        <p className="text-muted-foreground">Loading values…</p>
      ) : valuesQuery.isError ? (
        <p className="text-negative">Could not load indicator values.</p>
      ) : selectedYear === null ? (
        <p className="text-muted-foreground">No values stored for this indicator.</p>
      ) : (
        <>
          {snapshot && (
            <div className="grid gap-3 sm:grid-cols-3">
              <Metric label={`Median ${nounSingular}`} value={`${fmt(snapshot.median)}`} />
              <Metric label="Highest" value={fmt(snapshot.hi.value)} caption={snapshot.hi.name} />
              <Metric label="Lowest" value={fmt(snapshot.lo.value)} caption={snapshot.lo.name} />
            </div>
          )}

          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">
              {indicator?.name} — {selectedYear}
            </h3>
            <div className="h-96">
              {mapReady ? (
                <EChart option={choroplethOption} />
              ) : (
                <div className="grid h-full place-items-center text-sm text-muted-foreground">
                  Loading map…
                </div>
              )}
            </div>
            <div className="mt-3 space-y-1">
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Year</span>
                <span className="tabular-nums text-foreground">{selectedYear}</span>
              </div>
              <Slider
                min={availableYears[0]}
                max={availableYears[availableYears.length - 1]}
                value={[selectedYear]}
                onValueChange={([year]) => setYearOverride(year)}
                disabled={availableYears.length <= 1}
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border bg-card p-4">
              <h3 className="mb-2 font-semibold">Top 10 {nounPlural}</h3>
              <div className="h-80">
                <EChart option={topOption} />
              </div>
            </div>
            <div className="rounded-lg border bg-card p-4">
              <h3 className="mb-2 font-semibold">Bottom 10 {nounPlural}</h3>
              <div className="h-80">
                <EChart option={bottomOption} />
              </div>
            </div>
          </div>

          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">{indicator?.name} over time</h3>
            <MultiSelect
              label={`${nounPlural} (max ${maxTrendRegions})`}
              options={availableRegions}
              selected={trendRegions}
              onChange={setTrendRegions}
              max={maxTrendRegions}
              placeholder={`Select ${nounPlural}`}
              triggerClassName="w-full sm:w-96"
            />
            <div className="mt-3 h-80">
              {trendSeries.size === 0 ? (
                <div className="grid h-full place-items-center text-sm text-muted-foreground">
                  Select at least one {nounSingular} to display the trend.
                </div>
              ) : (
                <EChart option={trendOption} />
              )}
            </div>
          </div>

          {indicator && <div className="rounded-lg border bg-card p-4">{renderAbout(indicator)}</div>}
        </>
      )}
    </div>
  );
}
