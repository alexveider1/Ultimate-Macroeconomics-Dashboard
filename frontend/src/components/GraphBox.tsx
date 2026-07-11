import type * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { Info, Settings2 } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import {
  useCountryNameMap,
  useForecast,
  useForecastModels,
  useInterpretPlot,
  useWorldBankIndicatorInfo,
  useWorldBankIndicatorValues,
} from "@/api/hooks";
import type { WorldBankIndicatorInfo } from "@/api/types";
import {
  buildDistributionOption,
  type DistributionType,
  type Orientation,
  type ReferenceLine,
} from "@/charts/distribution";
import { buildForecastTrendOption, type ForecastPoint } from "@/charts/forecastTrend";
import { buildTimeTrendOption, type TrendPoint } from "@/charts/timeTrend";
import { useWorldMap } from "@/charts/useWorldMap";
import { buildWorldChoroplethOption, type ChoroplethRow } from "@/charts/worldChoropleth";
import { EChart } from "@/components/charts/EChart";
import { ForecastControls, type ForecastSettings } from "@/components/ForecastControls";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useUiStore } from "@/store/uiStore";
import { useTheme } from "@/theme/useTheme";

const CURRENT_YEAR = new Date().getFullYear();

/** Natural log, or null for non-positive values (matching the Streamlit ln transform). */
function ln(value: number): number | null {
  return value > 0 ? Math.log(value) : null;
}

interface GraphBoxProps {
  indicatorId: string;
  name: string;
}

/**
 * Reusable World Bank indicator card: world choropleth (left) + selector-driven
 * time-trend / distribution chart (right), with a log toggle, year filter, and a
 * metadata expander. Forecasting + LLM plot descriptions come in a later milestone.
 */
export function GraphBox({ indicatorId, name }: GraphBoxProps) {
  const { config } = useTheme();
  const selectedCountries = useUiStore((state) => state.selectedCountries);
  const countryNames = useCountryNameMap();
  const { mapName, ready: mapReady } = useWorldMap();

  const valuesQuery = useWorldBankIndicatorValues(indicatorId);
  const infoQuery = useWorldBankIndicatorInfo(indicatorId);

  const [rightPlot, setRightPlot] = useState<"time trend" | "distribution">("time trend");
  const [showMarkers, setShowMarkers] = useState(false);
  const [distributionType, setDistributionType] = useState<DistributionType>("histogram");
  const [orientation, setOrientation] = useState<Orientation>("vertical");
  const [useLog, setUseLog] = useState(false);
  const [yearOverride, setYearOverride] = useState<number | null>(null);
  const [showMeta, setShowMeta] = useState(false);

  // Forecasting (M3): per-country forecast points, run against the trend history.
  const modelsQuery = useForecastModels();
  const forecastMutation = useForecast();
  const [forecastByCode, setForecastByCode] = useState<Map<string, ForecastPoint[]>>(new Map());
  const [forecastError, setForecastError] = useState<string | null>(null);

  // LLM plot description (M3): captures the right chart's PNG for the vision endpoint.
  const rightChartRef = useRef<echarts.ECharts | null>(null);
  const interpretMutation = useInterpretPlot();
  const [description, setDescription] = useState<string | null>(null);

  const points = useMemo(() => valuesQuery.data?.points ?? [], [valuesQuery.data]);
  const resolvedName = infoQuery.data?.name ?? valuesQuery.data?.name ?? name;
  const units = infoQuery.data?.units ?? "";

  const availableYears = useMemo(() => {
    const years = new Set<number>();
    for (const point of points) if (point.value !== null) years.add(point.year);
    return [...years].sort((a, b) => a - b);
  }, [points]);

  const { minYear, maxYear, selectedYear } = useMemo(() => {
    if (availableYears.length === 0) {
      const fallback = CURRENT_YEAR - 1;
      return { minYear: fallback, maxYear: fallback, selectedYear: fallback };
    }
    const lo = availableYears[0];
    const hi = availableYears[availableYears.length - 1];
    let year = yearOverride ?? Math.min(Math.max(CURRENT_YEAR - 1, lo), hi);
    year = Math.min(Math.max(year, lo), hi);
    return { minYear: lo, maxYear: hi, selectedYear: year };
  }, [availableYears, yearOverride]);

  const valueLabel = useMemo(() => {
    const base = units || "Value";
    return useLog ? `ln(${base})` : base;
  }, [units, useLog]);

  const mapRows = useMemo<ChoroplethRow[]>(() => {
    const byCode = new Map<string, number>();
    for (const point of points) {
      if (point.year !== selectedYear || point.value === null) continue;
      const code = point.economy.toUpperCase();
      if (code.length !== 3) continue;
      byCode.set(code, point.value);
    }
    const rows: ChoroplethRow[] = [];
    for (const [code, raw] of byCode) {
      const value = useLog ? ln(raw) : raw;
      if (value === null) continue;
      rows.push({ code, name: countryNames.get(code) ?? code, value });
    }
    return rows;
  }, [points, selectedYear, useLog, countryNames]);

  const trendSeries = useMemo<Map<string, TrendPoint[]>>(() => {
    const wanted = new Set(selectedCountries.map((code) => code.toUpperCase()));
    const series = new Map<string, TrendPoint[]>();
    for (const point of points) {
      const code = point.economy.toUpperCase();
      if (!wanted.has(code) || point.value === null) continue;
      const value = useLog ? ln(point.value) : point.value;
      if (value === null) continue;
      if (!series.has(code)) series.set(code, []);
      series.get(code)!.push({ year: point.year, value });
    }
    return series;
  }, [points, selectedCountries, useLog]);

  // Raw (never log-transformed) history per selected country — the forecaster
  // fits on actual values; the log toggle only affects the display below.
  const rawTrendSeries = useMemo<Map<string, TrendPoint[]>>(() => {
    const wanted = new Set(selectedCountries.map((code) => code.toUpperCase()));
    const series = new Map<string, TrendPoint[]>();
    for (const point of points) {
      const code = point.economy.toUpperCase();
      if (!wanted.has(code) || point.value === null) continue;
      if (!series.has(code)) series.set(code, []);
      series.get(code)!.push({ year: point.year, value: point.value });
    }
    return series;
  }, [points, selectedCountries]);

  // Forecast points for display: ln-transform (dropping non-positive) when the
  // log toggle is on, so the forecast overlays the log-scaled history correctly.
  const displayForecast = useMemo<Map<string, ForecastPoint[]>>(() => {
    if (!useLog) return forecastByCode;
    const out = new Map<string, ForecastPoint[]>();
    for (const [code, pts] of forecastByCode) {
      const mapped: ForecastPoint[] = [];
      for (const p of pts) {
        const yhat = ln(p.yhat);
        const lo = ln(p.yhat_lower);
        const hi = ln(p.yhat_upper);
        if (yhat === null || lo === null || hi === null) continue;
        mapped.push({ ds: p.ds, yhat, yhat_lower: lo, yhat_upper: hi });
      }
      if (mapped.length) out.set(code, mapped);
    }
    return out;
  }, [forecastByCode, useLog]);

  const { distValues, referenceLines } = useMemo(() => {
    const values: number[] = [];
    const byCode = new Map<string, number>();
    for (const point of points) {
      if (point.year !== selectedYear || point.value === null) continue;
      const code = point.economy.toUpperCase();
      if (code.length !== 3) continue;
      const value = useLog ? ln(point.value) : point.value;
      if (value === null) continue;
      values.push(value);
      byCode.set(code, value);
    }
    const refs: ReferenceLine[] = [];
    for (const code of selectedCountries.map((entry) => entry.toUpperCase())) {
      const value = byCode.get(code);
      if (value !== undefined) refs.push({ label: countryNames.get(code) ?? code, value });
    }
    return { distValues: values, referenceLines: refs };
  }, [points, selectedYear, useLog, selectedCountries, countryNames]);

  const choroplethOption = useMemo<EChartsOption>(
    () =>
      buildWorldChoroplethOption(mapRows, {
        mapName,
        valueLabel,
        tokens: {
          sequential: config.series.sequential,
          mapLand: config.semantic.mapLand,
          mapBorder: config.semantic.mapCoastline,
        },
      }),
    [mapRows, mapName, valueLabel, config],
  );

  const hasForecast = displayForecast.size > 0;

  const rightOption = useMemo<EChartsOption>(() => {
    if (rightPlot === "time trend") {
      const labelFor = (code: string) => countryNames.get(code) ?? code;
      if (hasForecast) {
        return buildForecastTrendOption(trendSeries, displayForecast, {
          showMarkers,
          valueLabel,
          labelFor,
          colorway: config.series.colorway,
          confidenceBandAlpha: config.charts.confidenceBandAlpha,
        });
      }
      return buildTimeTrendOption(trendSeries, { showMarkers, valueLabel, labelFor });
    }
    return buildDistributionOption(distValues, {
      plotType: distributionType,
      orientation,
      valueLabel,
      referenceLines,
      tokens: {
        bar: config.series.colorway[0],
        reference: config.semantic.referenceLine,
        selected: config.semantic.selectedMarker,
      },
    });
  }, [
    rightPlot,
    trendSeries,
    displayForecast,
    hasForecast,
    showMarkers,
    valueLabel,
    countryNames,
    distValues,
    distributionType,
    orientation,
    referenceLines,
    config,
  ]);

  const runForecast = async (settings: ForecastSettings) => {
    setForecastError(null);
    const codes = [...rawTrendSeries.keys()];
    if (codes.length === 0) {
      setForecastError("Select countries with data to forecast.");
      return;
    }
    if (codes.length > 20) {
      setForecastError("Forecasting is limited to 20 series at a time.");
      return;
    }
    const result = new Map<string, ForecastPoint[]>();
    const skipped: string[] = [];
    for (const code of codes) {
      const pts = [...rawTrendSeries.get(code)!]
        .filter((p) => Number.isFinite(p.value))
        .sort((a, b) => a.year - b.year)
        .slice(-settings.pointsToUse);
      if (pts.length < 6) {
        skipped.push(countryNames.get(code) ?? code);
        continue;
      }
      try {
        const resp = await forecastMutation.mutateAsync({
          model_type: settings.model,
          dates: pts.map((p) => `${p.year}-01-01`),
          values: pts.map((p) => p.value),
          n_prev: pts.length,
          n_predict: settings.pointsToPredict,
          alpha: settings.alpha,
          model_params: settings.modelParams,
        });
        result.set(code, resp.forecast);
      } catch (error) {
        setForecastError(`Forecast service failed: ${(error as Error).message}`);
        return;
      }
    }
    setForecastByCode(result);
    if (skipped.length > 0) {
      setForecastError(`Skipped series with fewer than 6 points: ${skipped.join(", ")}`);
    }
  };

  const describePlot = async (mode: "no_hallucinations" | "creative") => {
    const instance = rightChartRef.current;
    if (!instance) return;
    const dataUrl = instance.getDataURL({
      type: "png",
      pixelRatio: 2,
      backgroundColor: config.chrome.background,
    });
    const base64 = dataUrl.split(",")[1] ?? "";
    const context = `Indicator: ${resolvedName}. Year: ${selectedYear}. Chart: ${rightPlot}${
      hasForecast ? " with forecast" : ""
    }.`;
    setDescription(null);
    try {
      const resp = await interpretMutation.mutateAsync({
        image_base64: base64,
        mode,
        chart_context: context,
      });
      setDescription(resp.description || "No interpretation returned.");
    } catch (error) {
      setDescription(`Plot description failed: ${(error as Error).message}`);
    }
  };

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="mb-3 flex items-start justify-between gap-2">
        <h3 className="font-semibold">{resolvedName}</h3>
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" aria-label="Chart settings">
              <Settings2 className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="max-h-[80vh] space-y-3 overflow-y-auto">
            <div className="space-y-1">
              <Label>Right-side chart</Label>
              <Select
                aria-label="Right-side chart"
                value={rightPlot}
                onChange={(event) =>
                  setRightPlot(event.target.value as "time trend" | "distribution")
                }
              >
                <option value="time trend">Time trend</option>
                <option value="distribution">Distribution</option>
              </Select>
            </div>
            {rightPlot === "time trend" ? (
              <>
                <label className="flex items-center justify-between text-sm">
                  <span>Highlight points</span>
                  <Switch checked={showMarkers} onCheckedChange={setShowMarkers} />
                </label>
                <div className="space-y-1 border-t pt-3">
                  <Label>Time-series forecasting</Label>
                  <ForecastControls
                    models={modelsQuery.data?.models}
                    running={forecastMutation.isPending}
                    hasForecast={hasForecast}
                    onRun={runForecast}
                    onClear={() => {
                      setForecastByCode(new Map());
                      setForecastError(null);
                    }}
                  />
                </div>
              </>
            ) : (
              <>
                <div className="space-y-1">
                  <Label>Distribution type</Label>
                  <Select
                    aria-label="Distribution type"
                    value={distributionType}
                    onChange={(event) =>
                      setDistributionType(event.target.value as DistributionType)
                    }
                  >
                    <option value="histogram">Histogram</option>
                    <option value="density">Density</option>
                    <option value="box">Box plot</option>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label>Orientation</Label>
                  <Select
                    aria-label="Orientation"
                    value={orientation}
                    onChange={(event) => setOrientation(event.target.value as Orientation)}
                  >
                    <option value="vertical">Vertical</option>
                    <option value="horizontal">Horizontal</option>
                  </Select>
                </div>
              </>
            )}
          </PopoverContent>
        </Popover>
      </div>

      {valuesQuery.isLoading ? (
        <div className="grid h-72 place-items-center text-sm text-muted-foreground">Loading…</div>
      ) : valuesQuery.isError ? (
        <div className="grid h-72 place-items-center text-sm text-negative">
          Could not load indicator data.
        </div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <div className="h-72">
                {mapReady ? (
                  <EChart option={choroplethOption} />
                ) : (
                  <div className="grid h-full place-items-center text-sm text-muted-foreground">
                    Loading map…
                  </div>
                )}
              </div>
              <label className="mt-2 flex items-center gap-2 text-sm">
                <Switch checked={useLog} onCheckedChange={setUseLog} />
                Apply log transformation
              </label>
            </div>
            <div className="h-72">
              {distValues.length === 0 && rightPlot === "distribution" && trendSeries.size === 0 ? (
                <div className="grid h-full place-items-center text-sm text-muted-foreground">
                  No data for this selection.
                </div>
              ) : (
                <EChart
                  option={rightOption}
                  onReady={(instance) => {
                    rightChartRef.current = instance;
                  }}
                />
              )}
            </div>
          </div>

          {forecastError && <p className="mt-2 text-xs text-negative">{forecastError}</p>}

          <div className="mt-3 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-muted-foreground">Describe this chart:</span>
              <Button
                size="sm"
                variant="outline"
                disabled={interpretMutation.isPending}
                onClick={() => void describePlot("no_hallucinations")}
              >
                Factual reading
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={interpretMutation.isPending}
                onClick={() => void describePlot("creative")}
              >
                Analyst take
              </Button>
              {interpretMutation.isPending && (
                <span className="text-xs text-muted-foreground">Analysing…</span>
              )}
            </div>
            {description && (
              <div className="whitespace-pre-wrap rounded-md border bg-background p-3 text-sm text-foreground">
                {description}
              </div>
            )}
          </div>

          <div className="mt-4 space-y-1">
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>Year</span>
              <span className="tabular-nums text-foreground">{selectedYear}</span>
            </div>
            <Slider
              min={minYear}
              max={maxYear}
              value={[selectedYear]}
              onValueChange={([year]) => setYearOverride(year)}
              disabled={availableYears.length === 0}
            />
            <p className="text-xs text-muted-foreground">
              Applies to the map and distribution chart.
            </p>
          </div>

          <button
            type="button"
            className="mt-3 flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
            onClick={() => setShowMeta((value) => !value)}
          >
            <Info className="h-4 w-4" />
            {showMeta ? "Hide metadata" : "Show metadata"}
          </button>
          {showMeta && <MetadataSection info={infoQuery.data} />}
        </>
      )}
    </div>
  );
}

function MetadataSection({ info }: { info: WorldBankIndicatorInfo | undefined }) {
  if (!info) return null;
  const fields: [string, string | null][] = [
    ["Units", info.units],
    ["Source", info.source],
    ["Development relevance", info.development_relevance],
    ["Limitations & exceptions", info.limitations_and_exceptions],
    ["Statistical concept & methodology", info.statistical_concept_and_methodology],
  ];
  const shown = fields.filter(([, value]) => value && value.trim());
  return (
    <div className="mt-2 space-y-2 rounded-md border bg-background p-3 text-sm">
      {shown.length === 0 ? (
        <p className="text-muted-foreground">No metadata available.</p>
      ) : (
        shown.map(([label, value]) => (
          <div key={label}>
            <p className="font-medium">{label}</p>
            <p className="text-muted-foreground">{value}</p>
          </div>
        ))
      )}
    </div>
  );
}
