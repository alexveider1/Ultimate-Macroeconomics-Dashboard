import { useMemo, useState } from "react";

import { useYahooMetadata, useYahooPrices } from "@/api/hooks";
import type { OhlcvPoint, YahooMetadataOut } from "@/api/types";
import { buildCandlestickOption } from "@/charts/candlestick";
import {
  buildCorrelationHeatmapOption,
  computeReturnsCorrelation,
} from "@/charts/correlationHeatmap";
import { buildPriceTrendOption, type PricePoint } from "@/charts/priceTrend";
import { buildTreemapOption, type TreeNode } from "@/charts/treemap";
import { EChart } from "@/components/charts/EChart";
import { MultiSelect } from "@/components/MultiSelect";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { useTheme } from "@/theme/useTheme";

const DEFAULT_COMPANIES = ["META", "AAPL", "AMZN", "GOOGL", "MSFT"];
const CURRENT_YEAR = new Date().getFullYear();

function tickerLabel(meta: YahooMetadataOut | undefined, ticker: string): string {
  const name = meta?.asset_name;
  return name && name !== ticker ? `${ticker} — ${name}` : ticker;
}

/** Yahoo Finance dashboard — company trend, candlestick, sector treemap, correlations, indices. */
export function YahooPage() {
  const metaQuery = useYahooMetadata();
  const pricesQuery = useYahooPrices();
  const { config } = useTheme();

  const metaByTicker = useMemo(
    () => new Map((metaQuery.data ?? []).map((m) => [m.ticker, m])),
    [metaQuery.data],
  );
  const prices = useMemo(() => pricesQuery.data ?? [], [pricesQuery.data]);

  const categoryOf = (ticker: string) => (metaByTicker.get(ticker)?.category ?? "").toLowerCase();

  const companyTickers = useMemo(() => {
    const set = new Set<string>();
    for (const p of prices) if (categoryOf(p.ticker) === "companies") set.add(p.ticker);
    return [...set].sort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prices, metaByTicker]);

  const indexTickers = useMemo(() => {
    const set = new Set<string>();
    for (const p of prices) if (categoryOf(p.ticker) === "indices") set.add(p.ticker);
    return [...set].sort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prices, metaByTicker]);

  const companyDefault = useMemo(() => {
    const present = DEFAULT_COMPANIES.filter((t) => companyTickers.includes(t));
    return present.length ? present : companyTickers.slice(0, 6);
  }, [companyTickers]);

  const [selectedCompanies, setSelectedCompanies] = useState<string[]>([]);
  const companies = selectedCompanies.length ? selectedCompanies : companyDefault;

  const [candleTicker, setCandleTicker] = useState<string>("");
  const effectiveCandle =
    candleTicker || (companyTickers.includes("NVDA") ? "NVDA" : (companyDefault[0] ?? ""));

  const [selectedIndices, setSelectedIndices] = useState<string[]>([]);
  const indices = selectedIndices.length ? selectedIndices : indexTickers.slice(0, 3);

  const companyYears = useMemo(() => {
    const set = new Set<number>();
    for (const p of prices) {
      if (categoryOf(p.ticker) === "companies" && p.close !== null) {
        set.add(Number(p.date.slice(0, 4)));
      }
    }
    return [...set].filter((y) => Number.isFinite(y)).sort((a, b) => a - b);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prices, metaByTicker]);

  const [yearOverride, setYearOverride] = useState<number | null>(null);
  const selectedYear = useMemo(() => {
    if (companyYears.length === 0) return null;
    const lo = companyYears[0];
    const hi = companyYears[companyYears.length - 1];
    const year = yearOverride ?? Math.min(Math.max(CURRENT_YEAR - 1, lo), hi);
    return Math.min(Math.max(year, lo), hi);
  }, [companyYears, yearOverride]);

  const seriesFor = (tickers: string[]): Map<string, PricePoint[]> => {
    const wanted = new Set(tickers);
    const series = new Map<string, PricePoint[]>();
    for (const p of prices) {
      if (!wanted.has(p.ticker) || p.close === null) continue;
      const label = tickerLabel(metaByTicker.get(p.ticker), p.ticker);
      if (!series.has(label)) series.set(label, []);
      series.get(label)!.push({ date: p.date, value: p.close });
    }
    return series;
  };

  const companyTrendOption = useMemo(
    () => buildPriceTrendOption(seriesFor(companies), { valueLabel: "Close" }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prices, companies, metaByTicker],
  );

  const indicesTrendOption = useMemo(
    () => buildPriceTrendOption(seriesFor(indices), { valueLabel: "Close", area: true }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prices, indices, metaByTicker],
  );

  const candleOption = useMemo(() => {
    const candles = prices
      .filter(
        (p): p is OhlcvPoint & { open: number; high: number; low: number; close: number } =>
          p.ticker === effectiveCandle &&
          p.open !== null &&
          p.high !== null &&
          p.low !== null &&
          p.close !== null,
      )
      .map((p) => ({ date: p.date, open: p.open, high: p.high, low: p.low, close: p.close }));
    return buildCandlestickOption(candles, {
      upColor: config.semantic.positive,
      downColor: config.semantic.negative,
    });
  }, [prices, effectiveCandle, config]);

  const treemapOption = useMemo(() => {
    if (selectedYear === null) return {};
    const latestByTicker = new Map<string, { close: number; volume: number }>();
    for (const p of prices) {
      if (categoryOf(p.ticker) !== "companies") continue;
      if (Number(p.date.slice(0, 4)) !== selectedYear || p.close === null) continue;
      latestByTicker.set(p.ticker, { close: p.close, volume: p.volume ?? 0 });
    }
    const bySector = new Map<string, TreeNode[]>();
    for (const [ticker, { close, volume }] of latestByTicker) {
      const meta = metaByTicker.get(ticker);
      const sector = meta?.sector || "Unknown";
      const size = volume > 0 ? volume : Math.abs(close) || 1;
      if (!bySector.has(sector)) bySector.set(sector, []);
      bySector.get(sector)!.push({
        name: ticker,
        value: size,
        detail: `${meta?.asset_name ?? ticker} · close ${close.toLocaleString()}`,
      });
    }
    const nodes: TreeNode[] = [...bySector.entries()].map(([sector, children]) => ({
      name: sector,
      children,
    }));
    return buildTreemapOption(nodes, `Companies by sector — ${selectedYear}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prices, selectedYear, metaByTicker]);

  const heatmapOption = useMemo(() => {
    if (selectedYear === null) return {};
    const rows = prices
      .filter(
        (p) =>
          categoryOf(p.ticker) === "companies" &&
          !p.ticker.startsWith("^") &&
          Number(p.date.slice(0, 4)) === selectedYear &&
          p.close !== null,
      )
      .map((p) => ({ date: p.date, key: p.ticker, close: p.close! }));
    const corr = computeReturnsCorrelation(rows);
    return buildCorrelationHeatmapOption(corr, {
      diverging: config.series.diverging,
      labelFor: (ticker) => tickerLabel(metaByTicker.get(ticker), ticker),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prices, selectedYear, metaByTicker, config]);

  if (metaQuery.isLoading || pricesQuery.isLoading) {
    return <p className="text-muted-foreground">Loading Yahoo Finance data…</p>;
  }
  if (!prices.length || !companyTickers.length) {
    return <p className="text-muted-foreground">No Yahoo Finance data available.</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Yahoo Finance Dashboard</h2>
        <p className="max-w-3xl text-muted-foreground">
          Company trends, candlestick, sector treemap, correlation heatmap and index trends.
        </p>
      </div>

      <section className="space-y-2 rounded-lg border bg-card p-4">
        <h3 className="font-semibold">Company close trend (all history)</h3>
        <MultiSelect
          options={companyTickers.map((t) => ({
            value: t,
            label: tickerLabel(metaByTicker.get(t), t),
          }))}
          selected={companies}
          onChange={setSelectedCompanies}
          max={20}
          placeholder="Select companies"
          triggerClassName="w-full sm:w-96"
        />
        <div className="h-96">
          <EChart option={companyTrendOption} />
        </div>
      </section>

      <section className="space-y-2 rounded-lg border bg-card p-4">
        <h3 className="font-semibold">Company candlestick (all history)</h3>
        <div className="max-w-xs">
          <Select
            aria-label="Candlestick company"
            value={effectiveCandle}
            onChange={(event) => setCandleTicker(event.target.value)}
          >
            {companyTickers.map((t) => (
              <option key={t} value={t}>
                {tickerLabel(metaByTicker.get(t), t)}
              </option>
            ))}
          </Select>
        </div>
        <div className="h-96">
          <EChart option={candleOption} />
        </div>
      </section>

      {selectedYear !== null && (
        <section className="space-y-2 rounded-lg border bg-card p-4">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Year filter (treemap &amp; correlation)</span>
            <span className="tabular-nums text-foreground">{selectedYear}</span>
          </div>
          <Slider
            min={companyYears[0]}
            max={companyYears[companyYears.length - 1]}
            value={[selectedYear]}
            onValueChange={([year]) => setYearOverride(year)}
            disabled={companyYears.length <= 1}
          />
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="h-96">
              <EChart option={treemapOption} />
            </div>
            <div className="h-96">
              <EChart option={heatmapOption} />
            </div>
          </div>
        </section>
      )}

      {indexTickers.length > 0 && (
        <section className="space-y-2 rounded-lg border bg-card p-4">
          <h3 className="font-semibold">Index close trend (all history)</h3>
          <MultiSelect
            options={indexTickers.map((t) => ({
              value: t,
              label: tickerLabel(metaByTicker.get(t), t),
            }))}
            selected={indices}
            onChange={setSelectedIndices}
            placeholder="Select indices"
            triggerClassName="w-full sm:w-96"
          />
          <div className="h-96">
            <EChart option={indicesTrendOption} />
          </div>
        </section>
      )}
    </div>
  );
}
