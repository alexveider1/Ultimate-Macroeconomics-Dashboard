import { type ReactNode, useMemo, useState } from "react";

import { useCryptoMetadata, useCryptoPrices } from "@/api/hooks";
import type { CryptoCandle } from "@/api/types";
import { buildCandlestickOption } from "@/charts/candlestick";
import {
  buildCorrelationHeatmapOption,
  computeReturnsCorrelation,
} from "@/charts/correlationHeatmap";
import { buildPriceTrendOption, type PricePoint } from "@/charts/priceTrend";
import { EChart } from "@/components/charts/EChart";
import { DataTable } from "@/components/DataTable";
import { MultiSelect } from "@/components/MultiSelect";
import { useTheme } from "@/theme/useTheme";

const TOP_N_TREND = 5;

function num(value: number | null, opts?: Intl.NumberFormatOptions): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toLocaleString(undefined, opts);
}

/** Crypto (Binance) dashboard — overview, top-coin trend, BTC candlestick, correlations. */
export function CryptoPage() {
  const metaQuery = useCryptoMetadata();
  const pricesQuery = useCryptoPrices();
  const { config } = useTheme();

  const meta = useMemo(
    () => [...(metaQuery.data ?? [])].sort((a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9)),
    [metaQuery.data],
  );
  const prices = useMemo(() => pricesQuery.data ?? [], [pricesQuery.data]);

  const baseAssets = useMemo(() => {
    const set = new Set<string>();
    for (const p of prices) if (p.base_asset) set.add(p.base_asset);
    return [...set].sort();
  }, [prices]);

  const defaultSelection = useMemo(
    () =>
      meta
        .map((m) => m.base_asset)
        .filter((a): a is string => Boolean(a) && baseAssets.includes(a!))
        .slice(0, TOP_N_TREND),
    [meta, baseAssets],
  );

  const [selected, setSelected] = useState<string[]>([]);
  const effectiveSelection = selected.length ? selected : defaultSelection;

  const trendSeries = useMemo<Map<string, PricePoint[]>>(() => {
    const wanted = new Set(effectiveSelection);
    const series = new Map<string, PricePoint[]>();
    for (const p of prices) {
      if (!p.base_asset || !wanted.has(p.base_asset) || p.close === null) continue;
      if (!series.has(p.base_asset)) series.set(p.base_asset, []);
      series.get(p.base_asset)!.push({ date: p.date, value: p.close });
    }
    return series;
  }, [prices, effectiveSelection]);

  const trendOption = useMemo(
    () => buildPriceTrendOption(trendSeries, { valueLabel: "Close (USDT)", logY: true }),
    [trendSeries],
  );

  const btcOption = useMemo(() => {
    const candles = prices
      .filter((p): p is CryptoCandle & { open: number; high: number; low: number; close: number } =>
        p.base_asset === "BTC" &&
        p.open !== null &&
        p.high !== null &&
        p.low !== null &&
        p.close !== null,
      )
      .map((p) => ({ date: p.date, open: p.open, high: p.high, low: p.low, close: p.close }));
    return buildCandlestickOption(candles, {
      upColor: config.semantic.positive,
      downColor: config.semantic.negative,
      valueLabel: "BTC/USDT",
    });
  }, [prices, config]);

  const heatmapOption = useMemo(() => {
    const corr = computeReturnsCorrelation(
      prices
        .filter((p) => p.base_asset && p.close !== null)
        .map((p) => ({ date: p.date, key: p.base_asset!, close: p.close! })),
    );
    return buildCorrelationHeatmapOption(corr, { diverging: config.series.diverging });
  }, [prices, config]);

  const overviewRows = useMemo<Record<string, ReactNode>[]>(
    () =>
      meta.map((m) => {
        const pct = m.price_change_percent_24h;
        const pctNode =
          pct === null ? (
            "—"
          ) : (
            <span className={pct >= 0 ? "text-positive" : "text-negative"}>
              {pct >= 0 ? "+" : ""}
              {pct.toLocaleString(undefined, { maximumFractionDigits: 2 })}%
            </span>
          );
        return {
          rank: m.rank ?? "—",
          coin: m.base_asset ?? "—",
          pair: m.symbol,
          last: num(m.last_price, { maximumFractionDigits: 6 }),
          pct: pctNode,
          volume: num(m.quote_volume_24h, { maximumFractionDigits: 0 }),
          trades: num(m.trade_count_24h, { maximumFractionDigits: 0 }),
        };
      }),
    [meta],
  );

  if (metaQuery.isLoading || pricesQuery.isLoading) {
    return <p className="text-muted-foreground">Loading crypto data…</p>;
  }
  if (!meta.length || !prices.length) {
    return (
      <p className="text-muted-foreground">
        No crypto data available. It is ingested on a clean boot of the
        <code> downloader_general </code> container.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Crypto (Binance) Dashboard</h2>
        <p className="max-w-3xl text-muted-foreground">
          The most actively traded Binance USDT pairs: market overview, top-coin price dynamics, a
          Bitcoin candlestick, and a return-correlation heatmap.
        </p>
      </div>

      <section className="space-y-2">
        <h3 className="font-semibold">Market overview (ranked by 24h quote volume)</h3>
        <DataTable
          columns={[
            { key: "rank", header: "Rank", align: "right" },
            { key: "coin", header: "Coin" },
            { key: "pair", header: "Pair" },
            { key: "last", header: "Last price", align: "right" },
            { key: "pct", header: "24h %", align: "right" },
            { key: "volume", header: "24h volume (USDT)", align: "right" },
            { key: "trades", header: "24h trades", align: "right" },
          ]}
          rows={overviewRows}
        />
      </section>

      <section className="space-y-2 rounded-lg border bg-card p-4">
        <h3 className="font-semibold">Top coins price dynamics (log scale)</h3>
        <MultiSelect
          options={baseAssets.map((a) => ({ value: a, label: a }))}
          selected={effectiveSelection}
          onChange={setSelected}
          max={15}
          placeholder="Select coins"
          triggerClassName="w-full sm:w-80"
        />
        <div className="h-96">
          <EChart option={trendOption} />
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="space-y-2 rounded-lg border bg-card p-4">
          <h3 className="font-semibold">Bitcoin candlestick (all history)</h3>
          <div className="h-96">
            <EChart option={btcOption} />
          </div>
        </section>
        <section className="space-y-2 rounded-lg border bg-card p-4">
          <h3 className="font-semibold">Coin return correlation heatmap</h3>
          <div className="h-96">
            <EChart option={heatmapOption} />
          </div>
        </section>
      </div>
    </div>
  );
}
