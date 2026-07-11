import type { EChartsOption } from "echarts";

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface CandlestickOptions {
  /** Up (bullish) colour — theme `positive`. */
  upColor: string;
  /** Down (bearish) colour — theme `negative`. */
  downColor: string;
  valueLabel?: string;
}

/**
 * OHLC candlestick over a category (date) axis with an inside+slider dataZoom
 * for the long histories. Up/down colours come from the theme's positive /
 * negative tokens.
 */
export function buildCandlestickOption(
  candles: Candle[],
  { upColor, downColor, valueLabel = "Price" }: CandlestickOptions,
): EChartsOption {
  const clean = candles
    .filter(
      (c) =>
        Number.isFinite(c.open) &&
        Number.isFinite(c.high) &&
        Number.isFinite(c.low) &&
        Number.isFinite(c.close),
    )
    .sort((a, b) => a.date.localeCompare(b.date));

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    grid: { left: 8, right: 16, top: 16, bottom: 64, containLabel: true },
    xAxis: { type: "category", data: clean.map((c) => c.date.slice(0, 10)), boundaryGap: true },
    yAxis: { type: "value", scale: true, name: valueLabel },
    dataZoom: [
      { type: "inside", start: 60, end: 100 },
      { type: "slider", start: 60, end: 100, bottom: 12 },
    ],
    series: [
      {
        type: "candlestick",
        // ECharts candlestick datum order is [open, close, low, high].
        data: clean.map((c) => [c.open, c.close, c.low, c.high]),
        itemStyle: {
          color: upColor,
          color0: downColor,
          borderColor: upColor,
          borderColor0: downColor,
        },
      },
    ],
  };
}
