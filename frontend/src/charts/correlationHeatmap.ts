import type { EChartsOption } from "echarts";

export interface ReturnRow {
  date: string;
  key: string;
  close: number;
}

export interface CorrelationMatrix {
  labels: string[];
  /** `matrix[i][j]` = Pearson correlation of returns for labels[i] vs labels[j]. */
  matrix: number[][];
}

/** Pearson correlation of two aligned arrays; 0 when fewer than two points. */
function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i += 1) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i += 1) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const denom = Math.sqrt(dx * dy);
  return denom === 0 ? 0 : num / denom;
}

/**
 * Build a returns-correlation matrix from long-format close prices: per key,
 * compute daily percentage returns, then correlate each pair on their
 * overlapping dates (mirrors the Streamlit pivot+corr).
 */
export function computeReturnsCorrelation(rows: ReturnRow[]): CorrelationMatrix {
  const byKey = new Map<string, { date: string; close: number }[]>();
  for (const row of rows) {
    if (!Number.isFinite(row.close)) continue;
    if (!byKey.has(row.key)) byKey.set(row.key, []);
    byKey.get(row.key)!.push({ date: row.date, close: row.close });
  }

  // Per key: sorted returns keyed by date.
  const returnsByKey = new Map<string, Map<string, number>>();
  for (const [key, series] of byKey) {
    series.sort((a, b) => a.date.localeCompare(b.date));
    const returns = new Map<string, number>();
    for (let i = 1; i < series.length; i += 1) {
      const prev = series[i - 1].close;
      if (prev !== 0 && Number.isFinite(prev)) {
        returns.set(series[i].date, series[i].close / prev - 1);
      }
    }
    if (returns.size > 0) returnsByKey.set(key, returns);
  }

  const labels = [...returnsByKey.keys()].sort();
  const matrix = labels.map((rowKey) =>
    labels.map((colKey) => {
      if (rowKey === colKey) return 1;
      const a = returnsByKey.get(rowKey)!;
      const b = returnsByKey.get(colKey)!;
      const xs: number[] = [];
      const ys: number[] = [];
      for (const [date, value] of a) {
        const other = b.get(date);
        if (other !== undefined) {
          xs.push(value);
          ys.push(other);
        }
      }
      return pearson(xs, ys);
    }),
  );

  return { labels, matrix };
}

export interface HeatmapOptions {
  /** Three-stop diverging ramp `[low, mid, high]` for the -1..1 scale. */
  diverging: [string, string, string];
  /** Optional label prettifier (e.g. ticker → "AAPL - Apple"). */
  labelFor?: (key: string) => string;
}

/** Pearson-correlation heatmap of every series pair, coloured on a -1..1 diverging scale. */
export function buildCorrelationHeatmapOption(
  { labels, matrix }: CorrelationMatrix,
  { diverging, labelFor }: HeatmapOptions,
): EChartsOption {
  const display = labels.map((l) => (labelFor ? labelFor(l) : l));
  const data: [number, number, number][] = [];
  for (let y = 0; y < labels.length; y += 1) {
    for (let x = 0; x < labels.length; x += 1) {
      data.push([x, y, Number(matrix[y][x].toFixed(3))]);
    }
  }

  return {
    tooltip: {
      position: "top",
      formatter: (params: unknown) => {
        const p = params as { data: [number, number, number] };
        const [x, y, v] = p.data;
        return `${display[y]} × ${display[x]}<br/>corr: ${v.toFixed(2)}`;
      },
    },
    grid: { left: 8, right: 8, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: "category", data: display, axisLabel: { rotate: 60, fontSize: 10 } },
    yAxis: { type: "category", data: display, axisLabel: { fontSize: 10 } },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "center",
      inRange: { color: diverging },
    },
    series: [
      {
        type: "heatmap",
        data,
        emphasis: { itemStyle: { shadowBlur: 6 } },
        progressive: 0,
      },
    ],
  };
}
