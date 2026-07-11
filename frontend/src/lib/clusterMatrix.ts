/**
 * Pure feature-matrix transforms for the clustering sandbox, ported from
 * `app/pages/13_clustering_sandbox.py`. Kept framework-free so they're unit
 * tested directly. The page fetches per-indicator World Bank values, builds a
 * `country × indicator` matrix for one year (absolute or YoY change), handles
 * missing values, optionally normalises, then POSTs to the clustering service.
 */

import type { IndicatorPoint } from "@/api/types";

export type FeatureMode = "absolute" | "relative_change";
export type MissingStrategy = "drop" | "mean" | "median";
export type Normalization = "none" | "zscore" | "minmax";

/** `year → (economy → value)` for one indicator's non-null observations. */
export type YearMaps = Map<number, Map<string, number>>;

export function buildYearMaps(points: IndicatorPoint[]): YearMaps {
  const maps: YearMaps = new Map();
  for (const p of points) {
    if (p.value === null || !Number.isFinite(p.value)) continue;
    const code = p.economy.toUpperCase();
    if (!maps.has(p.year)) maps.set(p.year, new Map());
    maps.get(p.year)!.set(code, p.value);
  }
  return maps;
}

/** Years usable for one indicator under a feature mode (change mode needs year−1). */
export function indicatorYears(maps: YearMaps, mode: FeatureMode): Set<number> {
  const years = new Set<number>();
  for (const year of maps.keys()) {
    if (mode === "relative_change" && !maps.has(year - 1)) continue;
    years.add(year);
  }
  return years;
}

/** Intersect the usable-year sets across every selected indicator. */
export function commonYears(
  indicators: { maps: YearMaps; mode: FeatureMode }[],
): number[] {
  let common: Set<number> | null = null;
  for (const { maps, mode } of indicators) {
    const years = indicatorYears(maps, mode);
    if (common === null) {
      common = years;
    } else {
      const prev: Set<number> = common;
      common = new Set([...years].filter((y) => prev.has(y)));
    }
  }
  return [...(common ?? [])].sort((a, b) => a - b);
}

/** The feature value for one economy/year: absolute, or `(cur − prev) / prev`. */
export function featureValue(
  maps: YearMaps,
  year: number,
  mode: FeatureMode,
  economy: string,
): number | null {
  const cur = maps.get(year)?.get(economy);
  if (cur === undefined) return null;
  if (mode === "absolute") return cur;
  const prev = maps.get(year - 1)?.get(economy);
  if (prev === undefined || prev === 0) return null;
  return (cur - prev) / prev;
}

export interface MatrixRow {
  economy: string;
  [indicatorId: string]: string | number | null;
}

/** Build the `economy × indicator` matrix (union of economies, per-cell feature value). */
export function buildMatrix(
  indicators: { id: string; maps: YearMaps; mode: FeatureMode }[],
  year: number,
): MatrixRow[] {
  const economies = new Set<string>();
  for (const { maps } of indicators) {
    for (const code of maps.get(year)?.keys() ?? []) economies.add(code);
  }
  const rows: MatrixRow[] = [];
  for (const economy of economies) {
    const row: MatrixRow = { economy };
    for (const { id, maps, mode } of indicators) {
      row[id] = featureValue(maps, year, mode, economy);
    }
    rows.push(row);
  }
  return rows;
}

/** Drop rows with any null feature, or impute nulls with the column mean/median. */
export function applyMissing(
  rows: MatrixRow[],
  cols: string[],
  strategy: MissingStrategy,
): MatrixRow[] {
  if (strategy === "drop") {
    return rows.filter((row) => cols.every((c) => typeof row[c] === "number"));
  }
  const fill: Record<string, number> = {};
  for (const c of cols) {
    const values = rows.map((r) => r[c]).filter((v): v is number => typeof v === "number");
    fill[c] = values.length === 0 ? 0 : columnFill(values, strategy);
  }
  return rows.map((row) => {
    const out: MatrixRow = { economy: row.economy };
    for (const c of cols) out[c] = typeof row[c] === "number" ? (row[c] as number) : fill[c];
    return out;
  });
}

function columnFill(values: number[], strategy: "mean" | "median"): number {
  if (strategy === "mean") return values.reduce((a, b) => a + b, 0) / values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/** Z-score or min-max normalise the given columns in place; `"none"` is a passthrough. */
export function normalize(
  rows: MatrixRow[],
  cols: string[],
  mode: Normalization,
): MatrixRow[] {
  if (mode === "none" || rows.length === 0) return rows;
  const stats: Record<string, { a: number; b: number }> = {};
  for (const c of cols) {
    const values = rows.map((r) => r[c]).filter((v): v is number => typeof v === "number");
    if (mode === "zscore") {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, values.length - 1);
      stats[c] = { a: mean, b: Math.sqrt(variance) };
    } else {
      const min = Math.min(...values);
      const max = Math.max(...values);
      stats[c] = { a: min, b: max - min };
    }
  }
  return rows.map((row) => {
    const out: MatrixRow = { economy: row.economy };
    for (const c of cols) {
      const v = row[c];
      if (typeof v !== "number") {
        out[c] = v;
        continue;
      }
      const { a, b } = stats[c];
      out[c] = b === 0 ? 0 : mode === "zscore" ? (v - a) / b : (v - a) / b;
    }
    return out;
  });
}
