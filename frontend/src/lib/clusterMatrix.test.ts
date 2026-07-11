import { describe, expect, it } from "vitest";

import type { IndicatorPoint } from "@/api/types";

import {
  applyMissing,
  buildMatrix,
  buildYearMaps,
  commonYears,
  featureValue,
  normalize,
  type MatrixRow,
} from "./clusterMatrix";

const points = (rows: [string, number, number | null][]): IndicatorPoint[] =>
  rows.map(([economy, year, value]) => ({ economy, year, value }));

describe("clusterMatrix", () => {
  it("buildYearMaps drops nulls and upper-cases economies", () => {
    const maps = buildYearMaps(points([["usa", 2000, 5], ["USA", 2001, null], ["deu", 2000, 3]]));
    expect(maps.get(2000)!.get("USA")).toBe(5);
    expect(maps.has(2001)).toBe(false);
  });

  it("featureValue computes absolute and YoY change", () => {
    const maps = buildYearMaps(points([["USA", 2000, 100], ["USA", 2001, 110]]));
    expect(featureValue(maps, 2001, "absolute", "USA")).toBe(110);
    expect(featureValue(maps, 2001, "relative_change", "USA")).toBeCloseTo(0.1);
    // No prior year → null in change mode.
    expect(featureValue(maps, 2000, "relative_change", "USA")).toBeNull();
  });

  it("commonYears intersects across indicators, respecting change mode", () => {
    const a = buildYearMaps(points([["USA", 2000, 1], ["USA", 2001, 2], ["USA", 2002, 3]]));
    const b = buildYearMaps(points([["USA", 2001, 1], ["USA", 2002, 2]]));
    // Absolute: intersection of {2000,2001,2002} and {2001,2002} = {2001,2002}.
    expect(commonYears([{ maps: a, mode: "absolute" }, { maps: b, mode: "absolute" }])).toEqual([
      2001, 2002,
    ]);
    // Change mode on `a` needs year−1, dropping 2000.
    expect(commonYears([{ maps: a, mode: "relative_change" }, { maps: b, mode: "absolute" }])).toEqual(
      [2001, 2002],
    );
  });

  it("buildMatrix unions economies and fills per-cell feature values", () => {
    const a = buildYearMaps(points([["USA", 2001, 10], ["DEU", 2001, 20]]));
    const b = buildYearMaps(points([["USA", 2001, 30]]));
    const matrix = buildMatrix(
      [
        { id: "x", maps: a, mode: "absolute" },
        { id: "y", maps: b, mode: "absolute" },
      ],
      2001,
    );
    const usa = matrix.find((r) => r.economy === "USA")!;
    const deu = matrix.find((r) => r.economy === "DEU")!;
    expect(usa.x).toBe(10);
    expect(usa.y).toBe(30);
    expect(deu.y).toBeNull(); // missing in indicator y
  });

  it("applyMissing drops incomplete rows or imputes the mean", () => {
    const rows: MatrixRow[] = [
      { economy: "A", x: 1, y: 2 },
      { economy: "B", x: 3, y: null },
    ];
    expect(applyMissing(rows, ["x", "y"], "drop")).toHaveLength(1);
    const imputed = applyMissing(rows, ["x", "y"], "mean");
    // Only "B".y was null; the column mean over present values (2) fills it.
    expect(imputed[1].y).toBe(2);
  });

  it("normalize z-scores a column to mean 0", () => {
    const rows: MatrixRow[] = [
      { economy: "A", x: 0 },
      { economy: "B", x: 10 },
    ];
    const z = normalize(rows, ["x"], "zscore");
    expect((z[0].x as number) + (z[1].x as number)).toBeCloseTo(0);
    const mm = normalize(rows, ["x"], "minmax");
    expect(mm[0].x).toBe(0);
    expect(mm[1].x).toBe(1);
  });
});
