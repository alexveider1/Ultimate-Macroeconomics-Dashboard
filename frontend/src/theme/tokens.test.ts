import { describe, expect, it } from "vitest";

import { assertValidTheme, hexToHslTriplet, ThemeValidationError } from "./tokens";
import type { ThemeConfig } from "./types";

function validTheme(): ThemeConfig {
  return {
    label: "Test",
    mode: "dark",
    fontFamily: "Inter",
    chrome: {
      background: "#0e1117",
      surface: "#1e1e1e",
      border: "#2a2f3a",
      text: "#fafafa",
      textMuted: "#94a3b8",
      primary: "#10c8f1",
      primaryText: "#0e1117",
    },
    series: {
      colorway: ["#10c8f1", "#ff6b6b"],
      sequential: ["#0e1117", "#10c8f1"],
      diverging: ["#60a5fa", "#1e293b", "#f87171"],
    },
    semantic: {
      positive: "#22c55e",
      negative: "#ef4444",
      referenceLine: "#94a3b8",
      selectedMarker: "#fde047",
      mapCoastline: "#7dd3fc",
      mapLand: "#1e293b",
      sectors: { agriculture: "#34d399", manufacturing: "#f59e0b", services: "#60a5fa" },
    },
    charts: {
      confidenceBandAlpha: 0.18,
      gridLine: "#2a2f3a",
      axisLine: "#3a4150",
      tooltipBackground: "#1e1e1e",
      tooltipText: "#fafafa",
    },
    wordcloud: { background: "#0e1117", colors: ["#440154", "#fde725"] },
  };
}

describe("assertValidTheme", () => {
  it("accepts a complete theme", () => {
    expect(() => assertValidTheme("test", validTheme())).not.toThrow();
  });

  it("throws listing a missing chrome token (fail loud)", () => {
    const broken = validTheme() as unknown as { chrome: Record<string, unknown> };
    delete broken.chrome.primary;
    expect(() => assertValidTheme("test", broken)).toThrowError(ThemeValidationError);
    expect(() => assertValidTheme("test", broken)).toThrowError(/chrome\.primary/);
  });

  it("throws when confidenceBandAlpha is not numeric", () => {
    const broken = validTheme() as unknown as { charts: Record<string, unknown> };
    broken.charts.confidenceBandAlpha = "0.2";
    expect(() => assertValidTheme("test", broken)).toThrowError(/charts\.confidenceBandAlpha/);
  });

  it("throws on a malformed diverging scale", () => {
    const broken = validTheme() as unknown as { series: Record<string, unknown> };
    broken.series.diverging = ["#000", "#fff"];
    expect(() => assertValidTheme("test", broken)).toThrowError(/series\.diverging/);
  });
});

describe("hexToHslTriplet", () => {
  it("converts black and white", () => {
    expect(hexToHslTriplet("#000000")).toBe("0 0% 0%");
    expect(hexToHslTriplet("#ffffff")).toBe("0 0% 100%");
  });

  it("expands 3-digit hex and returns an H S% L% triplet", () => {
    expect(hexToHslTriplet("#fff")).toBe("0 0% 100%");
    expect(hexToHslTriplet("#10c8f1")).toMatch(/^\d+ \d+% \d+%$/);
  });
});
