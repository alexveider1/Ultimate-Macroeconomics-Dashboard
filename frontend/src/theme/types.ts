/**
 * The runtime theme contract, mirroring `_container_data/ui_themes.yaml` served
 * by the BFF at `/config/themes`. Every colour in the app comes from here — the
 * frontend hard-codes none. `tokens.ts` validates a fetched theme against this
 * shape and fails loud on any missing token.
 */

export interface ChromeTokens {
  background: string;
  surface: string;
  border: string;
  text: string;
  textMuted: string;
  primary: string;
  primaryText: string;
}

export interface SeriesTokens {
  /** Categorical palette for chart series. */
  colorway: string[];
  /** Two-stop sequential scale `[low, high]`. */
  sequential: [string, string];
  /** Three-stop diverging scale `[low, mid, high]`. */
  diverging: [string, string, string];
}

export interface SectorTokens {
  agriculture: string;
  manufacturing: string;
  services: string;
}

export interface SemanticTokens {
  positive: string;
  negative: string;
  referenceLine: string;
  selectedMarker: string;
  mapCoastline: string;
  mapLand: string;
  sectors: SectorTokens;
}

export interface ChartTokens {
  /** Opacity (0..1) of forecast confidence bands. */
  confidenceBandAlpha: number;
  gridLine: string;
  axisLine: string;
  tooltipBackground: string;
  tooltipText: string;
}

export interface WordcloudTokens {
  background: string;
  /** Explicit colour ramp (not a matplotlib colormap name). */
  colors: string[];
}

export interface ThemeConfig {
  label: string;
  mode: "dark" | "light";
  fontFamily: string;
  chrome: ChromeTokens;
  series: SeriesTokens;
  semantic: SemanticTokens;
  charts: ChartTokens;
  wordcloud: WordcloudTokens;
}

/** Response shape of `GET /config/themes`. */
export interface ThemesResponse {
  active: string;
  themes: Record<string, ThemeConfig>;
}
