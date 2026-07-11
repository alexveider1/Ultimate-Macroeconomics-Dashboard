/**
 * Theme token plumbing: strict validation, hex→HSL conversion, and the CSS
 * variable map the whole UI reads through Tailwind. A palette swap only rewrites
 * these variables — no component names a colour.
 *
 * `assertValidTheme` mirrors the old Streamlit `get_color` KeyError contract:
 * a theme missing any required token throws instead of silently falling back.
 */

import type { ThemeConfig } from "./types";

/** Thrown when a fetched theme is missing a required token — fail loud. */
export class ThemeValidationError extends Error {}

function req(obj: Record<string, unknown>, path: string, key: string, missing: string[]): void {
  const value = obj[key];
  if (value === undefined || value === null || value === "") {
    missing.push(`${path}.${key}`);
  }
}

/**
 * Validate untrusted theme data (as received from the BFF) against the required
 * token contract; throw listing any gaps, else narrow to `ThemeConfig`.
 */
export function assertValidTheme(name: string, theme: unknown): asserts theme is ThemeConfig {
  if (!theme || typeof theme !== "object") {
    throw new ThemeValidationError(`Theme "${name}" is missing or malformed`);
  }
  const t = theme as Record<string, unknown>;
  const group = (key: string): Record<string, unknown> => (t[key] ?? {}) as Record<string, unknown>;
  const missing: string[] = [];

  const chrome = group("chrome");
  for (const k of ["background", "surface", "border", "text", "textMuted", "primary", "primaryText"]) {
    req(chrome, "chrome", k, missing);
  }
  const semantic = group("semantic");
  for (const k of ["positive", "negative", "referenceLine", "selectedMarker", "mapCoastline", "mapLand"]) {
    req(semantic, "semantic", k, missing);
  }
  const sectors = (semantic.sectors ?? {}) as Record<string, unknown>;
  for (const k of ["agriculture", "manufacturing", "services"]) {
    req(sectors, "semantic.sectors", k, missing);
  }
  const charts = group("charts");
  for (const k of ["gridLine", "axisLine", "tooltipBackground", "tooltipText"]) {
    req(charts, "charts", k, missing);
  }
  if (typeof charts.confidenceBandAlpha !== "number") {
    missing.push("charts.confidenceBandAlpha");
  }
  const series = group("series");
  if (!Array.isArray(series.colorway) || series.colorway.length === 0) {
    missing.push("series.colorway");
  }
  if (!Array.isArray(series.sequential) || series.sequential.length !== 2) {
    missing.push("series.sequential");
  }
  if (!Array.isArray(series.diverging) || series.diverging.length !== 3) {
    missing.push("series.diverging");
  }
  const wordcloud = group("wordcloud");
  if (!Array.isArray(wordcloud.colors) || wordcloud.colors.length === 0) {
    missing.push("wordcloud.colors");
  }

  if (missing.length > 0) {
    throw new ThemeValidationError(
      `Theme "${name}" is missing required token(s): ${missing.join(", ")}`,
    );
  }
}

/** Convert a `#rgb` / `#rrggbb` hex string to an `"H S% L%"` triplet for `hsl()`. */
export function hexToHslTriplet(hex: string): string {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) {
    h = h
      .split("")
      .map((c) => c + c)
      .join("");
  }
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let s = 0;
  let hue = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        hue = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        hue = (b - r) / d + 2;
        break;
      default:
        hue = (r - g) / d + 4;
    }
    hue /= 6;
  }
  return `${Math.round(hue * 360)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`;
}

/**
 * Build the CSS-variable map for a theme. Chrome tokens follow the shadcn
 * `hsl(var(--…))` convention (so its components are styled); chart/semantic
 * tokens are raw hex vars consumed by Tailwind utilities and charts.
 */
export function buildCssVariables(theme: ThemeConfig): Record<string, string> {
  const { chrome, semantic } = theme;
  const hsl = hexToHslTriplet;
  return {
    "--font-sans": theme.fontFamily,

    // Chrome (shadcn convention).
    "--background": hsl(chrome.background),
    "--foreground": hsl(chrome.text),
    "--card": hsl(chrome.surface),
    "--card-foreground": hsl(chrome.text),
    "--popover": hsl(chrome.surface),
    "--popover-foreground": hsl(chrome.text),
    "--primary": hsl(chrome.primary),
    "--primary-foreground": hsl(chrome.primaryText),
    "--secondary": hsl(chrome.surface),
    "--secondary-foreground": hsl(chrome.text),
    "--muted": hsl(chrome.surface),
    "--muted-foreground": hsl(chrome.textMuted),
    "--accent": hsl(chrome.surface),
    "--accent-foreground": hsl(chrome.text),
    "--border": hsl(chrome.border),
    "--input": hsl(chrome.border),
    "--ring": hsl(chrome.primary),

    // Chart / domain semantic tokens (raw hex).
    "--positive": semantic.positive,
    "--negative": semantic.negative,
    "--reference-line": semantic.referenceLine,
    "--selected-marker": semantic.selectedMarker,
  };
}

/** Apply a validated theme to the document root: CSS variables + light/dark class. */
export function applyThemeToDocument(theme: ThemeConfig): void {
  const root = document.documentElement;
  for (const [name, value] of Object.entries(buildCssVariables(theme))) {
    root.style.setProperty(name, value);
  }
  root.classList.toggle("dark", theme.mode === "dark");
  root.dataset.theme = theme.mode;
}
