import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { getJson } from "@/api/http";

import { ThemeContext, type ThemeContextValue } from "./context";
import { registerEchartsTheme } from "./echartsTheme";
import { assertValidTheme, applyThemeToDocument } from "./tokens";
import type { ThemesResponse } from "./types";

const THEME_STORAGE_KEY = "umd.theme";

/** Validate + apply a theme by name: CSS variables + light/dark class + ECharts theme. */
function applyTheme(name: string, data: ThemesResponse): void {
  const theme = data.themes[name];
  assertValidTheme(name, theme); // fail loud on a missing token
  applyThemeToDocument(theme);
  registerEchartsTheme(theme);
}

function CenteredMessage({ children }: { children: ReactNode }) {
  return (
    <div className="grid min-h-screen place-items-center bg-background p-6 text-center text-muted-foreground">
      {children}
    </div>
  );
}

/**
 * Loads the theme palettes from the BFF, applies the active one (localStorage
 * override → config default), and exposes a runtime switcher. Blocks rendering
 * until the theme is applied so no component ever paints with placeholder colours;
 * a fetch/validation failure is surfaced loudly rather than silently degrading.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["config", "themes"],
    queryFn: () => getJson<ThemesResponse>("/config/themes"),
    staleTime: Infinity,
    retry: 1,
  });

  const [themeName, setThemeName] = useState<string | null>(null);
  const [applyError, setApplyError] = useState<string | null>(null);

  useEffect(() => {
    if (!data) return;
    try {
      const stored = localStorage.getItem(THEME_STORAGE_KEY);
      const initial = stored && data.themes[stored] ? stored : data.active;
      applyTheme(initial, data);
      setThemeName(initial);
      setApplyError(null);
    } catch (err) {
      setApplyError(err instanceof Error ? err.message : String(err));
    }
  }, [data]);

  const setTheme = useCallback(
    (name: string) => {
      if (!data || !data.themes[name]) return;
      try {
        applyTheme(name, data);
        localStorage.setItem(THEME_STORAGE_KEY, name);
        setThemeName(name);
      } catch (err) {
        setApplyError(err instanceof Error ? err.message : String(err));
      }
    },
    [data],
  );

  const value = useMemo<ThemeContextValue | null>(() => {
    if (!data || !themeName) return null;
    return {
      themeName,
      config: data.themes[themeName],
      options: Object.entries(data.themes).map(([name, theme]) => ({
        name,
        label: theme.label ?? name,
      })),
      setTheme,
    };
  }, [data, themeName, setTheme]);

  if (applyError) {
    return <CenteredMessage>Theme error: {applyError}</CenteredMessage>;
  }
  if (error) {
    return <CenteredMessage>Could not load theme configuration: {String(error)}</CenteredMessage>;
  }
  if (isLoading || !value) {
    return <CenteredMessage>Loading…</CenteredMessage>;
  }

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
