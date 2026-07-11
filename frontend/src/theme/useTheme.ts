import { useContext } from "react";

import { ThemeContext, type ThemeContextValue } from "./context";

/** Access the active theme + runtime switcher. Throws outside `ThemeProvider`. */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return ctx;
}
