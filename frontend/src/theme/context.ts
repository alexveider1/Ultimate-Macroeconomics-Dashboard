import { createContext } from "react";

import type { ThemeConfig } from "./types";

export interface ThemeOption {
  name: string;
  label: string;
}

export interface ThemeContextValue {
  /** Active theme name (key in the themes map). */
  themeName: string;
  /** The active theme's tokens. */
  config: ThemeConfig;
  /** All selectable themes, for the switcher. */
  options: ThemeOption[];
  /** Switch the active theme at runtime (persisted to localStorage). */
  setTheme: (name: string) => void;
}

export const ThemeContext = createContext<ThemeContextValue | null>(null);
