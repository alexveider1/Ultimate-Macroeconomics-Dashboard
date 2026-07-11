import { useTheme } from "@/theme/useTheme";

/**
 * Runtime theme switcher — proves the config-driven theming: picking a theme
 * re-applies the CSS variables + ECharts theme with no rebuild. A native select
 * for now; upgraded to a styled dropdown alongside the other form controls.
 */
export function ThemeSwitcher() {
  const { themeName, options, setTheme } = useTheme();
  return (
    <label className="flex items-center gap-2 text-sm text-muted-foreground">
      <span className="hidden sm:inline">Theme</span>
      <select
        aria-label="Theme"
        value={themeName}
        onChange={(event) => setTheme(event.target.value)}
        className="h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {options.map((option) => (
          <option key={option.name} value={option.name}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
