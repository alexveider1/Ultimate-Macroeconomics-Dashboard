import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Cross-page UI state. `selectedCountries` is shared across every World Bank
 * page (mirrors the Streamlit shared `wb_selected_countries` session key) and
 * persisted so the choice survives reloads. Defaults match the old
 * DEFAULT_COUNTRY_ALIASES.
 */
interface UiState {
  selectedCountries: string[];
  setSelectedCountries: (codes: string[]) => void;
}

export const MAX_COUNTRY_SELECTION = 10;

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCountries: ["USA", "CHN", "DEU"],
      setSelectedCountries: (codes) =>
        set({ selectedCountries: codes.slice(0, MAX_COUNTRY_SELECTION) }),
    }),
    { name: "umd.ui" },
  ),
);
