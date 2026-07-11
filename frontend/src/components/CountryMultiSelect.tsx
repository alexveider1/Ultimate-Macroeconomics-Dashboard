import { Check, ChevronDown, X } from "lucide-react";
import { useMemo, useState } from "react";

import { useWorldBankCountries } from "@/api/hooks";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { MAX_COUNTRY_SELECTION, useUiStore } from "@/store/uiStore";

/**
 * Shared country picker (writes to the Zustand `selectedCountries`, so the choice
 * carries across every dashboard page — mirrors the Streamlit shared session key).
 * Capped at MAX_COUNTRY_SELECTION.
 */
export function CountryMultiSelect() {
  const { data: countries } = useWorldBankCountries(false);
  const selected = useUiStore((state) => state.selectedCountries);
  const setSelected = useUiStore((state) => state.setSelectedCountries);
  const [search, setSearch] = useState("");

  const options = useMemo(
    () =>
      (countries ?? [])
        .filter((country) => country.name)
        .map((country) => ({ code: country.id.toUpperCase(), label: `${country.name} (${country.id})` }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    [countries],
  );

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return query ? options.filter((option) => option.label.toLowerCase().includes(query)) : options;
  }, [options, search]);

  const toggle = (code: string) => {
    if (selected.includes(code)) {
      setSelected(selected.filter((existing) => existing !== code));
    } else if (selected.length < MAX_COUNTRY_SELECTION) {
      setSelected([...selected, code]);
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm text-muted-foreground">
        Countries for time trends &amp; reference lines
      </label>
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" className="h-9 w-72 justify-between font-normal">
            <span className="truncate">
              {selected.length ? `${selected.length} selected` : "Select countries"}
            </span>
            <ChevronDown className="h-4 w-4 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-80 p-0">
          <div className="p-2">
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search countries…"
              className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
          <div className="max-h-64 overflow-y-auto p-1">
            {filtered.map((option) => {
              const isSelected = selected.includes(option.code);
              return (
                <button
                  type="button"
                  key={option.code}
                  onClick={() => toggle(option.code)}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-accent"
                >
                  <span
                    className={cn(
                      "flex h-4 w-4 items-center justify-center rounded-sm border",
                      isSelected
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input",
                    )}
                  >
                    {isSelected && <Check className="h-3 w-3" />}
                  </span>
                  <span className="truncate">{option.label}</span>
                </button>
              );
            })}
          </div>
          <div className="flex items-center justify-between border-t p-2 text-xs text-muted-foreground">
            <span>
              {selected.length}/{MAX_COUNTRY_SELECTION}
            </span>
            <button type="button" className="hover:text-foreground" onClick={() => setSelected([])}>
              Clear
            </button>
          </div>
        </PopoverContent>
      </Popover>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1">
          {selected.map((code) => (
            <span
              key={code}
              className="inline-flex items-center gap-1 rounded bg-secondary px-2 py-0.5 text-xs text-secondary-foreground"
            >
              {code}
              <button type="button" onClick={() => toggle(code)} aria-label={`Remove ${code}`}>
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
