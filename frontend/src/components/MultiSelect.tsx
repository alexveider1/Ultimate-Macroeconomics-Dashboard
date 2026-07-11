import { Check, ChevronDown, X } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

export interface MultiSelectOption {
  value: string;
  label: string;
}

export interface MultiSelectProps {
  label?: string;
  options: MultiSelectOption[];
  selected: string[];
  onChange: (next: string[]) => void;
  /** Cap on the number of selected values (chips beyond it can't be added). */
  max?: number;
  placeholder?: string;
  triggerClassName?: string;
  /** Show the removable chip row under the trigger. */
  showChips?: boolean;
}

/**
 * Generic searchable multi-select (Popover + checkbox list + chips). Shared by
 * the regional (states / NUTS regions) and market (tickers / coins) pages.
 */
export function MultiSelect({
  label,
  options,
  selected,
  onChange,
  max,
  placeholder = "Select…",
  triggerClassName,
  showChips = true,
}: MultiSelectProps) {
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return query
      ? options.filter((option) => option.label.toLowerCase().includes(query))
      : options;
  }, [options, search]);

  const labelByValue = useMemo(
    () => new Map(options.map((option) => [option.value, option.label])),
    [options],
  );

  const toggle = (value: string) => {
    if (selected.includes(value)) {
      onChange(selected.filter((existing) => existing !== value));
    } else if (max === undefined || selected.length < max) {
      onChange([...selected, value]);
    }
  };

  return (
    <div className="flex flex-col gap-1">
      {label && <label className="text-sm text-muted-foreground">{label}</label>}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className={cn("h-9 justify-between font-normal", triggerClassName)}
          >
            <span className="truncate">
              {selected.length ? `${selected.length} selected` : placeholder}
            </span>
            <ChevronDown className="h-4 w-4 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-80 p-0">
          <div className="p-2">
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search…"
              className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
          <div className="max-h-64 overflow-y-auto p-1">
            {filtered.map((option) => {
              const isSelected = selected.includes(option.value);
              return (
                <button
                  type="button"
                  key={option.value}
                  onClick={() => toggle(option.value)}
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
              {selected.length}
              {max !== undefined ? `/${max}` : ""}
            </span>
            <button type="button" className="hover:text-foreground" onClick={() => onChange([])}>
              Clear
            </button>
          </div>
        </PopoverContent>
      </Popover>
      {showChips && selected.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1">
          {selected.map((value) => (
            <span
              key={value}
              className="inline-flex items-center gap-1 rounded bg-secondary px-2 py-0.5 text-xs text-secondary-foreground"
            >
              {labelByValue.get(value) ?? value}
              <button type="button" onClick={() => toggle(value)} aria-label={`Remove ${value}`}>
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
