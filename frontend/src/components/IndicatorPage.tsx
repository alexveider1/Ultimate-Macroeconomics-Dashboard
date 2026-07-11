import { useMemo } from "react";

import { useDashboardConfig } from "@/api/hooks";
import { CountryMultiSelect } from "@/components/CountryMultiSelect";
import { GraphBox } from "@/components/GraphBox";

interface IndicatorPageProps {
  title: string;
  sectionKeys: string[];
  caption?: string;
}

/**
 * Generic config-driven dashboard page — the React equivalent of the Streamlit
 * `render_page_from_config`. Collects indicators from the given config sections
 * and renders one `GraphBox` per indicator under the shared country picker.
 */
export function IndicatorPage({ title, sectionKeys, caption }: IndicatorPageProps) {
  const { data, isLoading, isError } = useDashboardConfig();

  const items = useMemo(() => {
    if (!data) return [];
    const collected: { id: string; name: string }[] = [];
    for (const section of sectionKeys) {
      for (const item of data[section] ?? []) {
        if (item.id && item.name) collected.push({ id: item.id, name: item.name });
      }
    }
    return collected;
  }, [data, sectionKeys]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">{title}</h2>
        {caption && <p className="max-w-3xl text-muted-foreground">{caption}</p>}
      </div>

      <CountryMultiSelect />

      {isLoading ? (
        <p className="text-muted-foreground">Loading indicators…</p>
      ) : isError ? (
        <p className="text-negative">Could not load the dashboard config.</p>
      ) : items.length === 0 ? (
        <p className="text-muted-foreground">No indicators found for this page.</p>
      ) : (
        <div className="grid gap-6">
          {items.map((item) => (
            <GraphBox key={item.id} indicatorId={item.id} name={item.name} />
          ))}
        </div>
      )}
    </div>
  );
}
