import { useMemo, useState } from "react";

import { useDashboardConfig } from "@/api/hooks";
import { CountryMultiSelect } from "@/components/CountryMultiSelect";
import { GraphBox } from "@/components/GraphBox";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";

/** Pick any indicator (category → indicator) and render it as one GraphBox. */
export function CustomPlotPage() {
  const { data, isLoading } = useDashboardConfig();
  const sections = useMemo(() => (data ? Object.keys(data) : []), [data]);

  const [section, setSection] = useState<string | null>(null);
  const [indicatorId, setIndicatorId] = useState<string | null>(null);

  const activeSection = section ?? sections[0] ?? null;
  const items = (activeSection && data?.[activeSection]) || [];
  const activeItem = items.find((item) => item.id === indicatorId) ?? items[0] ?? null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Custom Plot Constructor</h2>
        <p className="text-muted-foreground">
          Pick any indicator and explore it as a two-panel dashboard: map on the left, a
          selector-driven chart on the right.
        </p>
      </div>

      <CountryMultiSelect />

      {isLoading ? (
        <p className="text-muted-foreground">Loading indicators…</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-4">
            <div className="space-y-1">
              <Label>Category</Label>
              <Select
                className="w-72"
                value={activeSection ?? ""}
                onChange={(event) => {
                  setSection(event.target.value);
                  setIndicatorId(null);
                }}
              >
                {sections.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            </div>
            <div className="space-y-1">
              <Label>Indicator</Label>
              <Select
                className="w-96"
                value={activeItem?.id ?? ""}
                onChange={(event) => setIndicatorId(event.target.value)}
              >
                {items.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} ({item.id})
                  </option>
                ))}
              </Select>
            </div>
          </div>

          {activeItem && (
            <GraphBox key={activeItem.id} indicatorId={activeItem.id} name={activeItem.name} />
          )}
        </>
      )}
    </div>
  );
}
