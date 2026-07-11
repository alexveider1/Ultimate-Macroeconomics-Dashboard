import type { EChartsOption } from "echarts";
import { useMemo } from "react";

import { useDashboardConfig, useWorldBankIndicatorValues } from "@/api/hooks";
import type { WorldBankIndicatorValues } from "@/api/types";
import { EChart } from "@/components/charts/EChart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useUiStore } from "@/store/uiStore";

/** Group `(economy, year, value)` points into one ECharts line series per economy. */
function buildLineOption(data: WorldBankIndicatorValues | undefined): EChartsOption {
  if (!data || data.points.length === 0) return {};
  const byEconomy = new Map<string, Map<number, number>>();
  const years = new Set<number>();
  for (const point of data.points) {
    if (point.value === null) continue;
    years.add(point.year);
    if (!byEconomy.has(point.economy)) byEconomy.set(point.economy, new Map());
    byEconomy.get(point.economy)!.set(point.year, point.value);
  }
  const sortedYears = [...years].sort((a, b) => a - b);
  const series = [...byEconomy.entries()].map(([economy, valueByYear]) => ({
    name: economy,
    type: "line" as const,
    showSymbol: false,
    connectNulls: true,
    data: sortedYears.map((year) => valueByYear.get(year) ?? null),
  }));
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 64, right: 24, top: 36, bottom: 40 },
    xAxis: { type: "category", data: sortedYears.map(String), boundaryGap: false },
    yAxis: { type: "value", scale: true },
    series,
  };
}

export function OverviewPage() {
  const selectedCountries = useUiStore((state) => state.selectedCountries);
  const dashboard = useDashboardConfig();

  const firstIndicator = useMemo(() => {
    if (!dashboard.data) return null;
    for (const items of Object.values(dashboard.data)) {
      const found = items.find((item) => item.id && item.name);
      if (found) return found;
    }
    return null;
  }, [dashboard.data]);

  const values = useWorldBankIndicatorValues(firstIndicator?.id, selectedCountries);
  const option = useMemo(() => buildLineOption(values.data), [values.data]);
  const hasData = Boolean(values.data && values.data.points.length > 0);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Overview</h2>
        <p className="text-muted-foreground">
          New TypeScript + ECharts frontend — foundation milestone. Colours, series and axes all
          come from the theme config; switch the theme in the header to see it apply live.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{firstIndicator?.name ?? "Sample indicator"}</CardTitle>
          <CardDescription>
            {firstIndicator ? `Series for ${selectedCountries.join(", ")}` : "Loading config…"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            {values.isLoading || dashboard.isLoading ? (
              <div className="grid h-full place-items-center text-sm text-muted-foreground">
                Loading data…
              </div>
            ) : values.error ? (
              <div className="grid h-full place-items-center text-sm text-negative">
                Could not load data: {String(values.error)}
              </div>
            ) : hasData ? (
              <EChart option={option} />
            ) : (
              <div className="grid h-full place-items-center text-sm text-muted-foreground">
                No data available for this indicator.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
