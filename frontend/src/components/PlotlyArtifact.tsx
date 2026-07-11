import { lazy, Suspense, useMemo } from "react";
import type { PlotParams } from "react-plotly.js";

/**
 * The agent's `plotly_agent` worker returns Plotly figure JSON (not ECharts), so
 * the chat page renders it with Plotly. Plotly is heavy, so it's lazy-loaded
 * (code-split) and only pulled in when a chat actually produces a plot. Built via
 * the react-plotly.js factory around the pre-bundled `plotly.js-dist-min`.
 */
const LazyPlot = lazy(async () => {
  const [factory, plotly] = await Promise.all([
    import("react-plotly.js/factory"),
    import("plotly.js-dist-min"),
  ]);
  const Plotly = (plotly as { default?: unknown }).default ?? plotly;
  return { default: factory.default(Plotly) };
});

export function PlotlyArtifact({ figureJson, title }: { figureJson: string; title?: string }) {
  const parsed = useMemo<{ data?: unknown[]; layout?: Record<string, unknown> } | null>(() => {
    try {
      return JSON.parse(figureJson);
    } catch {
      return null;
    }
  }, [figureJson]);

  if (!parsed) return <p className="text-sm text-negative">Plot artifact could not be parsed.</p>;

  const plotProps = {
    data: parsed.data ?? [],
    layout: { ...(parsed.layout ?? {}), autosize: true, margin: { t: 32, r: 16, b: 32, l: 48 } },
    useResizeHandler: true,
    style: { width: "100%", height: "400px" },
    config: { displayModeBar: false, responsive: true },
  } as unknown as PlotParams;

  return (
    <div className="space-y-1">
      {title && <p className="text-xs text-muted-foreground">Rendered plot: {title}</p>}
      <Suspense
        fallback={
          <div className="grid h-64 place-items-center text-sm text-muted-foreground">
            Loading plot…
          </div>
        }
      >
        <LazyPlot {...plotProps} />
      </Suspense>
    </div>
  );
}
