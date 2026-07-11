import type { EChartsOption } from "echarts";
import { useMemo, useState } from "react";

import { useNewsProjection } from "@/api/hooks";
import type { EmbeddingProjectionRequest } from "@/api/types";
import { buildClusterScatterOption, type ScatterPoint } from "@/charts/clusterScatter";
import { buildHistogramOption } from "@/charts/histogram";
import { useEchartsGl } from "@/charts/useEchartsGl";
import { EChart } from "@/components/charts/EChart";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { useTheme } from "@/theme/useTheme";

interface EmbeddingMapProps {
  collection: string;
  selectedId: string;
  selectedTitle: string;
}

/**
 * Server-projected embedding map (M3): the BFF scrolls the collection's vectors,
 * clusters + reduces them to 2D/3D, and returns coords + a distance distribution
 * from the selected article — so the ~1536-dim vectors never reach the browser.
 * Renders a cluster scatter (selected article highlighted) + a distance histogram.
 */
export function EmbeddingMap({ collection, selectedId, selectedTitle }: EmbeddingMapProps) {
  const { config } = useTheme();
  const projection = useNewsProjection(collection);

  const [method, setMethod] = useState("kmeans");
  const [reduction, setReduction] = useState("tsne");
  const [dim, setDim] = useState<2 | 3>(2);
  const [maxPoints, setMaxPoints] = useState(300);
  const [k, setK] = useState(4);

  const data = projection.data;
  const is3d = (data?.output_dim ?? 2) === 3;
  const glReady = useEchartsGl(is3d && Boolean(data));

  const run = () => {
    const body: EmbeddingProjectionRequest = {
      method,
      reduction_method: reduction,
      output_dim: dim,
      k,
      max_points: maxPoints,
      query_id: selectedId || null,
    };
    projection.mutate(body);
  };

  const scatterPoints = useMemo<ScatterPoint[]>(
    () =>
      (data?.points ?? []).map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
        cluster: p.cluster,
        label: p.title,
        detail: p.id,
      })),
    [data],
  );

  const scatterOption = useMemo<EChartsOption>(
    () =>
      buildClusterScatterOption(scatterPoints, {
        is3d: is3d && glReady,
        xLabel: "dim 1",
        yLabel: "dim 2",
        zLabel: "dim 3",
        colorway: config.series.colorway,
        highlightLabel: selectedId,
        highlightColor: config.semantic.selectedMarker,
      }),
    [scatterPoints, is3d, glReady, config, selectedId],
  );

  const distances = data?.distances ?? null;
  const histogramOption = useMemo<EChartsOption | null>(() => {
    if (!distances || distances.length === 0) return null;
    const mean = distances.reduce((a, b) => a + b, 0) / distances.length;
    return buildHistogramOption(distances, {
      color: config.series.colorway[0],
      valueLabel: "Cosine distance",
      markerValue: mean,
      markerColor: config.semantic.referenceLine,
      markerLabel: "mean",
    });
  }, [distances, config]);

  return (
    <section className="space-y-3 rounded-lg border bg-card p-4">
      <div>
        <h3 className="font-semibold">Embedding map</h3>
        <p className="text-xs text-muted-foreground">
          2D/3D projection of the topic&apos;s articles, clustered and reduced server-side. The
          selected article is highlighted; the histogram shows its cosine distance to the rest.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <div className="space-y-1">
          <Label>Cluster method</Label>
          <Select value={method} onChange={(e) => setMethod(e.target.value)}>
            <option value="kmeans">KMeans</option>
            <option value="dbscan">DBSCAN</option>
            <option value="hdbscan">HDBSCAN</option>
            <option value="hierarchical">Hierarchical</option>
          </Select>
        </div>
        <div className="space-y-1">
          <Label>Reducer</Label>
          <Select value={reduction} onChange={(e) => setReduction(e.target.value)}>
            <option value="tsne">t-SNE</option>
            <option value="umap">UMAP</option>
            <option value="pca">PCA</option>
          </Select>
        </div>
        <div className="space-y-1">
          <Label>Dimensions</Label>
          <Select value={String(dim)} onChange={(e) => setDim(Number(e.target.value) === 3 ? 3 : 2)}>
            <option value="2">2D</option>
            <option value="3">3D</option>
          </Select>
        </div>
        <div className="space-y-1">
          <Label>Max points</Label>
          <input
            type="number"
            min={50}
            max={1000}
            step={50}
            value={maxPoints}
            onChange={(e) => setMaxPoints(Math.min(1000, Math.max(4, Number(e.target.value))))}
            className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>
        <div className="space-y-1">
          <Label>k (kmeans)</Label>
          <input
            type="number"
            min={2}
            max={12}
            value={k}
            onChange={(e) => setK(Math.min(12, Math.max(2, Number(e.target.value))))}
            className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>
      </div>

      <Button onClick={run} disabled={projection.isPending}>
        {projection.isPending ? "Projecting…" : "Run embedding map"}
      </Button>

      {projection.isError && (
        <p className="text-sm text-negative">
          Projection failed: {(projection.error as Error).message}
        </p>
      )}
      {data?.message && <p className="text-sm text-muted-foreground">{data.message}</p>}

      {data && data.points.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="h-96">
            {is3d && !glReady ? (
              <div className="grid h-full place-items-center text-sm text-muted-foreground">
                Loading 3D renderer…
              </div>
            ) : (
              <EChart option={scatterOption} />
            )}
          </div>
          <div className="h-96">
            {histogramOption ? (
              <EChart option={histogramOption} />
            ) : (
              <div className="grid h-full place-items-center text-center text-sm text-muted-foreground">
                {selectedTitle
                  ? "Re-run to compute distances for the selected article."
                  : "Select an article to see its distance distribution."}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
