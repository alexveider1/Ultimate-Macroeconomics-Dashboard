import { useQueries } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { useMemo, useState } from "react";

import { useCluster, useClusterMethods, useCountryNameMap, useDashboardConfig } from "@/api/hooks";
import { getJson } from "@/api/http";
import type { ClusterResponse, WorldBankIndicatorValues } from "@/api/types";
import { buildClusterMapOption, type ClusterMapDatum } from "@/charts/clusterMap";
import { buildClusterScatterOption, type ScatterPoint } from "@/charts/clusterScatter";
import { useEchartsGl } from "@/charts/useEchartsGl";
import { useWorldMap } from "@/charts/useWorldMap";
import { DataTable } from "@/components/DataTable";
import { EChart } from "@/components/charts/EChart";
import { Metric } from "@/components/Metric";
import { MultiSelect } from "@/components/MultiSelect";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  applyMissing,
  buildMatrix,
  buildYearMaps,
  commonYears,
  normalize,
  type FeatureMode,
  type MissingStrategy,
  type Normalization,
  type YearMaps,
} from "@/lib/clusterMatrix";
import { useTheme } from "@/theme/useTheme";

const MAX_INDICATORS = 8;
const METHOD_LABELS: Record<string, string> = {
  kmeans: "KMeans",
  dbscan: "DBSCAN",
  meanshift: "Mean-Shift",
  hdbscan: "HDBSCAN",
  spectral: "Spectral",
  hierarchical: "Hierarchical",
};
const REDUCTION_LABELS: Record<string, string> = {
  tsne: "t-SNE",
  pca: "PCA",
  umap: "UMAP",
  kpca: "Kernel PCA",
};

/** Build the per-method tunables forwarded to the clustering service. */
function methodParams(method: string, k: number, eps: number, minSamples: number, nClusters: number): Record<string, unknown> {
  switch (method) {
    case "kmeans":
      return { k };
    case "dbscan":
      return { eps, min_samples: minSamples };
    case "hdbscan":
      return { hdbscan_min_cluster_size: Math.max(2, minSamples) };
    case "spectral":
      return { spectral_n_clusters: nClusters };
    case "hierarchical":
      return { hierarchical_n_clusters: nClusters };
    default:
      return {};
  }
}

export function ClusteringSandboxPage() {
  const { config } = useTheme();
  const { data: dashboard } = useDashboardConfig();
  const methodsQuery = useClusterMethods();
  const clusterMutation = useCluster();
  const countryNames = useCountryNameMap();
  const { mapName, ready: mapReady } = useWorldMap();

  const sections = useMemo(() => (dashboard ? Object.keys(dashboard) : []), [dashboard]);
  const [section, setSection] = useState<string | null>(null);
  const activeSection = section ?? sections[0] ?? null;
  const items = (activeSection && dashboard?.[activeSection]) || [];

  const [selected, setSelected] = useState<string[]>([]);
  const [modes, setModes] = useState<Record<string, FeatureMode>>({});
  const [yearOverride, setYearOverride] = useState<number | null>(null);
  const [missing, setMissing] = useState<MissingStrategy>("drop");
  const [normalization, setNormalization] = useState<Normalization>("zscore");
  const [method, setMethod] = useState("kmeans");
  const [reduction, setReduction] = useState("tsne");
  const [outputDim, setOutputDim] = useState<2 | 3>(2);
  const [k, setK] = useState(4);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [nClusters, setNClusters] = useState(4);
  const [result, setResult] = useState<ClusterResponse | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // Fetch each selected indicator's values (shares cache with GraphBox reads).
  const valueQueries = useQueries({
    queries: selected.map((id) => ({
      queryKey: ["worldbank", "indicator-values", id, "ALL"],
      queryFn: () => getJson<WorldBankIndicatorValues>(`/worldbank/indicators/${id}/values`),
      staleTime: 5 * 60 * 1000,
    })),
  });

  const yearMapsById = useMemo<Record<string, YearMaps>>(() => {
    const out: Record<string, YearMaps> = {};
    selected.forEach((id, i) => {
      const data = valueQueries[i]?.data;
      out[id] = data ? buildYearMaps(data.points) : new Map();
    });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, valueQueries.map((q) => q.dataUpdatedAt).join(",")]);

  const years = useMemo(() => {
    if (selected.length < 2) return [];
    return commonYears(
      selected.map((id) => ({ maps: yearMapsById[id] ?? new Map(), mode: modes[id] ?? "absolute" })),
    );
  }, [selected, yearMapsById, modes]);

  const selectedYear = useMemo(() => {
    if (years.length === 0) return null;
    if (yearOverride && years.includes(yearOverride)) return yearOverride;
    return years[years.length - 1];
  }, [years, yearOverride]);

  const anyLoading = valueQueries.some((q) => q.isLoading);
  const is3d = outputDim === 3 && selected.length >= 3;
  const glReady = useEchartsGl(is3d && Boolean(result));

  const runClustering = async () => {
    setRunError(null);
    if (selected.length < 2 || selectedYear === null) return;
    const indicators = selected.map((id) => ({
      id,
      maps: yearMapsById[id] ?? new Map(),
      mode: modes[id] ?? ("absolute" as FeatureMode),
    }));
    const matrix = buildMatrix(indicators, selectedYear);
    const prepared = applyMissing(matrix, selected, missing);
    if (prepared.length < 3) {
      setRunError("Too few economies remain after preprocessing. Try imputation or fewer indicators.");
      return;
    }
    const normalized = normalize(prepared, selected, normalization);
    const rows = normalized.map((row) => {
      const out: Record<string, unknown> = { economy: row.economy };
      for (const id of selected) out[id] = row[id];
      return out;
    });

    const effectiveDim = selected.length >= 3 ? outputDim : 2;
    try {
      const response = await clusterMutation.mutateAsync({
        method,
        dataframe: rows,
        feature_columns: selected,
        reduction_method: reduction,
        output_dim: effectiveDim,
        ...methodParams(method, k, eps, minSamples, nClusters),
      });
      setResult(response);
    } catch (error) {
      setRunError(`Clustering request failed: ${(error as Error).message}`);
    }
  };

  // Derive the scatter + map from the response.
  const { scatterPoints, mapRows, clusterCounts, nClustersFound } = useMemo(() => {
    if (!result) return { scatterPoints: [], mapRows: [], clusterCounts: [], nClustersFound: 0 };
    const cols = result.visualization_columns;
    const points: ScatterPoint[] = [];
    const map: ClusterMapDatum[] = [];
    const counts = new Map<string, number>();
    for (const row of result.dataframe) {
      const economy = String(row.economy ?? "");
      const cluster = String(row.cluster ?? "?");
      counts.set(cluster, (counts.get(cluster) ?? 0) + 1);
      const name = countryNames.get(economy.toUpperCase()) ?? economy;
      const x = Number(row[cols[0]]);
      const y = Number(row[cols[1]]);
      const z = cols[2] !== undefined ? Number(row[cols[2]]) : null;
      if (Number.isFinite(x) && Number.isFinite(y)) {
        points.push({ x, y, z, cluster, label: name, detail: economy });
      }
      if (economy.length === 3) map.push({ code: economy.toUpperCase(), name, cluster });
    }
    const countRows = [...counts.entries()]
      .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))
      .map(([cluster, count]) => ({ cluster, count: String(count) }));
    return { scatterPoints: points, mapRows: map, clusterCounts: countRows, nClustersFound: counts.size };
  }, [result, countryNames]);

  const scatterOption = useMemo<EChartsOption>(() => {
    const labels = result?.visualization_labels ?? [];
    return buildClusterScatterOption(scatterPoints, {
      is3d: is3d && glReady,
      xLabel: labels[0] ?? "x",
      yLabel: labels[1] ?? "y",
      zLabel: labels[2] ?? "z",
      colorway: config.series.colorway,
      highlightColor: config.semantic.selectedMarker,
    });
  }, [scatterPoints, is3d, glReady, result, config]);

  const mapOption = useMemo<EChartsOption>(
    () =>
      buildClusterMapOption(mapRows, {
        mapName,
        colorway: config.series.colorway,
        mapLand: config.semantic.mapLand,
        mapBorder: config.semantic.mapCoastline,
      }),
    [mapRows, mapName, config],
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Clustering Sandbox</h2>
        <p className="max-w-3xl text-muted-foreground">
          Build country clusters from World Bank indicators for one year (absolute value or
          year-over-year change), then project them to 2D/3D via t-SNE, PCA, UMAP or Kernel PCA.
        </p>
      </div>

      <div className="space-y-4 rounded-lg border bg-card p-4">
        <div className="flex flex-wrap gap-4">
          <div className="space-y-1">
            <Label>Category</Label>
            <Select
              className="w-72"
              value={activeSection ?? ""}
              onChange={(e) => {
                setSection(e.target.value);
                setSelected([]);
                setResult(null);
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
            <Label>Indicators (2–{MAX_INDICATORS})</Label>
            <MultiSelect
              options={items.map((item) => ({ value: item.id, label: `${item.name} (${item.id})` }))}
              selected={selected}
              onChange={setSelected}
              max={MAX_INDICATORS}
              placeholder="Pick indicators…"
              triggerClassName="w-96"
            />
          </div>
        </div>

        {selected.length >= 2 && (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {selected.map((id) => {
              const item = items.find((it) => it.id === id);
              return (
                <div key={id} className="space-y-1">
                  <Label className="truncate text-xs">{item?.name ?? id}</Label>
                  <Select
                    aria-label={`Feature mode ${id}`}
                    value={modes[id] ?? "absolute"}
                    onChange={(e) => setModes((prev) => ({ ...prev, [id]: e.target.value as FeatureMode }))}
                  >
                    <option value="absolute">Absolute value</option>
                    <option value="relative_change">Year-over-year change</option>
                  </Select>
                </div>
              );
            })}
          </div>
        )}

        {selected.length < 2 ? (
          <p className="text-sm text-muted-foreground">Select at least two indicators to continue.</p>
        ) : anyLoading ? (
          <p className="text-sm text-muted-foreground">Loading indicator data…</p>
        ) : years.length === 0 ? (
          <p className="text-sm text-negative">No common years with data for the selected indicators.</p>
        ) : (
          <>
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Year</span>
                <span className="tabular-nums text-foreground">{selectedYear}</span>
              </div>
              <Slider
                min={years[0]}
                max={years[years.length - 1]}
                value={[selectedYear ?? years[0]]}
                onValueChange={([y]) => setYearOverride(y)}
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-1">
                <Label>Missing values</Label>
                <Select value={missing} onChange={(e) => setMissing(e.target.value as MissingStrategy)}>
                  <option value="drop">Drop incomplete</option>
                  <option value="mean">Impute mean</option>
                  <option value="median">Impute median</option>
                </Select>
              </div>
              <div className="space-y-1">
                <Label>Normalization</Label>
                <Select
                  value={normalization}
                  onChange={(e) => setNormalization(e.target.value as Normalization)}
                >
                  <option value="none">None</option>
                  <option value="zscore">Z-score</option>
                  <option value="minmax">Min-max</option>
                </Select>
              </div>
              <div className="space-y-1">
                <Label>Algorithm</Label>
                <Select value={method} onChange={(e) => setMethod(e.target.value)}>
                  {(methodsQuery.data?.available_methods ?? Object.keys(METHOD_LABELS)).map((m) => (
                    <option key={m} value={m}>
                      {METHOD_LABELS[m] ?? m}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1">
                <Label>Scatter dimensions</Label>
                <Select
                  value={String(outputDim)}
                  onChange={(e) => setOutputDim(Number(e.target.value) === 3 ? 3 : 2)}
                >
                  <option value="2">2D</option>
                  <option value="3">3D {selected.length < 3 ? "(needs ≥3)" : ""}</option>
                </Select>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {(method === "kmeans" || method === "spectral" || method === "hierarchical") && (
                <NumberControl
                  label={method === "kmeans" ? "k clusters" : "n_clusters"}
                  min={2}
                  max={12}
                  value={method === "kmeans" ? k : nClusters}
                  onChange={(v) => (method === "kmeans" ? setK(v) : setNClusters(v))}
                />
              )}
              {method === "dbscan" && (
                <>
                  <NumberControl label="eps" min={0.05} max={3} step={0.05} value={eps} onChange={setEps} />
                  <NumberControl label="min_samples" min={2} max={25} value={minSamples} onChange={setMinSamples} />
                </>
              )}
              {method === "hdbscan" && (
                <NumberControl label="min_cluster_size" min={2} max={25} value={minSamples} onChange={setMinSamples} />
              )}
              {selected.length > outputDim && (
                <div className="space-y-1">
                  <Label>Dim-reduction</Label>
                  <Select value={reduction} onChange={(e) => setReduction(e.target.value)}>
                    {(methodsQuery.data?.available_reductions ?? Object.keys(REDUCTION_LABELS)).map((r) => (
                      <option key={r} value={r}>
                        {REDUCTION_LABELS[r] ?? r}
                      </option>
                    ))}
                  </Select>
                </div>
              )}
            </div>

            <Button onClick={() => void runClustering()} disabled={clusterMutation.isPending}>
              {clusterMutation.isPending ? "Clustering…" : "Run clustering"}
            </Button>
            {runError && <p className="text-sm text-negative">{runError}</p>}
          </>
        )}
      </div>

      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric label="Algorithm" value={(METHOD_LABELS[result.method_used] ?? result.method_used).toString()} />
            <Metric label="Projection" value={result.visualization_mode} />
            <Metric label="Economies clustered" value={String(result.dataframe.length)} />
            <Metric label="Distinct clusters" value={String(nClustersFound)} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-lg border bg-card p-4">
              <h3 className="mb-2 font-semibold">Cluster projection</h3>
              <div className="h-96">
                {is3d && !glReady ? (
                  <div className="grid h-full place-items-center text-sm text-muted-foreground">
                    Loading 3D renderer…
                  </div>
                ) : (
                  <EChart option={scatterOption} />
                )}
              </div>
            </div>
            <div className="rounded-lg border bg-card p-4">
              <h3 className="mb-2 font-semibold">Cluster map</h3>
              <div className="h-96">
                {mapReady ? (
                  <EChart option={mapOption} />
                ) : (
                  <div className="grid h-full place-items-center text-sm text-muted-foreground">
                    Loading map…
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Cluster sizes</h3>
            <DataTable
              columns={[
                { key: "cluster", header: "Cluster" },
                { key: "count", header: "Economies", align: "right" },
              ]}
              rows={clusterCounts}
              maxHeightClass="max-h-64"
            />
          </div>
        </>
      )}
    </div>
  );
}

function NumberControl({
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <Label>{label}</Label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      />
    </div>
  );
}
