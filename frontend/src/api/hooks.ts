/**
 * TanStack Query hooks over the BFF. One hook per read; query keys are stable so
 * navigation reuses cached data. More hooks are added as each page is built.
 */

import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { getJson, postJson } from "./http";
import type {
  ClusterMethodsOut,
  ClusterRequestBody,
  ClusterResponse,
  CountryOut,
  CryptoCandle,
  CryptoMetadataOut,
  DashboardConfig,
  EmbeddingProjectionRequest,
  EmbeddingProjectionResponse,
  EurostatIndicatorOut,
  ForecastModelsOut,
  ForecastRequestBody,
  ForecastResponse,
  FredIndicatorOut,
  NewsArticle,
  NewsCollectionsOut,
  NewsSearchResponse,
  OhlcvPoint,
  PlotInterpretResponse,
  RegionOut,
  RegionValuePoint,
  StateOut,
  WorldBankIndicatorInfo,
  WorldBankIndicatorValues,
  YahooMetadataOut,
} from "./types";

/** The World Bank page→section→indicator config that drives the dashboards. */
export function useDashboardConfig() {
  return useQuery({
    queryKey: ["config", "dashboard"],
    queryFn: () => getJson<DashboardConfig>("/config/dashboard"),
    staleTime: Infinity,
  });
}

/** Every World Bank economy (for the country picker / choropleth). */
export function useWorldBankCountries(includeAggregates = true) {
  return useQuery({
    queryKey: ["worldbank", "countries", includeAggregates],
    queryFn: () =>
      getJson<CountryOut[]>("/worldbank/countries", { include_aggregates: includeAggregates }),
  });
}

/** `Map<ISO3, country name>` for labelling trends / choropleth tooltips. */
export function useCountryNameMap(): Map<string, string> {
  const { data } = useWorldBankCountries(false);
  return useMemo(() => {
    const map = new Map<string, string>();
    for (const country of data ?? []) {
      if (country.name) map.set(country.id.toUpperCase(), country.name);
    }
    return map;
  }, [data]);
}

/** Resolved name + descriptive metadata for one indicator. */
export function useWorldBankIndicatorInfo(indicatorId: string | undefined) {
  return useQuery({
    queryKey: ["worldbank", "indicator-info", indicatorId],
    queryFn: () => getJson<WorldBankIndicatorInfo>(`/worldbank/indicators/${indicatorId}`),
    enabled: Boolean(indicatorId),
  });
}

/** `(economy, year, value)` observations for one indicator, optionally filtered. */
export function useWorldBankIndicatorValues(
  indicatorId: string | undefined,
  countries?: string[],
) {
  const codes = countries && countries.length > 0 ? countries.join(",") : undefined;
  return useQuery({
    queryKey: ["worldbank", "indicator-values", indicatorId, codes ?? "ALL"],
    queryFn: () =>
      getJson<WorldBankIndicatorValues>(`/worldbank/indicators/${indicatorId}/values`, {
        countries: codes,
      }),
    enabled: Boolean(indicatorId),
  });
}

// --- FRED US-state regional --------------------------------------------------

/** The US-state / DC catalogue (id = USPS postal code). */
export function useFredStates() {
  return useQuery({
    queryKey: ["fred", "states"],
    queryFn: () => getJson<StateOut[]>("/fred/states"),
    staleTime: Infinity,
  });
}

/** Every FRED state-indicator description row (for the category-grouped picker). */
export function useFredIndicators() {
  return useQuery({
    queryKey: ["fred", "indicators"],
    queryFn: () => getJson<FredIndicatorOut[]>("/fred/indicators"),
  });
}

/** `(state, year, value)` observations for one FRED indicator. */
export function useFredIndicatorValues(indicatorId: string | undefined) {
  return useQuery({
    queryKey: ["fred", "indicator-values", indicatorId],
    queryFn: () => getJson<RegionValuePoint[]>(`/fred/indicators/${indicatorId}/values`),
    enabled: Boolean(indicatorId),
  });
}

// --- Eurostat NUTS-2 regional ------------------------------------------------

/** The NUTS-2 region catalogue (id = NUTS-2 code). */
export function useEurostatRegions() {
  return useQuery({
    queryKey: ["eurostat", "regions"],
    queryFn: () => getJson<RegionOut[]>("/eurostat/regions"),
    staleTime: Infinity,
  });
}

/** Every Eurostat indicator description row (for the category-grouped picker). */
export function useEurostatIndicators() {
  return useQuery({
    queryKey: ["eurostat", "indicators"],
    queryFn: () => getJson<EurostatIndicatorOut[]>("/eurostat/indicators"),
  });
}

/** `(region, year, value)` observations for one Eurostat indicator. */
export function useEurostatIndicatorValues(indicatorId: string | undefined) {
  return useQuery({
    queryKey: ["eurostat", "indicator-values", indicatorId],
    queryFn: () => getJson<RegionValuePoint[]>(`/eurostat/indicators/${indicatorId}/values`),
    enabled: Boolean(indicatorId),
  });
}

// --- Yahoo Finance -----------------------------------------------------------

/** One master row per Yahoo ticker (category, sector, asset name…). */
export function useYahooMetadata() {
  return useQuery({
    queryKey: ["yahoo", "metadata"],
    queryFn: () => getJson<YahooMetadataOut[]>("/yahoo/metadata"),
  });
}

/** The full OHLCV history for every Yahoo ticker. */
export function useYahooPrices() {
  return useQuery({
    queryKey: ["yahoo", "prices"],
    queryFn: () => getJson<OhlcvPoint[]>("/yahoo/prices"),
  });
}

// --- Binance crypto ----------------------------------------------------------

/** One master row per Binance coin, ranked by 24h volume. */
export function useCryptoMetadata() {
  return useQuery({
    queryKey: ["crypto", "metadata"],
    queryFn: () => getJson<CryptoMetadataOut[]>("/crypto/metadata"),
  });
}

/** The full daily candle history for every Binance coin. */
export function useCryptoPrices() {
  return useQuery({
    queryKey: ["crypto", "prices"],
    queryFn: () => getJson<CryptoCandle[]>("/crypto/prices"),
  });
}

// --- News / RAG --------------------------------------------------------------

/** Every browsable Qdrant collection. */
export function useNewsCollections() {
  return useQuery({
    queryKey: ["news", "collections"],
    queryFn: () => getJson<NewsCollectionsOut>("/news/collections"),
    staleTime: Infinity,
  });
}

/** Browse up to `limit` stored documents from one collection. */
export function useNewsBrowse(collection: string | undefined, limit = 200) {
  return useQuery({
    queryKey: ["news", "browse", collection, limit],
    queryFn: () =>
      getJson<NewsArticle[]>(`/news/collections/${encodeURIComponent(collection!)}/articles`, {
        limit,
      }),
    enabled: Boolean(collection),
  });
}

export interface NewsSearchArgs {
  query: string;
  topic?: string;
  sentiment?: string;
  top_k?: number;
}

/** Semantic search over the RAG corpus (needs the BFF's OpenAI key). */
export function useNewsSearch() {
  return useMutation({
    mutationFn: (args: NewsSearchArgs) => postJson<NewsSearchResponse>("/news/search", args),
  });
}

/** Project + cluster one collection's embeddings server-side (M3 embedding map). */
export function useNewsProjection(collection: string | undefined) {
  return useMutation({
    mutationFn: (args: EmbeddingProjectionRequest) =>
      postJson<EmbeddingProjectionResponse>(
        `/news/collections/${encodeURIComponent(collection!)}/projection`,
        args,
      ),
  });
}

// --- Forecasting -------------------------------------------------------------

/** The forecaster's model list (drives the GraphBox forecasting dropdown). */
export function useForecastModels() {
  return useQuery({
    queryKey: ["forecast", "models"],
    queryFn: () => getJson<ForecastModelsOut>("/forecast/models"),
    staleTime: Infinity,
    retry: 1,
  });
}

/** Run one forecast (BFF proxies to the forecaster's `/predict`). */
export function useForecast() {
  return useMutation({
    mutationFn: (body: ForecastRequestBody) => postJson<ForecastResponse>("/forecast", body),
  });
}

// --- Clustering --------------------------------------------------------------

/** Clustering algorithms + dim-reduction methods the service supports. */
export function useClusterMethods() {
  return useQuery({
    queryKey: ["cluster", "methods"],
    queryFn: () => getJson<ClusterMethodsOut>("/cluster/methods"),
    staleTime: Infinity,
    retry: 1,
  });
}

/** Run one clustering request (BFF proxies to the clustering `/cluster`). */
export function useCluster() {
  return useMutation({
    mutationFn: (body: ClusterRequestBody) => postJson<ClusterResponse>("/cluster", body),
  });
}

// --- LLM plot interpretation -------------------------------------------------

export interface PlotInterpretArgs {
  image_base64: string;
  mode: "no_hallucinations" | "creative";
  chart_context: string;
}

/** Send a rendered-chart PNG to the agent's vision endpoint for a description. */
export function useInterpretPlot() {
  return useMutation({
    mutationFn: (args: PlotInterpretArgs) =>
      postJson<PlotInterpretResponse>("/agent/plots/interpret", args),
  });
}
