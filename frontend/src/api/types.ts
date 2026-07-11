/**
 * TypeScript mirror of the BFF's JSON contract (`bff/models.py`). Kept in sync by
 * hand — the BFF response models are the source of truth. Snake_case field names
 * are preserved to match the wire format exactly.
 */

// --- Frontend config -------------------------------------------------------

export interface IndicatorConfigItem {
  id: string;
  name: string;
  [key: string]: unknown;
}

/** `GET /config/dashboard`: section name → list of indicators. */
export type DashboardConfig = Record<string, IndicatorConfigItem[]>;

// --- World Bank ------------------------------------------------------------

export interface CountryOut {
  id: string;
  name: string | null;
  region: string | null;
  income_level: string | null;
  aggregate: boolean | null;
  latitude: number | null;
  longitude: number | null;
  capital_city: string | null;
}

export interface IndicatorPoint {
  economy: string;
  year: number;
  value: number | null;
}

export interface WorldBankIndicatorInfo {
  indicator_id: string;
  name: string | null;
  units: string | null;
  source: string | null;
  development_relevance: string | null;
  limitations_and_exceptions: string | null;
  statistical_concept_and_methodology: string | null;
}

export interface WorldBankIndicatorValues {
  indicator_id: string;
  name: string | null;
  points: IndicatorPoint[];
}

// --- Yahoo Finance ---------------------------------------------------------

export interface YahooMetadataOut {
  ticker: string;
  asset_name: string | null;
  category: string | null;
  short_name: string | null;
  sector: string | null;
  industry: string | null;
  currency: string | null;
  exchange: string | null;
}

export interface YahooMetadataDetail extends YahooMetadataOut {
  business_summary: string | null;
}

export interface OhlcvPoint {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  ticker: string;
}

// --- Binance crypto --------------------------------------------------------

export interface CryptoMetadataOut {
  symbol: string;
  base_asset: string | null;
  quote_asset: string | null;
  status: string | null;
  rank: number | null;
  description: string | null;
  last_price: number | null;
  price_change_percent_24h: number | null;
  high_24h: number | null;
  low_24h: number | null;
  quote_volume_24h: number | null;
  trade_count_24h: number | null;
}

export interface CryptoCandle {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  quote_volume: number | null;
  symbol: string;
  base_asset: string | null;
}

// --- FRED / Eurostat (regional) --------------------------------------------

export interface StateOut {
  id: string;
  name: string | null;
  fips: string | null;
  region: string | null;
  division: string | null;
}

export interface FredIndicatorOut {
  indicator_id: string;
  name: string | null;
  category: string | null;
  series_group: string | null;
  example_series_id: string | null;
  units: string | null;
  frequency: string | null;
  seasonal_adjustment: string | null;
  region_type: string | null;
  min_date: string | null;
  max_date: string | null;
  notes: string | null;
}

export interface RegionValuePoint {
  region: string;
  year: number;
  value: number | null;
}

export interface RegionOut {
  id: string;
  name: string | null;
  country_code: string | null;
  country_name: string | null;
  nuts1_id: string | null;
  level: number | null;
}

export interface EurostatIndicatorOut {
  indicator_id: string;
  name: string | null;
  category: string | null;
  dataset: string | null;
  filters: string | null;
  units: string | null;
  frequency: string | null;
  nuts_level: number | null;
  min_year: number | null;
  max_year: number | null;
  source_label: string | null;
  notes: string | null;
}

// --- News / RAG ------------------------------------------------------------

export interface NewsCollectionsOut {
  collections: string[];
}

export interface NewsArticle {
  id: string;
  title: string;
  text: string;
  url: string;
  published: string;
  source: string;
  topic: string;
  sentiment: string;
  collection: string;
}

export interface NewsSearchHit extends NewsArticle {
  score: number;
}

export interface NewsSearchResponse {
  articles: NewsSearchHit[];
  message: string | null;
}

// --- Forecasting -----------------------------------------------------------

/** `GET /forecast/models`: the forecaster's model list + which are enabled. */
export interface ForecastModelsOut {
  models: string[];
  [key: string]: unknown;
}

export interface ForecastRequestBody {
  model_type: string;
  dates: string[];
  values: number[];
  n_prev: number;
  n_predict: number;
  alpha: number;
  model_params: Record<string, number>;
}

export interface ForecastPoint {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

export interface ForecastResponse {
  model_used: string;
  forecast: ForecastPoint[];
}

// --- Clustering ------------------------------------------------------------

/** `GET /cluster/methods`: algorithms + dim-reduction methods the service exposes. */
export interface ClusterMethodsOut {
  available_methods: string[];
  available_reductions: string[];
}

/** Body for `POST /cluster` — the three required fields plus arbitrary tunables. */
export interface ClusterRequestBody {
  method: string;
  dataframe: Record<string, unknown>[];
  feature_columns: string[];
  [key: string]: unknown;
}

export interface ClusterResponse {
  method_used: string;
  dataframe: Record<string, unknown>[];
  visualization_mode: string;
  visualization_columns: string[];
  visualization_labels: string[];
}

// --- Plot interpretation (LLM vision) --------------------------------------

export interface PlotInterpretResponse {
  description: string;
  usage?: Record<string, unknown> | null;
  [key: string]: unknown;
}

// --- News embedding projection (M3) ----------------------------------------

export interface EmbeddingProjectionRequest {
  method: string;
  reduction_method: string;
  output_dim: 2 | 3;
  k: number;
  max_points: number;
  query_id?: string | null;
}

export interface EmbeddingProjectionPoint {
  id: string;
  title: string;
  cluster: string;
  x: number;
  y: number;
  z: number | null;
}

export interface EmbeddingProjectionResponse {
  points: EmbeddingProjectionPoint[];
  output_dim: number;
  mode: string;
  distances: number[] | null;
  query_id: string | null;
  query_title: string | null;
  message: string | null;
}
