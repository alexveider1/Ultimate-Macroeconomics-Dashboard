"""Pydantic request/response models — the BFF's typed JSON contract.

These are the shapes the (future) frontend consumes. They are deliberately
decoupled from the ORM ``schema.py`` models: the API surface stays stable even
if the underlying column layout shifts. Proxy endpoints (forecast / cluster /
agent) forward permissive bodies to the existing services, so their request
models allow extra keys and their responses are passed through untyped.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# World Bank
# --------------------------------------------------------------------------- #


class CountryOut(BaseModel):
    """One World Bank economy for the country picker / choropleth."""

    id: str
    name: str | None = None
    region: str | None = None
    income_level: str | None = None
    aggregate: bool | None = None
    latitude: float | None = None
    longitude: float | None = None
    capital_city: str | None = None


class IndicatorPoint(BaseModel):
    """One ``(economy, year, value)`` observation of a WB indicator."""

    economy: str
    year: int
    value: float | None = None


class WorldBankIndicatorInfo(BaseModel):
    """Descriptive metadata for one WB indicator series."""

    indicator_id: str
    name: str | None = None
    units: str | None = None
    source: str | None = None
    development_relevance: str | None = None
    limitations_and_exceptions: str | None = None
    statistical_concept_and_methodology: str | None = None


class WorldBankIndicatorValues(BaseModel):
    """A WB indicator slice: its resolved name plus the observation points."""

    indicator_id: str
    name: str | None = None
    points: list[IndicatorPoint]


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #


class YahooMetadataOut(BaseModel):
    """One Yahoo ticker's master row (without the long business summary)."""

    ticker: str
    asset_name: str | None = None
    category: str | None = None
    short_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    exchange: str | None = None


class YahooMetadataDetail(YahooMetadataOut):
    """Full Yahoo ticker master row including the business summary."""

    business_summary: str | None = None


class OhlcvPoint(BaseModel):
    """One daily OHLCV candle for a Yahoo ticker."""

    date: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None
    ticker: str


# --------------------------------------------------------------------------- #
# Binance crypto
# --------------------------------------------------------------------------- #


class CryptoMetadataOut(BaseModel):
    """One Binance spot pair's master row (ranked by trailing-24h volume)."""

    symbol: str
    base_asset: str | None = None
    quote_asset: str | None = None
    status: str | None = None
    rank: int | None = None
    description: str | None = None
    last_price: float | None = None
    price_change_percent_24h: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    quote_volume_24h: float | None = None
    trade_count_24h: int | None = None


class CryptoCandle(BaseModel):
    """One daily candle for a Binance pair."""

    date: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    quote_volume: float | None = None
    symbol: str
    base_asset: str | None = None


# --------------------------------------------------------------------------- #
# FRED US-state / Eurostat NUTS-2 (shared regional shapes)
# --------------------------------------------------------------------------- #


class StateOut(BaseModel):
    """One U.S. state / DC row."""

    id: str
    name: str | None = None
    fips: str | None = None
    region: str | None = None
    division: str | None = None


class FredIndicatorOut(BaseModel):
    """One FRED state-indicator description row."""

    indicator_id: str
    name: str | None = None
    category: str | None = None
    series_group: str | None = None
    example_series_id: str | None = None
    units: str | None = None
    frequency: str | None = None
    seasonal_adjustment: str | None = None
    region_type: str | None = None
    min_date: str | None = None
    max_date: str | None = None
    notes: str | None = None


class RegionValuePoint(BaseModel):
    """One ``(region, year, value)`` regional observation (state or NUTS-2)."""

    region: str
    year: int
    value: float | None = None


class RegionOut(BaseModel):
    """One Eurostat NUTS-2 region row."""

    id: str
    name: str | None = None
    country_code: str | None = None
    country_name: str | None = None
    nuts1_id: str | None = None
    level: int | None = None


class EurostatIndicatorOut(BaseModel):
    """One Eurostat indicator description row."""

    indicator_id: str
    name: str | None = None
    category: str | None = None
    dataset: str | None = None
    filters: str | None = None
    units: str | None = None
    frequency: str | None = None
    nuts_level: int | None = None
    min_year: int | None = None
    max_year: int | None = None
    source_label: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------- #
# News / RAG
# --------------------------------------------------------------------------- #


class NewsCollectionsOut(BaseModel):
    """The set of Qdrant collections available to browse / search."""

    collections: list[str]


class NewsArticle(BaseModel):
    """One stored news/RAG document as returned by the browse endpoint."""

    id: str
    title: str = ""
    text: str = ""
    url: str = ""
    published: str = ""
    source: str = ""
    topic: str = ""
    sentiment: str = ""
    collection: str = ""


class NewsSearchRequest(BaseModel):
    """Body for ``POST /news/search`` — semantic search over the corpus."""

    query: str = Field(min_length=1)
    topic: str | None = None
    sentiment: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class NewsSearchHit(NewsArticle):
    """A search result: a news article plus its similarity score."""

    score: float = 0.0


class NewsSearchResponse(BaseModel):
    """Merged, score-ranked search results."""

    articles: list[NewsSearchHit]
    message: str | None = None


# --------------------------------------------------------------------------- #
# Proxy request bodies (forwarded to the existing services)
# --------------------------------------------------------------------------- #


class ForecastRequest(BaseModel):
    """Body for ``POST /forecast`` — forwarded to the forecaster ``/predict``."""

    model_config = ConfigDict(extra="allow")

    model_type: str = "prophet"
    dates: list[str]
    values: list[float]
    n_prev: int
    n_predict: int
    alpha: float = 0.05
    model_params: dict[str, Any] = Field(default_factory=dict)


class ClusterRequest(BaseModel):
    """Body for ``POST /cluster`` — forwarded to the clustering ``/cluster``.

    Only the three universally-required fields are declared; the algorithm
    tunables (``k``, ``eps``, ``reduction_method``, …) are accepted as extras
    and forwarded verbatim, so this stays in lockstep with the clustering
    service without re-declaring its ~25 parameters here.
    """

    model_config = ConfigDict(extra="allow")

    method: str
    dataframe: list[dict[str, Any]]
    feature_columns: list[str]


class ChatMessage(BaseModel):
    """One prior chat turn forwarded to the agent."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Body for ``POST /agent/chat/stream`` — forwarded to the agent."""

    user_message: str
    chat_history: list[ChatMessage] = Field(default_factory=list)


class PlotInterpretRequest(BaseModel):
    """Body for ``POST /agent/plots/interpret`` — forwarded to the agent."""

    image_base64: str
    mode: str = "no_hallucinations"
    chart_context: str = ""


class AgentModelsOut(BaseModel):
    """The list of LLM model ids the agent currently knows about."""

    models: list[str]
