"""ORM models + pydantic schemas for the on-demand ingestion service.

The service re-fetches a single unit of data on demand from one of five
sources and writes it into the same Postgres tables that ``downloader_general``
populates on first boot:

* ``worldbank`` — one indicator → ``indicators`` (:class:`MacroIndicator`).
* ``yahoo`` — one ticker → ``yahoo_metadata`` + ``yahoo_historical_prices``.
* ``binance`` — one spot pair → ``binance_metadata`` + ``binance_historical_prices``.
* ``fred`` — one US-state indicator → ``state_indicators`` + ``state_indicator_values``.
* ``eurostat`` — one NUTS-2 dataset → ``eurostat_indicators`` + ``eurostat_indicator_values``.

The ORM models mirror ``_container_data/database_schema.yaml`` column-for-column
so ``Base.metadata.create_all`` is a no-op against the tables that already exist
(and creates them for an isolated test database).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, model_validator
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for this service's ORM models."""


class MacroIndicator(Base):
    """One ``(economy, year, indicator_id, db_id)`` cell from the World Bank.

    Mirrors the ``indicators`` table that ``downloader_general`` populates
    on first boot; this service re-fetches single indicators on demand.
    """

    __tablename__ = "indicators"

    economy: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    year: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    db_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False, index=True)


class YahooMetadata(Base):
    """Master row for one Yahoo Finance ticker (``yahoo_metadata``)."""

    __tablename__ = "yahoo_metadata"

    ticker: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    asset_name: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    short_name: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    business_summary: Mapped[str | None] = mapped_column(String, nullable=True)


class YahooHistoricalPrice(Base):
    """One daily OHLCV row for a Yahoo ticker (``yahoo_historical_prices``)."""

    __tablename__ = "yahoo_historical_prices"

    date: Mapped[datetime] = mapped_column(DateTime, primary_key=True, nullable=False)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    ticker: Mapped[str] = mapped_column(
        String, ForeignKey("yahoo_metadata.ticker"), primary_key=True, nullable=False
    )
    category: Mapped[str | None] = mapped_column(String, nullable=True)


class BinanceMetadata(Base):
    """Master row for one Binance spot pair (``binance_metadata``).

    On-demand rows leave ``rank`` NULL (there is no batch-wide popularity
    ranking when a single pair is fetched in isolation).
    """

    __tablename__ = "binance_metadata"

    symbol: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    base_asset: Mapped[str | None] = mapped_column(String, nullable=True)
    quote_asset: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_change_percent_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    quote_volume_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    trade_count_24h: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class BinanceHistoricalPrice(Base):
    """One daily candle for a Binance pair (``binance_historical_prices``)."""

    __tablename__ = "binance_historical_prices"

    date: Mapped[datetime] = mapped_column(DateTime, primary_key=True, nullable=False)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    quote_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    symbol: Mapped[str] = mapped_column(
        String, ForeignKey("binance_metadata.symbol"), primary_key=True, nullable=False
    )
    base_asset: Mapped[str | None] = mapped_column(String, nullable=True)


class State(Base):
    """One U.S. state / DC row (``states``); mirrors the FRED states catalogue."""

    __tablename__ = "states"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    fips: Mapped[str | None] = mapped_column(String, nullable=True)
    region: Mapped[str | None] = mapped_column(String, nullable=True)
    division: Mapped[str | None] = mapped_column(String, nullable=True)


class StateIndicator(Base):
    """Description row for one FRED state-indicator concept (``state_indicators``)."""

    __tablename__ = "state_indicators"

    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    series_group: Mapped[str | None] = mapped_column(String, nullable=True)
    example_series_id: Mapped[str | None] = mapped_column(String, nullable=True)
    units: Mapped[str | None] = mapped_column(String, nullable=True)
    frequency: Mapped[str | None] = mapped_column(String, nullable=True)
    seasonal_adjustment: Mapped[str | None] = mapped_column(String, nullable=True)
    region_type: Mapped[str | None] = mapped_column(String, nullable=True)
    min_date: Mapped[str | None] = mapped_column(String, nullable=True)
    max_date: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


class StateIndicatorValue(Base):
    """One ``(state, year, indicator_id)`` FRED observation (``state_indicator_values``)."""

    __tablename__ = "state_indicator_values"

    state: Mapped[str] = mapped_column(
        String, ForeignKey("states.id"), primary_key=True, nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)


class Region(Base):
    """One NUTS-2 region row (``eurostat_regions``); mirrors the Eurostat catalogue."""

    __tablename__ = "eurostat_regions"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    country_code: Mapped[str | None] = mapped_column(String, nullable=True)
    country_name: Mapped[str | None] = mapped_column(String, nullable=True)
    nuts1_id: Mapped[str | None] = mapped_column(String, nullable=True)
    level: Mapped[int | None] = mapped_column(Integer, nullable=True)


class EurostatIndicator(Base):
    """Description row for one Eurostat indicator concept (``eurostat_indicators``)."""

    __tablename__ = "eurostat_indicators"

    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset: Mapped[str | None] = mapped_column(String, nullable=True)
    filters: Mapped[str | None] = mapped_column(String, nullable=True)
    units: Mapped[str | None] = mapped_column(String, nullable=True)
    frequency: Mapped[str | None] = mapped_column(String, nullable=True)
    nuts_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_label: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


class EurostatIndicatorValue(Base):
    """One ``(region, year, indicator_id)`` Eurostat observation (``eurostat_indicator_values``)."""

    __tablename__ = "eurostat_indicator_values"

    region: Mapped[str] = mapped_column(
        String, ForeignKey("eurostat_regions.id"), primary_key=True, nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)


SourceLiteral = Literal["worldbank", "yahoo", "binance", "fred", "eurostat"]


class IngestRequest(BaseModel):
    """Body for ``POST /ingest`` — one of three sources.

    ``source`` selects which id fields are required:

    * ``worldbank`` → ``indicator_id`` (e.g. ``NY.GDP.MKTP.CD``) + ``db_id`` (e.g. 2).
    * ``yahoo`` → ``ticker`` (e.g. ``AAPL``).
    * ``binance`` → ``symbol`` (full spot pair, e.g. ``BTCUSDT``).
    * ``fred`` → ``series_id`` (a representative single-state series, e.g. ``CAUR``
      or ``MEHOINUSCAA672N``); the whole 50-state + DC panel is fetched from it.
    * ``eurostat`` → ``dataset`` (a Eurostat dataset code, e.g. ``nama_10r_2gdp``)
      plus optional ``filters`` pinning its extra dimensions (e.g.
      ``{"unit": "EUR_HAB"}``); the whole NUTS-2 region panel is fetched from it.

    ``source`` defaults to ``worldbank`` so the historical World-Bank-only
    request body (``{indicator_id, db_id}``) keeps working unchanged.
    """

    source: SourceLiteral = "worldbank"
    indicator_id: str | None = None
    db_id: int | None = None
    ticker: str | None = None
    symbol: str | None = None
    series_id: str | None = None
    dataset: str | None = None
    filters: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> "IngestRequest":
        if self.source == "worldbank":
            if not self.indicator_id or self.db_id is None:
                raise ValueError("worldbank source requires 'indicator_id' and 'db_id'")
        elif self.source == "yahoo":
            if not self.ticker:
                raise ValueError("yahoo source requires 'ticker'")
        elif self.source == "binance":
            if not self.symbol:
                raise ValueError("binance source requires 'symbol'")
        elif self.source == "fred":
            if not self.series_id:
                raise ValueError("fred source requires 'series_id'")
        elif self.source == "eurostat":
            if not self.dataset:
                raise ValueError("eurostat source requires 'dataset'")
        return self


class IngestResponse(BaseModel):
    """Response from ``POST /ingest``.

    Args:
        source: Which source served the request (echo of the request).
        identifier: The id that was ingested — ``indicator_id`` / ``ticker`` /
            ``symbol`` depending on ``source``.
        db_id: World Bank database id; ``None`` for yahoo / binance.
        rows_inserted: Number of price/observation rows written; ``0`` when the
            data was already present.
        status: ``success`` or ``already_downloaded``.
    """

    source: str
    identifier: str
    db_id: int | None = None
    rows_inserted: int
    status: str
