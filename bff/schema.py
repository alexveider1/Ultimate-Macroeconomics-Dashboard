"""Read-only ORM models for the tables the BFF serves.

Per the repo convention (each container owns its ``pyproject.toml`` /
``Dockerfile`` / build context, no shared package), the SQLAlchemy ``Mapped``
models are duplicated here rather than imported from a shared package. They
mirror ``_container_data/database_schema.yaml`` column-for-column for the tables
the dashboard reads, so ``Base.metadata.create_all`` is a no-op against the live
tables (and creates them for an isolated test database).

The BFF never writes, so these carry only the columns the read endpoints
project; the World Bank ``world_bank_countries`` catalogue uses dotted source column names
(``region.value`` etc.), mapped here to snake-case attributes.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for the BFF's read-only ORM models."""


# --------------------------------------------------------------------------- #
# World Bank
# --------------------------------------------------------------------------- #
class Country(Base):
    """One World Bank economy (``world_bank_countries``); populated from db=2 (WDI)."""

    __tablename__ = "world_bank_countries"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    value: Mapped[str | None] = mapped_column(String, nullable=True)
    aggregate: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    region_id: Mapped[str | None] = mapped_column("region.id", String, nullable=True)
    region_value: Mapped[str | None] = mapped_column("region.value", String, nullable=True)
    income_level_id: Mapped[str | None] = mapped_column("incomeLevel.id", String, nullable=True)
    income_level_value: Mapped[str | None] = mapped_column(
        "incomeLevel.value", String, nullable=True
    )
    capital_city: Mapped[str | None] = mapped_column("capitalCity", String, nullable=True)


class DatabaseIndicator(Base):
    """Human-readable indicator title per WB database (``world_bank_database_indicators``)."""

    __tablename__ = "world_bank_database_indicators"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    database_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)


class MetadataRow(Base):
    """Rich descriptive metadata for one WB indicator series (``world_bank_metadata``)."""

    __tablename__ = "world_bank_metadata"

    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    db_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    indicator_name: Mapped[str | None] = mapped_column(String, nullable=True)
    units: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    development_relevance: Mapped[str | None] = mapped_column(String, nullable=True)
    limitations_and_exceptions: Mapped[str | None] = mapped_column(String, nullable=True)
    statistical_concept_and_methodology: Mapped[str | None] = mapped_column(String, nullable=True)


class MacroIndicator(Base):
    """One ``(economy, year, indicator_id, db_id)`` World Bank cell (``world_bank_indicators``)."""

    __tablename__ = "world_bank_indicators"

    economy: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    year: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    db_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False, index=True)


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Binance crypto
# --------------------------------------------------------------------------- #
class BinanceMetadata(Base):
    """Master row for one Binance spot pair (``binance_metadata``)."""

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


# --------------------------------------------------------------------------- #
# FRED US-state
# --------------------------------------------------------------------------- #
class State(Base):
    """One U.S. state / DC row (``fred_states``)."""

    __tablename__ = "fred_states"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    fips: Mapped[str | None] = mapped_column(String, nullable=True)
    region: Mapped[str | None] = mapped_column(String, nullable=True)
    division: Mapped[str | None] = mapped_column(String, nullable=True)


class StateIndicator(Base):
    """Description row for one FRED state-indicator concept (``fred_state_indicators``)."""

    __tablename__ = "fred_state_indicators"

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
    """One ``(state, year, indicator_id)`` FRED observation (``fred_state_indicator_values``)."""

    __tablename__ = "fred_state_indicator_values"

    state: Mapped[str] = mapped_column(
        String, ForeignKey("fred_states.id"), primary_key=True, nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicator_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)


# --------------------------------------------------------------------------- #
# Eurostat EU NUTS-2
# --------------------------------------------------------------------------- #
class Region(Base):
    """One NUTS-2 region row (``eurostat_regions``)."""

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
