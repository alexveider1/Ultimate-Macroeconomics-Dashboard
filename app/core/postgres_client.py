"""Read-only Postgres helpers used by every dashboard page.

The app intentionally uses ``connectorx`` (not SQLAlchemy ORM) so bulk reads
land directly in a Polars DataFrame with no row-by-row Python overhead. The
helpers below build parameter-free SQL strings, push them through
``connectorx.read_sql``, and cache results via ``st.cache_data`` so repeated
clicks on the same page don't re-query Postgres.
"""

import logging
from pathlib import Path
from typing import Iterable

import connectorx as cx
import polars as pl
import streamlit as st

from core.app_logging import log_sql_query
from core.config import load_config
from core.settings import get_settings

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)
SETTINGS = get_settings()

_PG = CONFIG.postgres
# Database name source of truth is POSTGRES_DB in .env (the postgres image only
# reads it from there on first volume init). config.yaml's `database` stays as
# a fallback for environments that don't set the env var.
_PG_DATABASE = SETTINGS.postgres_db or _PG.database
SQL_URL = (
    f"postgresql://"
    f"{SETTINGS.postgres_llm_user}:{SETTINGS.postgres_llm_password}"
    f"@{_PG.host}:{_PG.port}/{_PG_DATABASE}"
)
POSTGRES_TARGET = f"{_PG.host}:{_PG.port}"

logger = logging.getLogger(__name__)


def _sql_string(value: str) -> str:
    """Quote and escape ``value`` for safe interpolation into a SQL literal.

    Strips surrounding quotes and doubles every single-quote inside, then
    wraps the result in single quotes. Used because ``connectorx`` does not
    take bound parameters.

    Args:
        value: Raw string from caller input or config.

    Returns:
        A safely quoted SQL string literal.
    """
    raw = str(value).strip()
    if len(raw) >= 2 and ((raw[0] == "'" and raw[-1] == "'") or (raw[0] == '"' and raw[-1] == '"')):
        raw = raw[1:-1]

    return "'" + raw.replace("'", "''") + "'"


def _normalize_country_codes(country_code: str | Iterable[str]) -> list[str]:
    """Normalise a country-code argument into a list of upper-case ISO codes.

    Args:
        country_code: A single code, an iterable of codes, or the string
            ``"ALL"`` (case-insensitive) to mean "no filter".

    Returns:
        List of codes — empty for the "all countries" case.
    """
    if isinstance(country_code, str):
        normalized = country_code.strip()
        if not normalized or normalized.upper() == "ALL":
            return []
        return [normalized]

    if isinstance(country_code, Iterable):
        normalized_codes = [str(code).strip() for code in country_code if str(code).strip()]
        if not normalized_codes or any(code.upper() == "ALL" for code in normalized_codes):
            return []
        return normalized_codes

    normalized = str(country_code).strip()
    return [] if not normalized or normalized.upper() == "ALL" else [normalized]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_postgres_data(
    sql_uri: str | None = SQL_URL,
    query: str | None = None,
    partition_on: str | None = None,
    partitions: int | None = None,
) -> pl.DataFrame:
    """Run ``query`` against Postgres via ``connectorx`` and return Polars rows.

    Args:
        sql_uri: Optional override; defaults to the URI derived from env.
        query: SQL string (already-quoted via :func:`_sql_string`).
        partition_on: Column to partition the read on for parallelism.
        partitions: Number of partitions; ignored unless ``partition_on`` is set.

    Returns:
        Polars DataFrame with the result rows.
    """
    log_sql_query(query or "", target=POSTGRES_TARGET)
    try:
        if partition_on and partitions:
            df = cx.read_sql(
                sql_uri,
                query,
                partition_on=partition_on,
                partition_num=partitions,
                return_type="polars",
            )
        else:
            df = cx.read_sql(sql_uri, query, return_type="polars")
        return df
    except Exception as exc:
        logger.warning("Partitioned read failed, retrying without partitions: %s", exc)
        df = cx.read_sql(sql_uri, query, return_type="polars")
        return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_indicator(indicator_code: str, country_code: str = "ALL") -> pl.DataFrame:
    """Return ``(year, economy, value)`` rows for one WB indicator.

    Args:
        indicator_code: World Bank indicator id (e.g. ``NY.GDP.MKTP.CD``).
        country_code: Single code, list, or ``"ALL"`` for no filter.

    Returns:
        Polars frame ordered by ``year`` then ``economy``.
    """
    country_codes = _normalize_country_codes(country_code)
    query = (
        "SELECT year, economy, value "
        "FROM indicators "
        f"WHERE indicator_id = {_sql_string(indicator_code)}"
    )

    if country_codes:
        query += " AND economy IN (" + ", ".join(_sql_string(code) for code in country_codes) + ")"

    query += " ORDER BY year, economy"

    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_yahoo_finance_timeseries(ticker: str) -> pl.DataFrame:
    """Return the OHLCV history for one Yahoo Finance ticker."""
    query = (
        "SELECT date, open, high, low, close, volume, ticker "
        "FROM yahoo_historical_prices "
        f"WHERE ticker = {_sql_string(ticker)}"
    )

    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_metadata(indicator_code: str) -> pl.DataFrame:
    """Return every metadata row stored for ``indicator_code``."""
    query = f"SELECT * FROM metadata WHERE indicator_id = {_sql_string(indicator_code)}"
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_indicator_name(
    indicator_code: str, preferred_database_id: str | int = "2"
) -> str:
    """Return the human-readable name for an indicator, preferring one DB id.

    Args:
        indicator_code: WB indicator id.
        preferred_database_id: When the indicator appears in multiple
            databases, the row matching this id wins.

    Returns:
        Description string, or ``""`` when no matching row is found.
    """
    preferred_db = str(preferred_database_id).strip()
    query = (
        "SELECT id, description, database_id "
        "FROM database_indicators "
        f"WHERE id = {_sql_string(indicator_code)} "
        "AND description IS NOT NULL AND description <> '' "
        "ORDER BY "
        f"CASE WHEN COALESCE(database_id::text, '') = {_sql_string(preferred_db)} THEN 0 ELSE 1 END, "
        "COALESCE(database_id::text, '') "
        "LIMIT 1"
    )
    df = fetch_postgres_data(query=query)
    if df.is_empty() or "description" not in df.columns:
        return ""

    raw_name = str(df["description"][0]).strip()
    return raw_name


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_country_codes() -> list[str]:
    """Return every distinct, non-empty ``economy`` code present in indicators."""
    query = (
        "SELECT DISTINCT economy "
        "FROM indicators "
        "WHERE economy IS NOT NULL AND economy <> '' "
        "ORDER BY economy"
    )
    df = fetch_postgres_data(query=query)
    if df.is_empty() or "economy" not in df.columns:
        return []

    return [str(code).strip() for code in df["economy"].to_list() if str(code).strip()]


@st.cache_data(ttl=3600, show_spinner=False)
def get_yahoo_metadata(ticker: str) -> pl.DataFrame:
    """Return the full metadata row (sector / industry / etc.) for ``ticker``."""
    query = (
        "SELECT ticker, asset_name, category, short_name, sector, industry, currency, "
        "exchange, business_summary "
        "FROM yahoo_metadata "
        f"WHERE ticker = {_sql_string(ticker)}"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_yahoo_historical_prices() -> pl.DataFrame:
    """Return the complete OHLCV history for every Yahoo ticker."""
    query = (
        "SELECT date, open, high, low, close, volume, ticker "
        "FROM yahoo_historical_prices "
        "WHERE date IS NOT NULL AND close IS NOT NULL AND ticker IS NOT NULL"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_yahoo_metadata() -> pl.DataFrame:
    """Return one metadata row per Yahoo ticker (no business summary)."""
    query = (
        "SELECT ticker, asset_name, category, short_name, sector, industry, currency, exchange "
        "FROM yahoo_metadata"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_binance_historical_prices() -> pl.DataFrame:
    """Return the complete daily OHLCV history for every Binance coin.

    Returns an empty frame (rather than raising) when the table doesn't exist —
    a deployment whose volume predates the crypto tables shouldn't crash the
    Crypto page.
    """
    query = (
        "SELECT date, open, high, low, close, volume, quote_volume, symbol, base_asset "
        "FROM binance_historical_prices "
        "WHERE date IS NOT NULL AND close IS NOT NULL AND symbol IS NOT NULL"
    )
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("Binance historical prices unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_binance_metadata() -> pl.DataFrame:
    """Return one master-data row per Binance coin (ranked by 24h volume).

    Empty frame on a missing table, mirroring
    :func:`get_all_binance_historical_prices`.
    """
    query = (
        "SELECT symbol, base_asset, quote_asset, status, rank, description, last_price, "
        "price_change_percent_24h, high_24h, low_24h, quote_volume_24h, trade_count_24h "
        "FROM binance_metadata"
    )
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("Binance metadata unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_country_mapping() -> pl.DataFrame:
    """Return ``(id, value)`` for every WB economy with both fields set."""
    query = "SELECT id, value FROM countries WHERE id IS NOT NULL AND value IS NOT NULL"
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_country_regions() -> pl.DataFrame:
    """Return ``(id, value, region)`` for every non-aggregate WB economy."""
    query = (
        'SELECT id, value, "region.value" AS region '
        "FROM countries WHERE id IS NOT NULL AND aggregate = false"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_states() -> pl.DataFrame:
    """Return the FRED states catalogue ``(id, name, fips, region, division)``.

    Empty frame (rather than raising) on a missing table, so a deployment whose
    volume predates the FRED tables doesn't crash the Regional Statistics page.
    """
    query = "SELECT id, name, fips, region, division FROM states"
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("FRED states unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_indicators_catalog() -> pl.DataFrame:
    """Return every FRED indicator description row (one per indicator concept)."""
    query = (
        "SELECT indicator_id, name, category, series_group, example_series_id, units, "
        "frequency, seasonal_adjustment, region_type, min_date, max_date, notes "
        "FROM state_indicators"
    )
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("FRED indicator catalogue unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_indicator_meta(indicator_id: str) -> pl.DataFrame:
    """Return the single description row for one FRED indicator."""
    query = (
        "SELECT indicator_id, name, category, series_group, example_series_id, units, "
        "frequency, seasonal_adjustment, region_type, min_date, max_date, notes "
        "FROM state_indicators "
        f"WHERE indicator_id = {_sql_string(indicator_id)}"
    )
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("FRED indicator meta unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_indicator(indicator_id: str, states: str | Iterable[str] = "ALL") -> pl.DataFrame:
    """Return ``(state, year, value)`` rows for one FRED indicator.

    Args:
        indicator_id: FRED indicator slug (e.g. ``"unemployment_rate"``).
        states: Single 2-letter abbrev, an iterable of them, or ``"ALL"``.

    Returns:
        Polars frame ordered by ``year`` then ``state``.
    """
    state_codes = _normalize_country_codes(states)
    query = (
        "SELECT state, year, value "
        "FROM state_indicator_values "
        f"WHERE indicator_id = {_sql_string(indicator_id)} AND value IS NOT NULL"
    )
    if state_codes:
        query += " AND state IN (" + ", ".join(_sql_string(code) for code in state_codes) + ")"
    query += " ORDER BY year, state"
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("FRED indicator values unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_eurostat_regions() -> pl.DataFrame:
    """Return the Eurostat NUTS-2 catalogue ``(id, name, country_code, country_name, nuts1_id, level)``.

    Empty frame (rather than raising) on a missing table, so a deployment whose
    volume predates the Eurostat tables doesn't crash the Regional Statistics page.
    """
    query = "SELECT id, name, country_code, country_name, nuts1_id, level FROM eurostat_regions"
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("Eurostat regions unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_eurostat_indicators_catalog() -> pl.DataFrame:
    """Return every Eurostat indicator description row (one per indicator concept)."""
    query = (
        "SELECT indicator_id, name, category, dataset, filters, units, frequency, "
        "nuts_level, min_year, max_year, source_label, notes "
        "FROM eurostat_indicators"
    )
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("Eurostat indicator catalogue unavailable: %s", exc)
        return pl.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_eurostat_indicator(indicator_id: str, regions: str | Iterable[str] = "ALL") -> pl.DataFrame:
    """Return ``(region, year, value)`` rows for one Eurostat indicator.

    Args:
        indicator_id: Eurostat indicator slug (e.g. ``"gdp_per_capita_pps"``).
        regions: Single NUTS-2 code, an iterable of them, or ``"ALL"``.

    Returns:
        Polars frame ordered by ``year`` then ``region``.
    """
    region_codes = _normalize_country_codes(regions)
    query = (
        "SELECT region, year, value "
        "FROM eurostat_indicator_values "
        f"WHERE indicator_id = {_sql_string(indicator_id)} AND value IS NOT NULL"
    )
    if region_codes:
        query += " AND region IN (" + ", ".join(_sql_string(code) for code in region_codes) + ")"
    query += " ORDER BY year, region"
    try:
        return fetch_postgres_data(query=query)
    except Exception as exc:
        logger.warning("Eurostat indicator values unavailable: %s", exc)
        return pl.DataFrame()
