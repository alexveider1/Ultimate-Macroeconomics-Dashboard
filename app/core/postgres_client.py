import os
import yaml
import connectorx as cx
import polars as pl
import streamlit as st
from typing import Iterable

from dotenv import load_dotenv
from core.app_logging import log_sql_query

CONFIG = yaml.safe_load(open("config.yaml"))
load_dotenv(".env")
SQL_URL = f"postgresql://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@{CONFIG.get('postgres').get('host')}:{CONFIG.get('postgres').get('port')}/{CONFIG.get('postgres').get('database')}"
POSTGRES_TARGET = (
    f"{CONFIG.get('postgres').get('host')}:{CONFIG.get('postgres').get('port')}"
)


def _sql_string(value: str) -> str:
    raw = str(value).strip()
    if len(raw) >= 2 and (
        (raw[0] == "'" and raw[-1] == "'") or (raw[0] == '"' and raw[-1] == '"')
    ):
        raw = raw[1:-1]

    return "'" + raw.replace("'", "''") + "'"


def _normalize_country_codes(country_code) -> list[str]:
    if isinstance(country_code, str):
        normalized = country_code.strip()
        if not normalized or normalized.upper() == "ALL":
            return []
        return [normalized]

    if isinstance(country_code, Iterable):
        normalized_codes = [
            str(code).strip() for code in country_code if str(code).strip()
        ]
        if not normalized_codes or any(
            code.upper() == "ALL" for code in normalized_codes
        ):
            return []
        return normalized_codes

    normalized = str(country_code).strip()
    return [] if not normalized or normalized.upper() == "ALL" else [normalized]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_postgres_data(
    sql_uri: str | None = SQL_URL,
    query: str | None = None,
    partition_on: str = None,
    partitions: int = None,
) -> pl.DataFrame:
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
    except Exception:
        df = cx.read_sql(sql_uri, query, return_type="polars")
        return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_indicator(
    indicator_code: str, country_code: str = "ALL"
) -> pl.DataFrame:
    country_codes = _normalize_country_codes(country_code)
    query = (
        "SELECT year, economy, value "
        "FROM indicators "
        f"WHERE indicator_id = {_sql_string(indicator_code)}"
    )

    if country_codes:
        query += (
            " AND economy IN ("
            + ", ".join(_sql_string(code) for code in country_codes)
            + ")"
        )

    query += " ORDER BY year, economy"

    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_yahoo_finance_timeseries(ticker: str) -> pl.DataFrame:
    query = (
        "SELECT date, open, high, low, close, volume, ticker "
        "FROM yahoo_historical_prices "
        f"WHERE ticker = {_sql_string(ticker)}"
    )

    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_metadata(indicator_code: str) -> pl.DataFrame:
    query = f"SELECT * FROM metadata WHERE indicator_id = {_sql_string(indicator_code)}"
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_indicator_name(
    indicator_code: str, preferred_database_id: str | int = "2"
) -> str:
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
    query = (
        "SELECT ticker, asset_name, category, short_name, sector, industry, currency, "
        "exchange, business_summary "
        "FROM yahoo_metadata "
        f"WHERE ticker = {_sql_string(ticker)}"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_yahoo_historical_prices() -> pl.DataFrame:
    query = (
        "SELECT date, open, high, low, close, volume, ticker "
        "FROM yahoo_historical_prices "
        "WHERE date IS NOT NULL AND close IS NOT NULL AND ticker IS NOT NULL"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_yahoo_metadata() -> pl.DataFrame:
    query = (
        "SELECT ticker, asset_name, category, short_name, sector, industry, currency, exchange "
        "FROM yahoo_metadata"
    )
    return fetch_postgres_data(query=query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_world_bank_country_mapping() -> pl.DataFrame:
    query = "SELECT id, value FROM countries WHERE id IS NOT NULL AND value IS NOT NULL"
    return fetch_postgres_data(query=query)
