"""Yahoo Finance fetch + Postgres upsert for one ticker at a time.

Called by ``downloader_extra``'s ``POST /ingest`` endpoint (``source=yahoo``)
when the agent asks for a ticker not yet in the database. Reuses the same
single-ticker logic as ``downloader_general``'s batch extractor (metadata via
``yf.Ticker(...).info``, full OHLCV via ``.history(period="max")``), writing the
``yahoo_metadata`` row first (FK target) then the ``yahoo_historical_prices``
rows. ``yfinance`` is synchronous, so the whole fetch+store runs on a worker
thread via :func:`asyncio.to_thread`.
"""

import asyncio
import logging
from typing import Any

import polars as pl
from schema import YahooHistoricalPrice, YahooMetadata
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session
import yfinance as yf

logger = logging.getLogger(__name__)


def _replace_yahoo_rows(
    metadata_row: dict[str, Any],
    price_rows: list[dict[str, Any]],
    ticker: str,
    sql_uri: str,
) -> None:
    """Delete any prior copy of ``ticker`` and insert the fresh metadata + prices.

    Historical rows are deleted before the metadata row to satisfy the
    ``yahoo_historical_prices.ticker → yahoo_metadata.ticker`` FK, then both are
    re-inserted in a single transaction.
    """
    engine = create_engine(sql_uri)
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(YahooHistoricalPrice).where(YahooHistoricalPrice.ticker == ticker)
            )
            session.execute(delete(YahooMetadata).where(YahooMetadata.ticker == ticker))
            session.add(YahooMetadata(**metadata_row))
            session.flush()  # land the FK target before the price rows
            session.add_all([YahooHistoricalPrice(**row) for row in price_rows])
    finally:
        engine.dispose()


def _fetch_and_store_yahoo_sync(ticker: str, sql_uri: str) -> int:
    """Blocking fetch+store for one ticker (run via :func:`asyncio.to_thread`)."""
    ticker_obj = yf.Ticker(ticker)

    hist = ticker_obj.history(period="max")
    if hist is None or hist.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    try:
        info: dict[str, Any] = ticker_obj.info or {}
    except Exception:
        # .info occasionally raises for thin/odd tickers; history already proved
        # the ticker is real, so fall back to a minimal metadata row.
        info = {}

    category = "Indices" if ticker.startswith("^") else "Companies"
    metadata_row: dict[str, Any] = {
        "ticker": ticker,
        "asset_name": info.get("shortName") or info.get("longName") or ticker,
        "category": category,
        "short_name": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "business_summary": info.get("longBusinessSummary"),
    }

    hist = hist.reset_index()
    hist["Date"] = hist["Date"].dt.tz_localize(None)

    df = (
        pl.from_pandas(hist)
        .select(
            pl.col("Date").alias("date"),
            pl.col("Open").alias("open"),
            pl.col("High").alias("high"),
            pl.col("Low").alias("low"),
            pl.col("Close").alias("close"),
            pl.col("Volume").cast(pl.Int64, strict=False).alias("volume"),
        )
        .with_columns(
            pl.lit(ticker).alias("ticker"),
            pl.lit(category).alias("category"),
        )
        .unique(subset=["date", "ticker"], keep="last", maintain_order=True)
    )

    price_rows = df.to_dicts()
    _replace_yahoo_rows(metadata_row, price_rows, ticker, sql_uri)
    logger.info("Stored %d price rows for Yahoo ticker %s", len(price_rows), ticker)
    return len(price_rows)


async def fetch_and_store_yahoo(ticker: str, sql_uri: str) -> int:
    """Fetch one Yahoo ticker and replace any prior copy in Postgres.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"^GSPC"``).
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.

    Returns:
        Number of price rows that were inserted.

    Raises:
        ValueError: When Yahoo returns no history for the ticker.
    """
    ticker = ticker.strip()
    return await asyncio.to_thread(_fetch_and_store_yahoo_sync, ticker, sql_uri)
