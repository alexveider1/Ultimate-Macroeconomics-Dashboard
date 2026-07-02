"""Binance fetch + Postgres upsert for one spot pair at a time.

Called by ``downloader_extra``'s ``POST /ingest`` endpoint (``source=binance``)
when the agent asks for a crypto pair not yet in the database. Validates the
full pair symbol against the live ``/exchangeInfo`` endpoint, synthesizes one
``binance_metadata`` row (leaving ``rank`` NULL — there is no batch-wide
popularity ranking for a single on-demand pair), then pages the pair's full
daily candle history into ``binance_historical_prices``. All HTTP goes through
the async :mod:`binance_client`; the blocking DB write is offloaded to a worker
thread so the event loop stays free.
"""

import asyncio
import logging
from typing import Any

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

import binance_client
from schema import BinanceHistoricalPrice, BinanceMetadata

logger = logging.getLogger(__name__)


def _replace_binance_rows(
    metadata_row: dict[str, Any],
    price_rows: list[dict[str, Any]],
    symbol: str,
    sql_uri: str,
) -> None:
    """Delete any prior copy of ``symbol`` and insert the fresh metadata + prices.

    Historical rows are deleted before the metadata row to satisfy the
    ``binance_historical_prices.symbol → binance_metadata.symbol`` FK, then both
    are re-inserted in a single transaction. Intended to be called via
    :func:`asyncio.to_thread`.
    """
    engine = create_engine(sql_uri)
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(BinanceHistoricalPrice).where(BinanceHistoricalPrice.symbol == symbol)
            )
            session.execute(delete(BinanceMetadata).where(BinanceMetadata.symbol == symbol))
            session.add(BinanceMetadata(**metadata_row))
            session.flush()  # land the FK target before the price rows
            session.add_all([BinanceHistoricalPrice(**row) for row in price_rows])
    finally:
        engine.dispose()


async def fetch_and_store_binance(symbol: str, sql_uri: str) -> int:
    """Fetch one Binance spot pair and replace any prior copy in Postgres.

    Args:
        symbol: Full spot pair symbol (e.g. ``"BTCUSDT"``).
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.

    Returns:
        Number of candle rows that were inserted.

    Raises:
        ValueError: When the symbol is not a known spot pair, or has no candles.
    """
    symbol = symbol.upper().strip()
    async with binance_client.build_async_client() as client:
        info = await binance_client.fetch_symbol_info(client, symbol)
        if info is None:
            raise ValueError(f"Unknown Binance spot pair: {symbol}")

        base_asset = str(info.get("baseAsset") or "")
        quote_asset = str(info.get("quoteAsset") or "")
        status = str(info.get("status") or "")

        ticker = await binance_client.fetch_24h_ticker_for_symbol(client, symbol)
        price_rows = await binance_client.fetch_klines(client, symbol, interval="1d")

    if not price_rows:
        raise ValueError(f"No historical candles found for Binance pair: {symbol}")

    metadata_row: dict[str, Any] = {
        "symbol": symbol,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "status": status,
        "rank": None,
        "description": (
            f"{base_asset}/{quote_asset} spot pair on Binance — "
            f"downloaded on demand (not part of the ranked top-coin set)."
        ),
        "last_price": binance_client._to_float(ticker.get("lastPrice")),
        "price_change_percent_24h": binance_client._to_float(ticker.get("priceChangePercent")),
        "high_24h": binance_client._to_float(ticker.get("highPrice")),
        "low_24h": binance_client._to_float(ticker.get("lowPrice")),
        "quote_volume_24h": binance_client._to_float(ticker.get("quoteVolume")),
        "trade_count_24h": ticker.get("count"),
    }

    for row in price_rows:
        row["symbol"] = symbol
        row["base_asset"] = base_asset

    await asyncio.to_thread(_replace_binance_rows, metadata_row, price_rows, symbol, sql_uri)
    logger.info("Stored %d candle rows for Binance pair %s", len(price_rows), symbol)
    return len(price_rows)
