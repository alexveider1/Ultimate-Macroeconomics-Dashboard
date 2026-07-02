"""Async Binance public-market-data client (httpx).

Talks to the documented Binance spot REST endpoints over a single shared
:class:`httpx.AsyncClient`. Every public coroutine returns plain
``list``/``dict`` objects shaped for the ingestion pipeline (so the Polars
schema cast in :mod:`src.utils.schema` keeps working unchanged):

- :func:`fetch_exchange_info` → raw ``symbols`` entries (caller filters them)
- :func:`fetch_24h_tickers`   → trailing-24h ticker stats for every symbol
- :func:`fetch_klines`        → ``binance_historical_prices`` rows (paged to the
  full history of the symbol)

Only public endpoints are used, so no API key is required. Retries are applied
by the caller via :func:`src.utils.wb_client.call_with_retries` (the generic
async retry already used by the World Bank pipeline), so this module stays a
thin transport layer.
"""

from datetime import datetime, timedelta
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.binance.com"
DEFAULT_TIMEOUT = 30.0
# Binance caps a single klines page at 1000 (spot); paging keeps the request
# count low even for coins with years of daily candles.
_KLINES_PAGE_LIMIT = 1000
_EPOCH = datetime(1970, 1, 1)


def build_async_client(base_url: str = DEFAULT_BASE_URL) -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` configured for the Binance API.

    Args:
        base_url: Root of the Binance REST API (e.g. ``https://api.binance.com``
            or the public ``https://data-api.binance.vision`` mirror).
    """
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=DEFAULT_TIMEOUT,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={"Accept": "application/json"},
    )


def _ms_to_naive_utc(open_time_ms: int) -> datetime:
    """Convert an epoch-millisecond timestamp to a naive UTC ``datetime``."""
    return _EPOCH + timedelta(milliseconds=open_time_ms)


def _to_float(value: Any) -> Optional[float]:
    """Parse a Binance numeric string to ``float``; ``None`` when missing/invalid."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


async def fetch_exchange_info(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Return the raw ``symbols`` catalogue from ``/api/v3/exchangeInfo``.

    Each entry carries ``symbol``, ``status``, ``baseAsset``, ``quoteAsset`` and
    ``isSpotTradingAllowed``; the caller is responsible for filtering down to the
    spot pairs it wants.
    """
    resp = await client.get("/api/v3/exchangeInfo")
    resp.raise_for_status()
    payload = resp.json()
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    return symbols or []


async def fetch_24h_tickers(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Return trailing-24h ticker stats for every symbol (``/api/v3/ticker/24hr``).

    Each entry carries ``symbol``, ``lastPrice``, ``priceChangePercent``,
    ``highPrice``, ``lowPrice``, ``quoteVolume`` and ``count``.
    """
    resp = await client.get("/api/v3/ticker/24hr")
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str = "1d",
    start_time: int = 0,
) -> list[dict[str, Any]]:
    """Fetch one symbol's full candle history (``binance_historical_prices`` rows).

    Pages forward from ``start_time`` (``0`` = earliest available) until the API
    returns a short/empty page, advancing the cursor past the last candle's open
    time each iteration. Each 12-field kline array is projected onto the schema
    columns; ``date`` is the candle open time as a naive UTC ``datetime``.

    Args:
        client: Shared async HTTP client.
        symbol: Spot pair symbol (e.g. ``"BTCUSDT"``).
        interval: Kline interval (``"1d"`` for daily candles).
        start_time: Epoch-millisecond lower bound for the first page.

    Returns:
        Rows shaped as ``{date, open, high, low, close, volume, quote_volume}``,
        ascending by ``date``.
    """
    rows: list[dict[str, Any]] = []
    cursor = start_time
    while True:
        resp = await client.get(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "limit": _KLINES_PAGE_LIMIT,
            },
        )
        resp.raise_for_status()
        page = resp.json()
        if not isinstance(page, list) or not page:
            break

        for kline in page:
            rows.append(
                {
                    "date": _ms_to_naive_utc(int(kline[0])),
                    "open": _to_float(kline[1]),
                    "high": _to_float(kline[2]),
                    "low": _to_float(kline[3]),
                    "close": _to_float(kline[4]),
                    "volume": _to_float(kline[5]),
                    "quote_volume": _to_float(kline[7]),
                }
            )

        last_open_time = int(page[-1][0])
        # Reached the most recent candle when the page isn't full; also guard
        # against a non-advancing cursor so we never loop forever.
        next_cursor = last_open_time + 1
        if len(page) < _KLINES_PAGE_LIMIT or next_cursor <= cursor:
            break
        cursor = next_cursor

    return rows


async def healthcheck(client: httpx.AsyncClient) -> bool:
    """Return ``True`` if the Binance API answers ``/api/v3/ping``."""
    try:
        resp = await client.get("/api/v3/ping")
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception("An error occured while testing connection to Binance API")
        return False
