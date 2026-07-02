"""Async Binance public-market-data client (httpx) — single-symbol subset.

A trimmed copy of ``downloader_general``'s client (per-service duplication, like
:mod:`wb_client`): the on-demand service only ever ingests one spot pair at a
time, so it validates the symbol and pulls just that pair's stats + candle
history. Only public spot endpoints are used, so no API key is required.

- :func:`fetch_symbol_info` → ``/api/v3/exchangeInfo?symbol=`` (validate + read
  ``baseAsset`` / ``quoteAsset`` / ``status``); ``None`` when the symbol is unknown.
- :func:`fetch_24h_ticker_for_symbol` → ``/api/v3/ticker/24hr?symbol=`` (single
  stats dict).
- :func:`fetch_klines` → ``binance_historical_prices`` rows (paged to full history).
"""

import logging
from datetime import datetime, timedelta
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
    """Return an ``httpx.AsyncClient`` configured for the Binance API."""
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


async def fetch_symbol_info(client: httpx.AsyncClient, symbol: str) -> Optional[dict[str, Any]]:
    """Validate a spot pair and return its ``exchangeInfo`` entry.

    Queries ``/api/v3/exchangeInfo?symbol=`` for the single pair. Binance answers
    HTTP 400 for an unknown symbol, which is treated as "not a valid pair" and
    surfaced as ``None`` so the caller can raise a clean error.

    Args:
        client: Shared async HTTP client.
        symbol: Full spot pair symbol (e.g. ``"BTCUSDT"``).

    Returns:
        The symbol's ``exchangeInfo`` dict (``baseAsset`` / ``quoteAsset`` /
        ``status`` / ``isSpotTradingAllowed`` ...), or ``None`` when unknown.
    """
    try:
        resp = await client.get("/api/v3/exchangeInfo", params={"symbol": symbol})
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        return None
    payload = resp.json()
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not symbols:
        return None
    return symbols[0]


async def fetch_24h_ticker_for_symbol(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    """Return trailing-24h ticker stats for one symbol (``/api/v3/ticker/24hr?symbol=``).

    Passing the ``symbol`` query parameter makes Binance return a single object
    (not the full-market list). Returns an empty dict on any error so the caller
    can still write a metadata row with NULL stats.
    """
    try:
        resp = await client.get("/api/v3/ticker/24hr", params={"symbol": symbol})
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        return {}
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str = "1d",
    start_time: int = 0,
) -> list[dict[str, Any]]:
    """Fetch one symbol's full candle history (``binance_historical_prices`` rows).

    Pages forward from ``start_time`` (``0`` = earliest available) until the API
    returns a short/empty page, advancing the cursor past the last candle's open
    time each iteration.

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
