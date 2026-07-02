"""Unit tests for the single-symbol Binance httpx client.

HTTP is faked with ``httpx.MockTransport`` (offline, deterministic); coroutines
are driven with ``asyncio.run`` so no pytest-asyncio plugin is required
(mirrors ``test_wb_client.py``).
"""

import asyncio
from datetime import datetime

import httpx

import binance_client


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.binance.test", transport=httpx.MockTransport(handler)
    )


def _run(handler, coro_factory):
    async def _main():
        async with _client(handler) as client:
            return await coro_factory(client)

    return asyncio.run(_main())


_SYMBOL_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "isSpotTradingAllowed": True,
        }
    ]
}


def test_fetch_symbol_info_valid():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("symbol") == "BTCUSDT"
        return httpx.Response(200, json=_SYMBOL_INFO)

    info = _run(handler, lambda c: binance_client.fetch_symbol_info(c, "BTCUSDT"))
    assert info is not None
    assert info["baseAsset"] == "BTC"
    assert info["quoteAsset"] == "USDT"


def test_fetch_symbol_info_unknown_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        # Binance answers HTTP 400 for an unknown symbol.
        return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})

    info = _run(handler, lambda c: binance_client.fetch_symbol_info(c, "FOOBARUSDT"))
    assert info is None


def test_fetch_24h_ticker_for_symbol():
    ticker = {
        "symbol": "BTCUSDT",
        "lastPrice": "60000",
        "priceChangePercent": "1.5",
        "highPrice": "61000",
        "lowPrice": "59000",
        "quoteVolume": "100",
        "count": 10,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ticker)

    out = _run(handler, lambda c: binance_client.fetch_24h_ticker_for_symbol(c, "BTCUSDT"))
    assert out["lastPrice"] == "60000"
    assert binance_client._to_float(out["lastPrice"]) == 60000.0


def test_fetch_24h_ticker_error_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"msg": "bad"})

    out = _run(handler, lambda c: binance_client.fetch_24h_ticker_for_symbol(c, "FOOUSDT"))
    assert out == {}


def test_fetch_klines_paginates_and_shapes():
    limit = binance_client._KLINES_PAGE_LIMIT
    day_ms = 86_400_000

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("startTime", "0"))
        if start == 0:
            page = [
                [i * day_ms, "1", "2", "0.5", "1.5", "10", i * day_ms + day_ms - 1, "15", 3]
                for i in range(limit)
            ]
            return httpx.Response(200, json=page)
        return httpx.Response(
            200,
            json=[
                [limit * day_ms, "1", "2", "0.5", "1.5", "10", limit * day_ms + day_ms - 1, "15", 3]
            ],
        )

    rows = _run(handler, lambda c: binance_client.fetch_klines(c, "BTCUSDT", interval="1d"))
    assert len(rows) == limit + 1
    first = rows[0]
    assert first["date"] == datetime(1970, 1, 1)
    assert first["open"] == 1.0
    assert first["close"] == 1.5
    assert first["quote_volume"] == 15.0
