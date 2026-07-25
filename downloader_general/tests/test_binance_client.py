"""Unit tests for the async Binance market-data client and selection logic.

All HTTP is faked with ``httpx.MockTransport`` so the tests are offline and
deterministic; async coroutines are driven with ``asyncio.run`` to avoid a
pytest-asyncio dependency (mirrors ``test_wb_client.py``).
"""

import asyncio
from datetime import datetime

import httpx
from src.extractors.binance_download import BinanceDownloader, _is_leveraged_token
from src.utils import binance_client


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.binance.test", transport=httpx.MockTransport(handler)
    )


def _run(handler, coro_factory):
    async def _main():
        async with _client(handler) as client:
            return await coro_factory(client)

    return asyncio.run(_main())


_EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "isSpotTradingAllowed": True,
        },
        {
            "symbol": "ETHUSDT",
            "status": "TRADING",
            "baseAsset": "ETH",
            "quoteAsset": "USDT",
            "isSpotTradingAllowed": True,
        },
        # Excluded: stablecoin base asset
        {
            "symbol": "USDCUSDT",
            "status": "TRADING",
            "baseAsset": "USDC",
            "quoteAsset": "USDT",
            "isSpotTradingAllowed": True,
        },
        # Excluded: leveraged token
        {
            "symbol": "BTCUPUSDT",
            "status": "TRADING",
            "baseAsset": "BTCUP",
            "quoteAsset": "USDT",
            "isSpotTradingAllowed": True,
        },
        # Excluded: non-USDT quote
        {
            "symbol": "ETHBTC",
            "status": "TRADING",
            "baseAsset": "ETH",
            "quoteAsset": "BTC",
            "isSpotTradingAllowed": True,
        },
        # Excluded: not trading
        {
            "symbol": "FOOUSDT",
            "status": "BREAK",
            "baseAsset": "FOO",
            "quoteAsset": "USDT",
            "isSpotTradingAllowed": True,
        },
    ]
}

_TICKERS = [
    {
        "symbol": "BTCUSDT",
        "lastPrice": "60000",
        "priceChangePercent": "1.5",
        "highPrice": "61000",
        "lowPrice": "59000",
        "quoteVolume": "100",
        "count": 10,
    },
    {
        "symbol": "ETHUSDT",
        "lastPrice": "3000",
        "priceChangePercent": "-2.0",
        "highPrice": "3100",
        "lowPrice": "2900",
        "quoteVolume": "500",
        "count": 20,
    },
    {"symbol": "USDCUSDT", "quoteVolume": "999999", "count": 1},
    {"symbol": "BTCUPUSDT", "quoteVolume": "888888", "count": 1},
]


def test_is_leveraged_token():
    assert _is_leveraged_token("BTCUP")
    assert _is_leveraged_token("ETHDOWN")
    assert _is_leveraged_token("ETHBULL")
    assert _is_leveraged_token("ADABEAR")
    assert not _is_leveraged_token("BTC")
    assert not _is_leveraged_token("ETH")


def test_fetch_exchange_info_returns_symbols():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_EXCHANGE_INFO)

    symbols = _run(handler, lambda c: binance_client.fetch_exchange_info(c))
    assert {s["symbol"] for s in symbols} == {
        "BTCUSDT",
        "ETHUSDT",
        "USDCUSDT",
        "BTCUPUSDT",
        "ETHBTC",
        "FOOUSDT",
    }


def test_fetch_klines_paginates_and_shapes():
    # First page is full (limit rows) -> a second page is fetched; second is short -> stop.
    limit = binance_client._KLINES_PAGE_LIMIT
    day_ms = 86_400_000

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("startTime", "0"))
        if start == 0:
            page = [
                [
                    i * day_ms,
                    "1",
                    "2",
                    "0.5",
                    "1.5",
                    "10",
                    i * day_ms + day_ms - 1,
                    "15",
                    3,
                    "0",
                    "0",
                    "0",
                ]
                for i in range(limit)
            ]
            return httpx.Response(200, json=page)
        # Second page: a single trailing candle, then exhaustion.
        return httpx.Response(
            200,
            json=[
                [
                    limit * day_ms,
                    "1",
                    "2",
                    "0.5",
                    "1.5",
                    "10",
                    limit * day_ms + day_ms - 1,
                    "15",
                    3,
                    "0",
                    "0",
                    "0",
                ]
            ],
        )

    rows = _run(handler, lambda c: binance_client.fetch_klines(c, "BTCUSDT", interval="1d"))
    assert len(rows) == limit + 1
    first = rows[0]
    assert first["date"] == datetime(1970, 1, 1)
    assert first["open"] == 1.0
    assert first["high"] == 2.0
    assert first["low"] == 0.5
    assert first["close"] == 1.5
    assert first["volume"] == 10.0
    assert first["quote_volume"] == 15.0


def _downloader() -> BinanceDownloader:
    config = {
        "base_url": "https://api.binance.test",
        "quote_asset": "USDT",
        "top_n": 30,
        "kline_interval": "1d",
        "max_parallel_symbols": 2,
        "exclude_base_assets": ["USDC", "FDUSD", "TUSD"],
    }
    dl = BinanceDownloader.__new__(BinanceDownloader)
    dl.quote_asset = "USDT"
    dl.top_n = 30
    dl.kline_interval = "1d"
    dl.exclude_base_assets = {a.upper() for a in config["exclude_base_assets"]}
    dl.download_max_retries = 0
    dl.download_retry_delay_seconds = 0
    return dl


def test_valid_base_assets_filters():
    dl = _downloader()
    valid = dl._valid_base_assets(_EXCHANGE_INFO["symbols"])
    assert valid == {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}


def test_select_top_symbols_ranks_by_quote_volume():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/exchangeInfo"):
            return httpx.Response(200, json=_EXCHANGE_INFO)
        if request.url.path.endswith("/ticker/24hr"):
            return httpx.Response(200, json=_TICKERS)
        return httpx.Response(404, json={})

    dl = _downloader()
    rows = _run(handler, lambda c: dl.select_top_symbols(c))
    # Only the two valid pairs survive filtering; ETH outranks BTC (higher volume).
    assert [r["symbol"] for r in rows] == ["ETHUSDT", "BTCUSDT"]
    assert rows[0]["rank"] == 1
    assert rows[0]["base_asset"] == "ETH"
    assert rows[0]["quote_volume_24h"] == 500.0
    assert rows[1]["rank"] == 2
    assert "ETH/USDT" in rows[0]["description"]
