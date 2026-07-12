"""End-to-end store tests for the Yahoo + Binance ingestion paths.

Each test drives ``fetch_and_store_*`` against a real Postgres (testcontainer)
with the external API faked — ``httpx.MockTransport`` for Binance and a
monkeypatched ``yfinance.Ticker`` for Yahoo — so the full
fetch → transform → FK-ordered write path is exercised offline.
"""

from __future__ import annotations

import asyncio

import binance_client
import client_binance
import client_yahoo
import httpx
import pandas as pd
import pytest
from schema import (
    BinanceHistoricalPrice,
    BinanceMetadata,
    YahooHistoricalPrice,
    YahooMetadata,
)
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

_DAY_MS = 86_400_000

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
_TICKER_24H = {
    "symbol": "BTCUSDT",
    "lastPrice": "60000",
    "priceChangePercent": "1.5",
    "highPrice": "61000",
    "lowPrice": "59000",
    "quoteVolume": "100",
    "count": 10,
}


def _binance_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/exchangeInfo"):
        if request.url.params.get("symbol") == "BTCUSDT":
            return httpx.Response(200, json=_SYMBOL_INFO)
        return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
    if path.endswith("/ticker/24hr"):
        return httpx.Response(200, json=_TICKER_24H)
    if path.endswith("/klines"):
        page = [
            [i * _DAY_MS, "1", "2", "0.5", "1.5", "10", i * _DAY_MS + _DAY_MS - 1, "15", 3]
            for i in range(3)
        ]
        return httpx.Response(200, json=page)
    return httpx.Response(404, json={})


def test_fetch_and_store_binance(engine: Engine, postgres_uri: str, monkeypatch) -> None:
    monkeypatch.setattr(
        binance_client,
        "build_async_client",
        lambda *a, **k: httpx.AsyncClient(
            base_url="https://api.binance.test", transport=httpx.MockTransport(_binance_handler)
        ),
    )

    rows = asyncio.run(client_binance.fetch_and_store_binance("btcusdt", postgres_uri))
    assert rows == 3

    with Session(engine) as session:
        meta = session.execute(
            select(BinanceMetadata).where(BinanceMetadata.symbol == "BTCUSDT")
        ).scalar_one()
        assert meta.base_asset == "BTC"
        assert meta.rank is None
        assert meta.last_price == 60000.0
        n_prices = len(
            session.execute(
                select(BinanceHistoricalPrice).where(BinanceHistoricalPrice.symbol == "BTCUSDT")
            )
            .scalars()
            .all()
        )
        assert n_prices == 3

        session.execute(
            delete(BinanceHistoricalPrice).where(BinanceHistoricalPrice.symbol == "BTCUSDT")
        )
        session.execute(delete(BinanceMetadata).where(BinanceMetadata.symbol == "BTCUSDT"))
        session.commit()


def test_fetch_and_store_binance_unknown_symbol_raises(postgres_uri: str, monkeypatch) -> None:
    monkeypatch.setattr(
        binance_client,
        "build_async_client",
        lambda *a, **k: httpx.AsyncClient(
            base_url="https://api.binance.test", transport=httpx.MockTransport(_binance_handler)
        ),
    )
    with pytest.raises(ValueError):
        asyncio.run(client_binance.fetch_and_store_binance("FOOBARUSDT", postgres_uri))


class _FakeTicker:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, period: str = "max") -> pd.DataFrame:
        # Build the tz-aware index via DatetimeIndex (not pd.date_range, which
        # segfaults under pandas 3.0.x when called inside a worker thread on
        # this platform); this mirrors how yfinance hands back a tz-aware index.
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"]).tz_localize("UTC")
        df = pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0],
                "High": [2.0, 3.0, 4.0],
                "Low": [0.5, 1.0, 2.0],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            },
            index=idx,
        )
        df.index.name = "Date"
        return df

    @property
    def info(self) -> dict:
        return {
            "shortName": "Apple Inc.",
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currency": "USD",
            "exchange": "NMS",
            "longBusinessSummary": "Apple makes phones.",
        }


def test_fetch_and_store_yahoo(engine: Engine, postgres_uri: str, monkeypatch) -> None:
    monkeypatch.setattr(client_yahoo.yf, "Ticker", lambda t: _FakeTicker(t))

    rows = asyncio.run(client_yahoo.fetch_and_store_yahoo("AAPL", postgres_uri))
    assert rows == 3

    with Session(engine) as session:
        meta = session.execute(
            select(YahooMetadata).where(YahooMetadata.ticker == "AAPL")
        ).scalar_one()
        assert meta.short_name == "Apple Inc."
        assert meta.category == "Companies"
        assert meta.sector == "Technology"
        n_prices = len(
            session.execute(
                select(YahooHistoricalPrice).where(YahooHistoricalPrice.ticker == "AAPL")
            )
            .scalars()
            .all()
        )
        assert n_prices == 3

        session.execute(delete(YahooHistoricalPrice).where(YahooHistoricalPrice.ticker == "AAPL"))
        session.execute(delete(YahooMetadata).where(YahooMetadata.ticker == "AAPL"))
        session.commit()
