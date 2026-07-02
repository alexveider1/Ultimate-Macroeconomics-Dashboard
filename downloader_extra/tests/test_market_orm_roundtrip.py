"""ORM round-trips for the Yahoo + Binance tables through a real Postgres.

Exercises the metadata→historical FK ordering for both market sources.
"""

from __future__ import annotations

from datetime import datetime

from schema import (
    BinanceHistoricalPrice,
    BinanceMetadata,
    YahooHistoricalPrice,
    YahooMetadata,
)
from sqlalchemy import delete, select
from sqlalchemy.orm import Session


def test_yahoo_metadata_then_prices_roundtrip(session: Session) -> None:
    session.add(YahooMetadata(ticker="ZZTEST", asset_name="Z Test", category="Companies"))
    session.flush()
    session.add_all(
        [
            YahooHistoricalPrice(
                date=datetime(2021, 1, 1),
                close=10.0,
                volume=100,
                ticker="ZZTEST",
                category="Companies",
            ),
            YahooHistoricalPrice(
                date=datetime(2021, 1, 2),
                close=11.0,
                volume=200,
                ticker="ZZTEST",
                category="Companies",
            ),
        ]
    )
    session.commit()

    rows = (
        session.execute(
            select(YahooHistoricalPrice)
            .where(YahooHistoricalPrice.ticker == "ZZTEST")
            .order_by(YahooHistoricalPrice.date)
        )
        .scalars()
        .all()
    )
    assert [(r.close, r.volume) for r in rows] == [(10.0, 100), (11.0, 200)]

    session.execute(delete(YahooHistoricalPrice).where(YahooHistoricalPrice.ticker == "ZZTEST"))
    session.execute(delete(YahooMetadata).where(YahooMetadata.ticker == "ZZTEST"))
    session.commit()


def test_binance_metadata_then_prices_roundtrip(session: Session) -> None:
    session.add(
        BinanceMetadata(
            symbol="ZZUSDT", base_asset="ZZ", quote_asset="USDT", status="TRADING", rank=None
        )
    )
    session.flush()
    session.add_all(
        [
            BinanceHistoricalPrice(
                date=datetime(2021, 1, 1),
                close=1.5,
                volume=10.0,
                quote_volume=15.0,
                symbol="ZZUSDT",
                base_asset="ZZ",
            ),
            BinanceHistoricalPrice(
                date=datetime(2021, 1, 2),
                close=1.6,
                volume=11.0,
                quote_volume=17.0,
                symbol="ZZUSDT",
                base_asset="ZZ",
            ),
        ]
    )
    session.commit()

    meta = session.execute(
        select(BinanceMetadata).where(BinanceMetadata.symbol == "ZZUSDT")
    ).scalar_one()
    assert meta.rank is None
    assert meta.base_asset == "ZZ"

    prices = (
        session.execute(
            select(BinanceHistoricalPrice)
            .where(BinanceHistoricalPrice.symbol == "ZZUSDT")
            .order_by(BinanceHistoricalPrice.date)
        )
        .scalars()
        .all()
    )
    assert [r.close for r in prices] == [1.5, 1.6]

    session.execute(delete(BinanceHistoricalPrice).where(BinanceHistoricalPrice.symbol == "ZZUSDT"))
    session.execute(delete(BinanceMetadata).where(BinanceMetadata.symbol == "ZZUSDT"))
    session.commit()
