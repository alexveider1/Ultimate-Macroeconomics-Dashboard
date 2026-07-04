"""End-to-end tests for the Yahoo Finance + Binance crypto read routers."""

from datetime import datetime

from fastapi.testclient import TestClient
from schema import (
    BinanceHistoricalPrice,
    BinanceMetadata,
    YahooHistoricalPrice,
    YahooMetadata,
)
from sqlalchemy.orm import Session


def _seed_yahoo(session: Session) -> None:
    # Parent (metadata) is flushed before the child prices so the FK is satisfied
    # — SQLAlchemy doesn't order FK-only tables within a single flush.
    session.add(
        YahooMetadata(
            ticker="AAPL",
            asset_name="Apple Inc.",
            category="Companies",
            sector="Technology",
            business_summary="Designs phones.",
        )
    )
    session.flush()
    session.add_all(
        [
            YahooHistoricalPrice(
                date=datetime(2026, 1, 2), close=190.0, ticker="AAPL", volume=1000
            ),
            YahooHistoricalPrice(
                date=datetime(2026, 1, 3), close=192.0, ticker="AAPL", volume=1100
            ),
        ]
    )
    session.commit()


def _seed_crypto(session: Session) -> None:
    session.add_all(
        [
            BinanceMetadata(symbol="BTCUSDT", base_asset="BTC", quote_asset="USDT", rank=1),
            BinanceMetadata(symbol="ETHUSDT", base_asset="ETH", quote_asset="USDT", rank=2),
        ]
    )
    session.flush()
    session.add(BinanceHistoricalPrice(date=datetime(2026, 1, 2), close=95000.0, symbol="BTCUSDT"))
    session.commit()


def test_yahoo_metadata_list_hides_summary(client: TestClient, session: Session) -> None:
    _seed_yahoo(session)
    body = client.get("/yahoo/metadata").json()
    assert len(body) == 1
    assert body[0]["ticker"] == "AAPL"
    assert "business_summary" not in body[0]


def test_yahoo_metadata_detail_includes_summary(client: TestClient, session: Session) -> None:
    _seed_yahoo(session)
    body = client.get("/yahoo/metadata/AAPL").json()
    assert body["business_summary"] == "Designs phones."


def test_yahoo_metadata_detail_404(client: TestClient, session: Session) -> None:
    _seed_yahoo(session)
    assert client.get("/yahoo/metadata/NOPE").status_code == 404


def test_yahoo_prices_for_ticker(client: TestClient, session: Session) -> None:
    _seed_yahoo(session)
    body = client.get("/yahoo/prices/AAPL").json()
    assert [row["close"] for row in body] == [190.0, 192.0]


def test_crypto_metadata_ranked(client: TestClient, session: Session) -> None:
    _seed_crypto(session)
    body = client.get("/crypto/metadata").json()
    assert [row["symbol"] for row in body] == ["BTCUSDT", "ETHUSDT"]


def test_crypto_prices_for_symbol(client: TestClient, session: Session) -> None:
    _seed_crypto(session)
    body = client.get("/crypto/prices/btcusdt").json()  # lower-case is upper-cased.
    assert len(body) == 1
    assert body[0]["symbol"] == "BTCUSDT"
