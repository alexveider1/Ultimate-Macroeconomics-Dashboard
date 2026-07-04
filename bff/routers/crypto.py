"""Binance crypto read endpoints (metadata + daily candle history)."""

from db import get_session
from fastapi import APIRouter, Depends
from models import CryptoCandle, CryptoMetadataOut
from schema import BinanceHistoricalPrice, BinanceMetadata
from sqlalchemy import select
from sqlalchemy.orm import Session

router = APIRouter(prefix="/crypto", tags=["crypto"])


@router.get("/metadata", response_model=list[CryptoMetadataOut])
def list_metadata(session: Session = Depends(get_session)) -> list[CryptoMetadataOut]:
    """Return one master row per Binance coin, ranked by 24h volume."""
    rows = (
        session.execute(
            select(BinanceMetadata).order_by(BinanceMetadata.rank.is_(None), BinanceMetadata.rank)
        )
        .scalars()
        .all()
    )
    return [CryptoMetadataOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/prices", response_model=list[CryptoCandle])
def list_all_prices(session: Session = Depends(get_session)) -> list[CryptoCandle]:
    """Return the complete daily candle history for every Binance coin."""
    stmt = (
        select(BinanceHistoricalPrice)
        .where(BinanceHistoricalPrice.date.is_not(None))
        .where(BinanceHistoricalPrice.close.is_not(None))
        .where(BinanceHistoricalPrice.symbol.is_not(None))
        .order_by(BinanceHistoricalPrice.symbol, BinanceHistoricalPrice.date)
    )
    return [
        CryptoCandle.model_validate(row, from_attributes=True)
        for row in session.execute(stmt).scalars().all()
    ]


@router.get("/prices/{symbol}", response_model=list[CryptoCandle])
def get_prices(symbol: str, session: Session = Depends(get_session)) -> list[CryptoCandle]:
    """Return the daily candle history for one Binance pair."""
    stmt = (
        select(BinanceHistoricalPrice)
        .where(BinanceHistoricalPrice.symbol == symbol.upper())
        .order_by(BinanceHistoricalPrice.date)
    )
    return [
        CryptoCandle.model_validate(row, from_attributes=True)
        for row in session.execute(stmt).scalars().all()
    ]
