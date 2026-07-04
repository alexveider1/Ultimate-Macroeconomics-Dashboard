"""Yahoo Finance read endpoints (metadata + OHLCV history)."""

from db import get_session
from fastapi import APIRouter, Depends, HTTPException
from models import OhlcvPoint, YahooMetadataDetail, YahooMetadataOut
from schema import YahooHistoricalPrice, YahooMetadata
from sqlalchemy import select
from sqlalchemy.orm import Session

router = APIRouter(prefix="/yahoo", tags=["yahoo"])


@router.get("/metadata", response_model=list[YahooMetadataOut])
def list_metadata(session: Session = Depends(get_session)) -> list[YahooMetadataOut]:
    """Return one master row per Yahoo ticker (without the business summary)."""
    rows = session.execute(select(YahooMetadata).order_by(YahooMetadata.ticker)).scalars().all()
    return [YahooMetadataOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/metadata/{ticker}", response_model=YahooMetadataDetail)
def get_metadata(ticker: str, session: Session = Depends(get_session)) -> YahooMetadataDetail:
    """Return the full master row (incl. business summary) for one ticker."""
    row = session.get(YahooMetadata, ticker)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    return YahooMetadataDetail.model_validate(row, from_attributes=True)


@router.get("/prices", response_model=list[OhlcvPoint])
def list_all_prices(session: Session = Depends(get_session)) -> list[OhlcvPoint]:
    """Return the complete OHLCV history for every Yahoo ticker."""
    stmt = (
        select(YahooHistoricalPrice)
        .where(YahooHistoricalPrice.date.is_not(None))
        .where(YahooHistoricalPrice.close.is_not(None))
        .where(YahooHistoricalPrice.ticker.is_not(None))
        .order_by(YahooHistoricalPrice.ticker, YahooHistoricalPrice.date)
    )
    return [
        OhlcvPoint.model_validate(row, from_attributes=True)
        for row in session.execute(stmt).scalars().all()
    ]


@router.get("/prices/{ticker}", response_model=list[OhlcvPoint])
def get_prices(ticker: str, session: Session = Depends(get_session)) -> list[OhlcvPoint]:
    """Return the OHLCV history for one Yahoo ticker."""
    stmt = (
        select(YahooHistoricalPrice)
        .where(YahooHistoricalPrice.ticker == ticker)
        .order_by(YahooHistoricalPrice.date)
    )
    return [
        OhlcvPoint.model_validate(row, from_attributes=True)
        for row in session.execute(stmt).scalars().all()
    ]
