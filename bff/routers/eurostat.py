"""Eurostat EU NUTS-2 read endpoints (regions catalogue, indicators, values)."""

from db import get_session
from fastapi import APIRouter, Depends, HTTPException, Query
from models import EurostatIndicatorOut, RegionOut, RegionValuePoint
from schema import EurostatIndicator, EurostatIndicatorValue, Region
from sqlalchemy import select
from sqlalchemy.orm import Session
from utils import parse_code_filter

router = APIRouter(prefix="/eurostat", tags=["eurostat"])


@router.get("/regions", response_model=list[RegionOut])
def list_regions(session: Session = Depends(get_session)) -> list[RegionOut]:
    """Return the NUTS-2 region catalogue."""
    rows = session.execute(select(Region).order_by(Region.id)).scalars().all()
    return [RegionOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/indicators", response_model=list[EurostatIndicatorOut])
def list_indicators(session: Session = Depends(get_session)) -> list[EurostatIndicatorOut]:
    """Return every Eurostat indicator description row."""
    rows = (
        session.execute(select(EurostatIndicator).order_by(EurostatIndicator.indicator_id))
        .scalars()
        .all()
    )
    return [EurostatIndicatorOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/indicators/{indicator_id}", response_model=EurostatIndicatorOut)
def get_indicator(
    indicator_id: str, session: Session = Depends(get_session)
) -> EurostatIndicatorOut:
    """Return the description row for one Eurostat indicator."""
    row = session.get(EurostatIndicator, indicator_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown Eurostat indicator: {indicator_id}")
    return EurostatIndicatorOut.model_validate(row, from_attributes=True)


@router.get("/indicators/{indicator_id}/values", response_model=list[RegionValuePoint])
def get_indicator_values(
    indicator_id: str,
    regions: str | None = Query(default=None, description="Comma-separated NUTS-2 codes or ALL"),
    session: Session = Depends(get_session),
) -> list[RegionValuePoint]:
    """Return ``(region, year, value)`` observations for one Eurostat indicator."""
    codes = parse_code_filter(regions)
    stmt = (
        select(
            EurostatIndicatorValue.region,
            EurostatIndicatorValue.year,
            EurostatIndicatorValue.value,
        )
        .where(EurostatIndicatorValue.indicator_id == indicator_id)
        .where(EurostatIndicatorValue.value.is_not(None))
    )
    if codes:
        stmt = stmt.where(EurostatIndicatorValue.region.in_(codes))
    stmt = stmt.order_by(EurostatIndicatorValue.year, EurostatIndicatorValue.region)
    return [
        RegionValuePoint(region=row.region, year=row.year, value=row.value)
        for row in session.execute(stmt).all()
    ]
