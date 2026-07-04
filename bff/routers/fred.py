"""FRED US-state read endpoints (states catalogue, indicators, values)."""

from db import get_session
from fastapi import APIRouter, Depends, HTTPException, Query
from models import FredIndicatorOut, RegionValuePoint, StateOut
from schema import State, StateIndicator, StateIndicatorValue
from sqlalchemy import select
from sqlalchemy.orm import Session
from utils import parse_code_filter

router = APIRouter(prefix="/fred", tags=["fred"])


@router.get("/states", response_model=list[StateOut])
def list_states(session: Session = Depends(get_session)) -> list[StateOut]:
    """Return the US-state / DC catalogue."""
    rows = session.execute(select(State).order_by(State.id)).scalars().all()
    return [StateOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/indicators", response_model=list[FredIndicatorOut])
def list_indicators(session: Session = Depends(get_session)) -> list[FredIndicatorOut]:
    """Return every FRED state-indicator description row."""
    rows = (
        session.execute(select(StateIndicator).order_by(StateIndicator.indicator_id))
        .scalars()
        .all()
    )
    return [FredIndicatorOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/indicators/{indicator_id}", response_model=FredIndicatorOut)
def get_indicator(indicator_id: str, session: Session = Depends(get_session)) -> FredIndicatorOut:
    """Return the description row for one FRED indicator."""
    row = session.get(StateIndicator, indicator_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown FRED indicator: {indicator_id}")
    return FredIndicatorOut.model_validate(row, from_attributes=True)


@router.get("/indicators/{indicator_id}/values", response_model=list[RegionValuePoint])
def get_indicator_values(
    indicator_id: str,
    states: str | None = Query(default=None, description="Comma-separated state abbrevs or ALL"),
    session: Session = Depends(get_session),
) -> list[RegionValuePoint]:
    """Return ``(state, year, value)`` observations for one FRED indicator."""
    codes = parse_code_filter(states)
    stmt = (
        select(
            StateIndicatorValue.state,
            StateIndicatorValue.year,
            StateIndicatorValue.value,
        )
        .where(StateIndicatorValue.indicator_id == indicator_id)
        .where(StateIndicatorValue.value.is_not(None))
    )
    if codes:
        stmt = stmt.where(StateIndicatorValue.state.in_(codes))
    stmt = stmt.order_by(StateIndicatorValue.year, StateIndicatorValue.state)
    return [
        RegionValuePoint(region=row.state, year=row.year, value=row.value)
        for row in session.execute(stmt).all()
    ]
