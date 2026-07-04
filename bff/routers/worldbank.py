"""World Bank read endpoints (indicators, catalogue, countries)."""

from db import get_session
from fastapi import APIRouter, Depends, Query
from models import (
    CountryOut,
    IndicatorPoint,
    WorldBankIndicatorInfo,
    WorldBankIndicatorValues,
)
from schema import Country, DatabaseIndicator, MacroIndicator, MetadataRow
from sqlalchemy import case, distinct, select
from sqlalchemy.orm import Session
from utils import parse_code_filter

router = APIRouter(prefix="/worldbank", tags=["worldbank"])

PREFERRED_DB_ID = 2  # World Development Indicators.


def _resolve_indicator_name(
    session: Session, indicator_id: str, preferred_db_id: int = PREFERRED_DB_ID
) -> str | None:
    """Return the human-readable title, preferring the WDI database row."""
    stmt = (
        select(DatabaseIndicator.description)
        .where(DatabaseIndicator.id == indicator_id)
        .where(DatabaseIndicator.description.is_not(None))
        .where(DatabaseIndicator.description != "")
        .order_by(
            case((DatabaseIndicator.database_id == preferred_db_id, 0), else_=1),
            DatabaseIndicator.database_id,
        )
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


@router.get("/countries", response_model=list[CountryOut])
def list_countries(
    include_aggregates: bool = Query(default=True),
    session: Session = Depends(get_session),
) -> list[CountryOut]:
    """Return every World Bank economy (optionally excluding aggregate regions)."""
    stmt = select(Country).order_by(Country.id)
    if not include_aggregates:
        stmt = stmt.where(Country.aggregate.is_(False))
    rows = session.execute(stmt).scalars().all()
    return [
        CountryOut(
            id=row.id,
            name=row.value,
            region=row.region_value,
            income_level=row.income_level_value,
            aggregate=row.aggregate,
            latitude=row.latitude,
            longitude=row.longitude,
            capital_city=row.capital_city,
        )
        for row in rows
    ]


@router.get("/countries/codes", response_model=list[str])
def list_country_codes(session: Session = Depends(get_session)) -> list[str]:
    """Return every distinct, non-empty ``economy`` code present in indicators."""
    stmt = (
        select(distinct(MacroIndicator.economy))
        .where(MacroIndicator.economy.is_not(None))
        .where(MacroIndicator.economy != "")
        .order_by(MacroIndicator.economy)
    )
    return [code for code in session.execute(stmt).scalars().all() if code]


@router.get("/indicators/{indicator_id}", response_model=WorldBankIndicatorInfo)
def get_indicator_info(
    indicator_id: str,
    session: Session = Depends(get_session),
) -> WorldBankIndicatorInfo:
    """Return the resolved name + descriptive metadata for one WB indicator."""
    meta = session.execute(
        select(MetadataRow)
        .where(MetadataRow.indicator_id == indicator_id)
        .order_by(
            case((MetadataRow.db_id == PREFERRED_DB_ID, 0), else_=1),
            MetadataRow.db_id,
        )
        .limit(1)
    ).scalar_one_or_none()

    name = _resolve_indicator_name(session, indicator_id)
    if name is None and meta is not None:
        name = meta.indicator_name

    return WorldBankIndicatorInfo(
        indicator_id=indicator_id,
        name=name,
        units=meta.units if meta else None,
        source=meta.source if meta else None,
        development_relevance=meta.development_relevance if meta else None,
        limitations_and_exceptions=meta.limitations_and_exceptions if meta else None,
        statistical_concept_and_methodology=(
            meta.statistical_concept_and_methodology if meta else None
        ),
    )


@router.get("/indicators/{indicator_id}/values", response_model=WorldBankIndicatorValues)
def get_indicator_values(
    indicator_id: str,
    countries: str | None = Query(default=None, description="Comma-separated ISO codes or ALL"),
    db_id: int | None = Query(default=None, description="Restrict to one WB database id"),
    session: Session = Depends(get_session),
) -> WorldBankIndicatorValues:
    """Return ``(economy, year, value)`` observations for one WB indicator."""
    codes = parse_code_filter(countries)
    stmt = select(MacroIndicator.economy, MacroIndicator.year, MacroIndicator.value).where(
        MacroIndicator.indicator_id == indicator_id
    )
    if db_id is not None:
        stmt = stmt.where(MacroIndicator.db_id == db_id)
    if codes:
        stmt = stmt.where(MacroIndicator.economy.in_(codes))
    stmt = stmt.order_by(MacroIndicator.year, MacroIndicator.economy)

    points = [
        IndicatorPoint(economy=row.economy, year=row.year, value=row.value)
        for row in session.execute(stmt).all()
    ]
    return WorldBankIndicatorValues(
        indicator_id=indicator_id,
        name=_resolve_indicator_name(session, indicator_id),
        points=points,
    )
