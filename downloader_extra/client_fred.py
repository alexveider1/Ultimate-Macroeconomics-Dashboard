"""FRED fetch + Postgres upsert for one state indicator at a time.

Called by ``downloader_extra``'s ``POST /ingest`` endpoint when the agent asks
about a US-state indicator not yet in the database. The request carries a
representative single-state series id (e.g. ``CAUR``); this module resolves it to
a GeoFRED series group, fetches the whole annual 50-state + DC panel in one call,
and replaces any prior copy of that indicator in Postgres.

The stored ``indicator_id`` slug is the upper-cased series id (on-demand items
have no curated slug, mirroring how the Binance path uses the pair symbol); the
human-readable group title is stored as ``name``. The ``states`` catalogue is
assumed to already exist (populated by ``downloader_general``); on-demand values
reference it via the enforced FK.
"""

import asyncio
import logging

from fred_client import (
    build_async_client,
    fetch_regional_panel,
    fetch_series_group,
    parse_regional_panel,
    synthesize_notes,
)
from schema import StateIndicator, StateIndicatorValue
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _replace_fred_indicator(
    description: dict, value_rows: list[dict], indicator_id: str, sql_uri: str
) -> None:
    """Delete any existing copy of the indicator and insert the fresh rows.

    Runs the (blocking) SQLAlchemy work in a single transaction; intended to be
    called via :func:`asyncio.to_thread` so the event loop stays free.
    """
    engine = create_engine(sql_uri)
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(StateIndicatorValue).where(StateIndicatorValue.indicator_id == indicator_id)
            )
            session.execute(
                delete(StateIndicator).where(StateIndicator.indicator_id == indicator_id)
            )
            session.add(StateIndicator(**description))
            session.add_all([StateIndicatorValue(**row) for row in value_rows])
    finally:
        engine.dispose()


async def fetch_and_store_fred(series_id: str, sql_uri: str, api_key: str) -> int:
    """Fetch one FRED state indicator and replace any prior copy in Postgres.

    Args:
        series_id: A representative single-state FRED series (e.g. ``CAUR``).
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.
        api_key: FRED API key.

    Returns:
        Number of state-year rows inserted.

    Raises:
        ValueError: When the series has no state-level group or returns no data.
    """
    async with build_async_client(api_key) as client:
        group = await fetch_series_group(client, series_id)
        if group is None or group.get("region_type") != "state":
            raise ValueError(f"'{series_id}' is not a state-level FRED series")
        panel = await fetch_regional_panel(
            client,
            series_group=group["series_group"],
            region_type=group["region_type"],
            start_date=group["min_date"],
            end_date=group["max_date"],
            units=group["units"],
            season=group.get("season", "NSA"),
        )

    rows, _ = parse_regional_panel(panel)
    if not rows:
        raise ValueError(f"No state panel data found for FRED series: {series_id}")

    indicator_id = series_id.strip().upper()
    description = {
        "indicator_id": indicator_id,
        "name": group.get("title") or indicator_id,
        "category": "On-demand (FRED)",
        "series_group": str(group.get("series_group") or ""),
        "example_series_id": series_id,
        "units": group.get("units"),
        "frequency": group.get("frequency"),
        "seasonal_adjustment": group.get("season"),
        "region_type": group.get("region_type"),
        "min_date": group.get("min_date"),
        "max_date": group.get("max_date"),
        "notes": synthesize_notes(group),
    }
    value_rows = [
        {
            "state": row["state"],
            "year": int(row["year"]),
            "value": float(row["value"]) if row["value"] is not None else None,
            "indicator_id": indicator_id,
        }
        for row in rows
    ]

    await asyncio.to_thread(_replace_fred_indicator, description, value_rows, indicator_id, sql_uri)
    return len(value_rows)
