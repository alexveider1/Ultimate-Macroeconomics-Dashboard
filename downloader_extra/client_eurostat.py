"""Eurostat fetch + Postgres upsert for one NUTS-2 dataset at a time.

Called by ``downloader_extra``'s ``POST /ingest`` endpoint when the agent asks
about an EU regional indicator not yet in the database. The request carries a
Eurostat ``dataset`` code (e.g. ``nama_10r_2gdp``) plus optional ``filters`` that
pin the dataset's extra dimensions; this module fetches the whole NUTS-2 region
panel in one keyless call and replaces any prior copy of that dataset in Postgres.

The stored ``indicator_id`` slug is the lower-cased dataset code (on-demand items
have no curated slug, mirroring how the Binance/FRED paths use the pair symbol /
series id); the human-readable dataset label is stored as ``name``. The
``eurostat_regions`` catalogue is assumed to already exist (populated by
``downloader_general``); values are filtered to codes present in it so the enforced
FK holds.
"""

import asyncio
import json
import logging

from eurostat_client import build_async_client, fetch_dataset, parse_jsonstat, synthesize_notes
from schema import EurostatIndicator, EurostatIndicatorValue, Region
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_NUTS_LEVEL = 2
_ON_DEMAND_CATEGORY = "On-demand (Eurostat)"


def _existing_region_ids(session: Session) -> set[str]:
    """Return the set of NUTS-2 codes already present in ``eurostat_regions``."""
    return set(session.execute(select(Region.id)).scalars().all())


def _replace_eurostat_indicator(
    description: dict, value_rows: list[dict], indicator_id: str, sql_uri: str
) -> int:
    """Delete any existing copy of the indicator, filter to known regions, and insert.

    Runs the (blocking) SQLAlchemy work in a single transaction; intended to be
    called via :func:`asyncio.to_thread` so the event loop stays free. Returns the
    number of value rows actually inserted (after the region-FK filter).
    """
    engine = create_engine(sql_uri)
    try:
        with Session(engine) as session, session.begin():
            region_ids = _existing_region_ids(session)
            kept = [row for row in value_rows if row["region"] in region_ids]
            session.execute(
                delete(EurostatIndicatorValue).where(
                    EurostatIndicatorValue.indicator_id == indicator_id
                )
            )
            session.execute(
                delete(EurostatIndicator).where(EurostatIndicator.indicator_id == indicator_id)
            )
            session.add(EurostatIndicator(**description))
            session.add_all([EurostatIndicatorValue(**row) for row in kept])
        return len(kept)
    finally:
        engine.dispose()


async def fetch_and_store_eurostat(
    dataset: str, filters: dict[str, str] | None, sql_uri: str
) -> int:
    """Fetch one Eurostat NUTS-2 dataset and replace any prior copy in Postgres.

    Args:
        dataset: Eurostat dataset code (e.g. ``nama_10r_2gdp``).
        filters: Dimensions pinned to a single category (e.g. ``{"unit": "EUR_HAB"}``).
            Any dimension left unpinned collapses to its first category.
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.

    Returns:
        Number of region-year rows inserted.

    Raises:
        ValueError: When the dataset returns no NUTS-2 panel data.
    """
    filters = filters or {}
    async with build_async_client() as client:
        payload = await fetch_dataset(
            client, dataset, geo_level=f"nuts{_NUTS_LEVEL}", filters=filters
        )

    if not payload:
        raise ValueError(f"Eurostat dataset '{dataset}' returned no data")

    rows, meta = parse_jsonstat(payload, level=_NUTS_LEVEL)
    if not rows:
        raise ValueError(f"No NUTS-2 panel data found for Eurostat dataset: {dataset}")

    indicator_id = dataset.strip().lower()
    years = [int(r["year"]) for r in rows]
    min_year, max_year = min(years), max(years)
    description = {
        "indicator_id": indicator_id,
        "name": meta.get("source_label") or dataset,
        "category": _ON_DEMAND_CATEGORY,
        "dataset": dataset,
        "filters": json.dumps(filters, sort_keys=True),
        "units": meta.get("units"),
        "frequency": meta.get("frequency"),
        "nuts_level": _NUTS_LEVEL,
        "min_year": min_year,
        "max_year": max_year,
        "source_label": meta.get("source_label"),
        "notes": synthesize_notes(meta, dataset, filters, min_year, max_year),
    }
    value_rows = [
        {
            "region": row["region"],
            "year": int(row["year"]),
            "value": float(row["value"]) if row["value"] is not None else None,
            "indicator_id": indicator_id,
        }
        for row in rows
    ]

    return await asyncio.to_thread(
        _replace_eurostat_indicator, description, value_rows, indicator_id, sql_uri
    )
