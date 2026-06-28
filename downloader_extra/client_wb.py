"""World Bank fetch + Postgres upsert for one indicator at a time.

Called by ``downloader_extra``'s ``POST /ingest`` endpoint when the agent
asks for an indicator not yet in the database. Fetches the observations over
an async ``httpx`` client (:mod:`wb_client`, which replaced the ``wbgapi``
dependency) and replaces any prior copy of the indicator in Postgres.
"""

import asyncio
import logging

import polars as pl
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

import wb_client
from schema import MacroIndicator

logger = logging.getLogger(__name__)


def _replace_indicator_rows(
    rows: list[dict], indicator_id: str, wb_db_id: int, sql_uri: str
) -> None:
    """Delete any existing copy of the indicator and insert the fresh rows.

    Runs the (blocking) SQLAlchemy work in a single transaction; intended to be
    called via :func:`asyncio.to_thread` so the event loop stays free.

    Args:
        rows: Row dicts ready for ``MacroIndicator(**row)``.
        indicator_id: World Bank indicator id.
        wb_db_id: World Bank database id.
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.
    """
    engine = create_engine(sql_uri)
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(MacroIndicator).where(
                    MacroIndicator.indicator_id == indicator_id,
                    MacroIndicator.db_id == wb_db_id,
                )
            )
            session.add_all([MacroIndicator(**row) for row in rows])
    finally:
        engine.dispose()


async def fetch_and_store_indicator(indicator_id: str, wb_db_id: int, sql_uri: str) -> int:
    """Fetch one WB indicator and replace any prior copy in Postgres.

    The function is idempotent: it first deletes all existing rows for the
    ``(indicator_id, db_id)`` pair, then inserts the fresh fetch in a single
    transaction.

    Args:
        indicator_id: World Bank indicator id.
        wb_db_id: World Bank database id.
        sql_uri: SQLAlchemy URI for the Postgres superuser connection.

    Returns:
        Number of rows that were inserted.

    Raises:
        ValueError: When the WB API returns no usable data for the indicator.
    """
    async with wb_client.build_async_client() as client:
        rows = await wb_client.fetch_indicator_data(client, indicator_id, wb_db_id)

    if not rows:
        raise ValueError(f"No data found for indicator id: {indicator_id}")

    df_transformed = (
        pl.DataFrame(rows)
        .with_columns(
            [
                pl.lit(indicator_id).alias("indicator_id"),
                pl.lit(wb_db_id).alias("db_id"),
            ]
        )
        .drop_nulls(subset=["economy", "year"])
        .with_columns(
            [
                pl.col("year").cast(pl.Int32, strict=False),
                pl.col("value").cast(pl.Float64, strict=False),
            ]
        )
    )

    if df_transformed.is_empty():
        raise ValueError(
            f"No non-null rows found for indicator id: {indicator_id} in db: {wb_db_id}"
        )

    df_transformed = df_transformed.unique(
        subset=["economy", "year", "indicator_id", "db_id"],
        keep="last",
        maintain_order=True,
    )

    rows_to_insert = df_transformed.to_dicts()
    await asyncio.to_thread(
        _replace_indicator_rows, rows_to_insert, indicator_id, wb_db_id, sql_uri
    )
    return len(rows_to_insert)
