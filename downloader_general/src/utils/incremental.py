"""Read helpers for incremental (append-only) scheduled updates.

The post-ingest scheduler refreshes each source by appending only rows newer
than what Postgres already holds. These helpers read the current per-item maxima
and catalogue rows from the value tables with lightweight SQLAlchemy Core selects
(no ORM models, no raw ``text()`` — see the repo DB-access rule), so an extractor
can decide exactly what to fetch.

**Safety contract:** a *missing* table (the source was never initially ingested)
returns ``None`` so the caller can fall back to a full ``run()``; any other error
(a transient connection blip) is allowed to propagate so the scheduler's
per-source ``try/except`` retries next interval instead of a caller mistaking it
for "empty" and re-running a destructive full ingest.
"""

from typing import Any

from sqlalchemy import column, create_engine, func, inspect, select, table


def group_max(
    sql_uri: str,
    table_name: str,
    group_cols: list[str],
    period_col: str,
) -> dict[tuple[Any, ...], Any] | None:
    """Return ``{group_key_tuple: max(period_col)}`` grouped by ``group_cols``.

    Args:
        sql_uri: Postgres connection URI.
        table_name: Value table to scan (e.g. ``yahoo_historical_prices``).
        group_cols: Columns to group by (the per-item key). Each result key is a
            tuple with one element per group column, in the given order.
        period_col: The column whose per-group maximum is returned (e.g. ``date``
            or ``year``).

    Returns:
        A mapping from each group key to its maximum period value; ``{}`` when the
        table exists but is empty; ``None`` when the table does not exist yet.

    Raises:
        Exception: Any error other than a missing table (e.g. a connection
            failure) propagates unchanged.
    """
    tbl = table(table_name, *[column(c) for c in group_cols], column(period_col))
    stmt = select(*[tbl.c[c] for c in group_cols], func.max(tbl.c[period_col])).group_by(
        *[tbl.c[c] for c in group_cols]
    )
    engine = create_engine(sql_uri)
    try:
        if not inspect(engine).has_table(table_name):
            return None  # source never ingested — caller falls back to a full run
        with engine.connect() as conn:
            rows = conn.execute(stmt).all()
    finally:
        engine.dispose()

    return {tuple(row[:-1]): row[-1] for row in rows}


def read_rows(
    sql_uri: str,
    table_name: str,
    cols: list[str],
) -> list[dict[str, Any]] | None:
    """Return ``cols`` for every row of ``table_name`` as a list of dicts.

    Used to read a catalogue table (e.g. ``state_indicators`` /
    ``eurostat_indicators``) so an incremental update can recover each concept's
    source id / dataset / filters before re-fetching.

    Args:
        sql_uri: Postgres connection URI.
        table_name: Catalogue table to read.
        cols: Columns to select, in order.

    Returns:
        One dict per row keyed by column name; ``None`` when the table does not
        exist yet. Transient errors propagate (see module docstring).
    """
    tbl = table(table_name, *[column(c) for c in cols])
    stmt = select(*[tbl.c[c] for c in cols])
    engine = create_engine(sql_uri)
    try:
        if not inspect(engine).has_table(table_name):
            return None
        with engine.connect() as conn:
            rows = conn.execute(stmt).all()
    finally:
        engine.dispose()

    return [dict(zip(cols, row)) for row in rows]
