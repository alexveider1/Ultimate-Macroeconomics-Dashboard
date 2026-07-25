"""Integration tests for :func:`write_polars_to_table`.

Covers the dedup contract that guards against upstream APIs returning the same
primary key twice (e.g. the World Bank ``/indicator?source=2`` catalogue lists
the 12 WGI series twice) — those must not trip the table's PK constraint.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sqlalchemy import create_engine, text
from src.utils.schema import bootstrap_schema_group, write_polars_to_table

_SCHEMA: dict[str, Any] = {
    "databases": {
        "test_group": {
            "database_indicators": {
                "columns": {
                    "id": {"type": "TEXT"},
                    "description": {"type": "TEXT"},
                    "database_id": {"type": "INTEGER"},
                },
                "primary_key": ["id", "database_id"],
            }
        }
    }
}

_TABLE_DEF = _SCHEMA["databases"]["test_group"]["database_indicators"]


def _row_count(sql_uri: str, table: str) -> int:
    engine = create_engine(sql_uri)
    try:
        with engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())
    finally:
        engine.dispose()


def test_write_dedups_duplicate_primary_keys(postgres_uri: str) -> None:
    """Duplicate (id, database_id) rows are dropped instead of raising."""
    bootstrap_schema_group(postgres_uri, _SCHEMA, "test_group")

    df = pl.DataFrame(
        {
            "id": ["AG.CON.FERT.ZS", "GOV_WGI_CC_EST", "GOV_WGI_CC_EST"],
            "description": ["Fertilizer", "Control of Corruption", "Control of Corruption"],
            "database_id": [2, 2, 2],
        }
    )

    write_polars_to_table(df, postgres_uri, "database_indicators", _TABLE_DEF)

    assert _row_count(postgres_uri, "database_indicators") == 2


def test_write_keeps_same_id_across_different_database_ids(postgres_uri: str) -> None:
    """The composite PK means the same id under two databases is not a dup."""
    bootstrap_schema_group(postgres_uri, _SCHEMA, "test_group")

    df = pl.DataFrame(
        {
            "id": ["GOV_WGI_CC_EST", "GOV_WGI_CC_EST"],
            "description": ["Control of Corruption", "Control of Corruption"],
            "database_id": [1, 2],
        }
    )

    write_polars_to_table(df, postgres_uri, "database_indicators", _TABLE_DEF)

    assert _row_count(postgres_uri, "database_indicators") == 2
