"""YAML-driven Postgres schema bootstrap + typed polars writes.

``database_schema.yaml`` declares one logical "group" per source (``world_bank``,
``yahoo_finance``...). On first boot the matching group is bootstrapped via
:func:`bootstrap_schema_group`, which drops and recreates every table with the
declared columns, primary keys, and foreign keys. Per-row writes go through
:func:`write_polars_to_table`, which projects the input DataFrame onto the
declared columns + dtypes before handing it to ``polars.DataFrame.write_database``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl
from sqlalchemy import (
    DDL,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    create_engine,
)
import yaml

logger = logging.getLogger(__name__)


_SQL_TO_SQLALCHEMY = {
    "TEXT": String,
    "INTEGER": Integer,
    "BIGINT": BigInteger,
    "DOUBLE PRECISION": Float,
    "BOOLEAN": Boolean,
    "TIMESTAMP WITHOUT TIME ZONE": DateTime,
}

_SQL_TO_POLARS = {
    "TEXT": pl.Utf8,
    "INTEGER": pl.Int32,
    "BIGINT": pl.Int64,
    "DOUBLE PRECISION": pl.Float64,
    "BOOLEAN": pl.Boolean,
    "TIMESTAMP WITHOUT TIME ZONE": pl.Datetime,
}


def load_database_schema(path: str | Path) -> Dict[str, Any]:
    """Load and parse a YAML schema file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed schema as a Python dict.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_table_definition(schema: Dict[str, Any], group: str, table: str) -> Dict[str, Any]:
    """Look up the column/PK/FK definition for one table inside a schema group.

    Args:
        schema: Top-level schema dict (output of :func:`load_database_schema`).
        group: Logical group name (e.g. ``world_bank``).
        table: Table name within that group.

    Returns:
        Table definition dict.
    """
    return schema["databases"][group][table]


def _polars_dtype_for(sql_type: str):
    """Map a SQL type string from the YAML schema to a Polars dtype."""
    return _SQL_TO_POLARS[sql_type.upper()]


def _sa_type_for(sql_type: str):
    """Map a SQL type string from the YAML schema to a SQLAlchemy column type."""
    return _SQL_TO_SQLALCHEMY[sql_type.upper()]


def cast_dataframe_to_schema(df: pl.DataFrame, table_def: Dict[str, Any]) -> pl.DataFrame:
    """Project ``df`` onto the schema-declared columns and cast dtypes.

    Columns missing from the input are added as nulls; columns not in the
    schema are dropped. Casts are non-strict so unparseable values become
    NULL rather than raising.

    Args:
        df: Source DataFrame.
        table_def: Table definition from the YAML schema (must have a
            ``columns`` dict keyed by column name with a ``type`` field).

    Returns:
        A new DataFrame with exactly the schema columns in declaration order.
    """
    columns = table_def["columns"]
    select_exprs = []
    for col_name, col_info in columns.items():
        target_dtype = _polars_dtype_for(col_info["type"])
        if col_name in df.columns:
            select_exprs.append(pl.col(col_name).cast(target_dtype, strict=False).alias(col_name))
        else:
            select_exprs.append(pl.lit(None).cast(target_dtype).alias(col_name))
    return df.select(select_exprs)


def _topo_sort_tables(tables: Dict[str, Dict[str, Any]]) -> List[str]:
    """Topologically sort tables so FK targets are created before dependents.

    A cycle (which shouldn't happen with a well-designed schema) is reported
    via ``logger.warning`` and the remaining tables are appended in arbitrary
    order so creation still proceeds.

    Args:
        tables: ``{table_name: table_def}`` mapping for one schema group.

    Returns:
        Table names in safe creation order.
    """
    deps: Dict[str, set] = {name: set() for name in tables}
    for name, table_def in tables.items():
        for fk in table_def.get("foreign_keys") or []:
            ref = fk["references_table"]
            if ref != name and ref in tables:
                deps[name].add(ref)

    ordered: List[str] = []
    remaining = {n: set(d) for n, d in deps.items()}
    while remaining:
        ready = sorted(n for n, d in remaining.items() if not d)
        if not ready:
            logger.warning(
                "FK cycle detected while ordering tables; remaining=%s",
                list(remaining),
            )
            ordered.extend(remaining.keys())
            break
        for n in ready:
            ordered.append(n)
            remaining.pop(n)
            for d in remaining.values():
                d.discard(n)
    return ordered


def _build_metadata(schema: Dict[str, Any], group: str) -> Tuple[MetaData, List[str]]:
    """Build a SQLAlchemy ``MetaData`` from the YAML declaration for one group.

    Args:
        schema: Parsed YAML schema.
        group: Group name (e.g. ``world_bank``).

    Returns:
        Tuple of ``(metadata, creation_order)`` where ``creation_order``
        respects FK dependencies.
    """
    metadata = MetaData()
    tables_def = schema["databases"][group]
    creation_order = _topo_sort_tables(tables_def)

    for table_name in creation_order:
        table_def = tables_def[table_name]
        columns = table_def["columns"]
        pk_cols = set(table_def.get("primary_key") or [])

        sa_columns = [
            Column(col_name, _sa_type_for(col_info["type"]), nullable=col_name not in pk_cols)
            for col_name, col_info in columns.items()
        ]

        constraints = []
        if pk_cols:
            constraints.append(PrimaryKeyConstraint(*table_def["primary_key"]))
        for fk in table_def.get("foreign_keys") or []:
            if not fk.get("enforce", True):
                continue
            constraints.append(
                ForeignKeyConstraint(
                    fk["columns"],
                    [f"{fk['references_table']}.{c}" for c in fk["references_columns"]],
                )
            )

        Table(table_name, metadata, *sa_columns, *constraints)

    return metadata, creation_order


def bootstrap_schema_group(sql_uri: str, schema: Dict[str, Any], group: str) -> None:
    """Drop and recreate every table in ``schema.databases[group]``.

    Tables are dropped in reverse FK order with ``CASCADE`` then recreated in
    forward FK order so foreign-key targets exist when their dependents are
    created.

    Args:
        sql_uri: Postgres URI for a superuser (or owner of all the tables).
        schema: Parsed YAML schema.
        group: Group name within ``schema``.
    """
    metadata, creation_order = _build_metadata(schema, group)
    engine = create_engine(sql_uri)
    try:
        with engine.begin() as conn:
            for table_name in reversed(creation_order):
                conn.execute(DDL(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
            metadata.create_all(bind=conn)
    finally:
        engine.dispose()
    logger.info(
        "Bootstrapped '%s' schema; created tables in order: %s",
        group,
        creation_order,
    )


def write_polars_to_table(
    df: pl.DataFrame,
    sql_uri: str,
    table_name: str,
    table_def: Dict[str, Any],
) -> None:
    """Cast ``df`` to the schema's columns/dtypes and append to ``table_name``.

    Assumes the table has already been created via
    :func:`bootstrap_schema_group`. Polars opens its own connection from the
    URI, so the caller doesn't need to manage one.

    Args:
        df: Source DataFrame; no-ops when empty.
        sql_uri: Postgres URI.
        table_name: Destination table.
        table_def: Schema definition for ``table_name``.
    """
    if df.is_empty():
        return
    cast_df = cast_dataframe_to_schema(df, table_def)
    cast_df.write_database(
        table_name,
        connection=sql_uri,
        if_table_exists="append",
        engine="sqlalchemy",
    )
