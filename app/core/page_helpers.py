"""Shared helpers used across `app/pages/`.

These extract patterns previously duplicated verbatim in every dashboard
page (10× copies of `_prepare_indicator_slice`). Keeping the cleaning
logic in one place ensures the dashboard pages stay in sync if the
World Bank schema or our normalization rules ever change.
"""

from __future__ import annotations

import polars as pl

from core.postgres_client import get_world_bank_indicator

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"year", "economy", "value"})


def prepare_indicator_slice(
    df: pl.DataFrame,
    value_col: str = "value",
) -> pl.DataFrame:
    """Normalize a raw World Bank indicator frame into the canonical shape.

    - Casts ``year`` to ``Int64`` and uppercases ``economy``.
    - Aliases ``value`` to ``value_col`` so callers can join multiple
      indicators side-by-side without column collisions.
    - Drops rows with null key/value, deduplicates per (year, economy)
      by mean — World Bank source dumps occasionally include duplicate
      rows from overlapping country aggregates.
    - Returns an empty frame if input is empty or missing required cols
      (callers must check ``.is_empty()``).
    """
    if df.is_empty() or not _REQUIRED_COLUMNS.issubset(set(df.columns)):
        return pl.DataFrame()

    return (
        df.select(
            [
                pl.col("year").cast(pl.Int64, strict=False).alias("year"),
                pl.col("economy").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Float64, strict=False).alias(value_col),
            ]
        )
        .filter(
            pl.col("year").is_not_null()
            & pl.col("economy").is_not_null()
            & pl.col(value_col).is_not_null()
        )
        .group_by(["year", "economy"])
        .agg(pl.col(value_col).mean().alias(value_col))
        .sort(["year", "economy"])
    )


def fetch_indicator_slice(
    indicator_id: str,
    country_code: str | list[str] = "ALL",
    value_col: str = "value",
) -> pl.DataFrame:
    """Fetch a World Bank indicator and run it through ``prepare_indicator_slice``."""
    df = get_world_bank_indicator(indicator_id, country_code=country_code)
    return prepare_indicator_slice(df, value_col=value_col)
