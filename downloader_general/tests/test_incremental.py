"""Tests for the incremental read helpers (``src/utils/incremental.py``).

A file-based SQLite database stands in for Postgres: ``group_max`` / ``read_rows``
each open their own engine from the URI, so the table must be persisted to disk
(an in-memory ``sqlite://`` would be a fresh empty DB per engine).
"""

from pathlib import Path

from sqlalchemy import create_engine, text

from src.utils.incremental import group_max, read_rows


def _make_prices_db(tmp_path: Path, *, with_rows: bool = True) -> str:
    """Create a ``prices`` table (optionally populated) and return its SQLite URI."""
    uri = f"sqlite:///{tmp_path / 'prices.db'}"
    engine = create_engine(uri)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE prices (date INTEGER, ticker TEXT, category TEXT, value REAL)")
        )
        if with_rows:
            conn.execute(
                text(
                    "INSERT INTO prices VALUES "
                    "(1,'AAA','C',1.0),(3,'AAA','C',2.0),(2,'BBB','C',5.0)"
                )
            )
    engine.dispose()
    return uri


def test_group_max_returns_per_key_maxima(tmp_path: Path) -> None:
    uri = _make_prices_db(tmp_path)
    result = group_max(uri, "prices", ["ticker", "category"], "date")
    assert result == {("AAA", "C"): 3, ("BBB", "C"): 2}


def test_group_max_single_group_col(tmp_path: Path) -> None:
    uri = _make_prices_db(tmp_path)
    result = group_max(uri, "prices", ["ticker"], "date")
    assert result == {("AAA",): 3, ("BBB",): 2}


def test_group_max_missing_table_returns_none(tmp_path: Path) -> None:
    # A DB file that exists but has no `prices` table -> None (never ingested).
    uri = f"sqlite:///{tmp_path / 'empty.db'}"
    create_engine(uri).dispose()
    assert group_max(uri, "prices", ["ticker"], "date") is None


def test_group_max_empty_table_returns_empty_dict(tmp_path: Path) -> None:
    uri = _make_prices_db(tmp_path, with_rows=False)
    assert group_max(uri, "prices", ["ticker"], "date") == {}


def test_read_rows_returns_dicts(tmp_path: Path) -> None:
    uri = _make_prices_db(tmp_path)
    rows = read_rows(uri, "prices", ["ticker", "value"])
    assert rows is not None
    assert {r["ticker"] for r in rows} == {"AAA", "BBB"}
    assert all(set(r.keys()) == {"ticker", "value"} for r in rows)


def test_read_rows_missing_table_returns_none(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'empty.db'}"
    create_engine(uri).dispose()
    assert read_rows(uri, "prices", ["ticker"]) is None
