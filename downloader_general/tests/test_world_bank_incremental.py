"""Regression test for the World Bank incremental (append-only) update transform.

Guards the bug where :meth:`WorldBankDownloader._update_indicator` referenced the
raw ``time`` column *after* an earlier ``select`` had already renamed it to
``year`` — a ``ColumnNotFoundError`` that made every scheduled WB update fail. The
transform must instead: rename ``time`` → ``year`` (Int64), attach
``indicator_id``/``db_id``, and keep only years strictly newer than ``last_year``.
"""

import asyncio
from typing import Any

import polars as pl
from src.extractors import world_bank_download
from src.extractors.world_bank_download import WorldBankDownloader


def _make_downloader() -> WorldBankDownloader:
    dl = WorldBankDownloader.__new__(WorldBankDownloader)
    dl.sql_uri = "postgresql://x"  # non-None so _require_sql_uri() passes
    dl.indicators_table_name = "world_bank_indicators"
    dl.download_max_retries = 1
    dl.download_retry_delay_seconds = 0
    return dl


def test_update_indicator_appends_only_newer_years(monkeypatch) -> None:
    records = [
        {"economy": "USA", "time": 2023, "value": 1.0},
        {"economy": "USA", "time": 2024, "value": 2.0},
        {"economy": "USA", "time": 2025, "value": 3.0},
        {"economy": "DEU", "time": 2025, "value": 4.0},
    ]

    async def _fake_call(**_: Any) -> list[dict[str, Any]]:
        return records

    captured: dict[str, pl.DataFrame] = {}

    def _fake_write(df: pl.DataFrame, *_a: Any, **_k: Any) -> None:
        captured["df"] = df

    monkeypatch.setattr(world_bank_download.wb_client, "call_with_retries", _fake_call)
    monkeypatch.setattr(world_bank_download, "write_polars_to_table", _fake_write)

    dl = _make_downloader()
    monkeypatch.setattr(dl, "_table_def", lambda _name: {})

    # last_year=2024 → only the two 2025 rows survive the year filter.
    asyncio.run(
        dl._update_indicator(client=None, indicator_id="NY.GDP.MKTP.CD", db=2, last_year=2024)
    )

    df = captured["df"]
    assert set(df.columns) == {"economy", "year", "value", "indicator_id", "db_id"}
    assert df["year"].dtype == pl.Int64
    assert sorted(df["year"].to_list()) == [2025, 2025]
    assert df["indicator_id"].unique().to_list() == ["NY.GDP.MKTP.CD"]
    assert df["db_id"].unique().to_list() == [2]


def test_update_indicator_no_write_when_nothing_newer(monkeypatch) -> None:
    records = [{"economy": "USA", "time": 2025, "value": 1.0}]

    async def _fake_call(**_: Any) -> list[dict[str, Any]]:
        return records

    wrote = {"called": False}

    def _fake_write(*_a: Any, **_k: Any) -> None:
        wrote["called"] = True

    monkeypatch.setattr(world_bank_download.wb_client, "call_with_retries", _fake_call)
    monkeypatch.setattr(world_bank_download, "write_polars_to_table", _fake_write)

    dl = _make_downloader()
    monkeypatch.setattr(dl, "_table_def", lambda _name: {})

    # Latest stored year already 2025 → filter drops everything → no write.
    asyncio.run(dl._update_indicator(client=None, indicator_id="X", db=2, last_year=2025))
    assert wrote["called"] is False
