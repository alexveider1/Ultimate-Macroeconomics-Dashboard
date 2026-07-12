"""Tests for the filesystem-safe timestamp/name helpers."""

from datetime import datetime, timezone

from naming import postgres_dump_name, utc_timestamp


def test_utc_timestamp_is_filesystem_safe() -> None:
    moment = datetime(2026, 7, 4, 12, 30, 0, tzinfo=timezone.utc)
    ts = utc_timestamp(moment)
    assert ts == "2026-07-04T12-30-00Z"
    assert ":" not in ts  # no colons -> safe on object stores / Windows


def test_utc_timestamp_normalizes_to_utc() -> None:
    # A non-UTC aware datetime is converted to UTC before formatting.
    tz = timezone.utc
    moment = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
    assert utc_timestamp(moment).endswith("Z")


def test_postgres_dump_name() -> None:
    assert postgres_dump_name("macro", "2026-07-04T12-30-00Z") == "macro_2026-07-04T12-30-00Z.dump"
