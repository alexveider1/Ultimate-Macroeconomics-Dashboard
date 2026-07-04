"""Filesystem-safe UTC timestamp helpers for backup artifact names."""

from datetime import datetime, timezone


def utc_timestamp(now: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp like ``2026-07-04T12-30-00Z``.

    Colons (illegal in filenames on some object stores / filesystems) are
    avoided by using ``-`` throughout; the trailing ``Z`` marks UTC.
    """
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def postgres_dump_name(database: str, timestamp: str) -> str:
    """Name a ``pg_dump -Fc`` artifact, e.g. ``macro_2026-07-04T12-30-00Z.dump``."""
    return f"{database}_{timestamp}.dump"
