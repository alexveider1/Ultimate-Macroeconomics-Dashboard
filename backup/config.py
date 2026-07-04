"""Typed view over the ``config.yaml`` slice the backup service reads.

The backup service reuses the existing top-level ``postgres`` and ``qdrant``
blocks for connection topology and adds its own ``backup`` block for scheduling,
the rclone target, and retention. No secrets live here — Postgres credentials
and the Qdrant API key come from the environment (see ``settings.py``) and the
cloud credentials live in the mounted ``rclone.conf``.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class PostgresConfig(BaseModel):
    """The ``postgres`` section. ``database`` falls back to ``POSTGRES_DB``."""

    host: str = "db"
    port: int = 5432
    database: str | None = None


class QdrantConfig(BaseModel):
    """The ``qdrant`` section — REST host/port for the snapshot API."""

    host: str = "vector_db"
    port: int = 6333


class BackupConfig(BaseModel):
    """The ``backup`` section — scheduling, rclone destination, retention.

    ``enabled`` defaults to ``False`` so a fresh deploy performs no backups
    until a cloud remote is configured. ``rclone_config_path``/``staging_dir``
    are container-relative to the service WORKDIR (``/app``); they resolve into
    the bind-mounted ``_container_data/backup`` host directory.
    """

    enabled: bool = False
    interval_minutes: float = 60.0
    run_on_start: bool = True
    rclone_remote: str = ""
    rclone_path: str = "macro-backups"
    rclone_config_path: str = "_container_data/rclone.conf"
    retention_days: int = 7
    staging_dir: str = "_container_data/staging"


class BackupServiceConfig(BaseModel):
    """The portion of ``config.yaml`` the backup service reads."""

    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    backup: BackupConfig = BackupConfig()


def load_config(path: Path) -> BackupServiceConfig:
    """Parse and validate ``config.yaml`` into a :class:`BackupServiceConfig`."""
    return BackupServiceConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
