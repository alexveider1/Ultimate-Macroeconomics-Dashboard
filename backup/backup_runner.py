"""Orchestrate a single backup run: dump + snapshot + upload + prune.

Consistency note: Postgres and Qdrant live in separate stores, so there is no
cross-store transactional snapshot. We run the two captures back-to-back in one
function so they are as close in time as possible; a backup set reflects each
store at the moment its own capture ran.
"""

import logging
from pathlib import Path
import shutil

from backup_postgres import dump_postgres
from backup_qdrant import snapshot_qdrant
from config import BackupServiceConfig
from naming import postgres_dump_name, utc_timestamp
import rclone_runner
from settings import Settings

logger = logging.getLogger(__name__)


def run_backup_once(config: BackupServiceConfig, settings: Settings) -> None:
    """Run one full backup cycle. Raises on failure (the caller decides retry).

    Staging is wiped before and after each run so ``rclone copy`` only ever
    uploads the current run's two artifacts (under ``postgres/`` and
    ``qdrant/`` subdirs).
    """
    backup = config.backup

    if not backup.rclone_remote:
        logger.error(
            "backup.rclone_remote is empty; skipping run. "
            "Set it in config.yaml and provide rclone.conf to enable uploads."
        )
        return
    if not Path(backup.rclone_config_path).exists():
        logger.error(
            "rclone config not found at %s; skipping run. "
            "Create it (rclone config) under _container_data/backup/.",
            backup.rclone_config_path,
        )
        return

    database = settings.postgres_db or config.postgres.database
    if not database:
        raise ValueError("No Postgres database name (set POSTGRES_DB or postgres.database).")

    staging = Path(backup.staging_dir)
    timestamp = utc_timestamp()
    if staging.exists():
        shutil.rmtree(staging)

    try:
        dump_postgres(
            host=config.postgres.host,
            port=config.postgres.port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=database,
            out_path=staging / "postgres" / postgres_dump_name(database, timestamp),
        )
        snapshot_qdrant(
            base_url=f"http://{config.qdrant.host}:{config.qdrant.port}",
            api_key=settings.qdrant_api_key,
            out_dir=staging / "qdrant",
        )
        rclone_runner.upload(
            config_path=backup.rclone_config_path,
            source=staging,
            remote=backup.rclone_remote,
            remote_path=backup.rclone_path,
        )
        rclone_runner.prune(
            config_path=backup.rclone_config_path,
            remote=backup.rclone_remote,
            remote_path=backup.rclone_path,
            retention_days=backup.retention_days,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    logger.info("Backup run complete (timestamp=%s)", timestamp)
