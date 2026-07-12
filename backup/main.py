"""Entry point for the backup service.

Long-running scheduler: on each tick it dumps Postgres, snapshots Qdrant, and
uploads both to the configured rclone remote, then prunes old remote backups.
When ``backup.enabled`` is false the process logs that and idles until SIGTERM
so ``restart: unless-stopped`` does not restart-loop a clean exit. Each run is
wrapped so one failure is logged and retried next interval rather than killing
the loop.
"""

import logging
from pathlib import Path
import signal
import sys
import threading
from types import FrameType

from backup_runner import run_backup_once
from config import BackupServiceConfig, load_config
from settings import Settings, get_settings

CONFIG_PATH = Path("config.yaml")

logger = logging.getLogger(__name__)

_stop = threading.Event()


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    logger.info("Received signal %s; shutting down.", signum)
    _stop.set()


def _setup_logging() -> None:
    container_data_dir = Path("_container_data")
    container_data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(container_data_dir / "app.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _safe_run(config: BackupServiceConfig, settings: Settings) -> None:
    """Run one backup, swallowing+logging any error so the loop survives."""
    try:
        run_backup_once(config, settings)
    except Exception:
        logger.exception("Backup run failed; will retry next interval")


def main() -> None:
    _setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = load_config(CONFIG_PATH)
    backup = config.backup

    if not backup.enabled:
        logger.info("Cloud backups are disabled (backup.enabled=false); idling.")
        _stop.wait()
        logger.info("Exit.")
        return

    settings = get_settings()
    interval_seconds = max(backup.interval_minutes * 60.0, 1.0)
    logger.info(
        "Cloud backups enabled: remote=%r path=%r every %s min (run_on_start=%s, retention=%sd)",
        backup.rclone_remote,
        backup.rclone_path,
        backup.interval_minutes,
        backup.run_on_start,
        backup.retention_days,
    )

    if backup.run_on_start:
        _safe_run(config, settings)

    while not _stop.wait(interval_seconds):
        _safe_run(config, settings)

    logger.info("Backup scheduler stopped.")


if __name__ == "__main__":
    main()
