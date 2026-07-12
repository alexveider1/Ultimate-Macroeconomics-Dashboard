"""Thin wrappers around the ``rclone`` CLI for upload + retention pruning.

The argv builders are pure functions so they can be unit-tested without a real
rclone binary or remote.
"""

import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


def build_copy_args(*, config_path: str, source: str, remote: str, remote_path: str) -> list[str]:
    """Build the ``rclone copy`` argv."""
    return [
        "rclone",
        "--config",
        config_path,
        "copy",
        source,
        f"{remote}:{remote_path}",
    ]


def build_prune_args(
    *, config_path: str, remote: str, remote_path: str, retention_days: int
) -> list[str]:
    """Build the ``rclone delete --min-age Nd`` argv."""
    return [
        "rclone",
        "--config",
        config_path,
        "delete",
        f"{remote}:{remote_path}",
        "--min-age",
        f"{retention_days}d",
    ]


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error("rclone failed (exit %s): %s", exc.returncode, (exc.stderr or "").strip())
        raise


def upload(*, config_path: str, source: Path, remote: str, remote_path: str) -> None:
    """``rclone copy`` a local dir/file to ``<remote>:<remote_path>``."""
    cmd = build_copy_args(
        config_path=config_path, source=str(source), remote=remote, remote_path=remote_path
    )
    logger.info("Uploading %s -> %s:%s", source, remote, remote_path)
    _run(cmd)
    logger.info("Upload complete")


def prune(*, config_path: str, remote: str, remote_path: str, retention_days: int) -> None:
    """Delete remote artifacts older than ``retention_days`` days.

    A negative ``retention_days`` disables pruning (keep everything).
    """
    if retention_days < 0:
        logger.info("Retention disabled (retention_days < 0); skipping prune")
        return
    cmd = build_prune_args(
        config_path=config_path,
        remote=remote,
        remote_path=remote_path,
        retention_days=retention_days,
    )
    logger.info("Pruning %s:%s older than %s days", remote, remote_path, retention_days)
    _run(cmd)
    logger.info("Prune complete")
