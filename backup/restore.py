"""Restore helper for cloud backups (manual, run on demand).

Run inside the backup container so it has ``rclone`` + ``pg_restore`` + the
mounted ``rclone.conf`` + the Postgres/Qdrant env vars, e.g.::

    docker compose run --rm backup python restore.py --list
    docker compose run --rm backup python restore.py --postgres macro_2026-07-04T12-00-00Z.dump
    docker compose run --rm backup python restore.py --qdrant <name>.snapshot

Postgres restore is automated (``pg_restore --clean --if-exists``). A Qdrant
*full* snapshot cannot be recovered into a live instance with a single API
call, so this tool downloads the snapshot and prints the exact restart-based
recovery steps.
"""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys

from config import load_config
from settings import get_settings

CONFIG_PATH = Path("config.yaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("restore")


def _rclone(config_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["rclone", "--config", config_path, *args], check=True, capture_output=True, text=True
    )


def _list(config_path: str, remote: str, remote_path: str) -> None:
    out = _rclone(config_path, "lsf", "-R", f"{remote}:{remote_path}")
    print(out.stdout or "(no files found)")


def _download(config_path: str, remote: str, remote_src: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s:%s -> %s", remote, remote_src, dest)
    _rclone(config_path, "copyto", f"{remote}:{remote_src}", str(dest))
    return dest


def _restore_postgres(file_name: str, download_dir: Path) -> None:
    config = load_config(CONFIG_PATH)
    settings = get_settings()
    backup = config.backup
    database = settings.postgres_db or config.postgres.database
    if not database:
        raise SystemExit("No Postgres database name (set POSTGRES_DB or postgres.database).")

    local = _download(
        backup.rclone_config_path,
        backup.rclone_remote,
        f"{backup.rclone_path}/postgres/{file_name}",
        download_dir / file_name,
    )
    env = {**os.environ, "PGPASSWORD": settings.postgres_password}
    cmd = [
        "pg_restore",
        "--clean",
        "--if-exists",
        "--no-owner",
        "--host",
        config.postgres.host,
        "--port",
        str(config.postgres.port),
        "--username",
        settings.postgres_user,
        "--dbname",
        database,
        str(local),
    ]
    logger.warning(
        "Restoring Postgres database %r from %s (this overwrites data!)", database, local
    )
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        # pg_restore often exits non-zero on benign "does not exist" notices with
        # --clean; surface stderr so the operator can judge.
        logger.error("pg_restore exited %s:\n%s", result.returncode, result.stderr.strip())
        raise SystemExit(result.returncode)
    logger.info("Postgres restore complete.")


def _restore_qdrant(file_name: str, download_dir: Path) -> None:
    config = load_config(CONFIG_PATH)
    backup = config.backup
    local = _download(
        backup.rclone_config_path,
        backup.rclone_remote,
        f"{backup.rclone_path}/qdrant/{file_name}",
        download_dir / file_name,
    )
    print(
        f"""
Downloaded Qdrant full snapshot to: {local}

A full-storage snapshot is restored by recovering it into Qdrant's storage at
startup (there is no single live-instance API call for a *full* snapshot).
Recommended procedure (brief Qdrant downtime):

  1. Stop writers so nothing mutates Qdrant during recovery:
       docker compose stop app agent

  2. Copy the snapshot into the Qdrant container:
       docker cp {local} qdrant_db:/qdrant/snapshots/{file_name}

  3. Recover into storage by starting Qdrant once with the recovery flag
     (this rebuilds /qdrant/storage from the snapshot):
       docker compose run --rm --entrypoint ./qdrant vector_db \\
         --storage-snapshot /qdrant/snapshots/{file_name}

  4. Bring the stack back up:
       docker compose up -d

See https://qdrant.tech/documentation/concepts/snapshots/ for details.
"""
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Restore Postgres/Qdrant cloud backups.")
    parser.add_argument("--list", action="store_true", help="List available backups on the remote.")
    parser.add_argument("--postgres", metavar="FILE", help="Postgres dump file name to restore.")
    parser.add_argument("--qdrant", metavar="FILE", help="Qdrant snapshot file name to restore.")
    parser.add_argument(
        "--download-dir",
        default="_container_data/restore",
        help="Local dir for downloaded artifacts (default: _container_data/restore).",
    )
    args = parser.parse_args(argv)

    config = load_config(CONFIG_PATH)
    backup = config.backup
    if not backup.rclone_remote:
        raise SystemExit("backup.rclone_remote is empty; nothing to restore from.")

    download_dir = Path(args.download_dir)

    if args.list:
        _list(backup.rclone_config_path, backup.rclone_remote, backup.rclone_path)
        return
    if args.postgres:
        _restore_postgres(args.postgres, download_dir)
    if args.qdrant:
        _restore_qdrant(args.qdrant, download_dir)
    if not (args.postgres or args.qdrant):
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
