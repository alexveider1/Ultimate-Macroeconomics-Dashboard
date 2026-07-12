"""Dump the Postgres database with ``pg_dump`` (custom compressed format)."""

import logging
import os
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


def dump_postgres(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    out_path: Path,
) -> Path:
    """Run ``pg_dump -Fc`` into ``out_path`` and return it.

    Uses the custom compressed format (restorable with ``pg_restore``). The
    password is passed via ``PGPASSWORD`` in the child environment so it never
    appears in the process argv. Raises :class:`subprocess.CalledProcessError`
    on failure (after logging stderr).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PGPASSWORD": password}
    cmd = [
        "pg_dump",
        "--format=custom",
        "--host",
        host,
        "--port",
        str(port),
        "--username",
        user,
        "--dbname",
        database,
        "--file",
        str(out_path),
    ]
    logger.info("Dumping Postgres %s@%s:%s/%s -> %s", user, host, port, database, out_path)
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error("pg_dump failed (exit %s): %s", exc.returncode, (exc.stderr or "").strip())
        raise
    logger.info("Postgres dump complete (%s bytes)", out_path.stat().st_size)
    return out_path
