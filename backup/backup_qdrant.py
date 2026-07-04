"""Create, download, and clean up a full-storage Qdrant snapshot via REST.

A *full* snapshot captures every collection in a single ``.snapshot`` artifact.
We create it, stream it to local staging, then delete it server-side so
Qdrant's ephemeral ``/qdrant/snapshots`` layer (not mounted to a volume) does
not accumulate copies.
"""

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Snapshot creation and download can take a while for large storage; use a
# generous read timeout with a short connect timeout.
_TIMEOUT = httpx.Timeout(600.0, connect=15.0)


def _headers(api_key: str) -> dict[str, str]:
    """Qdrant authenticates via the ``api-key`` header (empty key → no header)."""
    return {"api-key": api_key} if api_key else {}


def snapshot_qdrant(*, base_url: str, api_key: str, out_dir: Path) -> Path:
    """Create a full snapshot, stream it to ``out_dir``, delete it server-side.

    Returns the local snapshot file path (named as Qdrant assigns,
    ``<name>.snapshot``). A failure to delete the server-side copy is logged
    but does not fail the backup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = _headers(api_key)

    with httpx.Client(base_url=base_url, headers=headers, timeout=_TIMEOUT) as client:
        logger.info("Requesting Qdrant full snapshot at %s", base_url)
        resp = client.post("/snapshots", params={"wait": "true"})
        resp.raise_for_status()
        name = resp.json()["result"]["name"]
        logger.info("Qdrant snapshot created: %s", name)

        out_path = out_dir / name
        with client.stream("GET", f"/snapshots/{name}") as stream:
            stream.raise_for_status()
            with out_path.open("wb") as fh:
                for chunk in stream.iter_bytes():
                    fh.write(chunk)
        logger.info(
            "Qdrant snapshot downloaded -> %s (%s bytes)", out_path, out_path.stat().st_size
        )

        try:
            client.delete(f"/snapshots/{name}").raise_for_status()
            logger.info("Deleted server-side Qdrant snapshot %s", name)
        except httpx.HTTPError:
            logger.warning(
                "Failed to delete server-side Qdrant snapshot %s (left in place)",
                name,
                exc_info=True,
            )

    return out_path
