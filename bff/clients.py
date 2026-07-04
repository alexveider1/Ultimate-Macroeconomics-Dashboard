"""Thin async proxies to the existing forecaster / clustering / agent services.

The BFF re-exposes these services behind one origin so the frontend only ever
talks to the BFF. Base URLs resolve as ``<SERVICE>_BASE_URL`` env override →
Compose-network default (``http://<service>:<port>`` from ``config.yaml``). A
single shared ``httpx.AsyncClient`` (built at startup, closed on shutdown) is
reused for every call.
"""

import logging
import os

from fastapi import HTTPException
import httpx

logger = logging.getLogger(__name__)


def resolve_base_url(env_var: str, default: str) -> str:
    """Return the first usable base URL: env override → Compose default."""
    override = os.getenv(env_var)
    if override and override.strip():
        return override.strip().rstrip("/")
    return default.rstrip("/")


def _downstream_error(service: str, exc: httpx.HTTPError) -> HTTPException:
    """Map an ``httpx`` failure to a client-facing :class:`HTTPException`."""
    response = getattr(exc, "response", None)
    if response is not None:
        body = (response.text or "")[:300].strip()
        detail = f"{service} returned HTTP {response.status_code}" + (f": {body}" if body else "")
        return HTTPException(status_code=502, detail=detail)
    logger.warning("%s request failed: %s", service, exc)
    return HTTPException(status_code=503, detail=f"{service} is unavailable: {exc}")


async def post_json(
    client: httpx.AsyncClient,
    service: str,
    url: str,
    body: dict,
    timeout: float,
) -> dict:
    """POST ``body`` as JSON and return the parsed response, or raise 502/503."""
    try:
        response = await client.post(url, json=body, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise _downstream_error(service, exc) from exc


async def get_json(
    client: httpx.AsyncClient,
    service: str,
    url: str,
    timeout: float,
) -> dict:
    """GET and return the parsed JSON response, or raise 502/503."""
    try:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise _downstream_error(service, exc) from exc
