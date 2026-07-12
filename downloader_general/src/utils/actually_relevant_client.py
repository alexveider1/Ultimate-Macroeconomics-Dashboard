"""Sync HTTP client for the Actually Relevant curated-news API (httpx).

Talks to the two documented public JSON endpoints of
``https://actually-relevant-api.onrender.com`` (no API key):

- ``GET /api/issues``  → the topic taxonomy: 5 top-level macro-issues each with
  a ``children`` list; used to map any granular ``issue.slug`` (e.g.
  ``nuclear-war``) up to its macro parent (``existential-threats``).
- ``GET /api/stories`` → a paginated envelope
  ``{data, total, page, pageSize, totalPages}`` of *curated* records. Each item
  carries analysis text (``summary``, ``relevanceSummary``, ``relevanceReasons``,
  ``antifactors``, ``quote``, ``marketingBlurb``) plus ``issue{name,slug}`` and a
  ``sourceUrl`` link-out — the API does **not** serve the full source body.

The free onrender host drops rapid sequential requests, so callers wrap each
call in :func:`src.utils.downloads._call_with_retries` (exponential backoff +
jitter). This module stays a thin transport layer returning plain dicts/lists.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://actually-relevant-api.onrender.com"
DEFAULT_TIMEOUT = 60.0


def build_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    """Return an ``httpx.Client`` configured for the Actually Relevant API."""
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"Accept": "application/json"},
    )


def fetch_issues(client: httpx.Client, base_url: str = DEFAULT_BASE_URL) -> list[dict[str, Any]]:
    """Return the raw ``/api/issues`` taxonomy (list of macro-issue dicts)."""
    resp = client.get(f"{base_url.rstrip('/')}/api/issues", params={"format": "json"})
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


def fetch_stories_page(
    client: httpx.Client,
    page: int,
    page_size: int,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Return one ``/api/stories`` page envelope ``{data, total, ...}``."""
    resp = client.get(
        f"{base_url.rstrip('/')}/api/stories",
        params={"format": "json", "page": page, "pageSize": page_size},
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def build_macro_map(issues: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Map every issue slug (macro and child) to its macro ``(slug, name)``.

    Top-level entries map to themselves; each ``child`` slug maps up to its
    parent so a story tagged with a granular slug (``pandemics``) is bucketed
    into the macro collection (``existential-threats``).
    """
    mapping: dict[str, tuple[str, str]] = {}
    for top in issues:
        macro_slug = top.get("slug")
        macro_name = top.get("name")
        if not macro_slug or not macro_name:
            continue
        mapping[macro_slug] = (macro_slug, macro_name)
        for child in top.get("children") or []:
            child_slug = child.get("slug")
            if child_slug:
                mapping[child_slug] = (macro_slug, macro_name)
    return mapping
