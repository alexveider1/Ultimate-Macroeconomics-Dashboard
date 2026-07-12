"""Async World Bank v2 REST client (httpx) — single-indicator subset.

The on-demand ingestion service only ever needs one indicator's observations,
so this is a trimmed copy of ``downloader_general``'s client: it pages the
``/country/all/indicator`` data endpoint and drops aggregate economies
(``region.id == "NA"``) to match the old ``wbgapi`` ``skipAggs=True`` behaviour.
Null observations are kept (``skipBlanks=False`` parity).
"""

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

WB_API_BASE = "https://api.worldbank.org/v2"
_PER_PAGE = 1000
DEFAULT_TIMEOUT = 30.0

# Aggregate ISO3 codes, cached for the process lifetime (only /country once).
_aggregate_codes_cache: Optional[set[str]] = None


def build_async_client() -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` configured for the WB API."""
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={"Accept": "application/json"},
    )


async def _fetch_list(
    client: httpx.AsyncClient,
    path: str,
    params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Page through a v2 ``[header, [records]]`` list endpoint.

    Args:
        client: Shared async HTTP client.
        path: Path relative to :data:`WB_API_BASE`.
        params: Extra query parameters merged with ``format``/``per_page``.

    Returns:
        Every record across all pages; empty list on an XML error payload.
    """
    query: dict[str, Any] = {"format": "json", "per_page": _PER_PAGE}
    query.update(params or {})
    page = 1
    records: list[dict[str, Any]] = []
    while True:
        query["page"] = page
        resp = await client.get(f"{WB_API_BASE}/{path}", params=query)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError):
            break
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            break
        header, page_records = payload[0], payload[1]
        records.extend(page_records)
        if page >= int(header.get("pages", 1)):
            break
        page += 1
    return records


async def aggregate_codes(client: httpx.AsyncClient) -> set[str]:
    """Return (and cache) the set of aggregate ISO3 codes (``region.id == "NA"``)."""
    global _aggregate_codes_cache
    if _aggregate_codes_cache is None:
        countries = await _fetch_list(client, "country")
        _aggregate_codes_cache = {
            country["id"]
            for country in countries
            if (country.get("region") or {}).get("id") == "NA"
        }
    return _aggregate_codes_cache


async def fetch_indicator_data(
    client: httpx.AsyncClient, indicator_id: str, db: int
) -> list[dict[str, Any]]:
    """Fetch one indicator's observations.

    Drops aggregate economies and non-ISO3 rows; keeps null observations and
    rows whose ``date`` parses as an integer year.

    Returns:
        Rows shaped as ``{"economy", "year", "value"}``.
    """
    aggregates = await aggregate_codes(client)
    records = await _fetch_list(client, f"country/all/indicator/{indicator_id}", {"source": db})
    rows: list[dict[str, Any]] = []
    for record in records:
        iso3 = record.get("countryiso3code", "")
        if len(iso3) != 3 or iso3 in aggregates:
            continue
        try:
            year = int(record["date"])
        except (ValueError, TypeError, KeyError):
            continue
        rows.append({"economy": iso3, "year": year, "value": record.get("value")})
    return rows
