"""Async World Bank v2 REST client (httpx).

Replaces the ``wbgapi`` dependency with direct calls to the documented v2
endpoints, served over a single shared :class:`httpx.AsyncClient`. Each public
coroutine returns plain ``list``/``dict`` objects shaped to match what the rest
of the ingestion pipeline already expects (so :func:`_polars_from_world_bank_records`
and the schema cast keep working unchanged):

- :func:`fetch_sources`        → ``databases`` table rows
- :func:`fetch_countries`      → ``countries`` table rows (aggregates dropped)
- :func:`fetch_series`         → ``database_indicators`` rows (``{id, value}``)
- :func:`fetch_indicator_data` → ``indicators`` rows (``{economy, time, value}``)
- :func:`fetch_series_metadata`/:func:`fetch_indicator_metadata` → ``metadata`` row

Aggregate economies (``region.id == "NA"``: "World", regions, income groups…)
are skipped to match the old ``skipAggs=True`` behaviour; null observations are
kept (``skipBlanks=False`` parity).
"""

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

WB_API_BASE = "https://api.worldbank.org/v2"
# WB starts rate-limiting beyond a few thousand records per page; 1000 keeps the
# page count low without tripping it.
_PER_PAGE = 1000
DEFAULT_TIMEOUT = 30.0

# Set of aggregate ISO3 codes ("World", regions, income groups…), cached for the
# lifetime of the process so we only hit /country once.
_aggregate_codes_cache: Optional[set[str]] = None
_raw_countries_cache: Optional[list[dict[str, Any]]] = None


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
        path: Path relative to :data:`WB_API_BASE` (e.g. ``"source"``).
        params: Extra query parameters merged with ``format``/``per_page``.

    Returns:
        Every record across all pages. Empty list when the endpoint returns
        nothing (or an XML error payload).
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
            # The API returns an XML error body (still HTTP 200) for some bad
            # requests; treat that as "no data".
            break
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            break
        header, page_records = payload[0], payload[1]
        records.extend(page_records)
        if page >= int(header.get("pages", 1)):
            break
        page += 1
    return records


async def _raw_countries(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Fetch (and cache) the raw ``/country`` catalogue."""
    global _raw_countries_cache
    if _raw_countries_cache is None:
        _raw_countries_cache = await _fetch_list(client, "country")
    return _raw_countries_cache


async def aggregate_codes(client: httpx.AsyncClient) -> set[str]:
    """Return the set of aggregate ISO3 codes (``region.id == "NA"``)."""
    global _aggregate_codes_cache
    if _aggregate_codes_cache is None:
        _aggregate_codes_cache = {
            country["id"]
            for country in await _raw_countries(client)
            if (country.get("region") or {}).get("id") == "NA"
        }
    return _aggregate_codes_cache


async def fetch_sources(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Return the WB source/database catalogue (``databases`` table rows).

    ``databid=y`` asks the API to include the alternate ``databid`` field that
    the schema keeps as text.
    """
    return await _fetch_list(client, "source", {"databid": "y"})


def _label(node: Optional[dict[str, Any]]) -> dict[str, Optional[str]]:
    """Normalise a ``{id, value}`` sub-object, trimming the (often padded) label."""
    node = node or {}
    value = node.get("value")
    return {
        "id": node.get("id") or None,
        "value": value.strip() if isinstance(value, str) else value,
    }


def _to_float(value: Any) -> Optional[float]:
    """Parse a coordinate string to ``float``; ``None`` when missing/invalid."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _country_record(country: dict[str, Any], is_aggregate: bool) -> dict[str, Any]:
    """Shape one ``/country`` record into the nested form the schema expects.

    Nested ``region``/``adminregion``/``lendingType``/``incomeLevel`` dicts get
    flattened to ``region.id`` / ``region.value`` … downstream by
    ``_flatten_record``.
    """
    capital = country.get("capitalCity")
    return {
        "id": country.get("id"),
        "value": country.get("name"),
        "aggregate": is_aggregate,
        "longitude": _to_float(country.get("longitude")),
        "latitude": _to_float(country.get("latitude")),
        "region": _label(country.get("region")),
        "adminregion": _label(country.get("adminregion")),
        "lendingType": _label(country.get("lendingType")),
        "incomeLevel": _label(country.get("incomeLevel")),
        "capitalCity": capital.strip() if isinstance(capital, str) else capital,
    }


async def fetch_countries(
    client: httpx.AsyncClient, *, skip_aggregates: bool = True
) -> list[dict[str, Any]]:
    """Return the ``countries`` table rows from ``/country``.

    Args:
        client: Shared async HTTP client.
        skip_aggregates: Drop aggregate economies (``region.id == "NA"``) to
            match the old ``economy.list(skipAggs=True)`` behaviour.
    """
    records: list[dict[str, Any]] = []
    for country in await _raw_countries(client):
        is_aggregate = (country.get("region") or {}).get("id") == "NA"
        if skip_aggregates and is_aggregate:
            continue
        records.append(_country_record(country, is_aggregate))
    return records


async def fetch_series(client: httpx.AsyncClient, db: int) -> list[dict[str, Any]]:
    """Return the indicator catalogue for one database (``database_indicators``).

    Shaped as ``{id, value}`` (``value`` is the indicator name) so the caller's
    ``rename({"value": "description"})`` keeps working.
    """
    raw = await _fetch_list(client, "indicator", {"source": db})
    return [{"id": item["id"], "value": item.get("name")} for item in raw]


async def fetch_indicator_data(
    client: httpx.AsyncClient, indicator_id: str, db: int
) -> list[dict[str, Any]]:
    """Fetch one indicator's observations (``indicators`` table rows).

    Drops aggregate economies and non-ISO3 rows; keeps null observations and
    rows whose ``date`` parses as an integer year.

    Returns:
        Rows shaped as ``{"economy", "time", "value"}``.
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
        rows.append({"economy": iso3, "time": year, "value": record.get("value")})
    return rows


def _metadata_row(meta: dict[str, Any]) -> dict[str, Optional[str]]:
    """Project a flat WB metatype dict onto the ``metadata`` table columns."""
    return {
        "indicator_name": meta.get("IndicatorName"),
        "units": meta.get("Unitofmeasure"),
        "source": meta.get("Source"),
        "development_relevance": meta.get("Developmentrelevance"),
        "limitations_and_exceptions": meta.get("Limitationsandexceptions"),
        "statistical_concept_and_methodology": meta.get("Statisticalconceptandmethodology"),
    }


async def fetch_series_metadata(
    client: httpx.AsyncClient, indicator_id: str, db: int
) -> Optional[dict[str, Optional[str]]]:
    """Fetch rich series metadata from the advanced ``/sources/.../metadata`` endpoint.

    Returns the ``metadata`` row dict, or ``None`` when the source has no
    metadata for the indicator (the API answers with an XML error then).
    """
    resp = await client.get(
        f"{WB_API_BASE}/sources/{db}/series/{indicator_id}/metadata",
        params={"format": "json"},
    )
    resp.raise_for_status()
    try:
        payload = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        metatype = payload["source"][0]["concept"][0]["variable"][0]["metatype"]
    except (KeyError, IndexError, TypeError):
        return None
    meta = {item["id"]: item.get("value") for item in metatype}
    return _metadata_row(meta)


async def fetch_indicator_metadata(
    client: httpx.AsyncClient, indicator_id: str, db: int
) -> Optional[dict[str, Optional[str]]]:
    """Fallback metadata from the standard ``/indicator/{id}`` endpoint.

    Used when the advanced metadata endpoint returns nothing. The standard
    endpoint exposes only name/unit/source/sourceNote, so the two long-form
    fields stay ``None``.
    """
    resp = await client.get(
        f"{WB_API_BASE}/indicator/{indicator_id}",
        params={"format": "json", "source": db},
    )
    resp.raise_for_status()
    try:
        payload = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return None
    info = payload[1][0]
    return {
        "indicator_name": info.get("name"),
        "units": info.get("unit") or "",
        "source": (info.get("source") or {}).get("value"),
        "development_relevance": info.get("sourceNote"),
        "limitations_and_exceptions": None,
        "statistical_concept_and_methodology": None,
    }


async def healthcheck(client: httpx.AsyncClient) -> bool:
    """Return ``True`` if the WB API answers ``/source`` with at least one row."""
    try:
        resp = await client.get(f"{WB_API_BASE}/source", params={"format": "json", "per_page": 1})
        resp.raise_for_status()
        payload = resp.json()
        return isinstance(payload, list) and len(payload) >= 2 and bool(payload[1])
    except Exception:
        logger.exception("An error occured while testing connection to World Bank API")
        return False


async def call_with_retries(
    operation_name: str,
    request_coro_factory,
    retry_delay_seconds: float,
    max_retries: int,
):
    """Await ``request_coro_factory()`` with bounded retry-on-exception.

    Args:
        operation_name: Label used in log messages so failures can be traced.
        request_coro_factory: Zero-arg callable returning a fresh coroutine on
            each attempt.
        retry_delay_seconds: Sleep between attempts.
        max_retries: Retries *after* the first attempt (total = ``max_retries + 1``).

    Returns:
        The coroutine's result on success, or ``None`` if every attempt raised.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            return await request_coro_factory()
        except Exception as exc:
            if attempt == max_retries:
                logger.exception(
                    "Operation '%s' failed after %d attempt(s), giving up",
                    operation_name,
                    attempt + 1,
                )
                return None
            logger.warning(
                "Retry %d/%d for operation '%s' failed: %s",
                attempt + 1,
                max_retries,
                operation_name,
                exc,
                exc_info=True,
            )
            await asyncio.sleep(retry_delay_seconds)
            attempt += 1
