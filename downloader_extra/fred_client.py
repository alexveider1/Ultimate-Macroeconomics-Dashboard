"""Trimmed async FRED / GeoFRED client for on-demand single-indicator ingestion.

A data-fetch-only copy of ``downloader_general/src/utils/fred_client.py``
(duplicated per service by design, like ``wb_client.py`` / ``binance_client.py``).
It resolves a representative state series id into its GeoFRED series group, then
fetches the whole annual cross-state panel in one call. Observations are mapped
to states by their FIPS ``code`` — the reliable key, since several FRED groups
use per-state series ids that do not encode the abbreviation.
"""

import json
from typing import Any, Optional

import httpx

FRED_API_BASE = "https://api.stlouisfed.org"
DEFAULT_TIMEOUT = 60.0
ANNUAL_FREQUENCY = "a"

# FIPS state code -> (USPS abbreviation, full name, Census region, Census division).
FIPS_TO_STATE: dict[str, tuple[str, str, str, str]] = {
    "01": ("AL", "Alabama", "South", "East South Central"),
    "02": ("AK", "Alaska", "West", "Pacific"),
    "04": ("AZ", "Arizona", "West", "Mountain"),
    "05": ("AR", "Arkansas", "South", "West South Central"),
    "06": ("CA", "California", "West", "Pacific"),
    "08": ("CO", "Colorado", "West", "Mountain"),
    "09": ("CT", "Connecticut", "Northeast", "New England"),
    "10": ("DE", "Delaware", "South", "South Atlantic"),
    "11": ("DC", "District of Columbia", "South", "South Atlantic"),
    "12": ("FL", "Florida", "South", "South Atlantic"),
    "13": ("GA", "Georgia", "South", "South Atlantic"),
    "15": ("HI", "Hawaii", "West", "Pacific"),
    "16": ("ID", "Idaho", "West", "Mountain"),
    "17": ("IL", "Illinois", "Midwest", "East North Central"),
    "18": ("IN", "Indiana", "Midwest", "East North Central"),
    "19": ("IA", "Iowa", "Midwest", "West North Central"),
    "20": ("KS", "Kansas", "Midwest", "West North Central"),
    "21": ("KY", "Kentucky", "South", "East South Central"),
    "22": ("LA", "Louisiana", "South", "West South Central"),
    "23": ("ME", "Maine", "Northeast", "New England"),
    "24": ("MD", "Maryland", "South", "South Atlantic"),
    "25": ("MA", "Massachusetts", "Northeast", "New England"),
    "26": ("MI", "Michigan", "Midwest", "East North Central"),
    "27": ("MN", "Minnesota", "Midwest", "West North Central"),
    "28": ("MS", "Mississippi", "South", "East South Central"),
    "29": ("MO", "Missouri", "Midwest", "West North Central"),
    "30": ("MT", "Montana", "West", "Mountain"),
    "31": ("NE", "Nebraska", "Midwest", "West North Central"),
    "32": ("NV", "Nevada", "West", "Mountain"),
    "33": ("NH", "New Hampshire", "Northeast", "New England"),
    "34": ("NJ", "New Jersey", "Northeast", "Middle Atlantic"),
    "35": ("NM", "New Mexico", "West", "Mountain"),
    "36": ("NY", "New York", "Northeast", "Middle Atlantic"),
    "37": ("NC", "North Carolina", "South", "South Atlantic"),
    "38": ("ND", "North Dakota", "Midwest", "West North Central"),
    "39": ("OH", "Ohio", "Midwest", "East North Central"),
    "40": ("OK", "Oklahoma", "South", "West South Central"),
    "41": ("OR", "Oregon", "West", "Pacific"),
    "42": ("PA", "Pennsylvania", "Northeast", "Middle Atlantic"),
    "44": ("RI", "Rhode Island", "Northeast", "New England"),
    "45": ("SC", "South Carolina", "South", "South Atlantic"),
    "46": ("SD", "South Dakota", "Midwest", "West North Central"),
    "47": ("TN", "Tennessee", "South", "East South Central"),
    "48": ("TX", "Texas", "South", "West South Central"),
    "49": ("UT", "Utah", "West", "Mountain"),
    "50": ("VT", "Vermont", "Northeast", "New England"),
    "51": ("VA", "Virginia", "South", "South Atlantic"),
    "53": ("WA", "Washington", "West", "Pacific"),
    "54": ("WV", "West Virginia", "South", "South Atlantic"),
    "55": ("WI", "Wisconsin", "Midwest", "East North Central"),
    "56": ("WY", "Wyoming", "West", "Mountain"),
}


def build_async_client(api_key: str) -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` with the FRED key baked into default params."""
    return httpx.AsyncClient(
        base_url=FRED_API_BASE,
        timeout=DEFAULT_TIMEOUT,
        params={"api_key": api_key, "file_type": "json"},
        headers={"Accept": "application/json"},
    )


async def _get_json(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> Any:
    """GET ``path`` and decode JSON, tolerating FRED's XML error bodies (→ None)."""
    resp = await client.get(path, params=params)
    resp.raise_for_status()
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        return None


async def fetch_series_group(client: httpx.AsyncClient, series_id: str) -> Optional[dict[str, Any]]:
    """Resolve a representative series id into its GeoFRED series group metadata."""
    payload = await _get_json(client, "geofred/series/group", {"series_id": series_id})
    if not isinstance(payload, dict):
        return None
    group = payload.get("series_group")
    return group if isinstance(group, dict) and group.get("series_group") else None


async def fetch_regional_panel(
    client: httpx.AsyncClient,
    *,
    series_group: str,
    region_type: str,
    start_date: str,
    end_date: str,
    units: str,
    season: str,
    frequency: str = ANNUAL_FREQUENCY,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch the whole annual cross-state panel for a series group in one call."""
    payload = await _get_json(
        client,
        "geofred/regional/data",
        {
            "series_group": series_group,
            "region_type": region_type,
            "date": end_date,
            "start_date": start_date,
            "units": units,
            "season": season or "NSA",
            "frequency": frequency,
        },
    )
    if not isinstance(payload, dict):
        return {}
    data = (payload.get("meta") or {}).get("data")
    return data if isinstance(data, dict) else {}


def parse_regional_panel(
    panel: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Flatten a regional panel into ``{state, year, value}`` rows + FRED names.

    Maps each observation to a state via its FIPS ``code`` and dedups to one
    value per ``(state, year)``, dropping unknown FIPS and null values.
    """
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for date_str, records in panel.items():
        try:
            year = int(str(date_str)[:4])
        except (ValueError, TypeError):
            continue
        for record in records or []:
            fips = str(record.get("code") or "").strip()
            mapping = FIPS_TO_STATE.get(fips)
            if mapping is None:
                continue
            abbrev = mapping[0]
            region_name = record.get("region")
            if isinstance(region_name, str) and region_name.strip():
                names.setdefault(fips, region_name.strip())
            value = record.get("value")
            if value is None:
                continue
            key = (abbrev, year)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"state": abbrev, "year": year, "value": value})
    return rows, names


def synthesize_notes(group: dict[str, Any]) -> str:
    """Compose a human-readable description from series-group metadata."""
    title = group.get("title") or "Indicator"
    units = group.get("units") or "n/a"
    frequency = group.get("frequency") or "n/a"
    season = group.get("season") or "NSA"
    min_date = group.get("min_date") or "?"
    max_date = group.get("max_date") or "?"
    return (
        f"{title} by U.S. state. Units: {units}. Native frequency: {frequency} "
        f"({season}); values are aggregated to annual. Coverage: {min_date} to "
        f"{max_date}. Source: Federal Reserve Bank of St. Louis (FRED) regional data."
    )
