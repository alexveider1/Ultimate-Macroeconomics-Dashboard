"""Async FRED / GeoFRED REST client (httpx).

State-level ("regional") data is fetched through the documented GeoFRED
endpoints served over a single shared :class:`httpx.AsyncClient`:

- :func:`fetch_series_group`   → resolve a representative series id (e.g. ``CAUR``)
  into its *series group* metadata (group id, units, frequency, seasonal
  adjustment, region type, date range). One group covers all 50 states + DC.
- :func:`fetch_regional_panel` → the whole annual cross-state panel for a group
  in a single call: ``{ "YYYY-01-01": [ {region, code, value, series_id}, … ] }``.
  Passing both ``start_date`` and ``date`` (end) returns every year in the range;
  ``frequency="a"`` aggregates monthly/quarterly/weekly series to annual.
- :func:`fetch_series`         → basic ``fred/series`` metadata for a single series
  (used only as a light fallback for extra descriptive text).

The reliable key that maps an observation to a state is the **FIPS ``code``**
returned by ``regional/data`` (California = ``"06"``) — *not* the per-state
``series_id``, because several FRED groups use series ids that do not encode the
state abbreviation (e.g. ``MEDDAYONMARCA``, ``EXPTOTCA``). :data:`FIPS_TO_STATE`
maps every FIPS code to its USPS abbreviation, name and Census region/division.

The API key travels as a query parameter baked into the client's default params
(``api_key`` + ``file_type=json``), so every request carries it automatically.
:func:`call_with_retries` is re-exported from :mod:`src.utils.wb_client` so the
FRED extractor uses the same bounded retry-on-exception wrapper.
"""

import json
import logging
from typing import Any, Optional

import httpx

from src.utils.wb_client import call_with_retries

__all__ = [
    "FIPS_TO_STATE",
    "build_async_client",
    "call_with_retries",
    "fetch_regional_panel",
    "fetch_series",
    "fetch_series_group",
    "healthcheck",
    "parse_regional_panel",
    "state_records_from_names",
    "synthesize_notes",
]

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org"
DEFAULT_TIMEOUT = 60.0
# GeoFRED uses "a" for annual aggregation of the underlying (possibly monthly /
# quarterly / weekly) series; every stored observation is annual.
ANNUAL_FREQUENCY = "a"

# FIPS state code -> (USPS abbreviation, full name, Census region, Census division).
# 50 states + District of Columbia; DC is grouped under "South" / "South Atlantic"
# following U.S. Census Bureau convention. This is the authoritative join reference
# because GeoFRED per-state series ids do not reliably encode the abbreviation.
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
    """Return an ``httpx.AsyncClient`` configured for the FRED / GeoFRED API.

    The API key and ``file_type=json`` are baked into the client's default query
    params so every request carries them; individual calls only add their own
    parameters, which httpx merges with these defaults.

    Args:
        api_key: FRED API key (from the ``FRED_API_KEY`` secret).
    """
    return httpx.AsyncClient(
        base_url=FRED_API_BASE,
        timeout=DEFAULT_TIMEOUT,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        params={"api_key": api_key, "file_type": "json"},
        headers={"Accept": "application/json"},
    )


async def _get_json(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> Any:
    """GET ``path`` and decode JSON, tolerating FRED's occasional XML error bodies.

    FRED answers some malformed requests with an XML error payload (still HTTP
    200 in a few cases, or a 4xx); ``raise_for_status`` handles HTTP errors and a
    JSON decode failure surfaces as ``None`` so callers can treat it as "no data".
    """
    resp = await client.get(path, params=params)
    resp.raise_for_status()
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        return None


async def fetch_series_group(client: httpx.AsyncClient, series_id: str) -> Optional[dict[str, Any]]:
    """Resolve a representative series id into its GeoFRED series group metadata.

    Args:
        client: Shared async HTTP client.
        series_id: Any single-state FRED series (e.g. ``"CAUR"``).

    Returns:
        The ``series_group`` dict (``series_group`` id, ``region_type``,
        ``units``, ``season``, ``frequency``, ``min_date``, ``max_date``,
        ``title``), or ``None`` when the series has no group.
    """
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
    """Fetch the full cross-state panel for a series group in one call.

    Passing both ``start_date`` and ``date`` (end) returns every period in the
    range; ``frequency="a"`` collapses the native frequency to annual.

    Args:
        client: Shared async HTTP client.
        series_group: GeoFRED series group id (e.g. ``"1224"``).
        region_type: Geographic granularity (``"state"``).
        start_date: Earliest observation date (ISO ``YYYY-MM-DD``).
        end_date: Latest observation date (ISO ``YYYY-MM-DD``).
        units: Units string exactly as reported by the series group.
        season: Seasonal-adjustment code from the series group (e.g. ``"SA"``).
        frequency: Aggregation frequency; defaults to annual (``"a"``).

    Returns:
        Mapping of ``date -> list of {region, code, value, series_id}`` rows.
        Empty dict when the endpoint returns nothing.
    """
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


async def fetch_series(client: httpx.AsyncClient, series_id: str) -> Optional[dict[str, Any]]:
    """Return the basic ``fred/series`` metadata dict for one series, if any."""
    payload = await _get_json(client, "fred/series", {"series_id": series_id})
    if not isinstance(payload, dict):
        return None
    seriess = payload.get("seriess")
    if isinstance(seriess, list) and seriess and isinstance(seriess[0], dict):
        return seriess[0]
    return None


def parse_regional_panel(
    panel: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Flatten a regional panel into long rows and collect FRED state names.

    Maps each observation to a state via its FIPS ``code`` (the reliable key),
    keeps only rows with a known FIPS and a non-null value, and dedups to one
    value per ``(state, year)``.

    Args:
        panel: ``{date: [ {region, code, value, series_id}, … ]}`` from
            :func:`fetch_regional_panel`.

    Returns:
        Tuple of ``(rows, names)`` where ``rows`` is a list of
        ``{"state", "year", "value"}`` dicts and ``names`` maps FIPS code to the
        state name as reported by FRED (for cross-checking the states table).
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


def state_records_from_names(fred_names: dict[str, str]) -> list[dict[str, Any]]:
    """Build the ``states`` table rows from the static reference + FRED names.

    Every state in :data:`FIPS_TO_STATE` is emitted so the ``states`` table is
    always complete (a per-indicator panel may omit a state, e.g. DC in some
    housing series). The display ``name`` uses FRED's own label when available,
    falling back to the static name.

    Args:
        fred_names: ``{fips: name}`` collected from a live regional panel.

    Returns:
        List of ``{"id", "name", "fips", "region", "division"}`` dicts.
    """
    records: list[dict[str, Any]] = []
    for fips, (abbrev, name, region, division) in FIPS_TO_STATE.items():
        records.append(
            {
                "id": abbrev,
                "name": fred_names.get(fips, name),
                "fips": fips,
                "region": region,
                "division": division,
            }
        )
    return records


def synthesize_notes(group: dict[str, Any]) -> str:
    """Compose a human-readable description from series-group metadata.

    The GeoFRED group endpoint exposes no prose notes, so — like the Binance
    downloader — we synthesize one from the documented fields.
    """
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


async def healthcheck(client: httpx.AsyncClient) -> bool:
    """Return ``True`` if GeoFRED resolves a known state series group."""
    try:
        group = await fetch_series_group(client, "CAUR")
        return bool(group)
    except Exception:
        logger.exception("An error occured while testing connection to FRED API")
        return False
