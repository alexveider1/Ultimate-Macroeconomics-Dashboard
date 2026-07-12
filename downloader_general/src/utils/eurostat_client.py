"""Async Eurostat dissemination (JSON-stat) client + NUTS helpers (httpx).

Regional data is fetched, keyless, from the documented Eurostat dissemination
API over a single shared :class:`httpx.AsyncClient`:

- :func:`fetch_dataset`  → one dataset's full region×year panel as **JSON-stat 2.0**.
  ``geoLevel=nuts2`` restricts ``geo`` to NUTS-2 codes; every non-geo/non-time
  dimension (``unit``, ``sex``, ``age``, ``isced11``, ``nace_r2`` …) is pinned to a
  single category via query filters so the response is one clean series per
  region-year. Omitting ``time`` returns every available year in one call.
- :func:`parse_jsonstat` → flatten that JSON-stat payload into long
  ``{region, year, value}`` rows via row-major stride decoding, keeping only the
  target NUTS level and only cells where every pinned dimension is at index 0.
- :func:`regions_from_geojson` → build the ``eurostat_regions`` catalogue from the
  bundled GISCO NUTS-2 GeoJSON so the catalogue exactly matches the choropleth
  polygons (``NAME_LATN`` is the reliable region name; ``NAME_ENGL`` carries the
  country name at level 2).

:func:`call_with_retries` is re-exported from :mod:`src.utils.wb_client` so the
Eurostat extractor uses the same bounded retry-on-exception wrapper as the WB and
FRED extractors.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from src.utils.wb_client import call_with_retries

__all__ = [
    "build_async_client",
    "call_with_retries",
    "fetch_dataset",
    "healthcheck",
    "parse_jsonstat",
    "regions_from_geojson",
    "synthesize_notes",
]

logger = logging.getLogger(__name__)

EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0"
DEFAULT_TIMEOUT = 60.0
# A representative single-state series probe used by the healthcheck.
_HEALTHCHECK_DATASET = "nama_10r_2gdp"


def build_async_client() -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` configured for the Eurostat API.

    ``format=JSON`` and ``lang=EN`` are baked into the client's default query
    params so every request carries them; individual calls only add ``geoLevel``
    plus their pinned dimension filters, which httpx merges with these defaults.
    """
    return httpx.AsyncClient(
        base_url=EUROSTAT_API_BASE,
        timeout=DEFAULT_TIMEOUT,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        params={"format": "JSON", "lang": "EN"},
        headers={"Accept": "application/json"},
    )


async def fetch_dataset(
    client: httpx.AsyncClient,
    dataset: str,
    *,
    geo_level: str = "nuts2",
    filters: Optional[dict[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Fetch one Eurostat dataset as a JSON-stat 2.0 payload.

    Args:
        client: Shared async HTTP client.
        dataset: Eurostat dataset code (e.g. ``"nama_10r_2gdp"``).
        geo_level: ``geoLevel`` filter selecting the NUTS granularity
            (``"nuts2"``). Restricts ``geo`` to codes at that level.
        filters: Non-geo/non-time dimensions pinned to a single category
            (e.g. ``{"unit": "EUR_HAB"}``). ``time`` is intentionally omitted so
            the whole year range comes back in one call.

    Returns:
        The parsed JSON-stat dict, or ``None`` when the endpoint answers with a
        non-JSON error body.
    """
    params: dict[str, str] = {"geoLevel": geo_level}
    params.update(filters or {})
    resp = await client.get(f"data/{dataset}", params=params)
    resp.raise_for_status()
    try:
        payload = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _strides(sizes: list[int]) -> list[int]:
    """Row-major strides for a JSON-stat ``size`` vector (last dim fastest)."""
    strides = [1] * len(sizes)
    for k in range(len(sizes) - 2, -1, -1):
        strides[k] = strides[k + 1] * sizes[k + 1]
    return strides


def _invert_index(category: dict[str, Any]) -> dict[int, str]:
    """Invert a JSON-stat ``category.index`` (code→pos) into pos→code."""
    return {int(pos): code for code, pos in category.get("index", {}).items()}


def parse_jsonstat(
    payload: dict[str, Any], *, level: int = 2
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flatten a JSON-stat payload into long ``{region, year, value}`` rows.

    Decodes each flat value index into its per-dimension positions using
    row-major strides, then keeps only cells where **every** dimension other than
    ``geo`` and ``time`` sits at index 0 (its pinned/first category), so an
    under-pinned dimension collapses deterministically to its first category
    rather than duplicating region-years. Only ``geo`` codes at the requested
    NUTS ``level`` (code length ``level + 2``) are kept; null values are dropped.

    Args:
        payload: JSON-stat 2.0 dict from :func:`fetch_dataset`.
        level: NUTS level whose codes to keep (2 → 4-character codes).

    Returns:
        Tuple ``(rows, meta)`` where ``rows`` is a list of
        ``{"region", "year", "value"}`` dicts and ``meta`` carries
        ``units`` / ``frequency`` / ``source_label`` extracted from the payload.
    """
    dims: list[str] = payload.get("id", [])
    sizes: list[int] = payload.get("size", [])
    if not dims or "geo" not in dims or "time" not in dims:
        return [], {"units": "", "frequency": "", "source_label": payload.get("label", "")}

    strides = _strides(sizes)
    stride_of = dict(zip(dims, strides))
    size_of = dict(zip(dims, sizes))
    dimension = payload.get("dimension", {})
    geo_pos = _invert_index(dimension.get("geo", {}).get("category", {}))
    time_pos = _invert_index(dimension.get("time", {}).get("category", {}))
    other_dims = [d for d in dims if d not in ("geo", "time")]
    code_len = level + 2

    values = payload.get("value", {})
    items = values.items() if isinstance(values, dict) else enumerate(values)

    rows: list[dict[str, Any]] = []
    for raw_index, value in items:
        if value is None:
            continue
        index = int(raw_index)
        # Skip cells where any pinned dimension is not at its first category.
        if any((index // stride_of[d]) % size_of[d] != 0 for d in other_dims):
            continue
        region = geo_pos.get((index // stride_of["geo"]) % size_of["geo"])
        if not region or len(region) != code_len:
            continue
        time_code = time_pos.get((index // stride_of["time"]) % size_of["time"], "")
        try:
            year = int(str(time_code)[:4])
        except (ValueError, TypeError):
            continue
        try:
            numeric = float(value)
        except (ValueError, TypeError):
            continue
        rows.append({"region": region, "year": year, "value": numeric})

    return rows, _extract_meta(payload)


def _extract_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull units / frequency / dataset label out of a JSON-stat payload."""
    dimension = payload.get("dimension", {})

    def _first_label(dim: str) -> str:
        category = dimension.get(dim, {}).get("category", {})
        labels = category.get("label", {})
        index = category.get("index", {})
        if not index:
            return ""
        first_code = next(iter(index))
        return str(labels.get(first_code, first_code))

    units = ""
    for unit_dim in ("unit", "currency"):
        if unit_dim in dimension:
            units = _first_label(unit_dim)
            break
    return {
        "units": units,
        "frequency": _first_label("freq"),
        "source_label": str(payload.get("label", "")),
    }


def regions_from_geojson(path: str | Path, *, level: int = 2) -> list[dict[str, Any]]:
    """Build ``eurostat_regions`` rows from the bundled GISCO NUTS GeoJSON.

    Uses ``NAME_LATN`` for the region name (the reliable Latin-alphabet label;
    ``NAME_ENGL`` at NUTS-2 holds the country name, which is reused for
    ``country_name``). ``nuts1_id`` is the parent NUTS-1 code (first 3 chars).

    Args:
        path: Path to the GISCO ``NUTS_RG_*_LEVL_{level}`` GeoJSON.
        level: NUTS level to keep (2).

    Returns:
        One ``{"id", "name", "country_code", "country_name", "nuts1_id", "level"}``
        dict per region, sorted by code.
    """
    geojson = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("LEVL_CODE") != level:
            continue
        code = props.get("NUTS_ID")
        if not code:
            continue
        records.append(
            {
                "id": code,
                "name": props.get("NAME_LATN") or props.get("NUTS_NAME") or code,
                "country_code": props.get("CNTR_CODE") or code[:2],
                "country_name": props.get("NAME_ENGL") or props.get("CNTR_CODE") or code[:2],
                "nuts1_id": code[:3],
                "level": int(props.get("LEVL_CODE", level)),
            }
        )
    records.sort(key=lambda r: r["id"])
    return records


def synthesize_notes(
    meta: dict[str, Any], dataset: str, filters: dict[str, str], min_year: int, max_year: int
) -> str:
    """Compose a human-readable description from the dataset metadata + filters."""
    label = meta.get("source_label") or dataset
    units = meta.get("units") or "n/a"
    frequency = meta.get("frequency") or "Annual"
    filter_text = ", ".join(f"{k}={v}" for k, v in filters.items()) or "none"
    return (
        f"{label}. Units: {units}. Frequency: {frequency}. Coverage: {min_year}-{max_year}. "
        f"Eurostat dataset '{dataset}' (filters: {filter_text}), by NUTS-2 region. "
        f"Source: Eurostat."
    )


async def healthcheck(client: httpx.AsyncClient) -> bool:
    """Return ``True`` if Eurostat returns a JSON-stat panel for a known dataset."""
    try:
        payload = await fetch_dataset(
            client, _HEALTHCHECK_DATASET, filters={"unit": "MIO_EUR", "time": "2021"}
        )
        return bool(payload and payload.get("value"))
    except Exception:
        logger.exception("An error occured while testing connection to Eurostat API")
        return False
