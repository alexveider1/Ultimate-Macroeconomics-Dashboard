"""Trimmed async Eurostat (JSON-stat) client for on-demand single-dataset ingestion.

A data-fetch-only copy of ``downloader_general/src/utils/eurostat_client.py``
(duplicated per service by design, like ``wb_client.py`` / ``fred_client.py``).
It fetches one Eurostat dataset's full NUTS-2 region×year panel in a single keyless
call and flattens the JSON-stat payload into long ``{region, year, value}`` rows via
row-major stride decoding, keeping only cells where every non-geo/non-time dimension
is at its first (pinned) category.
"""

import json
from typing import Any, Optional

import httpx

EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0"
DEFAULT_TIMEOUT = 60.0


def build_async_client() -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` configured for the Eurostat API."""
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
    """Fetch one Eurostat dataset as a JSON-stat 2.0 payload (keyless)."""
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

    Keeps only cells where every dimension other than ``geo`` and ``time`` is at
    index 0 (its pinned/first category) and only ``geo`` codes at the requested
    NUTS ``level`` (code length ``level + 2``); null values are dropped.
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
