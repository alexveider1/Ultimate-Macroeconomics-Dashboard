"""Unit tests for the async Eurostat (JSON-stat) httpx client + NUTS helpers.

All HTTP is faked with ``httpx.MockTransport`` so the tests are offline and
deterministic; async coroutines are driven with ``asyncio.run`` to avoid a
pytest-asyncio dependency (mirrors ``test_fred_client.py`` / ``test_wb_client.py``).
"""

import asyncio
import json

import httpx
from src.utils import eurostat_client


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=eurostat_client.EUROSTAT_API_BASE,
        params={"format": "JSON", "lang": "EN"},
    )


def _run(handler, coro_factory):
    async def _main():
        async with _client(handler) as client:
            return await coro_factory(client)

    return asyncio.run(_main())


def _jsonstat(dims, sizes, index, value, *, label="Test dataset", extra_dim=None):
    """Build a minimal JSON-stat 2.0 payload.

    ``index`` maps each dimension id to its ``{code: pos}`` dict; ``value`` is the
    flat value map (dict) or list. ``extra_dim`` optionally injects a category
    dict for a pinned dimension (e.g. unit) so meta extraction has labels.
    """
    dimension = {}
    for dim in dims:
        cats = index.get(dim, {"_": 0})
        dimension[dim] = {"category": {"index": cats, "label": {c: f"{dim}:{c}" for c in cats}}}
    if extra_dim:
        dimension.update(extra_dim)
    return {"id": dims, "size": sizes, "value": value, "label": label, "dimension": dimension}


# --- pure helpers -----------------------------------------------------------


def test_strides_row_major():
    assert eurostat_client._strides([1, 1, 2, 2]) == [4, 4, 2, 1]
    assert eurostat_client._strides([2, 3]) == [3, 1]


def test_parse_jsonstat_simple_panel_no_duplicates():
    """geo × time flatten with unit pinned (size 1) → one row per region-year."""
    payload = _jsonstat(
        dims=["freq", "unit", "geo", "time"],
        sizes=[1, 1, 2, 2],
        index={
            "geo": {"BE10": 0, "BE21": 1},
            "time": {"2020": 0, "2021": 1},
            "unit": {"EUR_HAB": 0},
            "freq": {"A": 0},
        },
        value={"0": 100.0, "1": 110.0, "2": 200.0, "3": 210.0},
    )
    rows, meta = eurostat_client.parse_jsonstat(payload, level=2)
    keyed = {(r["region"], r["year"]): r["value"] for r in rows}
    assert keyed == {
        ("BE10", 2020): 100.0,
        ("BE10", 2021): 110.0,
        ("BE21", 2020): 200.0,
        ("BE21", 2021): 210.0,
    }
    assert len(rows) == len(keyed)  # no duplicate (region, year)
    assert meta["units"] == "unit:EUR_HAB"
    assert meta["frequency"] == "freq:A"
    assert meta["source_label"] == "Test dataset"


def test_parse_jsonstat_underpinned_dimension_takes_first_category():
    """An unpinned dim (size>1) collapses to index 0 — no region-year duplication."""
    payload = _jsonstat(
        dims=["freq", "sex", "geo", "time"],
        sizes=[1, 2, 2, 2],
        index={
            "geo": {"BE10": 0, "BE21": 1},
            "time": {"2020": 0, "2021": 1},
            "sex": {"T": 0, "M": 1},
            "freq": {"A": 0},
        },
        # sex=0 → flat 0..3, sex=1 → flat 4..7 (must be ignored)
        value={str(i): float(i) for i in range(8)},
    )
    rows, _ = eurostat_client.parse_jsonstat(payload, level=2)
    keyed = {(r["region"], r["year"]): r["value"] for r in rows}
    assert keyed == {
        ("BE10", 2020): 0.0,
        ("BE10", 2021): 1.0,
        ("BE21", 2020): 2.0,
        ("BE21", 2021): 3.0,
    }
    assert len(rows) == 4  # sex=M cells (4..7) dropped, no duplicates


def test_parse_jsonstat_accepts_dense_value_list():
    payload = _jsonstat(
        dims=["freq", "unit", "geo", "time"],
        sizes=[1, 1, 2, 2],
        index={
            "geo": {"BE10": 0, "BE21": 1},
            "time": {"2020": 0, "2021": 1},
            "unit": {"PC": 0},
            "freq": {"A": 0},
        },
        value=[100.0, 110.0, 200.0, 210.0],
    )
    rows, _ = eurostat_client.parse_jsonstat(payload, level=2)
    assert {(r["region"], r["year"]) for r in rows} == {
        ("BE10", 2020),
        ("BE10", 2021),
        ("BE21", 2020),
        ("BE21", 2021),
    }


def test_parse_jsonstat_drops_nulls_and_wrong_level_codes():
    """Null values and geo codes not of the requested level (len 4) are dropped."""
    payload = _jsonstat(
        dims=["geo", "time"],
        sizes=[3, 1],
        index={
            "geo": {"BE10": 0, "BE": 1, "BE21": 2},  # BE is len-2 (NUTS-0)
            "time": {"2020": 0},
        },
        value={"0": 5.0, "1": 9.0, "2": None},  # BE10 kept, BE wrong-level, BE21 null
    )
    rows, _ = eurostat_client.parse_jsonstat(payload, level=2)
    assert rows == [{"region": "BE10", "year": 2020, "value": 5.0}]


def test_regions_from_geojson_uses_latn_name_and_country(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "properties": {
                    "NUTS_ID": "DE21",
                    "LEVL_CODE": 2,
                    "CNTR_CODE": "DE",
                    "NAME_LATN": "Oberbayern",
                    "NAME_ENGL": "Germany",
                }
            },
            {
                "properties": {
                    "NUTS_ID": "DE",  # NUTS-0 country feature — must be skipped
                    "LEVL_CODE": 0,
                    "CNTR_CODE": "DE",
                    "NAME_LATN": "Deutschland",
                    "NAME_ENGL": "Germany",
                }
            },
        ],
    }
    path = tmp_path / "nuts.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    records = eurostat_client.regions_from_geojson(path, level=2)
    assert len(records) == 1
    rec = records[0]
    assert rec == {
        "id": "DE21",
        "name": "Oberbayern",  # NAME_LATN, not the country in NAME_ENGL
        "country_code": "DE",
        "country_name": "Germany",
        "nuts1_id": "DE2",
        "level": 2,
    }


def test_synthesize_notes_includes_dataset_units_and_source():
    note = eurostat_client.synthesize_notes(
        {
            "source_label": "GDP by NUTS 2 region",
            "units": "Euro per inhabitant",
            "frequency": "Annual",
        },
        "nama_10r_2gdp",
        {"unit": "EUR_HAB"},
        2000,
        2024,
    )
    assert "GDP by NUTS 2 region" in note
    assert "Euro per inhabitant" in note
    assert "nama_10r_2gdp" in note
    assert "2000-2024" in note
    assert "Eurostat" in note


# --- HTTP coroutines --------------------------------------------------------


def test_fetch_dataset_passes_geolevel_and_filters():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/data/nama_10r_2gdp")
        assert request.url.params["geoLevel"] == "nuts2"
        assert request.url.params["unit"] == "MIO_EUR"
        assert request.url.params["format"] == "JSON"
        return httpx.Response(200, json={"id": ["geo", "time"], "size": [1, 1], "value": {}})

    payload = _run(
        handler,
        lambda c: eurostat_client.fetch_dataset(
            c, "nama_10r_2gdp", geo_level="nuts2", filters={"unit": "MIO_EUR"}
        ),
    )
    assert payload is not None
    assert payload["id"] == ["geo", "time"]


def test_fetch_dataset_returns_none_on_non_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<error>bad request</error>")

    assert _run(handler, lambda c: eurostat_client.fetch_dataset(c, "bogus")) is None


def test_healthcheck_true_and_false():
    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": ["geo", "time"], "size": [1, 1], "value": {"0": 1.0}}
        )

    def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    assert _run(ok_handler, eurostat_client.healthcheck) is True
    assert _run(bad_handler, eurostat_client.healthcheck) is False
