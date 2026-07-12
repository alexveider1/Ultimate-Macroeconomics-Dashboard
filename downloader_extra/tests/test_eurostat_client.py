"""Unit tests for the trimmed Eurostat JSON-stat client used by on-demand ingest.

Guards the duplicated ``parse_jsonstat`` copy from drifting away from the
canonical ``downloader_general/src/utils/eurostat_client.py`` implementation.
"""

import asyncio

import eurostat_client
import httpx


def _jsonstat(dims, sizes, index, value, *, label="Test dataset"):
    dimension = {
        dim: {
            "category": {
                "index": index.get(dim, {"_": 0}),
                "label": {c: f"{dim}:{c}" for c in index.get(dim, {"_": 0})},
            }
        }
        for dim in dims
    }
    return {"id": dims, "size": sizes, "value": value, "label": label, "dimension": dimension}


def test_parse_jsonstat_flattens_region_year_without_duplicates():
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
    assert meta["units"] == "unit:EUR_HAB"


def test_parse_jsonstat_drops_nulls_and_wrong_level():
    payload = _jsonstat(
        dims=["geo", "time"],
        sizes=[3, 1],
        index={"geo": {"BE10": 0, "BE": 1, "BE21": 2}, "time": {"2020": 0}},
        value={"0": 5.0, "1": 9.0, "2": None},
    )
    rows, _ = eurostat_client.parse_jsonstat(payload, level=2)
    assert rows == [{"region": "BE10", "year": 2020, "value": 5.0}]


def test_fetch_dataset_passes_params():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/data/nama_10r_2gdp")
        assert request.url.params["geoLevel"] == "nuts2"
        assert request.url.params["unit"] == "EUR_HAB"
        return httpx.Response(200, json={"id": ["geo", "time"], "size": [1, 1], "value": {}})

    async def _main():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=eurostat_client.EUROSTAT_API_BASE,
            params={"format": "JSON", "lang": "EN"},
        ) as client:
            return await eurostat_client.fetch_dataset(
                client, "nama_10r_2gdp", filters={"unit": "EUR_HAB"}
            )

    payload = asyncio.run(_main())
    assert payload is not None
