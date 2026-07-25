"""Unit tests for the async FRED / GeoFRED httpx client.

All HTTP is faked with ``httpx.MockTransport`` so the tests are offline and
deterministic; async coroutines are driven with ``asyncio.run`` to avoid a
pytest-asyncio dependency (mirrors ``test_wb_client.py``).
"""

import asyncio

import httpx
from src.utils import fred_client


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=fred_client.FRED_API_BASE,
        params={"api_key": "test", "file_type": "json"},
    )


def _run(handler, coro_factory):
    async def _main():
        async with _client(handler) as client:
            return await coro_factory(client)

    return asyncio.run(_main())


# --- pure helpers -----------------------------------------------------------


def test_fips_map_is_complete_and_unique():
    """50 states + DC, unique abbreviations, DC classified under South."""
    assert len(fred_client.FIPS_TO_STATE) == 51
    abbrevs = [v[0] for v in fred_client.FIPS_TO_STATE.values()]
    assert len(set(abbrevs)) == 51
    assert fred_client.FIPS_TO_STATE["06"][0] == "CA"
    assert fred_client.FIPS_TO_STATE["11"] == (
        "DC",
        "District of Columbia",
        "South",
        "South Atlantic",
    )


def test_parse_regional_panel_maps_by_fips_not_series_prefix():
    """A series id whose prefix is NOT the state abbrev still maps via FIPS code.

    ``MEDDAYONMARCA`` starts with 'ME' (Maine) but FIPS '06' is California — the
    parser must key on ``code``, not ``series_id``.
    """
    panel = {
        "2022-01-01": [
            {"region": "California", "code": "06", "value": 39.0, "series_id": "MEDDAYONMARCA"},
            {"region": "Texas", "code": "48", "value": 25.0, "series_id": "MEDDAYONMARTX"},
        ]
    }
    rows, names = fred_client.parse_regional_panel(panel)
    by_state = {r["state"]: r["value"] for r in rows}
    assert by_state == {"CA": 39.0, "TX": 25.0}
    assert all(r["year"] == 2022 for r in rows)
    assert names["06"] == "California"


def test_parse_regional_panel_drops_nulls_and_unknown_fips_and_dedups():
    panel = {
        "2020-01-01": [
            {"region": "California", "code": "06", "value": None, "series_id": "X"},  # null → drop
            {"region": "Nowhere", "code": "99", "value": 1.0, "series_id": "Y"},  # bad FIPS → drop
            {"region": "Texas", "code": "48", "value": 5.0, "series_id": "Z"},
            {"region": "Texas", "code": "48", "value": 7.0, "series_id": "Z"},  # dup (state,year)
        ]
    }
    rows, _ = fred_client.parse_regional_panel(panel)
    assert rows == [{"state": "TX", "year": 2020, "value": 5.0}]


def test_state_records_from_names_uses_fred_name_then_static_fallback():
    records = fred_client.state_records_from_names({"06": "Califas"})
    assert len(records) == 51
    by_id = {r["id"]: r for r in records}
    assert by_id["CA"]["name"] == "Califas"  # FRED name wins
    assert by_id["CA"]["region"] == "West"  # static enrichment
    assert by_id["TX"]["name"] == "Texas"  # static fallback when FRED name absent


def test_synthesize_notes_includes_title_and_units():
    note = fred_client.synthesize_notes(
        {"title": "Unemployment Rate", "units": "Percent", "frequency": "Monthly"}
    )
    assert "Unemployment Rate" in note
    assert "Percent" in note
    assert "FRED" in note


# --- HTTP coroutines --------------------------------------------------------


def test_fetch_series_group_returns_group():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/geofred/series/group"
        assert request.url.params["series_id"] == "CAUR"
        return httpx.Response(
            200,
            json={
                "series_group": {
                    "title": "Unemployment Rate",
                    "region_type": "state",
                    "series_group": "1224",
                    "season": "SA",
                    "units": "Percent",
                    "frequency": "Monthly",
                    "min_date": "1976-01-01",
                    "max_date": "2026-01-01",
                }
            },
        )

    group = _run(handler, lambda c: fred_client.fetch_series_group(c, "CAUR"))
    assert group is not None
    assert group["series_group"] == "1224"
    assert group["region_type"] == "state"


def test_fetch_series_group_none_when_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"series_group": {}})

    assert _run(handler, lambda c: fred_client.fetch_series_group(c, "BOGUS")) is None


def test_fetch_regional_panel_returns_data_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/geofred/regional/data"
        assert request.url.params["frequency"] == "a"
        return httpx.Response(
            200,
            json={
                "meta": {
                    "data": {
                        "2023-01-01": [
                            {
                                "region": "California",
                                "code": "06",
                                "value": 4.7,
                                "series_id": "CAUR",
                            }
                        ]
                    }
                }
            },
        )

    panel = _run(
        handler,
        lambda c: fred_client.fetch_regional_panel(
            c,
            series_group="1224",
            region_type="state",
            start_date="2023-01-01",
            end_date="2023-01-01",
            units="Percent",
            season="SA",
        ),
    )
    assert list(panel.keys()) == ["2023-01-01"]
    rows, _ = fred_client.parse_regional_panel(panel)
    assert rows == [{"state": "CA", "year": 2023, "value": 4.7}]


def test_healthcheck_true_and_false():
    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"series_group": {"series_group": "1224"}})

    def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    assert _run(ok_handler, fred_client.healthcheck) is True
    assert _run(bad_handler, fred_client.healthcheck) is False
