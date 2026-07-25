"""Unit tests for the async World Bank httpx client.

All HTTP is faked with ``httpx.MockTransport`` so the tests are offline and
deterministic; async coroutines are driven with ``asyncio.run`` to avoid a
pytest-asyncio dependency.
"""

import asyncio

import httpx
import pytest
from src.utils import wb_client


@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear the module-level /country caches around every test."""
    wb_client._raw_countries_cache = None
    wb_client._aggregate_codes_cache = None
    yield
    wb_client._raw_countries_cache = None
    wb_client._aggregate_codes_cache = None


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(handler, coro_factory):
    async def _main():
        async with _client(handler) as client:
            return await coro_factory(client)

    return asyncio.run(_main())


_COUNTRIES = [
    {
        "id": "USA",
        "name": "United States",
        "region": {"id": "NAC", "value": "North America "},
        "adminregion": {"id": "", "value": ""},
        "lendingType": {"id": "LNX", "value": "Not classified"},
        "incomeLevel": {"id": "HIC", "value": "High income"},
        "capitalCity": "Washington D.C. ",
        "longitude": "-77.032",
        "latitude": "38.8895",
    },
    {
        "id": "WLD",
        "name": "World",
        "region": {"id": "NA", "value": "Aggregates"},
        "adminregion": {"id": "", "value": ""},
        "lendingType": {"id": "", "value": "Aggregates"},
        "incomeLevel": {"id": "NA", "value": "Aggregates"},
        "capitalCity": "",
        "longitude": "",
        "latitude": "",
    },
]


def test_fetch_list_paginates():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(200, json=[{"page": 1, "pages": 2}, [{"id": "a"}]])
        return httpx.Response(200, json=[{"page": 2, "pages": 2}, [{"id": "b"}]])

    records = _run(handler, lambda c: wb_client._fetch_list(c, "source"))
    assert [r["id"] for r in records] == ["a", "b"]


def test_fetch_list_stops_on_xml_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<wb:error>nope</wb:error>")

    records = _run(handler, lambda c: wb_client._fetch_list(c, "source"))
    assert records == []


def test_fetch_sources_sends_databid():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["databid"] = request.url.params.get("databid")
        return httpx.Response(200, json=[{"pages": 1}, [{"id": "2", "name": "WDI"}]])

    sources = _run(handler, lambda c: wb_client.fetch_sources(c))
    assert seen["databid"] == "y"
    assert sources[0]["id"] == "2"


def test_fetch_countries_skips_aggregates_and_shapes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"pages": 1}, _COUNTRIES])

    countries = _run(handler, lambda c: wb_client.fetch_countries(c))
    assert len(countries) == 1
    usa = countries[0]
    assert usa["id"] == "USA"
    assert usa["value"] == "United States"
    assert usa["aggregate"] is False
    assert usa["region"] == {"id": "NAC", "value": "North America"}  # trimmed
    assert usa["capitalCity"] == "Washington D.C."  # trimmed
    assert usa["longitude"] == pytest.approx(-77.032)


def test_fetch_countries_can_keep_aggregates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"pages": 1}, _COUNTRIES])

    countries = _run(handler, lambda c: wb_client.fetch_countries(c, skip_aggregates=False))
    ids = {row["id"]: row["aggregate"] for row in countries}
    assert ids == {"USA": False, "WLD": True}
    # World coordinates are empty strings -> None
    world = next(row for row in countries if row["id"] == "WLD")
    assert world["longitude"] is None


def test_fetch_series_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        records = [{"id": "NY.GDP.MKTP.CD", "name": "GDP (current US$)", "unit": ""}]
        return httpx.Response(200, json=[{"pages": 1}, records])

    series = _run(handler, lambda c: wb_client.fetch_series(c, 2))
    assert series == [{"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"}]


def test_fetch_indicator_data_filters_and_keeps_nulls():
    data_records = [
        {"countryiso3code": "USA", "date": "2020", "value": 21.0},
        {"countryiso3code": "USA", "date": "2021", "value": None},  # null kept
        {"countryiso3code": "WLD", "date": "2020", "value": 84.0},  # aggregate dropped
        {"countryiso3code": "ZH", "date": "2020", "value": 1.0},  # 2-char dropped
        {"countryiso3code": "USA", "date": "n/a", "value": 9.0},  # bad date dropped
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/country"):
            return httpx.Response(200, json=[{"pages": 1}, _COUNTRIES])
        return httpx.Response(200, json=[{"pages": 1}, data_records])

    rows = _run(handler, lambda c: wb_client.fetch_indicator_data(c, "NY.GDP.MKTP.CD", 2))
    assert rows == [
        {"economy": "USA", "time": 2020, "value": 21.0},
        {"economy": "USA", "time": 2021, "value": None},
    ]


def test_fetch_series_metadata_parses_metatype():
    metatype = [
        {"id": "IndicatorName", "value": "GDP (current US$)"},
        {"id": "Unitofmeasure", "value": "US$"},
        {"id": "Source", "value": "WB national accounts"},
        {"id": "Developmentrelevance", "value": "matters"},
        {"id": "Limitationsandexceptions", "value": "caveats"},
        {"id": "Statisticalconceptandmethodology", "value": "method"},
    ]
    payload = {"source": [{"concept": [{"variable": [{"metatype": metatype}]}]}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    meta = _run(handler, lambda c: wb_client.fetch_series_metadata(c, "NY.GDP.MKTP.CD", 2))
    assert meta == {
        "indicator_name": "GDP (current US$)",
        "units": "US$",
        "source": "WB national accounts",
        "development_relevance": "matters",
        "limitations_and_exceptions": "caveats",
        "statistical_concept_and_methodology": "method",
    }


def test_fetch_series_metadata_returns_none_on_xml():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<wb:error>Data not found.</wb:error>")

    meta = _run(handler, lambda c: wb_client.fetch_series_metadata(c, "FAKE", 2))
    assert meta is None


def test_fetch_indicator_metadata_fallback_maps_fields():
    info = [
        {"pages": 1},
        [
            {
                "name": "GDP (current US$)",
                "unit": "",
                "source": {"id": "2", "value": "WDI"},
                "sourceNote": "note",
            }
        ],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=info)

    meta = _run(handler, lambda c: wb_client.fetch_indicator_metadata(c, "NY.GDP.MKTP.CD", 2))
    assert meta == {
        "indicator_name": "GDP (current US$)",
        "units": "",
        "source": "WDI",
        "development_relevance": "note",
        "limitations_and_exceptions": None,
        "statistical_concept_and_methodology": None,
    }


def test_healthcheck_true_then_false():
    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"pages": 1}, [{"id": "2"}]])

    def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    assert _run(ok_handler, lambda c: wb_client.healthcheck(c)) is True
    assert _run(bad_handler, lambda c: wb_client.healthcheck(c)) is False


def test_call_with_retries_succeeds_after_failures():
    attempts = {"n": 0}

    async def factory():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    result = asyncio.run(
        wb_client.call_with_retries(
            operation_name="t",
            request_coro_factory=factory,
            retry_delay_seconds=0,
            max_retries=3,
        )
    )
    assert result == "ok"
    assert attempts["n"] == 3


def test_call_with_retries_gives_up_returns_none():
    async def factory():
        raise httpx.ConnectError("boom")

    result = asyncio.run(
        wb_client.call_with_retries(
            operation_name="t",
            request_coro_factory=factory,
            retry_delay_seconds=0,
            max_retries=2,
        )
    )
    assert result is None


def test_compute_backoff_delay_grows_and_caps():
    # Zero base -> always zero (keeps retry unit tests fast).
    assert wb_client.compute_backoff_delay(0.0, 0) == 0.0
    assert wb_client.compute_backoff_delay(0.0, 5) == 0.0
    # Nominal doubles each attempt; jitter only ever adds time, never subtracts.
    for attempt in range(6):
        nominal = min(2.0 * (2**attempt), 60.0)
        delay = wb_client.compute_backoff_delay(2.0, attempt, max_delay=60.0, jitter=0.5)
        assert nominal <= delay <= nominal * 1.5
    # The nominal term is capped at max_delay regardless of attempt.
    assert wb_client.compute_backoff_delay(2.0, 20, max_delay=60.0, jitter=0.0) == 60.0


def test_fetch_list_retries_transient_page_failure(monkeypatch):
    # No real sleeping between page retries.
    monkeypatch.setattr(wb_client, "_PAGE_RETRY_BASE_DELAY", 0.0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # First two hits on page 1 return the WB rate-limit 400, then succeed.
        if calls["n"] < 3:
            return httpx.Response(400, text="Bad Request")
        return httpx.Response(200, json=[{"page": 1, "pages": 1}, [{"id": "a"}]])

    records = _run(handler, lambda c: wb_client._fetch_list(c, "source"))
    assert [r["id"] for r in records] == ["a"]
    assert calls["n"] == 3


def test_fetch_list_gives_up_after_page_retries_exhausted(monkeypatch):
    monkeypatch.setattr(wb_client, "_PAGE_RETRY_BASE_DELAY", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    with pytest.raises(httpx.HTTPStatusError):
        _run(handler, lambda c: wb_client._fetch_list(c, "source"))
