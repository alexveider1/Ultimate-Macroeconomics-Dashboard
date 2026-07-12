"""Unit tests for the on-demand async World Bank httpx client.

HTTP is faked with ``httpx.MockTransport`` (offline, deterministic); coroutines
are driven with ``asyncio.run`` so no pytest-asyncio plugin is required.
"""

import asyncio

import httpx
import pytest
import wb_client


@pytest.fixture(autouse=True)
def _reset_caches():
    wb_client._aggregate_codes_cache = None
    yield
    wb_client._aggregate_codes_cache = None


_COUNTRIES = [
    {"id": "USA", "name": "United States", "region": {"id": "NAC", "value": "North America"}},
    {"id": "WLD", "name": "World", "region": {"id": "NA", "value": "Aggregates"}},
]


def _run(handler, coro_factory):
    async def _main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await coro_factory(client)

    return asyncio.run(_main())


def test_aggregate_codes_cached():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[{"pages": 1}, _COUNTRIES])

    async def main(client):
        first = await wb_client.aggregate_codes(client)
        second = await wb_client.aggregate_codes(client)
        return first, second

    first, second = _run(handler, main)
    assert first == {"WLD"}
    assert second == {"WLD"}
    assert calls["n"] == 1  # /country fetched once, then cached


def test_fetch_indicator_data_filters_and_keeps_nulls():
    data_records = [
        {"countryiso3code": "USA", "date": "2020", "value": 21.0},
        {"countryiso3code": "USA", "date": "2021", "value": None},  # null kept
        {"countryiso3code": "WLD", "date": "2020", "value": 84.0},  # aggregate dropped
        {"countryiso3code": "ZH", "date": "2020", "value": 1.0},  # 2-char dropped
        {"countryiso3code": "USA", "date": "bad", "value": 9.0},  # bad date dropped
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/country"):
            return httpx.Response(200, json=[{"pages": 1}, _COUNTRIES])
        return httpx.Response(200, json=[{"pages": 1}, data_records])

    rows = _run(handler, lambda c: wb_client.fetch_indicator_data(c, "NY.GDP.MKTP.CD", 2))
    assert rows == [
        {"economy": "USA", "year": 2020, "value": 21.0},
        {"economy": "USA", "year": 2021, "value": None},
    ]
