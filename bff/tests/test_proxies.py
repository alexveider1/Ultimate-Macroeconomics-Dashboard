"""Tests for the forecaster / clustering / agent proxy routers + helpers.

A single ``httpx.MockTransport`` stands in for the three downstream services, so
these run with no network and no live containers.
"""

from collections.abc import Iterator

import clients
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest
from routers import agent, cluster, forecast


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/predict":
        return httpx.Response(200, json={"forecast": [1.0, 2.0], "model_type": "prophet"})
    if path == "/cluster":
        return httpx.Response(200, json={"labels": [0, 1, 0]})
    if path == "/models":
        return httpx.Response(200, json={"models": ["gpt-5.4", "gpt-5.4-mini", ""]})
    if path == "/plots/interpret":
        return httpx.Response(200, json={"description": "a line goes up", "mode": "creative"})
    if path == "/chat/stream":
        return httpx.Response(
            200,
            content=b'data: {"type": "final", "answer": "hi"}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    if path == "/boom":
        return httpx.Response(500, text="kaboom")
    raise httpx.ConnectError("refused")


@pytest.fixture()
def proxy_client() -> Iterator[TestClient]:
    app = FastAPI()
    for module in (forecast, cluster, agent):
        app.include_router(module.router)

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    app.state.forecaster_url = "http://forecaster:8001"
    app.state.clustering_url = "http://clustering:8002"
    app.state.agent_url = "http://agent:8000"

    with TestClient(app) as test_client:
        yield test_client


def test_resolve_base_url_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORECASTER_BASE_URL", "http://localhost:9001/")
    assert clients.resolve_base_url("FORECASTER_BASE_URL", "http://forecaster:8001") == (
        "http://localhost:9001"
    )


def test_resolve_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    assert clients.resolve_base_url("AGENT_BASE_URL", "http://agent:8000/") == "http://agent:8000"


def test_downstream_error_status() -> None:
    response = httpx.Response(503, text="down")
    exc = httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://x"), response=response)
    mapped = clients._downstream_error("agent", exc)
    assert isinstance(mapped, HTTPException)
    assert mapped.status_code == 502
    assert "503" in mapped.detail


def test_downstream_error_transport() -> None:
    mapped = clients._downstream_error("agent", httpx.ConnectError("refused"))
    assert mapped.status_code == 503


def test_forecast_proxy(proxy_client: TestClient) -> None:
    body = {
        "model_type": "prophet",
        "dates": ["2020-01-01", "2020-02-01"],
        "values": [1.0, 2.0],
        "n_prev": 2,
        "n_predict": 1,
    }
    response = proxy_client.post("/forecast", json=body)
    assert response.status_code == 200
    assert response.json()["forecast"] == [1.0, 2.0]


def test_cluster_proxy_forwards_extras(proxy_client: TestClient) -> None:
    body = {
        "method": "kmeans",
        "dataframe": [{"a": 1.0}, {"a": 2.0}],
        "feature_columns": ["a"],
        "k": 2,  # extra tunable, forwarded verbatim.
    }
    response = proxy_client.post("/cluster", json=body)
    assert response.status_code == 200
    assert response.json()["labels"] == [0, 1, 0]


def test_agent_models_filters_blanks(proxy_client: TestClient) -> None:
    response = proxy_client.get("/agent/models")
    assert response.status_code == 200
    assert response.json()["models"] == ["gpt-5.4", "gpt-5.4-mini"]


def test_agent_plot_interpret(proxy_client: TestClient) -> None:
    response = proxy_client.post(
        "/agent/plots/interpret", json={"image_base64": "eA==", "mode": "creative"}
    )
    assert response.status_code == 200
    assert response.json()["description"] == "a line goes up"


def test_agent_chat_stream_relays_sse(proxy_client: TestClient) -> None:
    response = proxy_client.post("/agent/chat/stream", json={"user_message": "hi"})
    assert response.status_code == 200
    assert 'data: {"type": "final"' in response.text
