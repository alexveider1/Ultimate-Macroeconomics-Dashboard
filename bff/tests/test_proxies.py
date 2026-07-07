"""Tests for the forecaster / clustering / agent proxy routers + helpers.

A single ``httpx.MockTransport`` stands in for the three downstream services, so
these run with no network and no live containers.
"""

from collections.abc import Iterator
import json
from types import SimpleNamespace

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


# --- Multimodal chat endpoint -------------------------------------------------


class _FakeTranscriptions:
    async def create(self, model: str, file: tuple) -> SimpleNamespace:  # noqa: A002
        return SimpleNamespace(text="transcribed words")


class _FakeWhisper:
    audio = SimpleNamespace(transcriptions=_FakeTranscriptions())

    async def close(self) -> None:  # pragma: no cover - parity with real client
        return None


@pytest.fixture()
def mm_client() -> Iterator[tuple[TestClient, list[dict]]]:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/chat/stream":
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=b'data: {"type": "final", "answer": "ok"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        if path == "/convert":
            return httpx.Response(200, json={"markdown": "# Report\n\nquarterly gdp"})
        raise httpx.ConnectError("refused")

    app = FastAPI()
    app.include_router(agent.router)
    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.agent_url = "http://agent:8000"
    app.state.docling_url = "http://docling:8006"
    app.state.docling_timeout = 30.0
    app.state.whisper_client = _FakeWhisper()
    app.state.whisper_model = "whisper-1"
    app.state.whisper_enabled = True

    with TestClient(app) as test_client:
        yield test_client, captured


def test_multimodal_text_file_folds_into_message(
    mm_client: tuple[TestClient, list[dict]],
) -> None:
    client, captured = mm_client
    resp = client.post(
        "/agent/chat/multimodal",
        data={"user_message": "summarize", "session_id": "s1"},
        files=[("files", ("notes.md", b"hello world", "text/markdown"))],
    )
    assert resp.status_code == 200
    assert 'data: {"type": "final"' in resp.text
    body = captured[-1]
    assert body["session_id"] == "s1"
    assert "summarize" in body["user_message"]
    assert "hello world" in body["user_message"]
    assert body["images"] == []


def test_multimodal_image_becomes_data_uri(
    mm_client: tuple[TestClient, list[dict]],
) -> None:
    client, captured = mm_client
    resp = client.post(
        "/agent/chat/multimodal",
        data={"user_message": "what is this"},
        files=[("files", ("chart.png", b"\x89PNGbytes", "image/png"))],
    )
    assert resp.status_code == 200
    body = captured[-1]
    assert len(body["images"]) == 1
    assert body["images"][0].startswith("data:image/png;base64,")
    assert "chart.png" in body["user_message"]


def test_multimodal_audio_is_transcribed(
    mm_client: tuple[TestClient, list[dict]],
) -> None:
    client, captured = mm_client
    resp = client.post(
        "/agent/chat/multimodal",
        data={"user_message": "listen"},
        files=[("files", ("memo.mp3", b"ID3audio", "audio/mpeg"))],
    )
    assert resp.status_code == 200
    assert "transcribed words" in captured[-1]["user_message"]


def test_multimodal_document_uses_docling(
    mm_client: tuple[TestClient, list[dict]],
) -> None:
    client, captured = mm_client
    resp = client.post(
        "/agent/chat/multimodal",
        data={"user_message": "read this"},
        files=[("files", ("report.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert resp.status_code == 200
    assert "quarterly gdp" in captured[-1]["user_message"]


def test_multimodal_audio_disabled_notes_gracefully(
    mm_client: tuple[TestClient, list[dict]],
) -> None:
    client, captured = mm_client
    client.app.state.whisper_enabled = False
    resp = client.post(
        "/agent/chat/multimodal",
        data={"user_message": "listen"},
        files=[("files", ("memo.mp3", b"ID3audio", "audio/mpeg"))],
    )
    assert resp.status_code == 200
    assert "not configured" in captured[-1]["user_message"]
