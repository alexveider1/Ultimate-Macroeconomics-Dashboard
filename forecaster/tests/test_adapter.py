"""Adapter tests: the forecaster forwards to Triton and reshapes the reply.

Triton itself is never contacted — ``main.infer_json`` is monkeypatched with a
fake so these run fast on CPU. They cover request translation (cleaned series
sent to the right model), response reshaping, model-toggle gating, and mapping a
``TritonError`` back onto the matching HTTP status.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
import main
import pytest
from triton_client import TritonError


def _base_request(model_type: str = "arima") -> dict[str, Any]:
    return {
        "model_type": model_type,
        "dates": [f"2020-{m:02d}-01" for m in range(1, 13)],
        "values": [float(v) for v in range(12)],
        "n_prev": 12,
        "n_predict": 3,
        "alpha": 0.05,
        "model_params": {"p": 1, "d": 1, "q": 1},
    }


def _fake_result(n: int) -> dict[str, list[Any]]:
    return {
        "ds": [f"2021-{i:02d}-01 00:00:00" for i in range(1, n + 1)],
        "yhat": [float(i) for i in range(n)],
        "yhat_lower": [float(i) - 1.0 for i in range(n)],
        "yhat_upper": [float(i) + 1.0 for i in range(n)],
    }


def test_predict_forwards_to_named_model_and_reshapes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_infer(_client: Any, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["model_name"] = model_name
        captured["payload"] = payload
        return _fake_result(payload["n_predict"])

    monkeypatch.setattr(main, "infer_json", fake_infer)

    with TestClient(main.app) as client:
        response = client.post("/predict", json=_base_request("arima"))

    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "arima"
    assert len(body["forecast"]) == 3
    assert body["forecast"][0].keys() == {"ds", "yhat", "yhat_lower", "yhat_upper"}

    # The cleaned history + params are forwarded to the arima-specific model.
    assert captured["model_name"] == "forecast_arima"
    assert captured["payload"]["n_predict"] == 3
    assert captured["payload"]["model_params"] == {"p": 1, "d": 1, "q": 1}
    assert len(captured["payload"]["dates"]) == len(captured["payload"]["values"]) == 12


def test_predict_rejects_bad_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "infer_json", lambda *_a, **_k: _fake_result(3))
    request = _base_request()
    request["dates"] = ["not-a-date"] * 12

    with TestClient(main.app) as client:
        response = client.post("/predict", json=request)

    assert response.status_code == 400
    assert "Invalid date format" in response.json()["detail"]


def test_predict_gates_disabled_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest disables Prophet + Chronos; the adapter must 400 before Triton.
    called = False

    def fake_infer(*_a: Any, **_k: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return _fake_result(3)

    monkeypatch.setattr(main, "infer_json", fake_infer)

    with TestClient(main.app) as client:
        response = client.post("/predict", json=_base_request("prophet"))

    assert response.status_code == 400
    assert "disabled" in response.json()["detail"]
    assert called is False


def test_predict_maps_triton_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_infer(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise TritonError(400, "history too short")

    monkeypatch.setattr(main, "infer_json", raising_infer)

    with TestClient(main.app) as client:
        response = client.post("/predict", json=_base_request("xgboost"))

    assert response.status_code == 400
    assert response.json()["detail"] == "history too short"


def test_models_lists_enabled_only() -> None:
    with TestClient(main.app) as client:
        models = client.get("/models").json()["available_models"]

    assert "arima" in models  # ARIMA family enabled
    assert "moving_average" in models and "xgboost" in models  # always on
    assert not any(m.startswith("chronos") for m in models)  # Chronos disabled
    assert "prophet" not in models  # Prophet disabled
