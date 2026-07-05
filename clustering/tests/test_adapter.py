"""Adapter tests: the clustering service forwards to Triton and reshapes the reply.

Triton is never contacted — ``main.infer_json`` is monkeypatched with a fake so
these run fast on CPU. They cover request forwarding, response reshaping, the
``/methods`` listing, and mapping a ``TritonError`` back onto its HTTP status.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
import main
import pytest
from triton_client import TritonError


def _base_request() -> dict[str, Any]:
    return {
        "method": "kmeans",
        "dataframe": [{"a": float(i), "b": float(i * 2)} for i in range(6)],
        "k": 2,
        "reduction_method": "pca",
        "output_dim": 2,
    }


def _fake_result(request: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(r) for r in request["dataframe"]]
    for i, row in enumerate(rows):
        row["cluster"] = i % 2
        row["__viz_x"] = float(i)
        row["__viz_y"] = float(-i)
    return {
        "method_used": request["method"],
        "dataframe": rows,
        "visualization_mode": "pca",
        "visualization_columns": ["__viz_x", "__viz_y"],
        "visualization_labels": ["PC 1", "PC 2"],
    }


def test_cluster_forwards_and_reshapes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_infer(_client: Any, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["model_name"] = model_name
        captured["payload"] = payload
        return _fake_result(payload)

    monkeypatch.setattr(main, "infer_json", fake_infer)

    with TestClient(main.app) as client:
        response = client.post("/cluster", json=_base_request())

    assert response.status_code == 200
    body = response.json()
    assert captured["model_name"] == "cluster"
    assert body["method_used"] == "kmeans"
    assert body["visualization_columns"] == ["__viz_x", "__viz_y"]
    assert all("cluster" in row for row in body["dataframe"])
    # The full validated request (incl. defaulted knobs) is forwarded.
    assert captured["payload"]["k"] == 2
    assert captured["payload"]["reduction_method"] == "pca"


def test_cluster_maps_triton_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_infer(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise TritonError(400, "k cannot be greater than the number of rows")

    monkeypatch.setattr(main, "infer_json", raising_infer)

    with TestClient(main.app) as client:
        response = client.post("/cluster", json=_base_request())

    assert response.status_code == 400
    assert "cannot be greater" in response.json()["detail"]


def test_cluster_rejects_empty_dataframe() -> None:
    request = _base_request()
    request["dataframe"] = []
    with TestClient(main.app) as client:
        response = client.post("/cluster", json=request)
    # Pydantic validation fires in the adapter before any Triton call.
    assert response.status_code == 422


def test_methods_endpoint_lists_algorithms() -> None:
    with TestClient(main.app) as client:
        payload = client.get("/methods").json()
    assert "kmeans" in payload["available_methods"]
    assert "hdbscan" in payload["available_methods"]
    assert set(payload["available_reductions"]) == {"tsne", "pca", "umap", "kpca"}
