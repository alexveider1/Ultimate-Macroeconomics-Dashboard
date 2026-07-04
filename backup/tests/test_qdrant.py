"""Tests for the Qdrant full-snapshot create/download/delete flow (mocked HTTP)."""

from pathlib import Path
from typing import Any

import backup_qdrant
import httpx
import pytest

SNAPSHOT = "full-snapshot-2026-07-04.snapshot"


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if request.method == "POST" and path == "/snapshots":
        return httpx.Response(200, json={"result": {"name": SNAPSHOT}})
    if request.method == "GET" and path == f"/snapshots/{SNAPSHOT}":
        return httpx.Response(200, content=b"QDRANTDATA")
    if request.method == "DELETE" and path == f"/snapshots/{SNAPSHOT}":
        return httpx.Response(200, json={"result": True})
    return httpx.Response(404)


def _install_mock(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording)
    real_cls = httpx.Client

    def fake_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.setdefault("transport", transport)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(backup_qdrant.httpx, "Client", fake_client)
    return seen


def test_snapshot_qdrant_downloads_and_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _install_mock(monkeypatch, _handler)
    out = backup_qdrant.snapshot_qdrant(
        base_url="http://vector_db:6333", api_key="secret", out_dir=tmp_path
    )
    assert out.name == SNAPSHOT
    assert out.read_bytes() == b"QDRANTDATA"
    methods = [(r.method, r.url.path) for r in seen]
    assert methods == [
        ("POST", "/snapshots"),
        ("GET", f"/snapshots/{SNAPSHOT}"),
        ("DELETE", f"/snapshots/{SNAPSHOT}"),
    ]
    # api-key header is attached on every call.
    assert all(r.headers.get("api-key") == "secret" for r in seen)


def test_snapshot_qdrant_survives_delete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(500)
        return _handler(request)

    _install_mock(monkeypatch, handler)
    # A failed server-side delete must not fail the backup.
    out = backup_qdrant.snapshot_qdrant(
        base_url="http://vector_db:6333", api_key="secret", out_dir=tmp_path
    )
    assert out.read_bytes() == b"QDRANTDATA"
