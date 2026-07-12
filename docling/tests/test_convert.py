"""Endpoint tests for the docling service.

Importing ``main`` imports the real docling API surface (via ``converter``), so
these tests double as a guard that the pinned docling version's imports resolve
and ``build_converter`` runs offline. The actual conversion is stubbed with a
fake converter on ``app.state`` — no cloud VLM, no network, no GPU.
"""

from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient
import main


class _FakeDocument:
    def export_to_markdown(self) -> str:
        return "# Heading\n\nconverted body"


class _FakeResult:
    document = _FakeDocument()


class _FakeConverter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def convert(self, stream: Any) -> _FakeResult:
        self.calls.append(stream.name)
        return _FakeResult()


def _client(converter: Any) -> TestClient:
    client = TestClient(main.app)
    client.__enter__()  # run lifespan (builds the real converter), then override it
    main.app.state.converter = converter
    return client


def test_health_and_formats() -> None:
    client = _client(_FakeConverter())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/formats").json() == {"formats": [".docx", ".pdf", ".pptx", ".xlsx"]}
    client.__exit__(None, None, None)


def test_convert_office_document_returns_markdown() -> None:
    fake = _FakeConverter()
    client = _client(fake)
    resp = client.post(
        "/convert",
        files={
            "file": ("report.docx", io.BytesIO(b"PK\x03\x04payload"), "application/octet-stream")
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["markdown"] == "# Heading\n\nconverted body"
    assert body["filename"] == "report.docx"
    assert body["format"] == "docx"
    assert fake.calls == ["report.docx"]
    client.__exit__(None, None, None)


def test_convert_rejects_unsupported_extension() -> None:
    client = _client(_FakeConverter())
    resp = client.post(
        "/convert",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 415
    client.__exit__(None, None, None)


def test_convert_rejects_empty_upload() -> None:
    client = _client(_FakeConverter())
    resp = client.post(
        "/convert",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert resp.status_code == 400
    client.__exit__(None, None, None)


def test_convert_maps_converter_failure_to_502() -> None:
    class _Boom:
        def convert(self, stream: Any) -> Any:
            raise RuntimeError("VLM unreachable")

    client = _client(_Boom())
    resp = client.post(
        "/convert",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 body"), "application/pdf")},
    )
    assert resp.status_code == 502
    client.__exit__(None, None, None)
