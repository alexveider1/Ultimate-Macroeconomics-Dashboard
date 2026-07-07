"""Unit tests for the multimodal upload dispatcher (pure logic, no network)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import multimodal


def _inputs(
    handler: Any = None, whisper: Any = None, whisper_enabled: bool = True
) -> multimodal.UploadInputs:
    transport = httpx.MockTransport(handler or (lambda r: httpx.Response(404)))
    return multimodal.UploadInputs(
        http_client=httpx.AsyncClient(transport=transport),
        docling_url="http://docling:8006",
        docling_timeout=10.0,
        whisper_client=whisper,
        whisper_model="whisper-1",
        whisper_enabled=whisper_enabled,
    )


def _run(
    inputs: multimodal.UploadInputs, files: list[tuple[str, str | None, bytes]]
) -> multimodal.ProcessedUploads:
    return asyncio.run(multimodal.process_uploads(inputs, files))


def test_text_file_decoded_into_block() -> None:
    result = _run(_inputs(), [("a.txt", "text/plain", b"line one")])
    assert result.images == []
    assert any("line one" in b for b in result.text_blocks)


def test_image_encoded_as_data_uri() -> None:
    result = _run(_inputs(), [("p.jpg", "image/jpeg", b"jpegdata")])
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/jpeg;base64,")
    assert any("p.jpg" in b for b in result.text_blocks)


def test_empty_file_skipped() -> None:
    result = _run(_inputs(), [("x.txt", "text/plain", b"")])
    assert any("empty" in b.lower() for b in result.text_blocks)


def test_unsupported_type_noted() -> None:
    result = _run(_inputs(), [("archive.zip", "application/zip", b"PK")])
    assert any("Unsupported" in b for b in result.text_blocks)


def test_document_conversion_failure_degrades() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="vlm down")

    result = _run(_inputs(handler=boom), [("d.pdf", "application/pdf", b"%PDF")])
    assert any("Could not read document" in b for b in result.text_blocks)


def test_audio_disabled_without_client() -> None:
    result = _run(
        _inputs(whisper=None, whisper_enabled=False),
        [("m.wav", "audio/wav", b"RIFF")],
    )
    assert any("not configured" in b for b in result.text_blocks)


def test_augment_message_joins_blocks() -> None:
    out = multimodal.augment_message("hello", ["--- Attached ---\nbody"])
    assert out.startswith("hello")
    assert "body" in out


def test_augment_message_empty_user_message() -> None:
    out = multimodal.augment_message("", ["block-a", "block-b"])
    assert out == "block-a\n\nblock-b"
