"""Tests for the chat multimodal-upload dispatcher.

Audio (Whisper) and document (docling) conversion are monkeypatched so the
dispatch logic is exercised without any network / OpenAI / docling call.
"""

from __future__ import annotations

import base64

from core import multimodal
from core.multimodal import ProcessedUploads, augment_message, process_uploads
import pytest


def test_text_file_decoded_into_block() -> None:
    result = process_uploads([("notes.txt", "text/plain", b"hello world")])
    assert result.images == []
    assert any("hello world" in block for block in result.text_blocks)
    assert any("notes.txt" in block for block in result.text_blocks)


def test_image_becomes_data_uri() -> None:
    raw = b"\x89PNG\r\n\x1a\n fake"
    result = process_uploads([("chart.png", "image/png", raw)])
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/png;base64,")
    encoded = result.images[0].split(",", 1)[1]
    assert base64.b64decode(encoded) == raw
    assert any("chart.png" in block for block in result.text_blocks)


def test_empty_upload_skipped() -> None:
    result = process_uploads([("empty.pdf", "application/pdf", b"")])
    assert result.images == []
    assert any("empty" in block.lower() for block in result.text_blocks)


def test_unsupported_extension_noted() -> None:
    result = process_uploads([("archive.zip", "application/zip", b"PK\x03\x04")])
    assert any("Unsupported" in block for block in result.text_blocks)


def test_audio_transcribed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multimodal, "transcribe_audio", lambda *_: "spoken text")
    result = process_uploads([("voice.wav", "audio/wav", b"RIFFfake")])
    assert result.images == []
    assert any("spoken text" in block for block in result.text_blocks)


def test_audio_failure_degrades_to_note(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_: object) -> str:
        raise RuntimeError("no whisper")

    monkeypatch.setattr(multimodal, "transcribe_audio", _boom)
    result = process_uploads([("voice.mp3", "audio/mpeg", b"ID3fake")])
    assert any("Could not transcribe" in block for block in result.text_blocks)


def test_document_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multimodal, "convert_document", lambda *_: "# Title\n\nbody")
    result = process_uploads([("report.pdf", "application/pdf", b"%PDF-1.4")])
    assert any("# Title" in block for block in result.text_blocks)
    assert any("report.pdf" in block for block in result.text_blocks)


def test_document_failure_degrades_to_note(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_: object) -> str:
        raise RuntimeError("docling down")

    monkeypatch.setattr(multimodal, "convert_document", _boom)
    result = process_uploads([("report.docx", None, b"PK\x03\x04")])
    assert any("Could not read document" in block for block in result.text_blocks)


def test_augment_message_appends_blocks() -> None:
    merged = augment_message("summarize this", ["--- Attached file: a.txt ---\nbody"])
    assert merged.startswith("summarize this")
    assert "body" in merged


def test_augment_message_without_text() -> None:
    merged = augment_message("", ["block one", "block two"])
    assert merged == "block one\n\nblock two"


def test_augment_message_no_blocks_is_identity() -> None:
    assert augment_message("hello", []) == "hello"


def test_supported_extensions_cover_each_kind() -> None:
    exts = multimodal.supported_upload_extensions()
    for expected in ("pdf", "png", "mp3", "txt", "docx"):
        assert expected in exts


def test_processed_uploads_defaults() -> None:
    empty = ProcessedUploads()
    assert empty.text_blocks == []
    assert empty.images == []
