"""Multimodal upload handling for the AI chat (file + voice input).

The Streamlit chat is the single place where non-text inputs are normalized
before the chat request reaches the agent (the app talks to the backends
directly — there is no BFF). Each attachment is routed by extension:

  - ``.json`` / ``.md`` / ``.txt`` → decoded UTF-8 text
  - images (``.png`` / ``.jpg`` / ``.jpeg`` / ``.webp`` / ``.gif``) → base64 data
    URI, forwarded as OpenAI vision content-parts (all project LLMs are visual)
  - audio (``.mp3`` / ``.wav`` / ``.m4a`` / ``.ogg`` / ``.webm`` / ``.flac``) →
    transcribed via the OpenAI-compatible Whisper endpoint (``config.whisper``)
  - documents (``.pdf`` / ``.docx`` / ``.pptx`` / ``.xlsx``) → Markdown via the
    ``docling`` service (PDF over a cloud OCR/VLM endpoint)

Extracted text is appended to ``user_message`` under clear delimiters (so every
worker sees it); images are returned separately for injection as vision parts.
A per-file failure degrades to an inline ``[note]`` rather than aborting the
whole request, so one bad attachment never sinks the others.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import logging
from pathlib import Path

import httpx

from core.api_client import resolve_docling_base_url
from core.config import load_config
from core.settings import get_settings

logger = logging.getLogger(__name__)

CONFIG = load_config(Path("config.yaml"))
SETTINGS = get_settings()

# docling PDF conversion runs a remote VLM, so give it a generous ceiling.
_DOCLING_TIMEOUT_SECONDS = 180.0
_WHISPER_TIMEOUT_SECONDS = 120.0

TEXT_EXTS = {".json", ".md", ".txt"}
IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"}
DOC_EXTS = {".pdf", ".docx", ".pptx", ".xlsx"}


@dataclass
class ProcessedUploads:
    """Result of processing a batch of uploads."""

    text_blocks: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


def _suffix(filename: str) -> str:
    """Return the lower-cased extension (``.pdf``) of a filename."""
    return Path(filename or "").suffix.lower()


def transcribe_audio(filename: str, data: bytes) -> str:
    """Transcribe audio bytes via the OpenAI-compatible Whisper endpoint.

    Posts a plain multipart request to ``{base_url}/audio/transcriptions`` (the
    OpenAI transcription contract) with ``httpx`` — no OpenAI SDK dependency.
    """
    whisper = CONFIG.whisper
    if not whisper.enabled or not SETTINGS.openai_api_key:
        raise RuntimeError("transcription is not configured")
    url = f"{whisper.base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {SETTINGS.openai_api_key}"}
    with httpx.Client(timeout=_WHISPER_TIMEOUT_SECONDS) as client:
        response = client.post(
            url,
            headers=headers,
            data={"model": whisper.model},
            files={"file": (filename, data)},
        )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("text", "") or "").strip()


def convert_document(filename: str, content_type: str | None, data: bytes) -> str:
    """Convert a document to Markdown via the ``docling`` service."""
    url = f"{resolve_docling_base_url()}/convert"
    with httpx.Client(timeout=_DOCLING_TIMEOUT_SECONDS) as client:
        response = client.post(
            url,
            files={"file": (filename, data, content_type or "application/octet-stream")},
        )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("markdown", "") or "").strip()


def _process_one(
    result: ProcessedUploads,
    filename: str,
    content_type: str | None,
    data: bytes,
) -> None:
    """Dispatch a single upload, mutating ``result`` in place."""
    ext = _suffix(filename)
    if not data:
        result.text_blocks.append(f"[Skipped empty attachment: {filename}]")
        return

    if ext in TEXT_EXTS:
        text = data.decode("utf-8", errors="replace").strip()
        result.text_blocks.append(f"--- Attached file: {filename} ---\n{text}")
        return

    if ext in IMAGE_MIME_BY_EXT:
        mime = IMAGE_MIME_BY_EXT[ext]
        encoded = base64.b64encode(data).decode("ascii")
        result.images.append(f"data:{mime};base64,{encoded}")
        result.text_blocks.append(f"[User attached image: {filename}]")
        return

    if ext in AUDIO_EXTS:
        try:
            transcript = transcribe_audio(filename, data)
        except Exception as exc:  # noqa: BLE001 - degrade one bad file, not the request
            logger.warning("Transcription failed for %s: %s", filename, exc)
            result.text_blocks.append(f"[Could not transcribe {filename}: {exc}]")
            return
        result.text_blocks.append(f"--- Transcribed audio: {filename} ---\n{transcript}")
        return

    if ext in DOC_EXTS:
        try:
            markdown = convert_document(filename, content_type, data)
        except Exception as exc:  # noqa: BLE001 - degrade one bad file, not the request
            logger.warning("Document conversion failed for %s: %s", filename, exc)
            result.text_blocks.append(f"[Could not read document {filename}: {exc}]")
            return
        result.text_blocks.append(
            f"--- Attached file: {filename} ({ext.lstrip('.')}) ---\n{markdown}"
        )
        return

    result.text_blocks.append(f"[Unsupported attachment type: {filename}]")


def process_uploads(files: list[tuple[str, str | None, bytes]]) -> ProcessedUploads:
    """Process every upload in order, returning extracted text + image URIs."""
    result = ProcessedUploads()
    for filename, content_type, data in files:
        _process_one(result, filename, content_type, data)
    return result


def augment_message(user_message: str, text_blocks: list[str]) -> str:
    """Append extracted-attachment text blocks to the user message."""
    if not text_blocks:
        return user_message
    parts = [user_message.strip()] if user_message.strip() else []
    parts.extend(text_blocks)
    return "\n\n".join(parts)


def supported_upload_extensions() -> list[str]:
    """Return every accepted attachment extension (no leading dot), sorted."""
    exts = TEXT_EXTS | set(IMAGE_MIME_BY_EXT) | AUDIO_EXTS | DOC_EXTS
    return sorted(ext.lstrip(".") for ext in exts)
