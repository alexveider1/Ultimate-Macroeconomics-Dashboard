"""Multimodal upload handling for the agent chat.

The BFF is the single place where non-text inputs are normalized before the chat
request reaches the agent. Each uploaded file is routed by extension:

  - ``.json`` / ``.md`` / ``.txt`` → decoded UTF-8 text
  - images (``.png`` / ``.jpg`` / ``.jpeg`` / ``.webp`` / ``.gif``) → base64 data
    URI, forwarded as OpenAI vision content-parts (all project LLMs are visual)
  - audio (``.mp3`` / ``.wav`` / ``.m4a`` / ``.ogg`` / ``.webm`` / ``.flac``) →
    transcribed via the OpenAI-compatible Whisper endpoint
  - documents (``.pdf`` / ``.docx`` / ``.pptx`` / ``.xlsx``) → Markdown via the
    ``docling`` service (PDF over the Triton-hosted VLM)

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
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

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


class WhisperClient(Protocol):
    """Structural type for the async OpenAI-compatible transcription client."""

    audio: Any


@dataclass
class UploadInputs:
    """Backends the dispatcher needs, gathered from ``app.state``."""

    http_client: httpx.AsyncClient
    docling_url: str
    docling_timeout: float
    whisper_client: WhisperClient | None
    whisper_model: str
    whisper_enabled: bool


@dataclass
class ProcessedUploads:
    """Result of processing a batch of uploads."""

    text_blocks: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


def _suffix(filename: str) -> str:
    """Return the lower-cased extension (``.pdf``) of a filename."""
    return Path(filename or "").suffix.lower()


async def transcribe_audio(client: WhisperClient, model: str, filename: str, data: bytes) -> str:
    """Transcribe audio bytes via the OpenAI-compatible Whisper endpoint."""
    result = await client.audio.transcriptions.create(  # type: ignore[attr-defined]
        model=model,
        file=(filename, data),
    )
    return str(getattr(result, "text", "") or "").strip()


async def convert_document(
    client: httpx.AsyncClient,
    docling_url: str,
    timeout: float,
    filename: str,
    content_type: str | None,
    data: bytes,
) -> str:
    """Convert a document to Markdown via the ``docling`` service."""
    url = f"{docling_url}/convert"
    response = await client.post(
        url,
        files={"file": (filename, data, content_type or "application/octet-stream")},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("markdown", "") or "").strip()


async def _process_one(
    inputs: UploadInputs,
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
        if not (inputs.whisper_enabled and inputs.whisper_client is not None):
            result.text_blocks.append(
                f"[Could not transcribe {filename}: transcription is not configured]"
            )
            return
        try:
            transcript = await transcribe_audio(
                inputs.whisper_client, inputs.whisper_model, filename, data
            )
        except Exception as exc:  # noqa: BLE001 - degrade one bad file, not the request
            logger.warning("Transcription failed for %s: %s", filename, exc)
            result.text_blocks.append(f"[Could not transcribe {filename}: {exc}]")
            return
        result.text_blocks.append(f"--- Transcribed audio: {filename} ---\n{transcript}")
        return

    if ext in DOC_EXTS:
        try:
            markdown = await convert_document(
                inputs.http_client,
                inputs.docling_url,
                inputs.docling_timeout,
                filename,
                content_type,
                data,
            )
        except Exception as exc:  # noqa: BLE001 - degrade one bad file, not the request
            logger.warning("Document conversion failed for %s: %s", filename, exc)
            result.text_blocks.append(f"[Could not read document {filename}: {exc}]")
            return
        result.text_blocks.append(
            f"--- Attached file: {filename} ({ext.lstrip('.')}) ---\n{markdown}"
        )
        return

    result.text_blocks.append(f"[Unsupported attachment type: {filename}]")


async def process_uploads(
    inputs: UploadInputs,
    files: list[tuple[str, str | None, bytes]],
) -> ProcessedUploads:
    """Process every upload in order, returning extracted text + image URIs."""
    result = ProcessedUploads()
    for filename, content_type, data in files:
        await _process_one(inputs, result, filename, content_type, data)
    return result


def augment_message(user_message: str, text_blocks: list[str]) -> str:
    """Append extracted-attachment text blocks to the user message."""
    if not text_blocks:
        return user_message
    parts = [user_message.strip()] if user_message.strip() else []
    parts.extend(text_blocks)
    return "\n\n".join(parts)
