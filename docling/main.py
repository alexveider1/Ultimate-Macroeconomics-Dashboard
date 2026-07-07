"""FastAPI service converting uploaded documents to Markdown via docling.

Contract:
  - ``POST /convert`` — multipart upload of one ``file``; returns
    ``{"markdown": str, "filename": str, "format": str}``.
  - ``GET /formats`` — the supported upload extensions.
  - ``GET /health`` — ``{"status": "ok"}`` for the Compose healthcheck.

PDF conversion offloads its VLM inference to the Triton-hosted granite-docling
model over Triton's OpenAI-compatible endpoint; Office formats are parsed
locally by docling's native backends. This service holds no secrets — the Triton
endpoint is keyless on the internal Compose network.
"""

from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path

from config import load_config
from converter import SUPPORTED_FORMATS, build_converter, resolve_vlm_url
from docling.datamodel.base_models import DocumentStream
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("DOCLING_CONFIG_PATH", "config.yaml"))
CONFIG = load_config(CONFIG_PATH)
DOCLING_CONFIG = CONFIG.docling
TRITON_CONFIG = CONFIG.triton


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared docling converter once on startup."""
    url = resolve_vlm_url(TRITON_CONFIG.host, TRITON_CONFIG.openai_port)
    logger.info("Docling service targeting Triton VLM '%s' at %s", TRITON_CONFIG.vlm_model, url)
    app.state.converter = build_converter(
        url=url,
        model=TRITON_CONFIG.vlm_model,
        timeout=DOCLING_CONFIG.convert_timeout_seconds,
    )
    yield


app = FastAPI(
    title="Docling Conversion API",
    description="Converts PDF/DOCX/PPTX/XLSX uploads to Markdown (PDF via Triton VLM).",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Docling Conversion API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/formats")
def list_formats() -> dict[str, list[str]]:
    """Return the accepted upload extensions."""
    return {"formats": sorted(SUPPORTED_FORMATS)}


def _suffix(filename: str) -> str:
    """Return the lower-cased extension (``.pdf``) of an uploaded filename."""
    return Path(filename or "").suffix.lower()


@app.post("/convert")
async def convert_document(file: UploadFile = File(...)) -> dict[str, str]:
    """Convert one uploaded document to Markdown.

    Rejects unsupported extensions with 415 and maps a docling failure to 502
    (the Triton VLM path is the usual culprit for PDFs).
    """
    filename = file.filename or "upload"
    suffix = _suffix(filename)
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported document type '{suffix or filename}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}."
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    converter = app.state.converter
    stream = DocumentStream(name=filename, stream=io.BytesIO(data))
    try:
        result = await run_in_threadpool(converter.convert, stream)
        markdown = result.document.export_to_markdown()
    except Exception as exc:  # noqa: BLE001 - surface any docling/VLM failure cleanly
        logger.warning("Docling conversion failed for %s: %s", filename, exc)
        raise HTTPException(status_code=502, detail=f"Document conversion failed: {exc}") from exc

    return {"markdown": markdown, "filename": filename, "format": suffix.lstrip(".")}
