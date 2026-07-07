"""Docling ``DocumentConverter`` construction + format routing.

Isolated from ``main.py`` so the (version-sensitive) docling API surface lives in
one place. Pinned to ``docling==2.110.0``: PDFs run through the ``VlmPipeline``
pointed at the Triton-hosted granite-docling VLM over its OpenAI-compatible
endpoint (``enable_remote_services=True`` — no local model weights, no GPU in this
container); Office formats (docx/pptx/xlsx) use docling's native backends and need
no VLM. If the docling API changes on upgrade, this is the single file to adjust.
"""

from __future__ import annotations

import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Extension -> docling InputFormat for the four document types the BFF routes
# here. Images are handled directly by the vision model (not docling); text
# files are decoded at the BFF. Everything else is rejected with a 415.
SUPPORTED_FORMATS: dict[str, InputFormat] = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
    ".xlsx": InputFormat.XLSX,
}


def resolve_vlm_url(host: str, openai_port: int) -> str:
    """Return the Triton OpenAI chat-completions endpoint (env override wins)."""
    override = os.environ.get("DOCLING_VLM_URL") or os.environ.get("TRITON_OPENAI_URL")
    if override:
        return override
    return f"http://{host}:{openai_port}/v1/chat/completions"


def build_converter(url: str, model: str, timeout: int) -> DocumentConverter:
    """Build the shared converter: remote VLM for PDF, native backends elsewhere.

    Args:
        url: Triton OpenAI-compatible chat-completions endpoint.
        model: OpenAI ``model`` field = the Triton model name (``granite_docling``).
        timeout: Per-request VLM call timeout in seconds.
    """
    vlm_options = VlmConvertOptions.from_preset(
        "granite_docling",
        engine_options=ApiVlmEngineOptions(
            runtime_type=VlmEngineType.API,
            url=url,
            params={
                "model": model,
                "temperature": 0.0,
                "max_tokens": 8192,
                "skip_special_tokens": False,
            },
            timeout=timeout,
        ),
    )
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            ),
        },
    )
