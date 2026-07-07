"""Agent proxy — models list, plot interpretation, and the SSE chat stream.

Also hosts the multimodal chat endpoint (``POST /agent/chat/multimodal``): a
multipart entry point that normalizes uploaded files (text / image / audio /
document) at the BFF, then streams to the agent's ``/chat/stream`` exactly like
the JSON path.
"""

import json
import logging
from typing import Any

import clients
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
import httpx
from models import AgentModelsOut, ChatRequest, PlotInterpretRequest
import multimodal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


def _stream_agent_chat(
    client: httpx.AsyncClient, agent_url: str, body: dict[str, Any]
) -> StreamingResponse:
    """Relay the agent's SSE chat stream verbatim, degrading to an error frame."""
    url = f"{agent_url}/chat/stream"

    async def event_stream():
        try:
            async with client.stream("POST", url, json=body, timeout=_STREAM_TIMEOUT) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            logger.warning("Agent chat stream failed: %s", exc)
            error = json.dumps({"type": "error", "answer": f"Agent is unavailable: {exc}"})
            yield f"data: {error}\n\n".encode()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/models", response_model=AgentModelsOut)
async def list_models(request: Request) -> AgentModelsOut:
    """Return the LLM model ids the agent currently knows about."""
    url = f"{request.app.state.agent_url}/models"
    payload = await clients.get_json(request.app.state.http_client, "agent", url, timeout=30.0)
    models = payload.get("models", []) or []
    return AgentModelsOut(models=[str(m) for m in models if str(m).strip()])


@router.post("/plots/interpret")
async def interpret_plot(payload: PlotInterpretRequest, request: Request) -> dict[str, Any]:
    """Proxy a rendered-chart image to the agent's vision endpoint."""
    url = f"{request.app.state.agent_url}/plots/interpret"
    return await clients.post_json(
        request.app.state.http_client,
        "agent",
        url,
        payload.model_dump(),
        timeout=90.0,
    )


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Relay the agent's Server-Sent-Events chat stream verbatim."""
    client: httpx.AsyncClient = request.app.state.http_client
    return _stream_agent_chat(client, request.app.state.agent_url, payload.model_dump())


def _parse_chat_history(raw: str) -> list[dict[str, Any]]:
    """Parse the multipart ``chat_history`` JSON string into turn dicts."""
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [m for m in parsed if isinstance(m, dict)]


@router.post("/chat/multimodal")
async def chat_multimodal(
    request: Request,
    user_message: str = Form(...),
    chat_history: str = Form("[]"),
    session_id: str | None = Form(None),
    files: list[UploadFile] = File(default=[]),
) -> StreamingResponse:
    """Normalize uploaded attachments, then stream the agent chat.

    Each file is routed by type at the BFF (text decoded, images base64-encoded
    for vision, audio transcribed via Whisper, documents converted via docling);
    extracted text is folded into ``user_message`` and images are forwarded as
    ``ChatRequest.images``. The response is the same SSE stream as ``/chat/stream``.
    """
    state = request.app.state
    uploads: list[tuple[str, str | None, bytes]] = []
    for upload in files:
        data = await upload.read()
        uploads.append((upload.filename or "upload", upload.content_type, data))

    inputs = multimodal.UploadInputs(
        http_client=state.http_client,
        docling_url=state.docling_url,
        docling_timeout=state.docling_timeout,
        whisper_client=state.whisper_client,
        whisper_model=state.whisper_model,
        whisper_enabled=state.whisper_enabled,
    )
    processed = await multimodal.process_uploads(inputs, uploads)

    payload = ChatRequest.model_validate(
        {
            "user_message": multimodal.augment_message(user_message, processed.text_blocks),
            "chat_history": _parse_chat_history(chat_history),
            "session_id": session_id,
            "images": processed.images,
        }
    )
    return _stream_agent_chat(state.http_client, state.agent_url, payload.model_dump())
