"""Agent proxy — models list, plot interpretation, and the SSE chat stream."""

import json
import logging
from typing import Any

import clients
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import httpx
from models import AgentModelsOut, ChatRequest, PlotInterpretRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


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
    url = f"{request.app.state.agent_url}/chat/stream"
    body = payload.model_dump()

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
