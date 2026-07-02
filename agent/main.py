"""FastAPI entry point for the agent service.

Exposes two endpoints:

* ``POST /chat/stream`` — server-sent events that wrap a :class:`MacroAgentGraph`
  run. The client receives ``step`` / ``token`` / ``final`` / ``error`` events.
* ``POST /plots/interpret`` — vision call that turns a rendered Plotly PNG into
  either a strict description or an analyst interpretation.

Token usage is tracked per request via :class:`agent.usage.UsageTracker` and
returned in the SSE ``final`` event so the dashboard can show it.
"""

import asyncio
from functools import lru_cache
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from starlette.responses import StreamingResponse

from agent.config import load_config
from agent.graph import MacroAgentGraph
from agent.schemas import (
    ChatRequest,
    PlotInterpretationRequest,
    PlotInterpretationResponse,
    TokenUsage,
)
from agent.settings import get_settings
from agent.tools import aclose_runtime_clients, configure_runtime
from agent.usage import UsageTracker

logger = logging.getLogger(__name__)

STREAM_TIMEOUT_SECONDS = 300

CONFIG_PATH = Path("config.yaml")
DATABASE_SCHEMA_PATH = Path("database_schema.yaml")
NEWS_TOPICS_PATH = Path("_configs/news_download_config.json")

CONFIG = load_config(CONFIG_PATH)
SETTINGS = get_settings()

SHARED_CFG = CONFIG.shared
AGENT_MODEL = SHARED_CFG.openai_llm_model
AGENT_MODEL_FAST = SHARED_CFG.openai_llm_model_fast
OPENAI_API_BASE_URL = SHARED_CFG.openai_base_url
OPENAI_API_KEY = SETTINGS.openai_api_key

PYTHON_SANDBOX_BASE_URL = f"http://python_sandbox:{CONFIG.python_sandbox.port}"
DOWNLOADER_EXTRA_BASE_URL = f"http://downloader_extra:{CONFIG.downloader_extra.port}"
QDRANT_URL = f"http://{CONFIG.qdrant.host}:{CONFIG.qdrant.port}"
QDRANT_API_KEY = SETTINGS.qdrant_api_key
POSTGRES_DATABASE_URI = (
    f"postgresql+psycopg2://"
    f"{SETTINGS.postgres_llm_user}:{SETTINGS.postgres_llm_password}"
    f"@{CONFIG.postgres.host}:{CONFIG.postgres.port}"
    f"/{SETTINGS.postgres_db or CONFIG.postgres.database}"
)
OPENAI_EMBEDDING_MODEL = SHARED_CFG.openai_embedding_model

configure_runtime(
    database_schema_path=DATABASE_SCHEMA_PATH,
    news_topics_path=NEWS_TOPICS_PATH,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    postgres_database_uri=POSTGRES_DATABASE_URI,
    python_sandbox_base_url=PYTHON_SANDBOX_BASE_URL,
    downloader_extra_base_url=DOWNLOADER_EXTRA_BASE_URL,
    openai_api_key=OPENAI_API_KEY or "",
    openai_base_url=OPENAI_API_BASE_URL or "",
    openai_embedding_model=OPENAI_EMBEDDING_MODEL,
)


def _require_api_key() -> str:
    """Return ``OPENAI_API_KEY`` or raise a clear error if it's missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OPENAI_API_KEY


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    """Return a process-wide sync OpenAI client (used for vision calls)."""
    return OpenAI(
        base_url=OPENAI_API_BASE_URL,
        api_key=_require_api_key(),
        max_retries=5,
    )


@lru_cache(maxsize=1)
def _get_macro_agent() -> MacroAgentGraph:
    """Return a process-wide :class:`MacroAgentGraph` instance."""
    return MacroAgentGraph(
        base_url=OPENAI_API_BASE_URL or "",
        model_name=AGENT_MODEL or "",
        fast_model_name=AGENT_MODEL_FAST or "",
        api_key=_require_api_key(),
    )


app = FastAPI(
    title="AI-Agent API",
    description="API for interacting with the AI-Agent.",
    version="0.1.0",
)


@app.on_event("shutdown")
async def _close_runtime_clients() -> None:
    """Close the shared httpx pool so uvicorn shuts down cleanly."""
    await aclose_runtime_clients()


@app.get("/")
def root() -> dict[str, str]:
    """Return ``{"status": "ok", "model": ...}`` for liveness + model echo."""
    return {"status": "ok", "model": AGENT_MODEL or "", "fast_model": AGENT_MODEL_FAST or ""}


@app.get("/health")
def health() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    """List models offered by the configured OpenAI-compatible endpoint."""
    if not OPENAI_API_KEY:
        return {"models": [AGENT_MODEL or ""]}
    try:
        models = _get_openai_client().models.list()
        return {"models": [m.id for m in models.data]}
    except OpenAIError as exc:
        logger.warning("Could not list OpenAI models: %s", exc)
        return {"models": [AGENT_MODEL or ""]}


@app.post("/chat/stream")
async def process_chat_stream(request: ChatRequest):
    """Stream the agent run as Server-Sent Events.

    Args:
        request: The ``ChatRequest`` payload with the new user message and
            the prior chat history.

    Returns:
        ``StreamingResponse`` emitting JSON-encoded events:
        ``step`` (worker boundary), ``token`` (incremental text from the
        final synthesis), ``final`` (full answer + artifacts + usage), and
        ``error`` (graceful timeout / failure).
    """
    agent = _get_macro_agent()
    chat_history = [m.model_dump() for m in request.chat_history]
    usage_tracker = UsageTracker()

    async def event_generator():
        """Inner generator that yields SSE-formatted ``data: ...`` strings."""
        try:
            async with asyncio.timeout(STREAM_TIMEOUT_SECONDS):
                async for event in agent.astream_events(
                    message=request.user_message,
                    chat_history=chat_history,
                    usage_tracker=usage_tracker,
                ):
                    event_type = event.get("type", "step")
                    if event_type == "step":
                        payload = {"type": "step", "node": event.get("node", "")}
                    elif event_type == "token":
                        payload = {"type": "token", "delta": event.get("delta", "")}
                    elif event_type == "final":
                        payload = {
                            "type": "final",
                            "answer": str(event.get("response", "")),
                            "model": AGENT_MODEL or "",
                            "artifacts": event.get("artifacts", {}),
                            "usage": usage_tracker.snapshot(default_model=AGENT_MODEL or ""),
                        }
                    elif event_type == "error":
                        payload = {
                            "type": "error",
                            "answer": str(event.get("response", "")),
                        }
                    else:
                        continue
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
        except asyncio.TimeoutError:
            logger.warning(
                "/chat/stream: agent stream exceeded %ss timeout",
                STREAM_TIMEOUT_SECONDS,
            )
            timeout_payload = {
                "type": "error",
                "answer": (
                    f"The agent took longer than {STREAM_TIMEOUT_SECONDS}s to respond and "
                    "was cancelled. Try a more specific question."
                ),
            }
            yield f"data: {json.dumps(timeout_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/plots/interpret", response_model=PlotInterpretationResponse)
async def interpret_plot(request: PlotInterpretationRequest):
    """Send a rendered chart image to the vision LLM for description.

    Args:
        request: PNG payload + mode (``no_hallucinations`` for strict
            description, anything else for interpretive analysis).

    Returns:
        :class:`PlotInterpretationResponse` with the model's description and
        per-call token usage.

    Raises:
        HTTPException: 502 on OpenAI errors, 500 on anything else.
    """
    try:
        client = _get_openai_client()

        if request.mode == "no_hallucinations":
            system_prompt = (
                "Read the chart. Only describe what is visible — direction, "
                "turning points, spikes, plateaus, line comparisons. No causes, "
                "no speculation. Reply in 3 short bullets, ~40 words total."
            )
            temperature = 0.0
        else:
            system_prompt = (
                "Macro-financial chart analyst. One bullet of observations, "
                "one of likely drivers. Mark hypotheses with 'likely'. "
                "Max 60 words total."
            )
            temperature = 0.5

        user_text = "Describe this chart."
        if request.chart_context.strip():
            user_text += f" Context: {request.chart_context.strip()}"

        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=AGENT_MODEL,
            temperature=temperature,
            max_tokens=200 if request.mode == "no_hallucinations" else 260,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{request.image_base64}"},
                        },
                    ],
                },
            ],
        )

        description = ""
        if completion.choices and completion.choices[0].message is not None:
            description = str(completion.choices[0].message.content or "").strip()

        usage = getattr(completion, "usage", None)
        token_usage = TokenUsage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            model=AGENT_MODEL or "",
        )

        return PlotInterpretationResponse(
            description=description or "No interpretation returned.",
            mode=request.mode,
            model=AGENT_MODEL or "",
            usage=token_usage,
        )
    except OpenAIError as exc:
        logger.exception("/plots/interpret: OpenAI call failed")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc
    except Exception as exc:
        logger.exception("/plots/interpret: unexpected error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
