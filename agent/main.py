import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from starlette.responses import StreamingResponse
import yaml

from agent.graph import MacroAgentGraph
from agent.schemas import (
    ChatRequest,
    PlotInterpretationRequest,
    PlotInterpretationResponse,
    TokenUsage,
)
from agent.tools import configure_runtime
from agent.usage import UsageTracker


CONFIG_PATH = "config.yaml"
ENV_FILE_PATH = ".env"
DATABASE_SCHEMA_PATH = "database_schema.yaml"
NEWS_TOPICS_PATH = "_configs/news_download_config.json"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

load_dotenv(dotenv_path=ENV_FILE_PATH)

SHARED_CFG = CONFIG.get("shared", {})
DEFAULT_AGENT_MODEL = SHARED_CFG.get("openai_llm_model") or ""
EMBEDDING_BASE_URL = SHARED_CFG.get("openai_base_url") or ""
# In the hosting build OPENAI_API_KEY is reserved for embeddings / RAG only.
# Chat / agentic LLM credentials are supplied per request via X-LLM-* headers.
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = SHARED_CFG.get(
    "openai_embedding_model", "text-embedding-3-small"
)

PYTHON_SANDBOX_BASE_URL = (
    f"http://python_sandbox:{CONFIG.get('python_sandbox', {}).get('port')}"
)
DOWNLOADER_EXTRA_BASE_URL = (
    f"http://downloader_extra:{CONFIG.get('downloader_extra', {}).get('port')}"
)
QDRANT_URL = (
    f"http://{CONFIG.get('qdrant', {}).get('host')}:"
    f"{CONFIG.get('qdrant', {}).get('port')}"
)
QDRANT_API_KEY = os.getenv("QDRANT__SERVICE__API_KEY", "")
POSTGRES_DATABASE_URI = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_LLM_USERNAME')}:{os.getenv('POSTGRES_LLM_PASSWORD')}"
    f"@{CONFIG.get('postgres', {}).get('host')}:{CONFIG.get('postgres', {}).get('port')}"
    f"/{CONFIG.get('postgres', {}).get('database')}"
)

configure_runtime(
    database_schema_path=DATABASE_SCHEMA_PATH,
    news_topics_path=NEWS_TOPICS_PATH,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    postgres_database_uri=POSTGRES_DATABASE_URI,
    python_sandbox_base_url=PYTHON_SANDBOX_BASE_URL,
    downloader_extra_base_url=DOWNLOADER_EXTRA_BASE_URL,
    openai_api_key=EMBEDDING_API_KEY,
    openai_base_url=EMBEDDING_BASE_URL,
    openai_embedding_model=OPENAI_EMBEDDING_MODEL,
)


def _resolve_llm_creds(
    base_url: str | None,
    api_key: str | None,
    model: str | None,
) -> tuple[str, str, str]:
    """Pull per-session LLM credentials out of request headers.

    Raises 401 if the caller has not supplied them. The chat / agentic LLM
    is intentionally NOT readable from the agent container's environment in
    this build — operators of the hosted dashboard never hold the user's key.
    """
    base_url = (base_url or "").strip()
    api_key = (api_key or "").strip()
    if not base_url or not api_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing LLM credentials. Provide X-LLM-Base-URL and "
                "X-LLM-API-Key request headers."
            ),
        )
    return base_url, api_key, (model or "").strip() or DEFAULT_AGENT_MODEL


app = FastAPI(
    title="AI-Agent API",
    description="API for interacting with the AI-Agent.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "model": DEFAULT_AGENT_MODEL}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models(
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
) -> dict[str, list[str]]:
    base_url = (x_llm_base_url or "").strip()
    api_key = (x_llm_api_key or "").strip()
    if not base_url or not api_key:
        return {"models": [DEFAULT_AGENT_MODEL] if DEFAULT_AGENT_MODEL else []}
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2)
        models = client.models.list()
        return {"models": [m.id for m in models.data]}
    except Exception:
        return {"models": [DEFAULT_AGENT_MODEL] if DEFAULT_AGENT_MODEL else []}


@app.post("/chat/stream")
async def process_chat_stream(
    request: ChatRequest,
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
):
    base_url, api_key, model = _resolve_llm_creds(
        x_llm_base_url, x_llm_api_key, x_llm_model
    )
    agent = MacroAgentGraph(base_url=base_url, model_name=model, api_key=api_key)
    chat_history = [m.model_dump() for m in request.chat_history]
    usage_tracker = UsageTracker()

    async def event_generator():
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
                    "model": model,
                    "artifacts": event.get("artifacts", {}),
                    "usage": usage_tracker.snapshot(default_model=model),
                }
            elif event_type == "error":
                payload = {
                    "type": "error",
                    "answer": str(event.get("response", "")),
                }
            else:
                continue
            yield f"data: {json.dumps(payload, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/plots/interpret", response_model=PlotInterpretationResponse)
async def interpret_plot(
    request: PlotInterpretationRequest,
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
):
    base_url, api_key, model = _resolve_llm_creds(
        x_llm_base_url, x_llm_api_key, x_llm_model
    )
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)

        if request.mode == "no_hallucinations":
            system_prompt = (
                "You are a chart-reading assistant. Describe only what is visible "
                "in the plot image. Focus on factual line behaviour over time: "
                "direction, turning points, relative volatility, plateaus, spikes, "
                "and comparisons between lines. Do not speculate about causes."
            )
            temperature = 0.0
        else:
            system_prompt = (
                "You are a macro-financial chart analyst. First summarise what the "
                "plot shows, then provide plausible interpretations. Clearly "
                "separate observations from hypotheses."
            )
            temperature = 0.5

        user_text = "Interpret this plot image."
        if request.chart_context.strip():
            user_text += f"\nContext: {request.chart_context.strip()}"

        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{request.image_base64}"
                            },
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
            model=model,
        )

        return PlotInterpretationResponse(
            description=description or "No interpretation returned.",
            mode=request.mode,
            model=model,
            usage=token_usage,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
