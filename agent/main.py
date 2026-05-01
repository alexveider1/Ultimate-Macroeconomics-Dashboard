import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from starlette.responses import StreamingResponse
import yaml

from agent.graph import MacroAgentGraph
from agent.schemas import (
    ChatRequest,
    PlotInterpretationRequest,
    PlotInterpretationResponse,
)
from agent.tools import configure_runtime


CONFIG_PATH = "config.yaml"
ENV_FILE_PATH = ".env"
DATABASE_SCHEMA_PATH = "database_schema.yaml"
NEWS_TOPICS_PATH = "_configs/news_download_config.json"

CONFIG = yaml.safe_load(open(CONFIG_PATH))

load_dotenv(dotenv_path=ENV_FILE_PATH)

SHARED_CFG = CONFIG.get("shared", {})
AGENT_MODEL = SHARED_CFG.get("openai_llm_model")
OPENAI_API_BASE_URL = SHARED_CFG.get("openai_base_url")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
OPENAI_EMBEDDING_MODEL = SHARED_CFG.get(
    "openai_embedding_model", "text-embedding-3-small"
)

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
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OPENAI_API_KEY


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    return OpenAI(
        base_url=OPENAI_API_BASE_URL,
        api_key=_require_api_key(),
        max_retries=5,
    )


@lru_cache(maxsize=1)
def _get_macro_agent() -> MacroAgentGraph:
    return MacroAgentGraph(
        base_url=OPENAI_API_BASE_URL or "",
        model_name=AGENT_MODEL or "",
        api_key=_require_api_key(),
    )


app = FastAPI(
    title="AI-Agent API",
    description="API for interacting with the AI-Agent.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "model": AGENT_MODEL or ""}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    if not OPENAI_API_KEY:
        return {"models": [AGENT_MODEL or ""]}
    try:
        models = _get_openai_client().models.list()
        return {"models": [m.id for m in models.data]}
    except Exception:
        return {"models": [AGENT_MODEL or ""]}


@app.post("/chat/stream")
async def process_chat_stream(request: ChatRequest):
    agent = _get_macro_agent()
    chat_history = [m.model_dump() for m in request.chat_history]

    async def event_generator():
        async for event in agent.astream_events(
            message=request.user_message,
            chat_history=chat_history,
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
async def interpret_plot(request: PlotInterpretationRequest):
    try:
        client = _get_openai_client()

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
            model=AGENT_MODEL,
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

        return PlotInterpretationResponse(
            description=description or "No interpretation returned.",
            mode=request.mode,
            model=AGENT_MODEL or "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
