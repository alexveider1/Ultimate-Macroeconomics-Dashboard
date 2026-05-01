import operator
from typing import Annotated, Any, Dict, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


def _merge_artifacts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge new artifact keys into the existing dict."""
    if not existing:
        return new
    merged = existing.copy()
    merged.update(new)
    return merged


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    worker_results: Annotated[Sequence[str], operator.add]
    current_plan: str
    next_worker: str
    isolated_worker_task: str
    artifacts: Annotated[Dict[str, Any], _merge_artifacts]
    last_worker: str
    retry_count: int
    trace: Annotated[list[str], operator.add]
    guardrail_blocked: bool
    guardrail_message: str


WORKER_LITERAL = Literal[
    "sql_agent",
    "plotly_agent",
    "table_agent",
    "rag_agent",
    "web_search",
    "downloader_agent",
    "chat_agent",
    "FINISH",
]


class SupervisorDecision(BaseModel):
    thought_process: str = Field(
        description=(
            "Deep, step-by-step reasoning about the current state. "
            "Analyze worker results, check for errors, evaluate quality, "
            "and decide on the next action. If retrying, explain what went wrong."
        )
    )
    updated_plan: str = Field(
        description=(
            "A numbered step-by-step plan for completing the user's request. "
            "Update as steps are completed or the approach changes."
        )
    )
    next_worker: WORKER_LITERAL = Field(
        description=(
            "The next worker to invoke: sql_agent, plotly_agent, table_agent, "
            "rag_agent, web_search, downloader_agent, chat_agent, or FINISH."
        )
    )
    isolated_worker_task: str = Field(
        description=(
            "If next_worker is a worker: a detailed, self-contained task description. "
            "Workers have NO memory of previous steps - include all context they need. "
            "If next_worker is FINISH: the complete, well-formatted final answer "
            "to present to the user (markdown allowed)."
        )
    )


class SQLGeneration(BaseModel):
    thought_process: str = Field(
        description=(
            "Step-by-step reasoning about which tables, columns, joins, "
            "filters, and aggregations are needed. Explain WHY this query "
            "is needed as the current step and what you learned from "
            "previous steps."
        )
    )
    sql_query: str = Field(
        description=(
            "An exact, executable PostgreSQL SELECT query. "
            "Must be read-only - no DDL, DML, or transaction statements."
        )
    )
    is_final_step: bool = Field(
        default=False,
        description=(
            "Set to true ONLY when this query fetches the actual data "
            "the user asked for (typically the indicators/metadata join "
            "or the final result set). Exploration queries (browsing "
            "databases, searching for indicator names) must be false."
        ),
    )


class PlotlyCodeGeneration(BaseModel):
    thought_process: str = Field(
        description="Step-by-step reasoning about how to visualize the data."
    )
    plotly_code: str = Field(
        description=(
            "Executable Python code using plotly to create a visualization. "
            "Input data is available as `data` (a list of dicts). "
            "The final figure must be assigned to a variable named `fig`. "
            "Do NOT call fig.show(). Import plotly modules as needed."
        )
    )
    title: str = Field(description="A concise, descriptive title for the chart.")


class PolarsCodeGeneration(BaseModel):
    thought_process: str = Field(
        description="Step-by-step reasoning on how to transform the data."
    )
    polars_code: str = Field(
        description=(
            "Executable Python Polars code. "
            "Assume `import polars as pl` is done and the input DataFrame "
            "is `df`. Do not use pandas. Assign the result to `result_df`."
        )
    )


class RAGSearchPlan(BaseModel):
    thought_process: str = Field(
        description="Reasoning about what to search for in the news vector DB."
    )
    search_query: str = Field(
        description="The semantic search query to find relevant news articles."
    )
    topic_filter: str | None = Field(
        default=None,
        description=(
            "Optional topic filter. One of: Economy Business and Finance, "
            "Science and Technology, Politics, Disaster and Accident, "
            "Education, Environment, Health, Social Issue."
        ),
    )
    sentiment_filter: str | None = Field(
        default=None,
        description="Optional sentiment filter: 'positive' or 'negative'.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results to retrieve.",
    )


class WebSearchPlan(BaseModel):
    thought_process: str = Field(
        description="Reasoning about what to search for on the internet."
    )
    search_queries: list[str] = Field(
        description="List of 1-3 search queries to execute.",
        min_length=1,
        max_length=3,
    )


class DownloadIndicatorPlan(BaseModel):
    thought_process: str = Field(
        description="Reasoning about which World Bank indicator to download."
    )
    indicator_id: str = Field(
        description="The World Bank indicator ID (e.g. 'NY.GDP.MKTP.CD')."
    )
    db_id: int = Field(description="The World Bank database ID (e.g. 2 for WDI).")


class ChatSynthesis(BaseModel):
    response: str = Field(
        description=(
            "A complete, well-formatted response for the user. "
            "Use markdown where appropriate."
        )
    )


class GuardrailDecision(BaseModel):
    is_inappropriate: bool = Field(
        description=(
            "True if the user's message contains harsh language, personal "
            "attacks, sexual content, requests for illegal/unethical content, "
            "or anything else outside the dashboard's scope of macroeconomics, "
            "politics, sociology, econometrics, data science and technology."
        )
    )
    reason: str = Field(
        default="",
        description="Short explanation of why the message was flagged.",
    )
    refusal_message: str = Field(
        default="",
        description=(
            "A polite, brief refusal in markdown to show the user when "
            "is_inappropriate is true. Empty otherwise."
        ),
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_message: str
    chat_history: list[ChatMessage] = Field(default_factory=list)


class PlotInterpretationRequest(BaseModel):
    image_base64: str
    mode: str = "no_hallucinations"
    chart_context: str = ""


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class PlotInterpretationResponse(BaseModel):
    description: str
    mode: str
    model: str
    usage: TokenUsage | None = None
