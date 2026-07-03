"""Pydantic schemas + LangGraph state shape for the multi-agent analyst.

Every structured-output LLM call in :mod:`agent.graph` materialises one of
these models via ``with_structured_output(...)``; ``AgentState`` is the graph
state shape itself and must remain a ``TypedDict`` per LangGraph's contract.
"""

import operator
from typing import Annotated, Any, Dict, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


def _merge_artifacts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge ``new`` artifact keys into ``existing``.

    Used as the channel reducer for ``AgentState.artifacts`` so a worker that
    returns ``{"figure": ...}`` doesn't clobber an earlier ``{"table": ...}``.

    Args:
        existing: Accumulated artifacts so far.
        new: New artifacts from the current step.

    Returns:
        Merged dict. ``new`` keys win on conflict.
    """
    if not existing:
        return new
    merged = existing.copy()
    merged.update(new)
    return merged


WORKER_STATUS_LITERAL = Literal[
    "SUCCESS",
    "EMPTY",
    "ERROR",
    "NEEDS_DOWNLOAD",
    "BLOCKED",
    "UNKNOWN",
]
"""Coarse outcome tag a worker returns alongside its worker_results string.

The supervisor can branch off this tag deterministically instead of regex-
matching prose; the per-worker semantics are:

* ``SUCCESS`` — produced useful output and/or wrote an artifact.
* ``EMPTY``   — ran cleanly but found nothing (zero rows, no articles, ...).
* ``ERROR``   — the worker itself failed (exception, sandbox error, ...).
* ``NEEDS_DOWNLOAD`` — sql_agent found the indicator in ``database_indicators``
  but the ``indicators`` table is empty for it; route to downloader_agent.
* ``BLOCKED`` — refused due to a precondition (e.g. no data in artifacts).
* ``UNKNOWN`` — pre-channel-default; should never appear after a worker runs.
"""


# Must stay a TypedDict — LangGraph requires the graph state schema to be a
# TypedDict (it uses Annotated[..., reducer] for channel merging). Every other
# data model in this service is a pydantic.BaseModel.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    worker_results: Annotated[Sequence[str], operator.add]
    current_plan: str
    next_worker: str
    isolated_worker_task: str
    artifacts: Annotated[Dict[str, Any], _merge_artifacts]
    last_worker: str
    last_worker_status: str
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
    """One supervisor turn: which worker to invoke next (or FINISH)."""

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
    """One SQL exploration/final step produced by the ``sql_agent`` worker."""

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
    """Code snippet produced by the ``plotly_agent`` worker for the sandbox."""

    thought_process: str = Field(
        description="Step-by-step reasoning about how to visualize the data."
    )
    plotly_code: str = Field(
        description=(
            "Executable Python code using plotly.graph_objects (imported as `go`) "
            "to create a visualization. Do NOT use plotly.express. "
            "Input data is available as `data` (a list of dicts). "
            "The final figure must be assigned to a variable named `fig`. "
            "Do NOT call fig.show()."
        )
    )
    title: str = Field(description="A concise, descriptive title for the chart.")


class PolarsCodeGeneration(BaseModel):
    """Polars transformation snippet produced by the ``table_agent`` worker."""

    thought_process: str = Field(description="Step-by-step reasoning on how to transform the data.")
    polars_code: str = Field(
        description=(
            "Executable Python Polars code. "
            "Assume `import polars as pl` is done and the input DataFrame "
            "is `df`. Do not use pandas. Assign the result to `result_df`."
        )
    )


class RAGSearchPlan(BaseModel):
    """Search request emitted by the ``rag_agent`` worker against Qdrant news."""

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
    """DuckDuckGo fallback search plan from the ``web_search`` worker."""

    thought_process: str = Field(description="Reasoning about what to search for on the internet.")
    search_queries: list[str] = Field(
        description="List of 1-3 search queries to execute.",
        min_length=1,
        max_length=3,
    )


class DownloadPlan(BaseModel):
    """On-demand ingestion request from the ``downloader_agent`` worker.

    One worker serves five sources; ``source`` selects which id fields apply.
    World Bank ids come from sql_agent's ``database_indicators`` lookup; Yahoo
    tickers, Binance pair symbols, FRED series ids and Eurostat dataset codes have
    no master catalogue, so they are inferred from the user's request (Apple →
    ``AAPL``, Solana → ``SOLUSDT``, state unemployment → ``CAUR``, EU regional GDP
    per capita → ``nama_10r_2gdp``).
    """

    thought_process: str = Field(
        description="Reasoning about what data to download and from which source."
    )
    source: Literal["worldbank", "yahoo", "binance", "fred", "eurostat"] = Field(
        description="Which data source to download from."
    )
    indicator_id: str | None = Field(
        default=None,
        description=(
            "World Bank indicator ID (e.g. 'NY.GDP.MKTP.CD'). Required when source='worldbank'."
        ),
    )
    db_id: int | None = Field(
        default=None,
        description="World Bank database ID (e.g. 2 for WDI). Required when source='worldbank'.",
    )
    ticker: str | None = Field(
        default=None,
        description=("Yahoo Finance ticker (e.g. 'AAPL', '^GSPC'). Required when source='yahoo'."),
    )
    symbol: str | None = Field(
        default=None,
        description=(
            "Full Binance spot pair symbol, USDT-quoted (e.g. 'BTCUSDT'). "
            "Required when source='binance'."
        ),
    )
    series_id: str | None = Field(
        default=None,
        description=(
            "A representative single-state FRED series id for the concept "
            "(e.g. 'CAUR' for state unemployment, 'CAPCPI' for per-capita income); "
            "the whole 50-state + DC panel is fetched from it. Required when source='fred'."
        ),
    )
    dataset: str | None = Field(
        default=None,
        description=(
            "Eurostat dataset code for the EU regional concept (e.g. 'nama_10r_2gdp' for "
            "regional GDP, 'lfst_r_lfu3rt' for unemployment, 'demo_r_pjanaggr3' for "
            "population); the whole NUTS-2 region panel is fetched from it. Required when "
            "source='eurostat'."
        ),
    )
    filters: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional Eurostat dimension filters pinning the dataset's extra dimensions to "
            "one category so a single clean series is returned (e.g. {'unit': 'EUR_HAB'} for "
            "GDP per capita, or {'sex': 'T', 'age': 'Y_GE15', 'unit': 'PC'} for an "
            "unemployment rate). Any dimension left out falls back to its first category. "
            "Only used when source='eurostat'."
        ),
    )


class ChatSynthesis(BaseModel):
    """Final markdown answer composed by the ``chat_agent`` worker."""

    response: str = Field(
        description=(
            "A complete, well-formatted response for the user. Use markdown where appropriate."
        )
    )


class GuardrailDecision(BaseModel):
    """Output of the guardrail screening step on each new user message."""

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
    """One turn of chat history (``role`` / ``content``)."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Body accepted by ``POST /chat/stream``."""

    user_message: str
    chat_history: list[ChatMessage] = Field(default_factory=list)


class PlotInterpretationRequest(BaseModel):
    """Body accepted by ``POST /plots/interpret``.

    Args:
        image_base64: Base64-encoded PNG of the rendered Plotly chart.
        mode: ``no_hallucinations`` for strict description; anything else
            switches to interpretive analyst mode.
        chart_context: Optional human-supplied caption fed to the LLM.
    """

    image_base64: str
    mode: str = "no_hallucinations"
    chart_context: str = ""


class TokenUsage(BaseModel):
    """Per-call token counts returned alongside every model response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class PlotInterpretationResponse(BaseModel):
    """Response from ``POST /plots/interpret``."""

    description: str
    mode: str
    model: str
    usage: TokenUsage | None = None
