"""LangGraph multi-agent orchestration for the AI analyst.

Defines one class per worker (guardrail, supervisor, sql, plotly, table, rag,
web search, downloader, chat) plus the top-level :class:`MacroAgentGraph` that
stitches them into a ``StateGraph``. The supervisor decides the next worker on
each tick by emitting a :class:`SupervisorDecision`; workers contribute
:class:`AgentState` deltas (messages, artifacts, trace lines, a coarse
``last_worker_status`` tag) that LangGraph's channel reducers merge into the
running state.
"""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .prompts import (
    GUARDRAIL_SYSTEM_PROMPT,
    sql_agent_step_prompt,
    supervisor_system_prompt,
)
from .schemas import (
    AgentState,
    ChatSynthesis,
    DownloadPlan,
    GuardrailDecision,
    PlotlyCodeGeneration,
    PolarsCodeGeneration,
    RAGSearchPlan,
    SQLGeneration,
    SupervisorDecision,
    WebSearchPlan,
)
from .tools import (
    encode_data_for_sandbox,
    execute_code_in_sandbox,
    get_database_schema_text,
    get_news_topics,
    ingest_data,
    run_sql_query,
    search_qdrant_news,
    web_search,
)

logger = logging.getLogger(__name__)

WORKER_NAMES = [
    "sql_agent",
    "plotly_agent",
    "table_agent",
    "rag_agent",
    "web_search",
    "downloader_agent",
    "chat_agent",
]

# Maximum number of recent (user, assistant) turns surfaced to a worker for
# follow-up disambiguation. Three turns is a good balance between context and
# prompt size — the supervisor still has the full history.
WORKER_HISTORY_TURNS = 3


def _format_chat_history(messages: list, max_turns: int = WORKER_HISTORY_TURNS) -> str:
    """Render the last ``max_turns`` messages as a compact text block.

    Workers don't see ``state["messages"]`` natively; passing the trailing
    chat turns into their prompts lets them disambiguate follow-ups like
    "now compare with Germany" without the supervisor having to spell that
    out in `isolated_worker_task`.
    """
    if not messages:
        return ""
    trimmed = messages[-max_turns:]
    lines: list[str] = []
    for msg in trimmed:
        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {str(msg.content).strip()}")
        elif isinstance(msg, AIMessage):
            text = str(msg.content).strip()
            if len(text) > 400:
                text = text[:400] + "…"
            lines.append(f"ASSISTANT: {text}")
    if not lines:
        return ""
    return "RECENT CHAT HISTORY (for disambiguating follow-ups):\n" + "\n".join(lines) + "\n\n"


# Regex shortcuts for the heuristic guardrail. Anything matching the first
# pattern is auto-allowed (no LLM call); anything matching the second is
# auto-rejected. Everything else escalates to the LLM screening step.
_GUARDRAIL_ALLOW_SHORT_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|"
    r"thanks|thank\s*you|ok|okay|cool|nice|got\s*it|"
    r"yes|no|sure|please|continue|more)[\s\.\!\?]*$",
    re.IGNORECASE,
)
_GUARDRAIL_BLOCK_RE = re.compile(
    r"\b(porn|nsfw|naked|nude|explicit|"
    r"kill\s+(yourself|myself)|suicide\s+(method|note|plan)|"
    r"how\s+to\s+(make|build|synthesi[sz]e)\s+(bomb|explosive|meth|cocaine|heroin|fentanyl)|"
    r"child\s+(porn|abuse|sexual)|cp\s+download|"
    r"malware|ransomware|keylogger|trojan|rootkit)\b",
    re.IGNORECASE,
)
# Topical keywords that mark a message as clearly in-scope for the dashboard.
# When present, the LLM screen is skipped — saves ~30s on routine questions.
_GUARDRAIL_IN_SCOPE_RE = re.compile(
    r"\b(gdp|inflation|unemployment|economy|economic|economi[ce]s?|finance|"
    r"financial|trade|export|import|currency|exchange\s*rate|interest\s*rate|"
    r"debt|deficit|budget|fiscal|monetary|bank|central\s*bank|fed|ecb|"
    r"stock|equit(y|ies)|market|index|indices|ticker|share|ohlc|nasdaq|s\s*&\s*p|"
    r"company|companies|earnings|dividend|portfolio|"
    r"world\s*bank|imf|wto|oecd|brics|eurozone|"
    r"demograph[a-z]+|population|fertility|mortality|life\s*expectancy|"
    r"education|literacy|enrol[lt]ment|"
    r"health|disease|hospital|"
    r"environment|emission|co2|climate|"
    r"politic[a-z]*|government|election|policy|sociolog[a-z]+|"
    r"econometric[a-z]*|regression|forecast|time\s*series|clustering|"
    r"chart|plot|graph|visuali[sz]e|table|data\s*science|"
    r"country|countries|nation|"
    r"explain|what\s+is|what\s+are|how\s+does|why|definition|formula|"
    r"compare|trend|over\s+time|since|between|"
    r"russia|usa|united\s*states|china|germany|france|uk|india|brazil|japan|"
    r"dashboard|agent|assistant|model|llm)\b",
    re.IGNORECASE,
)


class GuardrailAgent:
    """Screens the user's latest message with a heuristic-first, LLM-fallback flow.

    The fast path is a pair of regexes: short benign greetings and clearly
    in-scope keywords auto-pass without an LLM call (saves ~30s per turn on
    routine questions); obvious red-flag patterns auto-block. Only messages
    that don't match either side escalate to the structured-output LLM.
    """

    SYSTEM_PROMPT = GUARDRAIL_SYSTEM_PROMPT

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM used for the screening fallback."""
        self.llm = llm

    @staticmethod
    def _heuristic_verdict(message: str) -> str | None:
        """Return ``"allow"``, ``"block"``, or ``None`` (escalate to LLM)."""
        stripped = message.strip()
        if not stripped:
            return "allow"
        if _GUARDRAIL_BLOCK_RE.search(stripped):
            return "block"
        if _GUARDRAIL_ALLOW_SHORT_RE.match(stripped):
            return "allow"
        # A clearly in-scope keyword plus a reasonable length → auto-allow.
        # Short messages without in-scope keywords still escalate.
        if len(stripped) <= 600 and _GUARDRAIL_IN_SCOPE_RE.search(stripped):
            return "allow"
        return None

    async def ainvoke(self, state: AgentState) -> dict:
        """Screen the latest user message and emit a guardrail-decision delta."""
        last_user_msg = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                last_user_msg = str(msg.content)
                break
        if not last_user_msg.strip():
            return {
                "guardrail_blocked": False,
                "guardrail_message": "",
                "trace": ["guardrail: empty message — passing through"],
            }

        verdict = self._heuristic_verdict(last_user_msg)
        if verdict == "allow":
            return {
                "guardrail_blocked": False,
                "guardrail_message": "",
                "trace": ["guardrail: heuristic allow (no LLM call)"],
            }
        if verdict == "block":
            logger.info("guardrail: heuristic block — pattern match")
            refusal = (
                "I can only help with macroeconomics, finance, politics, "
                "sociology, data science and related dashboard topics. "
                "Please rephrase your question."
            )
            return {
                "guardrail_blocked": True,
                "guardrail_message": refusal,
                "trace": ["guardrail: heuristic block"],
            }

        # Ambiguous — fall back to the LLM screen.
        try:
            structured_llm = self.llm.with_structured_output(GuardrailDecision)
            decision: GuardrailDecision = await structured_llm.ainvoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=last_user_msg),
                ]
            )
            if decision.is_inappropriate:
                logger.info("guardrail: LLM blocked — %s", decision.reason)
                refusal = (
                    decision.refusal_message.strip()
                    or "I can only help with macroeconomics, politics, "
                    "sociology, data science and related dashboard topics. "
                    "Please rephrase your question."
                )
                return {
                    "guardrail_blocked": True,
                    "guardrail_message": refusal,
                    "trace": ["guardrail: LLM blocked"],
                }
            return {
                "guardrail_blocked": False,
                "guardrail_message": "",
                "trace": ["guardrail: LLM allow"],
            }
        except Exception as exc:
            logger.warning("guardrail: error – %s — passing through", exc)
            return {
                "guardrail_blocked": False,
                "guardrail_message": "",
                "trace": [f"guardrail: error ({exc}) — passing through"],
            }


class MacroSupervisorAgent:
    """Executive supervisor: plans, picks the next worker, decides FINISH."""

    def __init__(self, llm: ChatOpenAI, max_retries: int = 3):
        """Bind the structured-output LLM and per-worker retry budget."""
        self.llm = llm
        self.max_retries = max_retries

    @staticmethod
    def _summarize_artifacts(artifacts: dict[str, Any]) -> str:
        """Build a compact metadata-only summary; never include raw data rows."""
        if not artifacts:
            return "No artifacts stored yet."
        lines: list[str] = []
        for key, value in artifacts.items():
            if key in ("latest_data", "latest_table"):
                rows = value.get("rows", value.get("records", []))
                cols = value.get("columns", [])
                lines.append(f"- {key}: {len(rows)} rows, columns={cols}")
            elif key == "latest_plotly":
                lines.append(f"- latest_plotly: chart titled '{value.get('title', 'untitled')}'")
            elif key == "latest_rag_results":
                count = len(value) if isinstance(value, list) else "?"
                lines.append(f"- latest_rag_results: {count} articles")
            elif key == "latest_web_results":
                count = len(value) if isinstance(value, list) else "?"
                lines.append(f"- latest_web_results: {count} results")
            else:
                lines.append(f"- {key}: (stored)")
        return "\n".join(lines)

    @staticmethod
    def _trim_results_history(results: list[str], keep_verbatim: int = 2) -> str:
        """Keep the most recent ``keep_verbatim`` results verbatim, summarise the rest.

        Without this trim, ``worker_results`` (annotated with ``operator.add``)
        grows unboundedly: each subsequent supervisor decision re-reads every
        prior worker's full output, which is both slow and noisy.
        """
        if not results:
            return "No workers have been called yet."
        if len(results) <= keep_verbatim:
            return "\n---\n".join(results)
        older = results[:-keep_verbatim]
        recent = results[-keep_verbatim:]
        older_block = "\n".join(f"- (earlier) {r.splitlines()[0][:200]}" for r in older)
        return (
            f"EARLIER RESULTS (summarised — full text dropped to keep prompt small):\n"
            f"{older_block}\n\n"
            f"RECENT RESULTS (verbatim):\n" + "\n---\n".join(recent)
        )

    def _build_system_prompt(self, state: AgentState) -> str:
        """Compose the supervisor system prompt from the current state."""
        current_plan = state.get("current_plan") or "No plan created yet."
        last_worker = state.get("last_worker") or "None"
        retry_count = state.get("retry_count", 0)
        last_status = state.get("last_worker_status") or "UNKNOWN"

        results_history = self._trim_results_history(list(state.get("worker_results", [])))
        artifacts_summary = self._summarize_artifacts(state.get("artifacts", {}))

        retry_status = (
            f"Last worker: '{last_worker}', consecutive retries: {retry_count}/{self.max_retries}"
        )
        if retry_count >= self.max_retries:
            retry_instruction = (
                f"CRITICAL: Maximum retries ({self.max_retries}) reached for "
                f"'{last_worker}'. You MUST NOT call this worker again right now. "
                "Change your plan or use a different worker."
            )
        else:
            retry_instruction = (
                "If the last worker's result was poor or incorrect, "
                "you may retry with modified instructions."
            )

        return supervisor_system_prompt(
            current_plan=current_plan,
            results_history=results_history,
            artifacts_summary=artifacts_summary,
            retry_status=retry_status,
            retry_instruction=retry_instruction,
            last_worker_status=last_status,
        )

    async def ainvoke(self, state: AgentState) -> dict:
        """Pick the next worker (or FINISH) given the current state."""
        try:
            logger.info("Router: deciding next worker...")
            system_prompt = self._build_system_prompt(state)
            messages: list = [SystemMessage(content=system_prompt)]
            for msg in state.get("messages", []):
                messages.append(msg)

            structured_llm = self.llm.with_structured_output(SupervisorDecision)
            decision: SupervisorDecision = await structured_llm.ainvoke(messages)

            last_worker = state.get("last_worker")
            current_retries = state.get("retry_count", 0)

            if decision.next_worker == last_worker and decision.next_worker != "FINISH":
                new_retry = current_retries + 1
            else:
                new_retry = 0

            if new_retry > self.max_retries:
                logger.info(
                    "Router: max retries reached for '%s', falling back to chat_agent",
                    last_worker,
                )
                return {
                    "current_plan": (
                        decision.updated_plan + f"\n[System: max retries hit for {last_worker}]"
                    ),
                    "next_worker": "chat_agent",
                    "isolated_worker_task": (
                        f"The system repeatedly failed using the {last_worker} agent. "
                        "Inform the user and suggest what they could try differently."
                    ),
                    "last_worker": "chat_agent",
                    "retry_count": 0,
                    "trace": [f"Router: max retries for {last_worker}, falling back to chat_agent"],
                }

            logger.info("Router: selected '%s'", decision.next_worker)
            return {
                "current_plan": decision.updated_plan,
                "next_worker": decision.next_worker,
                "isolated_worker_task": decision.isolated_worker_task,
                "last_worker": decision.next_worker,
                "retry_count": new_retry,
                "trace": [f"Router: selected {decision.next_worker}"],
            }
        except Exception as exc:
            logger.exception("Router: critical error during decision")
            return {
                "next_worker": "FINISH",
                "isolated_worker_task": (
                    "I apologise, but I encountered an internal planning error. "
                    "Please try rephrasing your request."
                ),
                "trace": [f"Router: critical error – {exc}"],
            }


def _detect_market_needs_download(previous_steps: list[dict]) -> str | None:
    """Decide whether an empty SQL result means an untracked Yahoo/Binance asset.

    Unlike World Bank (which has the ``database_indicators`` master catalogue),
    Yahoo and Binance have no catalogue: the only signal that an asset can be
    downloaded on demand is that a lookup against its ``*_metadata`` /
    ``*_historical_prices`` tables came back empty. To avoid a false positive
    when the asset *is* tracked but simply has no rows for the requested filter,
    we suppress the signal if any ``*_metadata`` step returned rows.

    Args:
        previous_steps: The sql_agent's executed steps (each with ``query`` and
            ``result``), in order.

    Returns:
        ``"yahoo"`` / ``"binance"`` when the empty result looks like an
        untracked asset, otherwise ``None``.
    """

    def touched(substr: str) -> bool:
        return any(substr in s["query"].lower() for s in previous_steps)

    def meta_found(table: str) -> bool:
        return any(
            table in s["query"].lower() and s["result"].get("row_count", 0) > 0
            for s in previous_steps
        )

    if (touched("yahoo_metadata") or touched("yahoo_historical_prices")) and not meta_found(
        "yahoo_metadata"
    ):
        return "yahoo"
    if (touched("binance_metadata") or touched("binance_historical_prices")) and not meta_found(
        "binance_metadata"
    ):
        return "binance"
    return None


class SQLAgent:
    """Worker that issues up to ``MAX_SQL_STEPS`` read-only SELECTs against Postgres."""

    MAX_SQL_STEPS = 5

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM for SQL generation."""
        self.llm = llm

    @staticmethod
    def _build_history_block(previous_steps: list[dict]) -> str:
        if not previous_steps:
            return ""
        parts: list[str] = []
        for i, step in enumerate(previous_steps, 1):
            rows = step["result"].get("rows", [])
            sample = rows[:5]
            parts.append(
                f"--- Step {i} ---\n"
                f"Thought: {step['thought']}\n"
                f"Query: {step['query']}\n"
                f"Rows returned: {step['result'].get('row_count', 0)}\n"
                f"Columns: {step['result'].get('columns', [])}\n"
                f"Sample rows (first 5): {json.dumps(sample, default=str)[:1500]}"
            )
        return "\nPREVIOUS EXPLORATION STEPS:\n" + "\n\n".join(parts) + "\n"

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        chat_history_block = _format_chat_history(list(state.get("messages", [])))
        logger.info("sql_agent: starting multi-step exploration")
        try:
            schema_text = get_database_schema_text()
            structured_llm = self.llm.with_structured_output(SQLGeneration)

            previous_steps: list[dict] = []
            final_result = None

            for step_num in range(1, self.MAX_SQL_STEPS + 1):
                prompt = sql_agent_step_prompt(
                    schema_text=schema_text,
                    chat_history_block=chat_history_block,
                    task=task,
                    history_block=self._build_history_block(previous_steps),
                )
                gen: SQLGeneration = await structured_llm.ainvoke([SystemMessage(content=prompt)])

                result = await run_sql_query(gen.sql_query)

                if result.get("error"):
                    previous_steps.append(
                        {
                            "thought": gen.thought_process,
                            "query": gen.sql_query,
                            "result": {
                                "error": result["error"],
                                "rows": [],
                                "row_count": 0,
                                "columns": [],
                            },
                        }
                    )
                    continue

                previous_steps.append(
                    {
                        "thought": gen.thought_process,
                        "query": gen.sql_query,
                        "result": result,
                    }
                )

                if gen.is_final_step:
                    final_result = result
                    logger.info(
                        "sql_agent: final step reached at step %d — %d rows",
                        step_num,
                        result.get("row_count", 0),
                    )
                    break

            if final_result is None:
                for step in reversed(previous_steps):
                    if step["result"].get("rows"):
                        final_result = step["result"]
                        break

            if final_result is None or not final_result.get("rows"):
                step_lines = []
                indicator_match_step: dict | None = None
                for i, s in enumerate(previous_steps):
                    err = s["result"].get("error")
                    info = err if err else f"{s['result'].get('row_count', 0)} rows"
                    step_lines.append(f"  Step {i + 1}: {s['query'][:120]} -> {info}")
                    query_lower = s["query"].lower()
                    if "database_indicators" in query_lower and s["result"].get("row_count", 0) > 0:
                        indicator_match_step = s
                steps_summary = "\n".join(step_lines)

                if indicator_match_step is not None:
                    candidate_rows = indicator_match_step["result"].get("rows", [])[:5]
                    db_id_match = re.search(
                        r"database_id\s*=\s*(\d+)",
                        indicator_match_step["query"],
                        re.IGNORECASE,
                    )
                    db_id_value = db_id_match.group(1) if db_id_match else "2"
                    first_indicator_id = (
                        candidate_rows[0].get("id") if candidate_rows else "(unknown)"
                    )
                    candidate_lines = [
                        f"    indicator_id={row.get('id', '?')} | "
                        f"description={(row.get('description') or '')[:140]}"
                        for row in candidate_rows
                    ]
                    candidates_block = "\n".join(candidate_lines) or "    (no candidates extracted)"
                    return {
                        "worker_results": [
                            f"SQL_AGENT NEEDS_DOWNLOAD: indicator found in "
                            f"database_indicators (db_id={db_id_value}) but the "
                            f"`indicators` table has no rows for it yet — it has "
                            f"not been downloaded.\n"
                            f"Best match: indicator_id={first_indicator_id}, "
                            f"db_id={db_id_value}.\n"
                            f"All candidates from database_indicators:\n"
                            f"{candidates_block}\n"
                            f"Route to downloader_agent and pass the EXACT "
                            f"indicator_id and db_id of the candidate that best "
                            f"matches the user's request, then retry sql_agent.\n"
                            f"Steps taken:\n{steps_summary}"
                        ],
                        "last_worker_status": "NEEDS_DOWNLOAD",
                        "trace": [
                            f"sql_agent: NEEDS_DOWNLOAD "
                            f"(candidate={first_indicator_id}, db={db_id_value}, "
                            f"{len(previous_steps)} steps)"
                        ],
                    }

                market_source = _detect_market_needs_download(previous_steps)
                if market_source is not None:
                    if market_source == "yahoo":
                        asset_kind = "ticker"
                        infer_hint = "the Yahoo ticker (e.g. Apple → AAPL, S&P 500 → ^GSPC)"
                    else:
                        asset_kind = "spot pair"
                        infer_hint = (
                            "the FULL Binance pair symbol, USDT-quoted "
                            "(e.g. Bitcoin → BTCUSDT, Solana → SOLUSDT)"
                        )
                    return {
                        "worker_results": [
                            f"SQL_AGENT NEEDS_DOWNLOAD (source={market_source}): the "
                            f"requested {asset_kind} is not present in "
                            f"`{market_source}_metadata` — it is not tracked yet but "
                            f"CAN be downloaded on demand (there is no master "
                            f"catalogue for {market_source}).\n"
                            f"Route to downloader_agent with source={market_source} "
                            f"and {infer_hint} for the asset the user named, then "
                            f"retry sql_agent.\n"
                            f"Steps taken:\n{steps_summary}"
                        ],
                        "last_worker_status": "NEEDS_DOWNLOAD",
                        "trace": [
                            f"sql_agent: NEEDS_DOWNLOAD (source={market_source}, "
                            f"{len(previous_steps)} steps)"
                        ],
                    }

                return {
                    "worker_results": [
                        f"SQL_AGENT EMPTY: Could not retrieve data after "
                        f"{len(previous_steps)} steps.\nSteps taken:\n{steps_summary}"
                    ],
                    "last_worker_status": "EMPTY",
                    "trace": [f"sql_agent: empty after {len(previous_steps)} steps"],
                }

            truncated_note = " [TRUNCATED]" if final_result.get("truncated") else ""
            steps_trace = " → ".join(
                f"step{i + 1}({s['result'].get('row_count', '?')}rows)"
                for i, s in enumerate(previous_steps)
            )
            logger.info(
                "sql_agent: completed — %d rows after %d steps",
                final_result["row_count"],
                len(previous_steps),
            )
            return {
                "worker_results": [
                    f"SQL_AGENT SUCCESS: {final_result['row_count']} rows returned "
                    f"after {len(previous_steps)} steps. "
                    f"Columns: {final_result['columns']}.{truncated_note}"
                ],
                "artifacts": {"latest_data": final_result},
                "last_worker_status": "SUCCESS",
                "trace": [
                    f"sql_agent: {steps_trace} → "
                    f"final {final_result['row_count']} rows, "
                    f"cols={final_result['columns']}"
                ],
            }
        except Exception as exc:
            logger.exception("sql_agent: error")
            return {
                "worker_results": [f"SQL_AGENT ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"sql_agent: exception – {exc}"],
            }


class PlotlyAgent:
    """Worker that generates Plotly code and runs it in the python sandbox."""

    MAX_FIX_ATTEMPTS = 3

    PREAMBLE = """You are a Plotly visualisation expert.

RULES:
- Input data is available as `data` (list of dicts).
- Create a clear, informative chart. Use appropriate chart type.
- Assign the final figure to `fig`. Do NOT call fig.show().
- Use ONLY `plotly.graph_objects` (imported as `go`). Do NOT import or use `plotly.express`.
- Handle possible None/null values gracefully."""

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM for code generation."""
        self.llm = llm

    @classmethod
    def _build_plotly_prompt(
        cls,
        task: str,
        columns: list,
        rows_count: int,
        sample: list,
        attempts: list[dict],
    ) -> str:
        history_block = ""
        if attempts:
            parts: list[str] = []
            for i, att in enumerate(attempts, 1):
                parts.append(
                    f"--- Attempt {i} ({att['mode']}) ---\n"
                    f"Code:\n{att['code']}\n"
                    f"Issue:\n{att['issue'][:1500]}"
                )
            history_block = (
                "\n\nPREVIOUS FAILED ATTEMPTS (fix the issues described below):\n"
                + "\n\n".join(parts)
            )

        return f"""{cls.PREAMBLE}

================================================================
RUNTIME STATE for this chart (changes per call):

DATA SCHEMA:
- Columns: {columns}
- Total rows: {rows_count}
- Sample (first 3 rows): {json.dumps(sample, default=str)[:1500]}
{history_block}

YOUR TASK:
{task}"""

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        artifacts = state.get("artifacts", {})
        logger.info("plotly_agent: generating visualization")
        try:
            data = artifacts.get("latest_data") or artifacts.get("latest_table") or {}
            rows = data.get("rows", data.get("records", []))
            if not rows:
                return {
                    "worker_results": ["PLOTLY_AGENT BLOCKED: No data in artifacts to visualise."],
                    "last_worker_status": "BLOCKED",
                    "trace": ["plotly_agent: no data available"],
                }

            columns = data.get("columns", list(rows[0].keys()) if rows else [])
            sample = rows[:3]
            data_b64 = encode_data_for_sandbox(rows)
            structured_llm = self.llm.with_structured_output(PlotlyCodeGeneration)

            attempts: list[dict] = []
            last_issue = ""
            last_code = ""

            for attempt_num in range(1, self.MAX_FIX_ATTEMPTS + 1):
                prompt = self._build_plotly_prompt(
                    task=task,
                    columns=columns,
                    rows_count=len(rows),
                    sample=sample,
                    attempts=attempts,
                )
                gen: PlotlyCodeGeneration = await structured_llm.ainvoke(
                    [SystemMessage(content=prompt)]
                )
                last_code = gen.plotly_code

                sandbox_code = (
                    "import json, base64\n"
                    "import plotly.graph_objects as go\n\n"
                    f'data = json.loads(base64.b64decode("{data_b64}").decode())\n\n'
                    f"{gen.plotly_code}\n\n"
                    "_figure_json = fig.to_json()\n"
                    "print(json.dumps({'figure_json': _figure_json}))\n"
                )

                result = await execute_code_in_sandbox(sandbox_code)

                if not result.get("success"):
                    last_issue = (result.get("stderr") or "").strip()
                    attempts.append(
                        {
                            "code": gen.plotly_code,
                            "issue": f"sandbox error: {last_issue}",
                            "mode": "sandbox-error",
                        }
                    )
                    logger.info(
                        "plotly_agent: attempt %d sandbox failed — %s",
                        attempt_num,
                        last_issue[:120],
                    )
                    continue

                try:
                    sandbox_payload = json.loads(result["stdout"])
                    figure_json = str(sandbox_payload["figure_json"])
                except Exception as parse_exc:
                    last_issue = f"could not parse sandbox output: {parse_exc}"
                    attempts.append(
                        {
                            "code": gen.plotly_code,
                            "issue": last_issue,
                            "mode": "sandbox-output",
                        }
                    )
                    continue

                logger.info(
                    "plotly_agent: chart '%s' generated on attempt %d",
                    gen.title,
                    attempt_num,
                )
                return {
                    "worker_results": [
                        f"PLOTLY_AGENT SUCCESS: chart '{gen.title}' generated "
                        f"on attempt {attempt_num}."
                    ],
                    "artifacts": {
                        "latest_plotly": {
                            "figure_json": figure_json,
                            "title": gen.title,
                        }
                    },
                    "last_worker_status": "SUCCESS",
                    "trace": [
                        f"plotly_agent: chart '{gen.title}' generated (attempt {attempt_num})"
                    ],
                }

            return {
                "worker_results": [
                    f"PLOTLY_AGENT ERROR: all {self.MAX_FIX_ATTEMPTS} sandbox "
                    f"attempts failed.\nLast issue: "
                    f"{last_issue[:1500]}\nLast code:\n{last_code}"
                ],
                "last_worker_status": "ERROR",
                "trace": [
                    f"plotly_agent: failed after {self.MAX_FIX_ATTEMPTS} "
                    f"attempts – {last_issue[:120]}"
                ],
            }
        except Exception as exc:
            logger.exception("plotly_agent: error")
            return {
                "worker_results": [f"PLOTLY_AGENT ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"plotly_agent: exception – {exc}"],
            }


class TableAgent:
    """Worker that transforms tabular data via LLM-generated Polars snippets."""

    MAX_FIX_ATTEMPTS = 3

    PREAMBLE = """You are a senior data engineer using the Python `polars` library.

RULES:
- Write clean, idiomatic Polars code (pl.col, select, with_columns, group_by, agg…).
- `import polars as pl` and the DataFrame `df` already exist – do NOT recreate them.
- Assign the final transformed DataFrame to `result_df`.
- Do NOT use pandas."""

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM for Polars code generation."""
        self.llm = llm

    @classmethod
    def _build_polars_prompt(
        cls,
        task: str,
        schema_lines: str,
        columns: list,
        rows_count: int,
        attempts: list[dict],
    ) -> str:
        history_block = ""
        if attempts:
            parts: list[str] = []
            for i, att in enumerate(attempts, 1):
                parts.append(
                    f"--- Attempt {i} ---\n"
                    f"Code:\n{att['code']}\n"
                    f"Sandbox stderr:\n{att['stderr'][:1500]}"
                )
            history_block = (
                "\n\nPREVIOUS FAILED ATTEMPTS (study the tracebacks and fix them):\n"
                + "\n\n".join(parts)
            )

        return f"""{cls.PREAMBLE}

================================================================
RUNTIME STATE for this transform (changes per call):

INPUT DATA SCHEMA (variable: `df`):
{schema_lines}

Columns: {columns}
Total rows: {rows_count}
{history_block}

YOUR TASK:
{task}"""

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        artifacts = state.get("artifacts", {})
        logger.info("table_agent: starting data transformation")
        try:
            data = artifacts.get("latest_data") or artifacts.get("latest_table") or {}
            rows = data.get("rows", data.get("records", []))
            if not rows:
                return {
                    "worker_results": ["TABLE_AGENT BLOCKED: No input data in artifacts."],
                    "last_worker_status": "BLOCKED",
                    "trace": ["table_agent: no data available"],
                }

            columns = data.get("columns", list(rows[0].keys()))
            sample_row = rows[0]
            schema_lines = "\n".join(f"  - {k}: {type(v).__name__}" for k, v in sample_row.items())
            data_b64 = encode_data_for_sandbox(rows)
            structured_llm = self.llm.with_structured_output(PolarsCodeGeneration)

            attempts: list[dict] = []
            last_error = ""
            last_code = ""

            for attempt_num in range(1, self.MAX_FIX_ATTEMPTS + 1):
                prompt = self._build_polars_prompt(
                    task=task,
                    schema_lines=schema_lines,
                    columns=columns,
                    rows_count=len(rows),
                    attempts=attempts,
                )
                gen: PolarsCodeGeneration = await structured_llm.ainvoke(
                    [SystemMessage(content=prompt)]
                )
                last_code = gen.polars_code

                sandbox_code = (
                    "import json, base64\n"
                    "import polars as pl\n\n"
                    f'_raw = json.loads(base64.b64decode("{data_b64}").decode())\n'
                    "df = pl.DataFrame(_raw)\n\n"
                    f"{gen.polars_code}\n\n"
                    'print(json.dumps({"columns": result_df.columns, '
                    '"rows": result_df.to_dicts(), '
                    '"row_count": result_df.height}, default=str))\n'
                )

                result = await execute_code_in_sandbox(sandbox_code)

                if result.get("success"):
                    parsed = json.loads(result["stdout"])
                    logger.info(
                        "table_agent: %d rows on attempt %d",
                        parsed["row_count"],
                        attempt_num,
                    )
                    return {
                        "worker_results": [
                            f"TABLE_AGENT SUCCESS: {parsed['row_count']} rows, "
                            f"columns={parsed['columns']} (attempt {attempt_num})."
                        ],
                        "artifacts": {
                            "latest_data": {
                                "rows": parsed["rows"],
                                "columns": parsed["columns"],
                                "row_count": parsed["row_count"],
                                "truncated": False,
                                "query": f"[polars transformation] {task[:100]}",
                            }
                        },
                        "last_worker_status": "SUCCESS",
                        "trace": [
                            f"table_agent: {parsed['row_count']} rows, "
                            f"cols={parsed['columns']} (attempt {attempt_num})"
                        ],
                    }

                last_error = (result.get("stderr") or "").strip()
                attempts.append({"code": gen.polars_code, "stderr": last_error})
                logger.info(
                    "table_agent: attempt %d failed — %s",
                    attempt_num,
                    last_error[:120],
                )

            return {
                "worker_results": [
                    f"TABLE_AGENT ERROR: all {self.MAX_FIX_ATTEMPTS} sandbox "
                    f"attempts failed.\nLast stderr: {last_error[:1500]}\n"
                    f"Last code:\n{last_code}"
                ],
                "last_worker_status": "ERROR",
                "trace": [
                    f"table_agent: failed after {self.MAX_FIX_ATTEMPTS} "
                    f"attempts – {last_error[:120]}"
                ],
            }
        except Exception as exc:
            logger.exception("table_agent: error")
            return {
                "worker_results": [f"TABLE_AGENT ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"table_agent: exception – {exc}"],
            }


class RAGAgent:
    """Worker that searches the Qdrant news corpus for the supervisor's question."""

    PREAMBLE = """You are a news-retrieval specialist.
You plan semantic search queries against a Qdrant vector database of news articles."""

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM for search-plan generation."""
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        chat_history_block = _format_chat_history(list(state.get("messages", [])))
        logger.info("rag_agent: starting news search")
        try:
            topics = get_news_topics()
            system_prompt = f"""{self.PREAMBLE}

AVAILABLE TOPICS: {topics}
AVAILABLE SENTIMENTS: positive, negative

================================================================
RUNTIME STATE (changes per call):
{chat_history_block}YOUR TASK:
{task}"""

            structured_llm = self.llm.with_structured_output(RAGSearchPlan)
            plan: RAGSearchPlan = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt)]
            )

            logger.info("rag_agent: searching for '%s'", plan.search_query[:80])
            result = await search_qdrant_news(
                query=plan.search_query,
                topic_filter=plan.topic_filter,
                sentiment_filter=plan.sentiment_filter,
                top_k=plan.top_k,
            )

            articles = result.get("articles", [])
            if not articles:
                msg = result.get("error") or result.get("message") or "No articles found."
                logger.info("rag_agent: %s", msg)
                status = "ERROR" if result.get("error") else "EMPTY"
                return {
                    "worker_results": [f"RAG_AGENT {status}: {msg}"],
                    "last_worker_status": status,
                    "trace": [f"rag_agent: {msg}"],
                }

            summaries: list[str] = []
            for i, art in enumerate(articles, 1):
                url = art.get("url", "")
                url_line = f"   URL: {url}" if url else "   URL: N/A"
                summaries.append(
                    f"{i}. [{art.get('topic', '')}|{art.get('sentiment', '')}] "
                    f"{art.get('title', '(no title)')}\n"
                    f"{url_line}\n"
                    f"   Source: {art.get('source', '')} | Published: {art.get('published', '')}\n"
                    f"   {(art.get('text', '') or '')[:400]}"
                )

            logger.info("rag_agent: found %d articles", len(articles))
            return {
                "worker_results": [
                    f"RAG_AGENT SUCCESS: {len(articles)} articles found.\n"
                    + "\n---\n".join(summaries)
                ],
                "artifacts": {"latest_rag_results": articles},
                "last_worker_status": "SUCCESS",
                "trace": [f"rag_agent: {len(articles)} articles for '{plan.search_query[:80]}'"],
            }
        except Exception as exc:
            logger.exception("rag_agent: error")
            return {
                "worker_results": [f"RAG_AGENT ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"rag_agent: exception – {exc}"],
            }


class WebSearchAgent:
    """DuckDuckGo fallback worker for questions outside the RAG / SQL surfaces."""

    PREAMBLE = """You are a web-research specialist.
Generate 1-3 focused search queries to find information on the internet."""

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM for search-query generation."""
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        chat_history_block = _format_chat_history(list(state.get("messages", [])))
        logger.info("web_search: starting internet search")
        try:
            system_prompt = f"""{self.PREAMBLE}

================================================================
RUNTIME STATE (changes per call):
{chat_history_block}YOUR TASK:
{task}"""

            structured_llm = self.llm.with_structured_output(WebSearchPlan)
            plan: WebSearchPlan = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt)]
            )

            result = await web_search(plan.search_queries)
            logger.info("web_search: queries=%s", plan.search_queries)

            hits = result.get("results", [])
            if not hits:
                error_msg = result.get("error", "No results found.")
                status = "ERROR" if result.get("error") else "EMPTY"
                return {
                    "worker_results": [f"WEB_SEARCH {status}: {error_msg}"],
                    "last_worker_status": status,
                    "trace": [f"web_search: {error_msg}"],
                }

            summaries: list[str] = []
            for h in hits:
                summaries.append(
                    f"- {h.get('title', '')}\n  {h.get('body', '')}\n  {h.get('href', '')}"
                )

            return {
                "worker_results": [
                    f"WEB_SEARCH SUCCESS: {len(hits)} results.\n" + "\n".join(summaries)
                ],
                "artifacts": {"latest_web_results": hits},
                "last_worker_status": "SUCCESS",
                "trace": [f"web_search: {len(hits)} results for {plan.search_queries}"],
            }
        except Exception as exc:
            logger.exception("web_search: error")
            return {
                "worker_results": [f"WEB_SEARCH ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"web_search: exception – {exc}"],
            }


class DownloaderAgent:
    """Worker that on-demand-ingests one unit of data from WB / Yahoo / Binance / FRED."""

    EXTRACT_SYSTEM_PROMPT = (
        "You decide which single unit of data to download on demand and from "
        "which `source`, then output structured fields.\n"
        "- source='worldbank': the supervisor's task ALREADY contains the exact "
        "`indicator_id` (e.g. 'NY.GDP.MKTP.CD') and `db_id` (e.g. 2), discovered "
        "by sql_agent from the `database_indicators` master table — read them out "
        "VERBATIM, never invent them. Set indicator_id and db_id.\n"
        "- source='yahoo': there is NO catalogue to look up, so infer the correct "
        "Yahoo Finance `ticker` from the asset the user named using your own "
        "knowledge (e.g. Apple → AAPL, S&P 500 → ^GSPC, Palantir → PLTR). Set "
        "ticker.\n"
        "- source='binance': there is NO catalogue, so infer the FULL Binance "
        "spot pair `symbol`, quoted in USDT, from the coin the user named "
        "(e.g. Bitcoin → BTCUSDT, Solana → SOLUSDT, Litecoin → LTCUSDT). Set "
        "symbol.\n"
        "- source='fred': US-state indicators. There is NO catalogue, so infer a "
        "representative single-state FRED `series_id` for the concept the user "
        "named (e.g. state unemployment → CAUR, per-capita personal income → "
        "CAPCPI, personal consumption → CAPCE); the whole 50-state + DC panel is "
        "fetched from that one series. Set series_id.\n"
        "The supervisor's task states which source to use; output only the "
        "fields for that source."
    )

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM that extracts the download plan."""
        self.llm = llm

    @staticmethod
    def _build_payload(plan: DownloadPlan) -> tuple[str, dict]:
        """Map a ``DownloadPlan`` to ``(identifier, /ingest body)`` by source.

        Args:
            plan: The structured download plan from the LLM.

        Returns:
            ``(identifier, payload)`` where ``identifier`` is the human-readable
            id used in status messages and ``payload`` is the ``/ingest`` body.

        Raises:
            ValueError: When required fields for the source are missing.
        """
        if plan.source == "worldbank":
            if not plan.indicator_id or plan.db_id is None:
                raise ValueError("worldbank download needs indicator_id and db_id")
            return plan.indicator_id, {
                "source": "worldbank",
                "indicator_id": plan.indicator_id,
                "db_id": plan.db_id,
            }
        if plan.source == "yahoo":
            if not plan.ticker:
                raise ValueError("yahoo download needs a ticker")
            return plan.ticker, {"source": "yahoo", "ticker": plan.ticker}
        if plan.source == "binance":
            if not plan.symbol:
                raise ValueError("binance download needs a symbol")
            return plan.symbol, {"source": "binance", "symbol": plan.symbol}
        if plan.source == "fred":
            if not plan.series_id:
                raise ValueError("fred download needs a series_id")
            return plan.series_id, {"source": "fred", "series_id": plan.series_id}
        raise ValueError(f"Unknown download source: {plan.source}")

    async def ainvoke(self, state: AgentState) -> dict:
        """Extract the download plan and call ``downloader_extra/ingest``."""
        task = state["isolated_worker_task"]
        logger.info("downloader_agent: extracting download plan from task")
        try:
            structured_llm = self.llm.with_structured_output(DownloadPlan)
            plan: DownloadPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=self.EXTRACT_SYSTEM_PROMPT),
                    HumanMessage(content=f"SUPERVISOR TASK:\n{task}"),
                ]
            )

            identifier, payload = self._build_payload(plan)
            logger.info(
                "downloader_agent: calling /ingest source=%s identifier=%s",
                plan.source,
                identifier,
            )
            result = await ingest_data(payload)

            if not result.get("success", False):
                error = result.get("error") or result.get("detail", "Unknown error")
                return {
                    "worker_results": [
                        f"DOWNLOADER_AGENT ERROR: {error} "
                        f"(source={plan.source}, identifier={identifier})"
                    ],
                    "last_worker_status": "ERROR",
                    "trace": [f"downloader_agent: failed – {plan.source}/{identifier} – {error}"],
                }

            status = result.get("status", "success")
            rows_inserted = result.get("rows_inserted", 0)
            return {
                "worker_results": [
                    f"DOWNLOADER_AGENT SUCCESS: source={plan.source}, "
                    f"identifier={identifier}, rows_inserted={rows_inserted}, "
                    f"status={status}. The data for {identifier} is now stored in "
                    f"Postgres — route back to sql_agent to fetch it."
                ],
                "last_worker_status": "SUCCESS",
                "trace": [
                    f"downloader_agent: {plan.source}/{identifier} – "
                    f"{rows_inserted} rows, status={status}"
                ],
            }
        except Exception as exc:
            logger.exception("downloader_agent: error")
            return {
                "worker_results": [f"DOWNLOADER_AGENT ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"downloader_agent: exception – {exc}"],
            }


class ChatAgent:
    """Worker that turns accumulated worker results into a final markdown answer."""

    PREAMBLE = (
        "You are a helpful macroeconomic analyst assistant. "
        "Synthesise information, explain economic concepts, or provide "
        "conversational responses. Use markdown formatting. "
        "Be concise but thorough. "
        "For mathematical expressions, use Streamlit-compatible "
        "markdown math syntax: inline as `$expr$` and block as "
        "`$$expr$$`. Never use `\\(...\\)` or `\\[...\\]` delimiters — "
        "they will not render in the chat UI."
    )

    def __init__(self, llm: ChatOpenAI):
        """Bind the structured-output LLM for the final synthesis."""
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        chat_history_block = _format_chat_history(list(state.get("messages", [])))
        logger.info("chat_agent: synthesising response")
        try:
            user_prompt = f"{chat_history_block}USER TASK (from the supervisor):\n{task}"

            structured_llm = self.llm.with_structured_output(ChatSynthesis)
            synthesis: ChatSynthesis = await structured_llm.ainvoke(
                [
                    SystemMessage(content=self.PREAMBLE),
                    HumanMessage(content=user_prompt),
                ]
            )

            return {
                "worker_results": [f"CHAT_AGENT: {synthesis.response}"],
                "last_worker_status": "SUCCESS",
                "trace": [f"chat_agent: synthesised response ({len(synthesis.response)} chars)"],
            }
        except Exception as exc:
            logger.exception("chat_agent: error")
            return {
                "worker_results": [f"CHAT_AGENT ERROR: {exc}"],
                "last_worker_status": "ERROR",
                "trace": [f"chat_agent: exception – {exc}"],
            }


class MacroAgentGraph:
    """Builds and wraps the LangGraph multi-agent graph."""

    # How big each streamed chunk of the supervisor draft is, and the delay
    # between chunks. The dashboard chat UI just appends each delta, so we
    # cut the draft into small bursts to keep the streaming feel without
    # paying for another LLM call.
    DRAFT_STREAM_CHUNK_CHARS = 24
    DRAFT_STREAM_DELAY_SECONDS = 0.01

    def __init__(
        self,
        base_url: str,
        model_name: str,
        fast_model_name: str,
        api_key: str,
        max_retries: int = 3,
        recursion_limit: int = 30,
    ):
        """Build the strong/fast ``ChatOpenAI`` pair and assemble the StateGraph.

        The two models share the endpoint and credentials but differ in
        capability. ``smart_llm`` (``model_name``) does the reasoning-heavy
        work — planning, SQL/Plotly code generation and the final answer —
        while ``fast_llm`` (``fast_model_name``) handles the cheap in-scope
        screen and the lightweight workers to cut latency and cost. An empty
        ``fast_model_name`` falls back to the strong model, preserving the
        previous single-model behaviour. ``UsageTracker`` is attached at the
        graph level, so token accounting spans both models automatically.
        """

        def _build_llm(model: str) -> ChatOpenAI:
            return ChatOpenAI(
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=0,
                max_retries=3,
                stream_usage=True,
            )

        self.smart_llm = _build_llm(model_name)
        self.fast_llm = _build_llm(fast_model_name or model_name)
        self.max_retries = max_retries
        self.recursion_limit = recursion_limit

        # Strong model: planning, code generation and the final answer.
        self.supervisor = MacroSupervisorAgent(llm=self.smart_llm, max_retries=max_retries)
        self.sql_agent = SQLAgent(llm=self.smart_llm)
        self.plotly_agent = PlotlyAgent(llm=self.smart_llm)
        self.chat_agent = ChatAgent(llm=self.smart_llm)

        # Fast model: in-scope screening + lightweight workers.
        self.guardrail = GuardrailAgent(llm=self.fast_llm)
        self.table_agent = TableAgent(llm=self.fast_llm)
        self.rag_agent = RAGAgent(llm=self.fast_llm)
        self.web_search_agent = WebSearchAgent(llm=self.fast_llm)
        self.downloader_agent = DownloaderAgent(llm=self.fast_llm)

        self.graph = self._build_graph()

    def _build_graph(self):
        """Wire every worker into a ``StateGraph`` with conditional edges."""
        builder = StateGraph(AgentState)

        builder.add_node("guardrail", self.guardrail.ainvoke)
        builder.add_node("supervisor", self.supervisor.ainvoke)
        builder.add_node("sql_agent", self.sql_agent.ainvoke)
        builder.add_node("plotly_agent", self.plotly_agent.ainvoke)
        builder.add_node("table_agent", self.table_agent.ainvoke)
        builder.add_node("rag_agent", self.rag_agent.ainvoke)
        builder.add_node("web_search", self.web_search_agent.ainvoke)
        builder.add_node("downloader_agent", self.downloader_agent.ainvoke)
        builder.add_node("chat_agent", self.chat_agent.ainvoke)

        builder.set_entry_point("guardrail")

        builder.add_conditional_edges(
            "guardrail",
            lambda state: "blocked" if state.get("guardrail_blocked") else "ok",
            {"blocked": END, "ok": "supervisor"},
        )

        builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next_worker"],
            {name: name for name in WORKER_NAMES} | {"FINISH": END},
        )

        for name in WORKER_NAMES:
            builder.add_edge(name, "supervisor")

        return builder.compile()

    @staticmethod
    def _build_initial_state(
        message: str,
        chat_history: list[dict],
    ) -> dict:
        """Convert raw chat history + the current message into a LangGraph state."""
        messages: list = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))

        return {
            "messages": messages,
            "worker_results": [],
            "current_plan": "",
            "next_worker": "",
            "isolated_worker_task": "",
            "artifacts": {},
            "last_worker": "",
            "last_worker_status": "UNKNOWN",
            "retry_count": 0,
            "trace": [],
            "guardrail_blocked": False,
            "guardrail_message": "",
        }

    _LEAK_PATTERN = re.compile(
        r"(SQL_AGENT|PLOTLY_AGENT|TABLE_AGENT|RAG_AGENT|WEB_SEARCH|"
        r"DOWNLOADER_AGENT|CHAT_AGENT|sql_agent|plotly_agent|table_agent|"
        r"rag_agent|web_search|downloader_agent|chat_agent|"
        r"NEEDS_DOWNLOAD|last_worker_status|isolated_worker_task|"
        r"sandbox|traceback|Traceback)",
    )

    @classmethod
    def _sanitize_draft(cls, draft: str) -> str:
        """Drop obvious internal-detail leaks from the supervisor's FINISH draft.

        The supervisor is told never to leak worker names / retry / SQL /
        sandbox details, but smaller models occasionally slip. This is a
        last-mile guard: any line containing a leak token is dropped. We do
        NOT rewrite the prose — we only strip leaking lines and trim.
        """
        if not draft:
            return draft
        kept_lines: list[str] = []
        for line in draft.splitlines():
            if cls._LEAK_PATTERN.search(line):
                continue
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines).strip()
        return cleaned or draft.strip()

    async def _stream_supervisor_draft(self, draft: str):
        """Chunk-stream the supervisor's drafted FINISH answer to the user.

        Replaces the old ``_stream_final_synthesis`` LLM call. The supervisor
        already wrote the polished markdown answer in ``isolated_worker_task``
        when it picked FINISH, so we just emit it in small bursts to preserve
        the streaming-chat feel — no additional model call, no risk of the
        synthesis step altering numbers.
        """
        if not draft:
            return
        for i in range(0, len(draft), self.DRAFT_STREAM_CHUNK_CHARS):
            yield draft[i : i + self.DRAFT_STREAM_CHUNK_CHARS]
            if self.DRAFT_STREAM_DELAY_SECONDS > 0:
                await asyncio.sleep(self.DRAFT_STREAM_DELAY_SECONDS)

    async def astream_events(
        self,
        message: str,
        chat_history: list[dict] | None = None,
        usage_tracker: Any | None = None,
    ):
        """Yield events the API layer relays to the chat UI.

        Event types:
          - {"type": "step", "node": <node_name>}
          - {"type": "token", "delta": <str>}
          - {"type": "final", "response": <str>, "artifacts": {...}}
          - {"type": "error", "response": <str>}
        """
        state = self._build_initial_state(message, chat_history or [])
        accumulated_artifacts: dict[str, Any] = {}
        last_isolated_task = ""
        final_state: dict[str, Any] = {}

        graph_config: dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if usage_tracker is not None:
            graph_config["callbacks"] = [usage_tracker]

        try:
            async for chunk in self.graph.astream(state, config=graph_config):
                for node_name, output in chunk.items():
                    if not isinstance(output, dict):
                        continue

                    yield {"type": "step", "node": node_name}

                    if output.get("artifacts"):
                        accumulated_artifacts.update(output["artifacts"])

                    if output.get("isolated_worker_task"):
                        last_isolated_task = output["isolated_worker_task"]

                    if output.get("guardrail_blocked"):
                        final_state = {
                            "guardrail_blocked": True,
                            "guardrail_message": output.get("guardrail_message", ""),
                        }
                    elif output.get("guardrail_message"):
                        final_state["guardrail_message"] = output["guardrail_message"]

                    if output.get("worker_results"):
                        final_state.setdefault("worker_results", [])
                        final_state["worker_results"].extend(output["worker_results"])

                    if output.get("messages"):
                        final_state.setdefault("messages", [])
                        final_state["messages"].extend(output["messages"])

            yield {"type": "step", "node": "FINISH"}

            if final_state.get("guardrail_blocked"):
                refusal = (
                    final_state.get("guardrail_message")
                    or "I can only help with macroeconomics, politics, "
                    "sociology, data science and related dashboard topics."
                )
                for ch in refusal:
                    yield {"type": "token", "delta": ch}
                yield {
                    "type": "final",
                    "response": refusal,
                    "artifacts": {},
                }
                return

            draft = self._sanitize_draft(last_isolated_task) or ("I could not produce a response.")

            collected: list[str] = []
            try:
                async for delta in self._stream_supervisor_draft(draft):
                    collected.append(delta)
                    yield {"type": "token", "delta": delta}
            except Exception:
                logger.exception("Supervisor draft streaming failed")
                if not collected:
                    for ch in draft:
                        yield {"type": "token", "delta": ch}
                yield {
                    "type": "final",
                    "response": "".join(collected) or draft,
                    "artifacts": accumulated_artifacts,
                }
                return

            yield {
                "type": "final",
                "response": "".join(collected) or draft,
                "artifacts": accumulated_artifacts,
            }
        except Exception as exc:
            logger.exception("Graph astream failed")
            yield {
                "type": "error",
                "response": f"An error occurred: {exc}",
            }
