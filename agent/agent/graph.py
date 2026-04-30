import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .schemas import (
    AgentState,
    ChatSynthesis,
    DownloadIndicatorPlan,
    PlotlyCodeGeneration,
    PolarsCodeGeneration,
    RAGSearchPlan,
    SQLGeneration,
    SupervisorDecision,
    WebSearchPlan,
)
from .tools import (
    download_indicator,
    encode_data_for_sandbox,
    execute_code_in_sandbox,
    get_database_schema_text,
    get_news_topics,
    get_world_bank_catalog_text,
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


class MacroSupervisorAgent:
    def __init__(self, llm: ChatOpenAI, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    @staticmethod
    def _summarize_artifacts(artifacts: dict[str, Any]) -> str:
        """Produce a compact metadata-only summary – never include raw data rows."""
        if not artifacts:
            return "No artifacts stored yet."
        lines: list[str] = []
        for key, value in artifacts.items():
            if key in ("latest_data", "latest_table"):
                rows = value.get("rows", value.get("records", []))
                cols = value.get("columns", [])
                lines.append(f"- {key}: {len(rows)} rows, columns={cols}")
            elif key == "latest_plotly":
                lines.append(
                    f"- latest_plotly: chart titled '{value.get('title', 'untitled')}'"
                )
            elif key == "latest_rag_results":
                count = len(value) if isinstance(value, list) else "?"
                lines.append(f"- latest_rag_results: {count} articles")
            elif key == "latest_web_results":
                count = len(value) if isinstance(value, list) else "?"
                lines.append(f"- latest_web_results: {count} results")
            else:
                lines.append(f"- {key}: (stored)")
        return "\n".join(lines)

    def _build_system_prompt(self, state: AgentState) -> str:
        current_plan = state.get("current_plan") or "No plan created yet."
        last_worker = state.get("last_worker") or "None"
        retry_count = state.get("retry_count", 0)

        results_history = "\n---\n".join(state.get("worker_results", []))
        if not results_history:
            results_history = "No workers have been called yet."

        artifacts_summary = self._summarize_artifacts(state.get("artifacts", {}))

        retry_status = (
            f"Last worker: '{last_worker}', consecutive retries: "
            f"{retry_count}/{self.max_retries}"
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

        return f"""You are the executive supervisor of a macroeconomic dashboard multi-agent system.
Your role is to plan, delegate tasks to specialised workers, review their results, and deliver the final answer.

AVAILABLE WORKERS:
- sql_agent: Queries PostgreSQL (World Bank indicators + Yahoo Finance data).
  The sql_agent uses an INTERNAL multi-step exploration process:
    1) It first queries the `databases` table to identify the right database.
    2) It then searches `database_indicators` (filtered by database_id) using
       regexp/ILIKE on the `description` column to find the exact indicator.
    3) Finally it fetches data from `indicators` + `metadata` for the found indicator.
    4) It can also use the `countries` table for country metadata (names, regions, income levels).
  The sql_agent NEVER guesses indicator names — it always looks them up from the database.
  You do NOT need to tell it which indicator ID to use — just describe what data you need
  in plain language and it will find the right indicator through its exploration steps.
- plotly_agent: Generates Plotly visualizations from data stored in artifacts.
- table_agent: Transforms/reshapes data with Python Polars (data from artifacts).
- rag_agent: Semantic search over a Qdrant vector DB of news articles.
- web_search: Searches the live internet via DuckDuckGo.
- downloader_agent: Downloads NEW World Bank indicators not yet in the database.
- chat_agent: Provides conversational synthesis and general knowledge answers.

CURRENT PLAN:
{current_plan}

WORKER RESULTS HISTORY:
{results_history}

ARTIFACTS IN MEMORY:
{artifacts_summary}

RETRY STATUS:
{retry_status}
{retry_instruction}

INSTRUCTIONS:
1. Analyse the user's request and the current execution state.
2. Create or update a numbered step-by-step plan.
3. Choose the most appropriate next worker, or FINISH when the task is fully complete.
4. Write a detailed, self-contained task for the chosen worker.
   Workers have NO memory of prior steps – include every detail they need.
   If the worker needs prior data, tell it the artifact key (e.g. "data is in latest_data").
5. When routing to FINISH write the complete, well-formatted final answer
   for the user inside 'isolated_worker_task'. Use markdown formatting.
   Reference generated charts and tables when relevant.
   When the answer includes information from RAG (news articles), you MUST
   include the source URL for every article referenced. Format as markdown links.
6. For data-visualisation requests: first obtain data (sql_agent), then visualise (plotly_agent).
7. ROUTING TO sql_agent:
   a) Describe the data you need in plain language (e.g. "GDP per capita for Brazil 2000-2023").
   b) The sql_agent will internally explore databases → find indicators → fetch data.
      You do NOT need to provide indicator IDs or database IDs.
   c) If sql_agent returns an error about not finding indicators, first try refining
      your description (use broader terms, synonyms). Only after sql_agent has
      confirmed the indicator is NOT available in any database should you consider
      using downloader_agent.
   d) If sql_agent returns SQL_AGENT INDICATOR_NOT_DOWNLOADED, the indicator exists
      in the catalog (database_indicators) but has NOT been downloaded into the
      indicators table yet. In this case you MUST route to
      downloader_agent to download it, then route back to sql_agent to fetch the data.
   e) For Yahoo Finance data, simply describe the stock/index data needed — the
      sql_agent will query yahoo_historical_prices and yahoo_metadata directly.
8. FACT-FINDING STRATEGY (when the user asks about specific real-world facts, events,
   opinions, or context that is NOT answerable from numeric database data):
   a) ALWAYS route to rag_agent FIRST to search the news article database.
   b) Carefully evaluate rag_agent results for RELEVANCE to the user's specific question.
      If rag_agent returns "No articles found", returns results that are off-topic,
      only tangentially related, or do not actually answer the user's question,
      you MUST route to web_search as a follow-up to get a better answer.
   c) Do NOT accept low-relevance RAG results as sufficient — when in doubt,
      use web_search as a second source to supplement or replace RAG results.
   d) NEVER skip rag_agent and go directly to web_search for fact-based questions.
   e) When presenting facts from RAG results, always include the article source URLs.
9. If retrying, explicitly describe the previous error and what should change.
10. ROUTING TO downloader_agent (downloading NEW World Bank indicators):
   a) NEVER route directly to downloader_agent based on the user's request alone.
   b) You MUST first route to sql_agent to explore what databases and indicators
      are available. The sql_agent will search the `databases` table and then
      `database_indicators` to check if the requested data exists.
   c) Only if sql_agent explicitly reports that the indicator was NOT found in
      any database should you route to downloader_agent.
   d) After downloader_agent successfully downloads the indicator, route back to
      sql_agent to fetch the newly available data.
   e) The full sequence is: sql_agent (explore) → downloader_agent (download) → sql_agent (fetch)."""

    async def ainvoke(self, state: AgentState) -> dict:
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
                        decision.updated_plan
                        + f"\n[System: max retries hit for {last_worker}]"
                    ),
                    "next_worker": "chat_agent",
                    "isolated_worker_task": (
                        f"The system repeatedly failed using the {last_worker} agent. "
                        "Inform the user and suggest what they could try differently."
                    ),
                    "last_worker": "chat_agent",
                    "retry_count": 0,
                    "trace": [
                        f"Router: max retries for {last_worker}, falling back to chat_agent"
                    ],
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


class SQLAgent:
    MAX_SQL_STEPS = 5

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _build_step_prompt(
        self,
        task: str,
        schema_text: str,
        previous_steps: list[dict],
    ) -> str:
        history_block = ""
        if previous_steps:
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
            history_block = "\n\nPREVIOUS EXPLORATION STEPS:\n" + "\n\n".join(parts)

        return f"""You are a PostgreSQL expert for a macroeconomic database.

DATABASE SCHEMA:
{schema_text}

MANDATORY STEP-BY-STEP APPROACH FOR WORLD BANK DATA:
You MUST follow these steps IN ORDER. Do NOT skip steps or guess indicator names.

Step 1 — IDENTIFY THE DATABASE:
  Query the `databases` table to find which database is relevant to the user's request.
  Example: SELECT id, name, description FROM databases WHERE name ILIKE '%development%' OR description ILIKE '%gdp%';

Step 2 — FIND THE INDICATOR:
  Query `database_indicators` filtered by the `database_id` found in Step 1.
  There can be THOUSANDS of indicators per database — use ILIKE or regexp filtering
  on the `description` column to narrow down.
  Example: SELECT id, description FROM database_indicators
           WHERE database_id = 2 AND description ~* 'gdp.*per capita';
  If you get too many results, refine the filter. If zero results, broaden it.

Step 3 — FETCH THE DATA:
  Query `indicators` joined with `metadata` using the exact `indicator_id` (and
  optionally `db_id`) found in Step 2. This is the FINAL step — set is_final_step=true.
  Join with `countries` if the user needs country names instead of ISO codes.
  Example:
    SELECT i.economy, c.value AS country_name, i.year, i.value,
           m.indicator_name, m.units
    FROM indicators i
    JOIN metadata m ON i.indicator_id = m.indicator_id AND i.db_id = m.db_id
    LEFT JOIN countries c ON i.economy = c.id
    WHERE i.indicator_id = 'NY.GDP.PCAP.CD' AND i.db_id = 2
    ORDER BY i.year;

Step 4 (optional) — COUNTRY METADATA:
  Use the `countries` table if the user asks about regions, income levels,
  lending types, or needs to filter/group by these dimensions.
  Example: SELECT id, value, "region.value", "incomeLevel.value"
           FROM countries WHERE aggregate = false;

FOR YAHOO FINANCE DATA:
  Yahoo Finance tables (yahoo_historical_prices, yahoo_metadata) are simpler —
  you may query them directly in one step (set is_final_step=true).
  Always check yahoo_metadata first if unsure about available tickers.

RULES:
- NEVER invent or guess indicator IDs/names. Always look them up from the tables first.
- Only SELECT statements.
- The 'economy' column in the indicators table contains 3-letter ISO country codes.
- Use double quotes for identifiers with special characters (e.g. "region.value").
- Limit results to 500 rows unless the task explicitly asks for more.
- Use proper JOINs, aggregation and filtering.
- For exploration steps (Steps 1-2), set is_final_step = false.
- For the final data retrieval (Step 3), set is_final_step = true.
{history_block}

USER TASK:
{task}

Based on the previous steps (if any), generate the NEXT query in the sequence."""

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        logger.info("sql_agent: starting multi-step exploration")
        try:
            schema_text = get_database_schema_text()
            structured_llm = self.llm.with_structured_output(SQLGeneration)

            previous_steps: list[dict] = []
            final_result = None

            for step_num in range(1, self.MAX_SQL_STEPS + 1):
                prompt = self._build_step_prompt(task, schema_text, previous_steps)
                gen: SQLGeneration = await structured_llm.ainvoke(
                    [SystemMessage(content=prompt)]
                )

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
                found_indicator_no_data = False
                for i, s in enumerate(previous_steps):
                    err = s["result"].get("error")
                    info = err if err else f"{s['result'].get('row_count', 0)} rows"
                    step_lines.append(f"  Step {i + 1}: {s['query'][:120]} -> {info}")
                    query_lower = s["query"].lower()
                    if (
                        "database_indicators" in query_lower
                        and s["result"].get("row_count", 0) > 0
                    ):
                        found_indicator_no_data = True
                steps_summary = "\n".join(step_lines)

                if found_indicator_no_data:
                    return {
                        "worker_results": [
                            f"SQL_AGENT INDICATOR_NOT_DOWNLOADED: The indicator was "
                            f"found in database_indicators but returned 0 rows from "
                            f"the indicators table — it likely has not been downloaded "
                            f"yet. Use downloader_agent to download it, then retry "
                            f"sql_agent.\nSteps taken:\n{steps_summary}"
                        ],
                        "trace": [f"sql_agent: indicator found but not downloaded ({len(previous_steps)} steps)"],
                    }

                return {
                    "worker_results": [
                        f"SQL_AGENT ERROR: Could not retrieve data after "
                        f"{len(previous_steps)} steps.\nSteps taken:\n{steps_summary}"
                    ],
                    "trace": [f"sql_agent: failed after {len(previous_steps)} steps"],
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
                "trace": [
                    f"sql_agent: {steps_trace} → "
                    f"final {final_result['row_count']} rows, "
                    f"cols={final_result['columns']}"
                ],
            }
        except Exception as exc:
            return {
                "worker_results": [f"SQL_AGENT ERROR: {exc}"],
                "trace": [f"sql_agent: exception – {exc}"],
            }


class PlotlyAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        artifacts = state.get("artifacts", {})
        logger.info("plotly_agent: generating visualization")
        try:
            data = artifacts.get("latest_data") or artifacts.get("latest_table") or {}
            rows = data.get("rows", data.get("records", []))
            if not rows:
                return {
                    "worker_results": [
                        "PLOTLY_AGENT ERROR: No data in artifacts to visualise."
                    ],
                    "trace": ["plotly_agent: no data available"],
                }

            columns = data.get("columns", list(rows[0].keys()) if rows else [])
            sample = rows[:3]

            system_prompt = f"""You are a Plotly visualisation expert.

DATA SCHEMA:
- Columns: {columns}
- Total rows: {len(rows)}
- Sample (first 3 rows): {json.dumps(sample, default=str)[:1500]}

RULES:
- Input data is available as `data` (list of dicts, {len(rows)} rows).
- Create a clear, informative chart. Use appropriate chart type.
- Assign the final figure to `fig`. Do NOT call fig.show().
- Import any required plotly modules at the top.
- Handle possible None/null values gracefully.

YOUR TASK:
{task}"""

            structured_llm = self.llm.with_structured_output(PlotlyCodeGeneration)
            gen: PlotlyCodeGeneration = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt)]
            )

            data_b64 = encode_data_for_sandbox(rows)
            sandbox_code = (
                "import json, base64\n"
                "import plotly.graph_objects as go\n"
                "import plotly.express as px\n\n"
                f'data = json.loads(base64.b64decode("{data_b64}").decode())\n\n'
                f"{gen.plotly_code}\n\n"
                "print(fig.to_json())\n"
            )

            result = await execute_code_in_sandbox(sandbox_code)

            if not result.get("success"):
                return {
                    "worker_results": [
                        f"PLOTLY_AGENT ERROR: execution failed.\n"
                        f"stderr: {result.get('stderr', '')}\n"
                        f"Code:\n{gen.plotly_code}"
                    ],
                    "trace": [
                        f"plotly_agent: sandbox error – "
                        f"{(result.get('stderr') or '')[:120]}"
                    ],
                }

            figure_json = result.get("stdout", "").strip()
            return {
                "worker_results": [
                    f"PLOTLY_AGENT SUCCESS: chart '{gen.title}' generated."
                ],
                "artifacts": {
                    "latest_plotly": {
                        "figure_json": figure_json,
                        "title": gen.title,
                    }
                },
                "trace": [f"plotly_agent: created chart '{gen.title}'"],
            }
        except Exception as exc:
            return {
                "worker_results": [f"PLOTLY_AGENT ERROR: {exc}"],
                "trace": [f"plotly_agent: exception – {exc}"],
            }


class TableAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        artifacts = state.get("artifacts", {})
        logger.info("table_agent: starting data transformation")
        try:
            data = artifacts.get("latest_data") or artifacts.get("latest_table") or {}
            rows = data.get("rows", data.get("records", []))
            if not rows:
                return {
                    "worker_results": [
                        "TABLE_AGENT ERROR: No input data in artifacts."
                    ],
                    "trace": ["table_agent: no data available"],
                }

            columns = data.get("columns", list(rows[0].keys()))
            sample_row = rows[0]
            schema_lines = "\n".join(
                f"  - {k}: {type(v).__name__}" for k, v in sample_row.items()
            )

            system_prompt = f"""You are a senior data engineer using the Python `polars` library.

INPUT DATA SCHEMA (variable: `df`):
{schema_lines}

Columns: {columns}
Total rows: {len(rows)}

RULES:
- Write clean, idiomatic Polars code (pl.col, select, with_columns, group_by, agg…).
- `import polars as pl` and the DataFrame `df` already exist – do NOT recreate them.
- Assign the final transformed DataFrame to `result_df`.
- Do NOT use pandas.

YOUR TASK:
{task}"""

            structured_llm = self.llm.with_structured_output(PolarsCodeGeneration)
            gen: PolarsCodeGeneration = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt)]
            )

            data_b64 = encode_data_for_sandbox(rows)
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

            if not result.get("success"):
                return {
                    "worker_results": [
                        f"TABLE_AGENT ERROR: execution failed.\n"
                        f"stderr: {result.get('stderr', '')}\n"
                        f"Code:\n{gen.polars_code}"
                    ],
                    "trace": [
                        f"table_agent: sandbox error – "
                        f"{(result.get('stderr') or '')[:120]}"
                    ],
                }

            parsed = json.loads(result["stdout"])
            return {
                "worker_results": [
                    f"TABLE_AGENT SUCCESS: {parsed['row_count']} rows, "
                    f"columns={parsed['columns']}."
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
                "trace": [
                    f"table_agent: {parsed['row_count']} rows, cols={parsed['columns']}"
                ],
            }
        except Exception as exc:
            return {
                "worker_results": [f"TABLE_AGENT ERROR: {exc}"],
                "trace": [f"table_agent: exception – {exc}"],
            }


class RAGAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        logger.info("rag_agent: starting news search")
        try:
            topics = get_news_topics()
            system_prompt = f"""You are a news-retrieval specialist.
You plan semantic search queries against a Qdrant vector database of news articles.

AVAILABLE TOPICS: {topics}
AVAILABLE SENTIMENTS: positive, negative

YOUR TASK:
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
                msg = (
                    result.get("error") or result.get("message") or "No articles found."
                )
                logger.info("rag_agent: %s", msg)
                return {
                    "worker_results": [f"RAG_AGENT: {msg}"],
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
                "trace": [
                    f"rag_agent: {len(articles)} articles for "
                    f"'{plan.search_query[:80]}'"
                ],
            }
        except Exception as exc:
            logger.exception("rag_agent: error")
            return {
                "worker_results": [f"RAG_AGENT ERROR: {exc}"],
                "trace": [f"rag_agent: exception – {exc}"],
            }


class WebSearchAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        logger.info("web_search: starting internet search")
        try:
            system_prompt = f"""You are a web-research specialist.
Generate 1-3 focused search queries to find information on the internet.

YOUR TASK:
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
                return {
                    "worker_results": [f"WEB_SEARCH: {error_msg}"],
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
                "trace": [f"web_search: {len(hits)} results for {plan.search_queries}"],
            }
        except Exception as exc:
            return {
                "worker_results": [f"WEB_SEARCH ERROR: {exc}"],
                "trace": [f"web_search: exception – {exc}"],
            }


class DownloaderAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        logger.info("downloader_agent: starting indicator download")
        try:
            catalog_text = get_world_bank_catalog_text()
            system_prompt = f"""You are a World Bank data specialist.
Determine which indicator to download based on the task.

ALREADY DOWNLOADED INDICATORS:
{catalog_text}

Only download indicators NOT already in the list above.
Use valid World Bank indicator IDs (e.g. NY.GDP.MKTP.CD) and database IDs.

YOUR TASK:
{task}"""

            structured_llm = self.llm.with_structured_output(DownloadIndicatorPlan)
            plan: DownloadIndicatorPlan = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt)]
            )

            result = await download_indicator(plan.indicator_id, plan.db_id)
            logger.info(
                "downloader_agent: downloading indicator=%s db=%d",
                plan.indicator_id,
                plan.db_id,
            )

            if not result.get("success", False):
                error = result.get("error") or result.get("detail", "Unknown error")
                return {
                    "worker_results": [
                        f"DOWNLOADER_AGENT ERROR: {error} "
                        f"(indicator={plan.indicator_id}, db={plan.db_id})"
                    ],
                    "trace": [f"downloader_agent: failed – {error}"],
                }

            status = result.get("status", "success")
            rows_inserted = result.get("rows_inserted", 0)
            return {
                "worker_results": [
                    f"DOWNLOADER_AGENT SUCCESS: indicator={plan.indicator_id}, "
                    f"db={plan.db_id}, rows_inserted={rows_inserted}, status={status}."
                ],
                "trace": [
                    f"downloader_agent: {plan.indicator_id} – "
                    f"{rows_inserted} rows, status={status}"
                ],
            }
        except Exception as exc:
            return {
                "worker_results": [f"DOWNLOADER_AGENT ERROR: {exc}"],
                "trace": [f"downloader_agent: exception – {exc}"],
            }


class ChatAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        logger.info("chat_agent: synthesising response")
        try:
            system_prompt = (
                "You are a helpful macroeconomic analyst assistant. "
                "Synthesise information, explain economic concepts, or provide "
                "conversational responses. Use markdown formatting. "
                "Be concise but thorough."
            )

            structured_llm = self.llm.with_structured_output(ChatSynthesis)
            synthesis: ChatSynthesis = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=task),
                ]
            )

            return {
                "worker_results": [f"CHAT_AGENT: {synthesis.response}"],
                "trace": [
                    f"chat_agent: synthesised response ({len(synthesis.response)} chars)"
                ],
            }
        except Exception as exc:
            return {
                "worker_results": [f"CHAT_AGENT ERROR: {exc}"],
                "trace": [f"chat_agent: exception – {exc}"],
            }


class MacroAgentGraph:
    """Builds and wraps the LangGraph multi-agent graph."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        max_retries: int = 3,
        recursion_limit: int = 30,
    ):
        self.llm = ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_retries=3,
        )
        self.max_retries = max_retries
        self.recursion_limit = recursion_limit

        self.supervisor = MacroSupervisorAgent(llm=self.llm, max_retries=max_retries)
        self.sql_agent = SQLAgent(llm=self.llm)
        self.plotly_agent = PlotlyAgent(llm=self.llm)
        self.table_agent = TableAgent(llm=self.llm)
        self.rag_agent = RAGAgent(llm=self.llm)
        self.web_search_agent = WebSearchAgent(llm=self.llm)
        self.downloader_agent = DownloaderAgent(llm=self.llm)
        self.chat_agent = ChatAgent(llm=self.llm)

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("supervisor", self.supervisor.ainvoke)
        builder.add_node("sql_agent", self.sql_agent.ainvoke)
        builder.add_node("plotly_agent", self.plotly_agent.ainvoke)
        builder.add_node("table_agent", self.table_agent.ainvoke)
        builder.add_node("rag_agent", self.rag_agent.ainvoke)
        builder.add_node("web_search", self.web_search_agent.ainvoke)
        builder.add_node("downloader_agent", self.downloader_agent.ainvoke)
        builder.add_node("chat_agent", self.chat_agent.ainvoke)

        builder.set_entry_point("supervisor")

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
            "retry_count": 0,
            "trace": [],
        }

    @staticmethod
    def _build_progress_updates(trace: list[str]) -> list[str]:
        updates: list[str] = []
        for entry in trace:
            if entry.startswith("Router:"):
                updates.append(entry)
            elif ":" in entry:
                worker = entry.split(":", 1)[0].strip()
                detail = entry.split(":", 1)[-1].strip()
                if "exception" in detail or "error" in detail.lower():
                    updates.append(f"{worker}: error — {detail}")
                else:
                    updates.append(f"{worker}: {detail}")
            else:
                updates.append(entry)
        return updates

    @staticmethod
    def _extract_response(final_state: dict) -> dict:
        response = final_state.get("isolated_worker_task", "")
        if not response:
            results = list(final_state.get("worker_results", []))
            response = results[-1] if results else "I could not process your request."
        trace = list(final_state.get("trace", []))
        return {
            "response": response,
            "route": "orchestrated",
            "trace": trace,
            "progress_updates": MacroAgentGraph._build_progress_updates(trace),
            "artifacts": dict(final_state.get("artifacts", {})),
        }

    def invoke(self, message: str, chat_history: list[dict] | None = None) -> dict:
        state = self._build_initial_state(message, chat_history or [])
        try:
            final = self.graph.invoke(
                state, config={"recursion_limit": self.recursion_limit}
            )
        except Exception as exc:
            logger.exception("Graph invoke failed")
            return {
                "response": f"An error occurred: {exc}",
                "route": "error",
                "trace": [f"graph error: {exc}"],
                "artifacts": {},
            }
        return self._extract_response(final)

    async def ainvoke(
        self, message: str, chat_history: list[dict] | None = None
    ) -> dict:
        state = self._build_initial_state(message, chat_history or [])
        try:
            final = await self.graph.ainvoke(
                state, config={"recursion_limit": self.recursion_limit}
            )
        except Exception as exc:
            logger.exception("Graph ainvoke failed")
            return {
                "response": f"An error occurred: {exc}",
                "route": "error",
                "trace": [f"graph error: {exc}"],
                "artifacts": {},
            }
        return self._extract_response(final)

    async def astream_events(
        self, message: str, chat_history: list[dict] | None = None
    ):
        """Yield progress dicts as each graph node completes, then a final result."""
        state = self._build_initial_state(message, chat_history or [])
        accumulated_trace: list[str] = []
        accumulated_artifacts: dict[str, Any] = {}
        last_isolated_task = ""

        try:
            async for chunk in self.graph.astream(
                state, config={"recursion_limit": self.recursion_limit}
            ):
                for node_name, output in chunk.items():
                    if not isinstance(output, dict):
                        continue
                    trace_entries = output.get("trace", [])
                    accumulated_trace.extend(trace_entries)

                    if output.get("artifacts"):
                        accumulated_artifacts.update(output["artifacts"])

                    if output.get("isolated_worker_task"):
                        last_isolated_task = output["isolated_worker_task"]

                    progress = self._build_progress_updates(trace_entries)
                    yield {
                        "type": "progress",
                        "node": node_name,
                        "updates": progress,
                    }

            all_progress = self._build_progress_updates(accumulated_trace)
            yield {
                "type": "final",
                "response": last_isolated_task or "No answer returned.",
                "route": "orchestrated",
                "trace": accumulated_trace,
                "progress_updates": all_progress,
                "artifacts": accumulated_artifacts,
            }
        except Exception as exc:
            logger.exception("Graph astream failed")
            yield {
                "type": "error",
                "response": f"An error occurred: {exc}",
                "route": "error",
                "trace": accumulated_trace,
                "progress_updates": self._build_progress_updates(accumulated_trace),
                "artifacts": accumulated_artifacts,
            }
