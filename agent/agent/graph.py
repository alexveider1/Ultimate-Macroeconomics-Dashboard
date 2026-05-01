import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .schemas import (
    AgentState,
    ChatSynthesis,
    DownloadIndicatorPlan,
    GuardrailDecision,
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


class GuardrailAgent:
    """Screens the user's latest message before routing.

    Blocks harsh language, personal attacks, sexual or otherwise
    inappropriate content, and requests outside the dashboard's
    macroeconomics / politics / sociology / data-science scope.
    """

    SYSTEM_PROMPT = (
        "You are the safety screen for a macroeconomic dashboard assistant. "
        "Decide whether the user's most recent message is acceptable. "
        "It is acceptable when the user asks about economics (applied or "
        "theoretical), politics, sociology, econometrics, data science, "
        "technology, or anything else relevant to the dashboard — including "
        "casual greetings and meta questions about the assistant. "
        "Flag the message ONLY when it contains harsh language directed at "
        "the assistant or other people, personal attacks, sexual content, "
        "requests for illegal/unethical material, or topics clearly outside "
        "this scope (e.g. medical advice, malware, explicit content). "
        "If you flag the message, write a brief, polite refusal in markdown "
        "explaining you can only help with the dashboard's topics."
    )

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
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
        try:
            structured_llm = self.llm.with_structured_output(GuardrailDecision)
            decision: GuardrailDecision = await structured_llm.ainvoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=last_user_msg),
                ]
            )
            if decision.is_inappropriate:
                logger.info("guardrail: blocked — %s", decision.reason)
                refusal = (
                    decision.refusal_message.strip()
                    or "I can only help with macroeconomics, politics, "
                    "sociology, data science and related dashboard topics. "
                    "Please rephrase your question."
                )
                return {
                    "guardrail_blocked": True,
                    "guardrail_message": refusal,
                    "trace": ["guardrail: blocked"],
                }
            return {
                "guardrail_blocked": False,
                "guardrail_message": "",
                "trace": ["guardrail: passed"],
            }
        except Exception as exc:
            logger.warning("guardrail: error – %s — passing through", exc)
            return {
                "guardrail_blocked": False,
                "guardrail_message": "",
                "trace": [f"guardrail: error ({exc}) — passing through"],
            }


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
- sql_agent: Queries PostgreSQL. It serves TWO independent data domains and
  picks the right path based on the task you give it:
    A) WORLD BANK indicators — internal 3-step exploration:
       1) databases → identify the right World Bank database
       2) database_indicators (filtered by database_id) → find the indicator
          via ILIKE/regexp on `description`
       3) indicators + metadata → fetch the data series, optionally joined
          with `countries` for country names / regions / income levels
       It NEVER guesses indicator IDs — always looks them up.
    B) YAHOO FINANCE market data — simpler 1–2 step lookup:
       1) yahoo_metadata → find the right ticker(s) by `asset_name`,
          `short_name`, `sector`, `industry`, `category` (Indices/Companies)
       2) yahoo_historical_prices → fetch OHLCV history for those tickers
       Use this path for stocks, indices, sectors, ETFs, market data.
  Just describe what data you need in plain language; sql_agent decides
  which domain to query.
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
   Reference generated charts when relevant ("the chart above shows ...").
   IMPORTANT: NEVER embed the data table itself into the answer. Do NOT write
   markdown tables of the SQL/Polars rows. The UI renders the underlying
   data table separately below your answer in an expander, so just summarise
   the key numbers / takeaways in prose. A short markdown table is acceptable
   ONLY when summarising 2–4 hand-picked headline figures, never the full
   dataset.
   IMPORTANT: When a chart was produced by plotly_agent, the UI renders the
   chart itself directly in the chat above your answer. NEVER include the
   plotly Python code, JSON figure spec, raw figure description, axis lists,
   or any technical chart implementation details in the FINISH answer. Only
   reference the chart in prose ("the chart above shows ...") and summarise
   what the user can see in it.
   For mathematical expressions, use Streamlit-compatible markdown math:
   inline math as `$expr$` and block math as `$$expr$$`. Do NOT use
   `\\(...\\)` or `\\[...\\]` delimiters — they will not render.
   When the answer includes information from RAG (news articles), you MUST
   include the source URL for every article referenced. Format as markdown links.
6. For data-visualisation requests: first obtain data (sql_agent), then visualise (plotly_agent).
7. ROUTING TO chat_agent (DEFAULT for non-data tasks):
   a) Route to chat_agent for ANY request that does NOT require fetching,
      transforming, plotting, or searching for data — for example:
      definitions, conceptual explanations, theoretical questions about
      economics / econometrics / statistics / data science, "what does X
      mean?", "explain Y", greetings, meta-questions about the assistant,
      derivations / formulas, methodology discussions, and general
      conversational replies.
      For these, do NOT call sql_agent / rag_agent / web_search just to
      look something up — chat_agent already has general knowledge.
   b) After chat_agent returns its synthesis, FINISH and pass its response
      through to the user (you may lightly polish formatting but preserve
      content faithfully).
8. ROUTING TO sql_agent:
   a) Describe the data in plain language. State the data domain explicitly:
      "World Bank indicator: ..." OR "Yahoo Finance market data: ...".
      For ambiguous wording, infer from the request:
        - macro indicators (GDP, inflation, unemployment, debt, FX, demography,
          health, education, environment, governance) → World Bank
        - stocks, equities, tickers, indices (S&P 500, NASDAQ, ^GSPC),
          companies (AAPL, MSFT), OHLCV / closing prices / market cap →
          Yahoo Finance
   b) WORLD BANK PATH:
      - sql_agent will internally explore databases → indicators → fetch.
        You do NOT need to provide indicator IDs or database IDs.
      - If sql_agent reports the indicator is NOT found anywhere, try
        refining your description (broader terms, synonyms) before giving up.
      - If sql_agent returns SQL_AGENT INDICATOR_NOT_DOWNLOADED, the
        indicator exists in `database_indicators` but the data has not been
        downloaded yet. Route to downloader_agent, then back to sql_agent.
   c) YAHOO FINANCE PATH:
      - sql_agent will query yahoo_metadata + yahoo_historical_prices
        directly. Mention the ticker if the user gave one; otherwise
        describe the asset (e.g. "Apple stock", "S&P 500 index").
      - downloader_agent does NOT support Yahoo Finance. If the requested
        ticker is not present in yahoo_metadata, FINISH and tell the user
        the asset is not currently tracked by the dashboard.
9. FACT-FINDING STRATEGY (when the user asks about specific real-world facts, events,
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
10. If retrying, explicitly describe the previous error and what should change.
11. ROUTING TO downloader_agent (downloading NEW World Bank indicators):
   a) NEVER route directly to downloader_agent based on the user's request alone.
   b) You MUST first route to sql_agent so it can run the 3-step exploration
      (databases → database_indicators → indicators) and identify the exact
      `indicator_id` (e.g. 'NY.GDP.MKTP.CD') and `db_id` (e.g. 2) for the
      requested series.
   c) Trigger downloader_agent ONLY when sql_agent returns
      `SQL_AGENT INDICATOR_NOT_DOWNLOADED` — that response is guaranteed to
      include a "Best match: indicator_id=…, db_id=…" line plus a list of
      candidates extracted from `database_indicators`. Pick the candidate
      that best matches the user's request.
   d) The `isolated_worker_task` you give downloader_agent MUST contain the
      EXACT `indicator_id` and `db_id` you selected, in this literal form:
        indicator_id=<ID>
        db_id=<INT>
      (you may add a one-line description for context, but those two
      key=value lines are required and must be verbatim from the candidate
      list — never invent or paraphrase them).
   e) downloader_agent will call the downloader_extra `/ingest` endpoint,
      which fetches the entire (economy, year, value) table for that
      indicator from the World Bank API and persists it to the `indicators`
      table. It does NOT pick an indicator on its own — it strictly relies
      on the values you pass.
   f) Do NOT call downloader_agent for vague conceptual questions, Yahoo
      Finance assets, or when sql_agent has not yet identified a concrete
      indicator_id+db_id pair.
   g) After downloader_agent reports `DOWNLOADER_AGENT SUCCESS`, route back
      to sql_agent to fetch the newly available data.
   h) The full sequence is: sql_agent (explore → INDICATOR_NOT_DOWNLOADED)
      → downloader_agent (download with exact indicator_id+db_id)
      → sql_agent (fetch)."""

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

THIS DATABASE COVERS TWO INDEPENDENT DOMAINS — pick the right one FIRST:
  * WORLD BANK macro indicators → tables `databases`, `database_indicators`,
    `indicators`, `metadata`, `countries`. Use the 3-step plan below.
  * YAHOO FINANCE market data → tables `yahoo_metadata` and
    `yahoo_historical_prices`. Use the 1–2 step plan below. NEVER touch
    the World Bank tables for stock/index/ticker requests.

Inspect the user task; if it mentions tickers, stocks, equities, indices,
companies, OHLC/closing prices, market cap, "S&P", "NASDAQ", "Apple",
"^GSPC", "AAPL" etc. → YAHOO. Otherwise (GDP, inflation, unemployment,
demography, health, education, environment, governance) → WORLD BANK.

==================================================================
PLAN A — WORLD BANK (mandatory step order, do NOT skip or guess IDs):

Step 1 — IDENTIFY THE DATABASE:
  Query the `databases` table to find which database is relevant.
  Example: SELECT id, name, description FROM databases
           WHERE name ILIKE '%development%' OR description ILIKE '%gdp%';

Step 2 — FIND THE INDICATOR:
  Query `database_indicators` filtered by `database_id` from Step 1.
  Use ILIKE/regexp on `description` to narrow thousands of rows.
  Example: SELECT id, description FROM database_indicators
           WHERE database_id = 2 AND description ~* 'gdp.*per capita';

Step 3 — FETCH THE DATA (final, is_final_step=true):
  SELECT i.economy, c.value AS country_name, i.year, i.value,
         m.indicator_name, m.units
  FROM indicators i
  JOIN metadata m ON i.indicator_id = m.indicator_id AND i.db_id = m.db_id
  LEFT JOIN countries c ON i.economy = c.id
  WHERE i.indicator_id = 'NY.GDP.PCAP.CD' AND i.db_id = 2
  ORDER BY i.year;

Step 4 (optional) — COUNTRY METADATA:
  SELECT id, value, "region.value", "incomeLevel.value" FROM countries
  WHERE aggregate = false;

==================================================================
PLAN B — YAHOO FINANCE (1–2 steps):

Step 1 — RESOLVE THE TICKER (skip if the user already gave one):
  Query `yahoo_metadata` to find the right ticker by asset_name,
  short_name, sector, industry, or category ('Indices' / 'Companies').
  Example:
    SELECT ticker, asset_name, short_name, sector, industry, currency
    FROM yahoo_metadata
    WHERE short_name ILIKE '%apple%' OR asset_name ILIKE '%apple%';
  If zero rows: tell the router the asset is not tracked. Do NOT invent a
  ticker. (downloader_agent does NOT support Yahoo Finance.)

Step 2 — FETCH PRICE HISTORY (final, is_final_step=true):
  SELECT date, open, high, low, close, volume, ticker, category
  FROM yahoo_historical_prices
  WHERE ticker = 'AAPL'
    AND date >= '2020-01-01'
  ORDER BY date;
  Join with yahoo_metadata only if the answer needs descriptive fields
  (sector, currency, exchange).

==================================================================
RULES (apply to BOTH plans):
- Only SELECT statements.
- NEVER invent or guess World Bank indicator IDs or Yahoo tickers — look
  them up first.
- The 'economy' column in `indicators` holds 3-letter ISO country codes.
- Use double quotes for identifiers with special characters
  (e.g. "region.value").
- Limit results to 500 rows unless the task explicitly asks for more.
- For exploration steps (World Bank Step 1–2 / Yahoo Step 1),
  is_final_step = false. For the final data retrieval, is_final_step = true.
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
                indicator_match_step: dict | None = None
                for i, s in enumerate(previous_steps):
                    err = s["result"].get("error")
                    info = err if err else f"{s['result'].get('row_count', 0)} rows"
                    step_lines.append(f"  Step {i + 1}: {s['query'][:120]} -> {info}")
                    query_lower = s["query"].lower()
                    if (
                        "database_indicators" in query_lower
                        and s["result"].get("row_count", 0) > 0
                    ):
                        indicator_match_step = s
                steps_summary = "\n".join(step_lines)

                if indicator_match_step is not None:
                    candidate_rows = indicator_match_step["result"].get("rows", [])[:5]
                    db_id_match = re.search(
                        r"database_id\s*=\s*(\d+)",
                        indicator_match_step["query"],
                        re.IGNORECASE,
                    )
                    db_id_value = db_id_match.group(1) if db_id_match else "(unknown)"
                    first_indicator_id = (
                        candidate_rows[0].get("id") if candidate_rows else "(unknown)"
                    )
                    candidate_lines = [
                        f"    indicator_id={row.get('id', '?')} | "
                        f"description={(row.get('description') or '')[:140]}"
                        for row in candidate_rows
                    ]
                    candidates_block = (
                        "\n".join(candidate_lines) or "    (no candidates extracted)"
                    )
                    return {
                        "worker_results": [
                            f"SQL_AGENT INDICATOR_NOT_DOWNLOADED: indicator found in "
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
                        "trace": [
                            f"sql_agent: indicator found but not downloaded "
                            f"(candidate={first_indicator_id}, db={db_id_value}, "
                            f"{len(previous_steps)} steps)"
                        ],
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

    MAX_FIX_ATTEMPTS = 3

    @staticmethod
    def _build_plotly_prompt(
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

        return f"""You are a Plotly visualisation expert.

DATA SCHEMA:
- Columns: {columns}
- Total rows: {rows_count}
- Sample (first 3 rows): {json.dumps(sample, default=str)[:1500]}

RULES:
- Input data is available as `data` (list of dicts, {rows_count} rows).
- Create a clear, informative chart. Use appropriate chart type.
- Assign the final figure to `fig`. Do NOT call fig.show().
- Import any required plotly modules at the top.
- Handle possible None/null values gracefully.
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
                    "worker_results": [
                        "PLOTLY_AGENT ERROR: No data in artifacts to visualise."
                    ],
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
                    "import plotly.graph_objects as go\n"
                    "import plotly.express as px\n\n"
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
                    "trace": [
                        f"plotly_agent: chart '{gen.title}' generated "
                        f"(attempt {attempt_num})"
                    ],
                }

            return {
                "worker_results": [
                    f"PLOTLY_AGENT ERROR: all {self.MAX_FIX_ATTEMPTS} sandbox "
                    f"attempts failed.\nLast issue: "
                    f"{last_issue[:1500]}\nLast code:\n{last_code}"
                ],
                "trace": [
                    f"plotly_agent: failed after {self.MAX_FIX_ATTEMPTS} "
                    f"attempts – {last_issue[:120]}"
                ],
            }
        except Exception as exc:
            return {
                "worker_results": [f"PLOTLY_AGENT ERROR: {exc}"],
                "trace": [f"plotly_agent: exception – {exc}"],
            }


class TableAgent:
    MAX_FIX_ATTEMPTS = 3

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @staticmethod
    def _build_polars_prompt(
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

        return f"""You are a senior data engineer using the Python `polars` library.

INPUT DATA SCHEMA (variable: `df`):
{schema_lines}

Columns: {columns}
Total rows: {rows_count}

RULES:
- Write clean, idiomatic Polars code (pl.col, select, with_columns, group_by, agg…).
- `import polars as pl` and the DataFrame `df` already exist – do NOT recreate them.
- Assign the final transformed DataFrame to `result_df`.
- Do NOT use pandas.
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
                "trace": [
                    f"table_agent: failed after {self.MAX_FIX_ATTEMPTS} "
                    f"attempts – {last_error[:120]}"
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
    """Downloads a World Bank indicator's full table via the downloader_extra service.

    The supervisor is expected to have already used sql_agent to discover
    the exact `indicator_id` and `db_id` (via the `databases` →
    `database_indicators` exploration) and to have included those values
    verbatim in the worker task. This agent extracts them and calls
    `/ingest`, which fetches every (economy, year) row for that indicator
    from the World Bank API and persists them to the `indicators` table.
    """

    EXTRACT_SYSTEM_PROMPT = (
        "You extract the exact World Bank `indicator_id` (string, e.g. "
        "'NY.GDP.MKTP.CD') and `db_id` (integer database id, e.g. 2) from "
        "the supervisor's task description. The supervisor has ALREADY "
        "discovered these values via sql_agent's exploration of the "
        "`database_indicators` table — your job is purely to read them out "
        "of the task text. NEVER invent or guess values. If the task does "
        "not contain a clear indicator_id and db_id, return the closest "
        "literal values you can find."
    )

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def ainvoke(self, state: AgentState) -> dict:
        task = state["isolated_worker_task"]
        logger.info("downloader_agent: extracting indicator id from task")
        try:
            structured_llm = self.llm.with_structured_output(DownloadIndicatorPlan)
            plan: DownloadIndicatorPlan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=self.EXTRACT_SYSTEM_PROMPT),
                    HumanMessage(content=f"SUPERVISOR TASK:\n{task}"),
                ]
            )

            logger.info(
                "downloader_agent: calling /ingest indicator=%s db=%s",
                plan.indicator_id,
                plan.db_id,
            )
            result = await download_indicator(plan.indicator_id, plan.db_id)

            if not result.get("success", False):
                error = result.get("error") or result.get("detail", "Unknown error")
                return {
                    "worker_results": [
                        f"DOWNLOADER_AGENT ERROR: {error} "
                        f"(indicator={plan.indicator_id}, db={plan.db_id})"
                    ],
                    "trace": [
                        f"downloader_agent: failed – {plan.indicator_id}/"
                        f"{plan.db_id} – {error}"
                    ],
                }

            status = result.get("status", "success")
            rows_inserted = result.get("rows_inserted", 0)
            return {
                "worker_results": [
                    f"DOWNLOADER_AGENT SUCCESS: indicator={plan.indicator_id}, "
                    f"db={plan.db_id}, rows_inserted={rows_inserted}, "
                    f"status={status}. The full (economy, year, value) table for "
                    f"this indicator is now stored in the `indicators` table — "
                    f"route back to sql_agent to fetch it."
                ],
                "trace": [
                    f"downloader_agent: {plan.indicator_id}/{plan.db_id} – "
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
                "Be concise but thorough. "
                "For mathematical expressions, use Streamlit-compatible "
                "markdown math syntax: inline as `$expr$` and block as "
                "`$$expr$$`. Never use `\\(...\\)` or `\\[...\\]` delimiters — "
                "they will not render in the chat UI."
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

        self.guardrail = GuardrailAgent(llm=self.llm)
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
            "guardrail_blocked": False,
            "guardrail_message": "",
        }

    FINAL_SYNTHESIS_SYSTEM_PROMPT = (
        "You are the streaming output channel of the macroeconomic dashboard's "
        "router agent. The router has already composed the final answer in the "
        "SUPERVISOR DRAFT. Your job is to deliver that draft to the user, "
        "preserving its content as faithfully as possible.\n\n"
        "STRICT RULES:\n"
        "- Output the supervisor draft verbatim whenever it is well-formed. "
        "  Only adjust if the draft has obvious markdown/formatting glitches "
        "  or accidentally exposes implementation details (worker names, "
        "  retries, SQL, sandbox, tracebacks). In that case, fix the leak in "
        "  place but keep every fact, number, citation and URL intact.\n"
        "- Never invent new facts, numbers or sources that are not in the "
        "  draft or worker results.\n"
        "- Do NOT embed the full data table in your reply. The dashboard "
        "  renders it separately in an expander. A short markdown table of "
        "  2–4 hand-picked headline figures is acceptable.\n"
        "- When a chart artifact exists, the dashboard ALREADY renders the "
        "  rendered Plotly chart in the chat above your answer. Refer to it "
        "  as 'the chart above' and only describe what the user can see. "
        "  NEVER include Plotly code, JSON figure specs, fenced ```python``` "
        "  code blocks, axis-by-axis listings, or any technical chart "
        "  implementation details in your reply.\n"
        "- For mathematical expressions, use Streamlit-compatible markdown "
        "  math: inline as `$expr$` and block as `$$expr$$`. Never use "
        "  `\\(...\\)` or `\\[...\\]` — they will not render in the chat UI.\n"
        "- When citing news articles, keep their source URLs as markdown links.\n"
        "- Never apologise for technical issues. If the data was unavailable, "
        "  state it politely without revealing the internal failure mode."
    )

    @staticmethod
    def _format_worker_results(state: dict) -> str:
        results = state.get("worker_results") or []
        if not results:
            return "(no worker results)"
        return "\n---\n".join(str(r) for r in results)

    @staticmethod
    def _last_user_message(state: dict) -> str:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    async def _stream_final_synthesis(
        self,
        state: dict,
        supervisor_draft: str,
    ):
        """Re-emit the final answer as a token stream.

        Uses a separate, non-structured streaming call seeded with the
        supervisor's drafted answer plus the worker context, so the user
        sees tokens incrementally instead of one atomic blob.
        """
        artifacts = state.get("artifacts", {}) or {}
        has_plot = isinstance(artifacts.get("latest_plotly"), dict) and bool(
            artifacts["latest_plotly"].get("figure_json")
        )
        latest_data = artifacts.get("latest_data") or artifacts.get("latest_table")
        has_data = (
            isinstance(latest_data, dict)
            and bool(latest_data.get("rows") or latest_data.get("records"))
        )

        artifact_hints: list[str] = []
        if has_plot:
            artifact_hints.append("A plot has been rendered above your answer.")
        if has_data:
            artifact_hints.append(
                "A data table is shown below your answer in an expander."
            )
        artifact_block = (
            "\n".join(f"- {h}" for h in artifact_hints) or "- No artifacts."
        )

        user_question = self._last_user_message(state)
        worker_block = self._format_worker_results(state)
        draft_block = supervisor_draft.strip() or "(no draft)"

        user_prompt = (
            f"USER QUESTION:\n{user_question}\n\n"
            f"WORKER RESULTS (internal — do NOT quote verbatim):\n{worker_block}\n\n"
            f"ARTIFACTS AVAILABLE TO THE UI:\n{artifact_block}\n\n"
            f"SUPERVISOR DRAFT (deliver this to the user; preserve verbatim "
            f"unless it leaks internals or is malformed):\n{draft_block}\n\n"
            "Stream the final answer for the user now."
        )

        async for chunk in self.llm.astream(
            [
                SystemMessage(content=self.FINAL_SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        ):
            delta = getattr(chunk, "content", "") or ""
            if delta:
                yield delta

    async def astream_events(
        self, message: str, chat_history: list[dict] | None = None
    ):
        """Yield events the API layer relays to the chat UI.

        Event types:
          - {"type": "step", "node": <node_name>}
          - {"type": "token", "delta": <str>}
          - {"type": "final", "response": <str>, "artifacts": {...}}
          - {"type": "error", "response": <str>}

        Intermediate text from sub-agents is intentionally never emitted.
        """
        state = self._build_initial_state(message, chat_history or [])
        accumulated_artifacts: dict[str, Any] = {}
        last_isolated_task = ""
        final_state: dict[str, Any] = {}

        try:
            async for chunk in self.graph.astream(
                state, config={"recursion_limit": self.recursion_limit}
            ):
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

                    for key in ("worker_results",):
                        if output.get(key):
                            final_state.setdefault(key, [])
                            final_state[key].extend(output[key])

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

            stream_state = {
                "messages": state["messages"],
                "worker_results": final_state.get("worker_results", []),
                "artifacts": accumulated_artifacts,
            }

            collected: list[str] = []
            try:
                async for delta in self._stream_final_synthesis(
                    stream_state, last_isolated_task
                ):
                    collected.append(delta)
                    yield {"type": "token", "delta": delta}
            except Exception as exc:
                logger.exception("Final synthesis streaming failed")
                fallback = last_isolated_task or "I could not produce a response."
                if not collected:
                    for ch in fallback:
                        yield {"type": "token", "delta": ch}
                yield {
                    "type": "final",
                    "response": "".join(collected) or fallback,
                    "artifacts": accumulated_artifacts,
                }
                return

            full_answer = "".join(collected) or last_isolated_task
            yield {
                "type": "final",
                "response": full_answer,
                "artifacts": accumulated_artifacts,
            }
        except Exception as exc:
            logger.exception("Graph astream failed")
            yield {
                "type": "error",
                "response": f"An error occurred: {exc}",
            }
