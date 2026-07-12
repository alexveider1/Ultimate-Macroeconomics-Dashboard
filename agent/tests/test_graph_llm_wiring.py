"""Verify the strong/fast LLM split is wired to the right sub-agents."""

from agent.graph import MacroAgentGraph


def _build(fast: str = "fast-model") -> MacroAgentGraph:
    return MacroAgentGraph(
        base_url="https://example.invalid/v1",
        model_name="strong-model",
        fast_model_name=fast,
        api_key="sk-test",
    )


def test_strong_and_fast_models_are_distinct() -> None:
    graph = _build()
    assert graph.smart_llm is not graph.fast_llm
    assert graph.smart_llm.model_name == "strong-model"
    assert graph.fast_llm.model_name == "fast-model"


def test_strong_model_handles_planning_and_code() -> None:
    graph = _build()
    for agent in (graph.supervisor, graph.sql_agent, graph.plotly_agent, graph.chat_agent):
        assert agent.llm is graph.smart_llm


def test_fast_model_handles_screen_and_light_workers() -> None:
    graph = _build()
    for agent in (
        graph.guardrail,
        graph.table_agent,
        graph.rag_agent,
        graph.web_search_agent,
        graph.downloader_agent,
    ):
        assert agent.llm is graph.fast_llm


def test_empty_fast_model_falls_back_to_strong() -> None:
    graph = _build(fast="")
    assert graph.fast_llm.model_name == "strong-model"
