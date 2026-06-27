"""Smoke tests for the agent's pydantic schemas + tool helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import (
    ChatMessage,
    ChatRequest,
    PlotlyCodeGeneration,
    SupervisorDecision,
    WebSearchPlan,
)


def test_chat_request_accepts_history() -> None:
    req = ChatRequest(
        user_message="What is GDP?",
        chat_history=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ],
    )
    assert len(req.chat_history) == 2
    assert req.chat_history[0].role == "user"


def test_supervisor_decision_rejects_unknown_worker() -> None:
    with pytest.raises(ValidationError):
        SupervisorDecision(
            thought_process="...",
            updated_plan="1. step",
            next_worker="evil_worker",  # not in WORKER_LITERAL
            isolated_worker_task="...",
        )


def test_web_search_plan_enforces_query_count() -> None:
    WebSearchPlan(thought_process="...", search_queries=["a"])
    with pytest.raises(ValidationError):
        WebSearchPlan(thought_process="...", search_queries=[])
    with pytest.raises(ValidationError):
        WebSearchPlan(thought_process="...", search_queries=["a", "b", "c", "d"])


def test_plotly_code_generation_requires_fields() -> None:
    plan = PlotlyCodeGeneration(
        thought_process="line chart",
        plotly_code="fig = go.Figure()",
        title="GDP over time",
    )
    assert plan.title == "GDP over time"
    with pytest.raises(ValidationError):
        PlotlyCodeGeneration(thought_process="...", plotly_code="fig = ...")
