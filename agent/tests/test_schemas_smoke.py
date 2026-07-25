"""Smoke tests for the agent's pydantic schemas + tool helpers."""

from __future__ import annotations

from agent.schemas import (
    ChatMessage,
    ChatRequest,
    DownloadPlan,
    PlotlyCodeGeneration,
    SupervisorDecision,
    WebSearchPlan,
)
from pydantic import ValidationError
import pytest


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


def test_chat_request_defaults_images_empty() -> None:
    req = ChatRequest(user_message="What is GDP?")
    assert req.images == []


def test_chat_request_accepts_images() -> None:
    req = ChatRequest(
        user_message="Describe this chart",
        images=["data:image/png;base64,AAAA", "data:image/jpeg;base64,BBBB"],
    )
    assert len(req.images) == 2
    assert req.images[0].startswith("data:image/png;base64,")


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


def test_download_plan_accepts_each_source() -> None:
    wb = DownloadPlan(
        thought_process="...", source="worldbank", indicator_id="NY.GDP.MKTP.CD", db_id=2
    )
    assert wb.source == "worldbank" and wb.indicator_id == "NY.GDP.MKTP.CD"

    yahoo = DownloadPlan(thought_process="...", source="yahoo", ticker="AAPL")
    assert yahoo.ticker == "AAPL" and yahoo.symbol is None

    binance = DownloadPlan(thought_process="...", source="binance", symbol="BTCUSDT")
    assert binance.symbol == "BTCUSDT" and binance.db_id is None

    fred = DownloadPlan(thought_process="...", source="fred", series_id="CAUR")
    assert fred.series_id == "CAUR" and fred.ticker is None

    eurostat = DownloadPlan(
        thought_process="...",
        source="eurostat",
        dataset="nama_10r_2gdp",
        filters={"unit": "EUR_HAB"},
    )
    assert eurostat.dataset == "nama_10r_2gdp" and eurostat.filters == {"unit": "EUR_HAB"}


def test_download_plan_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        DownloadPlan.model_validate({"thought_process": "...", "source": "gdelt"})


def test_plotly_code_generation_requires_fields() -> None:
    plan = PlotlyCodeGeneration(
        thought_process="line chart",
        plotly_code="fig = go.Figure()",
        title="GDP over time",
    )
    assert plan.title == "GDP over time"
    with pytest.raises(ValidationError):
        PlotlyCodeGeneration.model_validate({"thought_process": "...", "plotly_code": "fig = ..."})
