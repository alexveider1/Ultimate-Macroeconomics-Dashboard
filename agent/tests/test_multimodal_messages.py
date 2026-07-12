"""Unit tests for multimodal message construction in the graph.

Covers the two pure helpers that carry image attachments into the LangGraph
conversation without leaking base64 into the text-only worker/guardrail paths:
``_message_text`` and ``MacroAgentGraph._build_initial_state``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.graph import MacroAgentGraph, _format_chat_history, _message_text

_IMG = "data:image/png;base64,AAAA"


def test_message_text_passes_through_plain_string() -> None:
    assert _message_text("hello world") == "hello world"


def test_message_text_extracts_text_parts_and_drops_images() -> None:
    content = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": _IMG}},
    ]
    out = _message_text(content)
    assert out == "describe this"
    assert "base64" not in out


def test_build_initial_state_plain_message_stays_string() -> None:
    state = MacroAgentGraph._build_initial_state("What is GDP?", [])
    last = state["messages"][-1]
    assert isinstance(last, HumanMessage)
    assert last.content == "What is GDP?"


def test_build_initial_state_with_images_builds_content_parts() -> None:
    state = MacroAgentGraph._build_initial_state(
        "Describe these", [], images=[_IMG, "data:image/jpeg;base64,BBBB"]
    )
    last = state["messages"][-1]
    assert isinstance(last.content, list)
    # one text part + two image parts
    assert last.content[0] == {"type": "text", "text": "Describe these"}
    image_parts = [p for p in last.content if p.get("type") == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"] == _IMG


def test_history_render_never_leaks_base64_from_multimodal_turn() -> None:
    state = MacroAgentGraph._build_initial_state("look", [], images=[_IMG])
    rendered = _format_chat_history(state["messages"])
    assert "USER: look" in rendered
    assert "base64" not in rendered
