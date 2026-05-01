"""Per-request LLM token-usage accounting.

Attached as a LangChain callback to every LLM invocation in a single
`/chat/stream` request, this tracker accumulates prompt/completion/total
tokens across all internal sub-agent calls (guardrail, supervisor, sql_agent,
plotly_agent, ...) plus the final synthesis stream. The aggregate is returned
in the `final` event of the SSE stream so the dashboard can display per-session
totals.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler


class UsageTracker(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.model = ""

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        model_name = self._extract_model_name(response)
        if model_name:
            self.model = model_name

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if isinstance(token_usage, dict) and any(token_usage.values()):
            return {
                "prompt_tokens": token_usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": token_usage.get("completion_tokens", 0) or 0,
                "total_tokens": token_usage.get("total_tokens", 0) or 0,
            }

        generations = getattr(response, "generations", None) or []
        for gen_list in generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                meta = getattr(msg, "usage_metadata", None) or {}
                if meta:
                    return {
                        "prompt_tokens": int(meta.get("input_tokens", 0) or 0),
                        "completion_tokens": int(meta.get("output_tokens", 0) or 0),
                        "total_tokens": int(meta.get("total_tokens", 0) or 0),
                    }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _extract_model_name(response: Any) -> str:
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            for key in ("model_name", "model"):
                value = llm_output.get(key)
                if value:
                    return str(value)
        return ""

    def snapshot(self, default_model: str = "") -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model or default_model,
        }
