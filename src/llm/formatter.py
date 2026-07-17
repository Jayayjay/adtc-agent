"""
Formats structured output (from HRM's reasoning path, or from a tool call
result) into a natural-language response via the LLM.
"""

from __future__ import annotations

from src.llm.client import LLMClient


def format_tool_result(model: LLMClient, task_description: str, structured_result: dict) -> str:
    """
    Used on the HRM REASONING path: HRM produces a structured plan/decision,
    this asks the LLM to render it as a clear response for the user.
    """
    prompt = (
        "You are formatting the result of a reasoning module into a clear, "
        "concise response for the user.\n\n"
        f"Original request: {task_description}\n"
        f"Structured result: {structured_result}\n\n"
        "Write a natural-language response reflecting this result. "
        "If it implies specific actions or tool calls, state them clearly."
    )
    return model.chat([{"role": "user", "content": prompt}])


def format_tool_execution_result(model: LLMClient, tool_name: str, tool_output) -> str:
    """Used after a tool call actually executes -- turns raw tool output into
    a user-facing sentence rather than dumping raw data."""
    prompt = (
        f"The tool '{tool_name}' returned this result:\n{tool_output}\n\n"
        "Summarize this result for the user in one or two sentences."
    )
    return model.chat([{"role": "user", "content": prompt}])
