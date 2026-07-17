"""
Template for a new domain-specific tool. Copy this file, rename the class,
fill in name/description/parameters/run(), then register it in
src/tools/registry.py's DEFAULT_TOOLS list.

This is where your actual Phase 3 task-domain tools go once you've picked a
concrete use case (file organizer, scheduling assistant, dev-workflow agent,
etc.) -- the built-in tools (filesystem, calculator, datetime) are generic
scaffolding, not your differentiator.
"""

from __future__ import annotations

from src.tools.base import BaseTool


class ExampleTool(BaseTool):
    name = "example_tool"
    description = "Describe what this tool does, in one sentence, for the LLM's prompt."
    parameters = {
        "some_arg": {"type": "string", "description": "Describe this argument."},
    }

    def run(self, some_arg: str) -> dict:
        # Replace with real logic.
        return {"echo": some_arg}
