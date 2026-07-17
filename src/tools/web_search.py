"""
Web search tool -- PLACEHOLDER ONLY.

Given the offline-first constraint of this submission (Healthcare & Medical /
IMCI decision support -- see src/tools/imci_protocol.py), a live web search
tool is the WRONG choice here. Not registered by default. This file exists
only as a consistent-interface placeholder in case a future variant needs it.
"""

from __future__ import annotations

from src.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "MOCK: placeholder web search. Not offline-compatible -- see module docstring."
    parameters = {"query": {"type": "string"}}

    def run(self, query: str) -> dict:
        return {
            "results": [],
            "warning": (
                "web_search is a mock/placeholder tool and does not perform a "
                "real search. Consider whether this tool belongs in an "
                "offline-first submission at all."
            ),
        }