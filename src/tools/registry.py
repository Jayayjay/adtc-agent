"""
Tool registry: holds all available tools, generates the manifest for LLM
prompting, dispatches execution, and provides simple mention-detection used
by the router (src/router/rule_router.py) to count how many distinct tools
a message seems to reference.
"""

from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ToolNotFoundError
from src.tools.base import BaseTool
from src.tools.filesystem import FilesystemTool
from src.tools.calculator import CalculatorTool
from src.tools.datetime import DateTimeTool
from src.tools.imci_triage_tool import IMCITriageTool

# arxiv/web_search deliberately excluded by default -- both require network
# access, which conflicts with the offline-first framing. Add them back only
# if your final task domain genuinely needs them and you've confirmed
# network access is acceptable in the judging environment.
#
# IMCITriageTool is the domain differentiator for this submission (Healthcare
# & Medical niche, IMCI protocol). Calculator/DateTime remain useful generic
# support (age-in-months arithmetic, follow-up date calculation).
DEFAULT_TOOL_CLASSES = [CalculatorTool, DateTimeTool, IMCITriageTool]


class ToolRegistry:
    def __init__(self, sandbox_root: str | Path = "data/sandbox"):
        self._tools: dict[str, BaseTool] = {}
        for cls in DEFAULT_TOOL_CLASSES:
            self.register(cls())
        self.register(FilesystemTool(sandbox_root=sandbox_root))

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ToolNotFoundError(f"No tool registered with name '{name}'")
        return self._tools[name]

    def execute(self, name: str, **kwargs):
        return self.get(name).run(**kwargs)

    def manifest(self) -> list[dict]:
        return [tool.to_manifest() for tool in self._tools.values()]

    def detect_mentions(self, message: str) -> list[str]:
        """
        Cheap heuristic: does the message mention a tool's name? Used by the
        router to estimate multi-tool complexity without needing a full LLM
        intent parse first (that would cost inference cycles just to route).
        """
        text = message.lower()
        return [name for name in self._tools if name.replace("_", " ") in text]
