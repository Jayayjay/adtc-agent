"""
Abstract base class all tools implement. Keeping this minimal and consistent
so registry.py can validate/dispatch generically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    # JSON-schema-ish parameter spec, used for LLM tool-call prompting and
    # for validators.py to sanity-check arguments before execution.
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool and return a JSON-serializable result."""
        raise NotImplementedError

    def to_manifest(self) -> dict:
        """Description block suitable for injecting into the LLM's system prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
