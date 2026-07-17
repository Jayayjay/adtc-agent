"""
Main Agent orchestration. This is the single place that wires together:
router -> (LLM fast path | HRM reasoning path) -> tools -> memory -> response

Kept deliberately thin -- each layer owns its own logic, Agent just sequences
the calls and handles the fast/reasoning branch.
"""

from __future__ import annotations

import logging

from src.config import SystemConfig
from src.core.exceptions import AgentError
from src.core.lifecycle import Lifecycle
from src.llm.formatter import format_tool_result
from src.router.rule_router import route, RoutePath
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.lifecycle = Lifecycle(config)
        self.tools = ToolRegistry()
        self.hrm = None  # lazily constructed -- see _get_hrm()

    def __enter__(self):
        self.lifecycle.startup()
        return self

    def __exit__(self, *exc):
        self.lifecycle.shutdown()

    def _get_hrm(self):
        if self.hrm is None:
            from src.hrm.state_machine import HRMSession
            self.hrm = HRMSession(self.config)
        return self.hrm

    def handle_message(self, message: str, session_id: str = "default") -> str:
        """
        Process one incoming message end-to-end. Returns the text response.
        """
        model = self.lifecycle.model_manager.primary

        # Relevant memory retrieval happens before routing so the router (and
        # eventually the LLM/HRM) can see prior context if needed.
        memory_hits = self.lifecycle.memory_manager.query(message, top_k=3)

        decision = route(message, requested_tools=self.tools.detect_mentions(message))
        logger.info("Routing decision: %s (%s)", decision.path, decision.reason)

        if decision.path == RoutePath.FAST:
            response = model.chat([
                {"role": "system", "content": self._system_prompt(memory_hits)},
                {"role": "user", "content": message},
            ])
        else:
            hrm = self._get_hrm()
            # TODO: extracting known_answers from free-text `message` via the
            # LLM's structured-extraction/tool-calling is real, unbuilt work
            # (see src/hrm/state_machine.py's solve_from_known_answers
            # docstring). Passing an empty dict here means the system will
            # immediately ask its first clarifying question rather than
            # guessing at unstated symptoms -- the safe default, not a bug.
            known_answers = {}
            age_months = self._extract_age_months(message)
            if age_months is None:
                response = (
                    "I need the child's age in months to begin an IMCI assessment. "
                    "Could you provide that?"
                )
            else:
                result = hrm.solve_from_known_answers(age_months, message, known_answers)
                if hasattr(result, "classification"):  # TriageResult
                    response = format_tool_result(model, message, {
                        "classification": result.classification.value,
                        "condition_label": result.condition_label,
                        "recommended_action": result.recommended_action,
                        "disclaimer": result.disclaimer,
                    })
                else:  # QuestionAction -- need more information
                    response = result.question_text

        self.lifecycle.memory_manager.store(session_id, message, response)
        return response

    @staticmethod
    def _system_prompt(memory_hits: list[dict]) -> str:
        from src.llm.prompts import build_system_prompt
        return build_system_prompt(memory_hits)

    @staticmethod
    def _extract_age_months(message: str) -> int | None:
        """
        Simple regex stopgap for pulling the child's age out of free text
        (e.g. "18-month-old", "2 years old", "18 months"). This is
        deliberately minimal -- real extraction should go through the LLM's
        structured output/tool-calling (see the TODO in handle_message),
        which would also handle the other known_answers fields, not just
        age. Returns None if no age is found, so the caller can ask
        explicitly rather than silently guessing.
        """
        import re
        months_match = re.search(r"(\d+)\s*[- ]?month", message, re.IGNORECASE)
        if months_match:
            return int(months_match.group(1))
        years_match = re.search(r"(\d+)\s*[- ]?year", message, re.IGNORECASE)
        if years_match:
            return int(years_match.group(1)) * 12
        return None