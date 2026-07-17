"""
Tool wrapper around src/tools/imci_protocol.py's deterministic rule engine.
This is the tool the LLM/HRM orchestration layer actually calls -- keeps the
safety-critical logic itself (imci_protocol.py) free of tool-registry/LLM
plumbing concerns.
"""

from __future__ import annotations

from dataclasses import asdict

from src.core.exceptions import ToolExecutionError
from src.tools.base import BaseTool
from src.tools.imci_protocol import ChildAssessment, assess


class IMCITriageTool(BaseTool):
    name = "imci_triage"
    description = (
        "Assess a child (2 months-5 years) against the IMCI protocol structure "
        "and return a classification (severe/moderate/mild) with recommended "
        "action. NOT a diagnosis -- protocol-following decision support only. "
        "See src/tools/imci_protocol.py for scope and required clinical "
        "verification notes."
    )
    parameters = {
        "age_months": {"type": "integer"},
        "danger_signs_present": {"type": "string", "description": "Comma-separated list, may be empty."},
        "cough_or_difficulty_breathing": {"type": "boolean", "default": False},
        "respiratory_rate_per_min": {"type": "integer", "default": None},
        "chest_indrawing": {"type": "boolean", "default": False},
        "stridor_when_calm": {"type": "boolean", "default": False},
        "diarrhea": {"type": "boolean", "default": False},
        "diarrhea_days": {"type": "integer", "default": None},
        "blood_in_stool": {"type": "boolean", "default": False},
        "child_lethargic_or_unconscious": {"type": "boolean", "default": False},
        "child_restless_or_irritable": {"type": "boolean", "default": False},
        "sunken_eyes": {"type": "boolean", "default": False},
        "not_able_to_drink_or_drinking_poorly": {"type": "boolean", "default": False},
        "drinking_eagerly_thirsty": {"type": "boolean", "default": False},
        "skin_pinch_goes_back_very_slowly": {"type": "boolean", "default": False},
        "skin_pinch_goes_back_slowly": {"type": "boolean", "default": False},
        "fever": {"type": "boolean", "default": False},
        "fever_days": {"type": "integer", "default": None},
        "stiff_neck": {"type": "boolean", "default": False},
        "ear_problem": {"type": "boolean", "default": False},
        "ear_pain": {"type": "boolean", "default": False},
        "ear_discharge_days": {"type": "integer", "default": None},
        "tender_swelling_behind_ear": {"type": "boolean", "default": False},
    }

    def run(self, age_months: int, danger_signs_present: str = "", **signs) -> dict:
        if age_months < 2 or age_months > 60:
            raise ToolExecutionError(
                self.name,
                "This scaffold models the 2-months-to-5-years IMCI age band only. "
                "The 0-2-months algorithm is structurally different and not modeled here.",
            )

        danger_list = [s.strip() for s in danger_signs_present.split(",") if s.strip()]

        child = ChildAssessment(
            age_months=age_months,
            danger_signs_present=danger_list,
            **{k: v for k, v in signs.items() if k in ChildAssessment.__dataclass_fields__},
        )
        result = assess(child)
        output = asdict(result)
        # Explicit conversion, not reliance on asdict()'s enum passthrough --
        # str(Classification.SEVERE) is "Classification.SEVERE" via Enum's
        # __str__, even though the object is also a str subclass. See the
        # discussion in this file's history / report notes for why this
        # matters beyond cosmetics.
        output["classification"] = result.classification.value
        return output
