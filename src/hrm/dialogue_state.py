"""
State representation for the adaptive question-asking dialogue HRM will
learn to orchestrate. See src/hrm/expert_policy.py for the full design
rationale (why this task, not "symptom module ordering", is what HRM
actually has a justified role in).

The dialogue: a health worker starts with an age and a free-text chief
complaint. The system asks ONE question at a time, updates its knowledge,
and decides the next question -- until either a danger sign forces an
immediate stop, or enough fields are known to call
src.tools.imci_protocol.assess() with a confident classification. This
mirrors how IMCI is actually taught and practiced (sequential elicitation,
not a form filled out in one shot).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SymptomCategory(str, Enum):
    DANGER_SIGNS = "danger_signs"
    COUGH = "cough"
    DIARRHEA = "diarrhea"
    FEVER = "fever"
    EAR = "ear"


# The full action space HRM's policy head predicts over: one of these field
# names (meaning "ask about this field next"), or the STOP sentinel (meaning
# "enough is known -- call imci_protocol.assess() now"). Defined ONCE here
# and imported by both src/hrm/expert_policy.py (question text lookup) and
# src/hrm/encoders.py / decoders.py (model input/output encoding) to avoid
# the field list drifting between the two -- see
# tests/unit/test_expert_policy.py's coverage assertion.
ACTION_FIELDS = [
    "danger_signs_present",
    "cough_or_difficulty_breathing", "stridor_when_calm", "chest_indrawing",
    "respiratory_rate_per_min",
    "diarrhea", "child_lethargic_or_unconscious", "sunken_eyes",
    "not_able_to_drink_or_drinking_poorly", "skin_pinch_goes_back_very_slowly",
    "child_restless_or_irritable", "drinking_eagerly_thirsty",
    "skin_pinch_goes_back_slowly", "diarrhea_days", "blood_in_stool",
    "fever", "stiff_neck",
    "ear_problem", "tender_swelling_behind_ear", "ear_discharge_days", "ear_pain",
]
ACTION_STOP = "STOP"
ACTION_SPACE = ACTION_FIELDS + [ACTION_STOP]  # 22 total actions


@dataclass
class QuestionAction:
    """A single question the orchestrator wants asked next, or a STOP signal."""
    field_name: str | None       # ChildAssessment field name to elicit, or None if stopping
    category: SymptomCategory | None
    question_text: str
    is_stop: bool = False        # True once enough info is known to classify


@dataclass
class DialogueState:
    age_months: int
    chief_complaint_text: str = ""
    known_fields: dict = field(default_factory=dict)   # field_name -> value, only fields asked so far
    asked_fields: set = field(default_factory=set)      # field_names already asked (even if answer was False)
    turn_count: int = 0

    def record_answer(self, field_name: str, value) -> None:
        self.known_fields[field_name] = value
        self.asked_fields.add(field_name)
        self.turn_count += 1

    def get(self, field_name: str, default=None):
        return self.known_fields.get(field_name, default)

    def has_danger_sign(self) -> bool:
        signs = self.known_fields.get("danger_signs_present", [])
        return bool(signs)