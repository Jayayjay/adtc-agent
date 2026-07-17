"""
Decodes HRM's raw output (a distribution/index over dialogue_state.ACTION_SPACE)
into a QuestionAction the agent can act on -- either "ask this field" or
"STOP, enough is known to classify."

REAL, working decoder now (not a stub) -- pairs with src/hrm/encoders.py's
concrete encoding scheme. Still needs an actual trained model to produce the
action_index this function consumes; see src/hrm/state_machine.py for how
the system runs on the expert policy (src/hrm/expert_policy.py) in the
meantime, and scripts/generate_hrm_training_data.py for how training data
for that eventual model gets generated.
"""

from __future__ import annotations

from src.hrm.dialogue_state import ACTION_SPACE, ACTION_STOP, QuestionAction, SymptomCategory
from src.hrm.expert_policy import QUESTION_TEXT, TOP_LEVEL_FIELD

_FIELD_TO_CATEGORY = {}
for _cat, _field in TOP_LEVEL_FIELD.items():
    _FIELD_TO_CATEGORY[_field] = _cat
# Sub-fields within each category -- map every ACTION_FIELDS entry to its
# category for QuestionAction construction (danger_signs_present handled
# separately since it isn't gated by a "category" the same way).
_SUBFIELD_CATEGORY = {
    "stridor_when_calm": SymptomCategory.COUGH,
    "chest_indrawing": SymptomCategory.COUGH,
    "respiratory_rate_per_min": SymptomCategory.COUGH,
    "child_lethargic_or_unconscious": SymptomCategory.DIARRHEA,
    "sunken_eyes": SymptomCategory.DIARRHEA,
    "not_able_to_drink_or_drinking_poorly": SymptomCategory.DIARRHEA,
    "skin_pinch_goes_back_very_slowly": SymptomCategory.DIARRHEA,
    "child_restless_or_irritable": SymptomCategory.DIARRHEA,
    "drinking_eagerly_thirsty": SymptomCategory.DIARRHEA,
    "skin_pinch_goes_back_slowly": SymptomCategory.DIARRHEA,
    "diarrhea_days": SymptomCategory.DIARRHEA,
    "blood_in_stool": SymptomCategory.DIARRHEA,
    "stiff_neck": SymptomCategory.FEVER,
    "tender_swelling_behind_ear": SymptomCategory.EAR,
    "ear_discharge_days": SymptomCategory.EAR,
    "ear_pain": SymptomCategory.EAR,
}
_FIELD_TO_CATEGORY.update(_SUBFIELD_CATEGORY)
_FIELD_TO_CATEGORY["danger_signs_present"] = SymptomCategory.DANGER_SIGNS


def decode_output(action_index: int) -> QuestionAction:
    """
    Args:
        action_index: index into dialogue_state.ACTION_SPACE, as produced by
            HRM's policy head (e.g. argmax over logits).

    Returns:
        A QuestionAction -- either asking about a specific field, or is_stop=True.
    """
    if not (0 <= action_index < len(ACTION_SPACE)):
        raise ValueError(f"action_index {action_index} out of range for ACTION_SPACE (len={len(ACTION_SPACE)})")

    action = ACTION_SPACE[action_index]
    if action == ACTION_STOP:
        return QuestionAction(field_name=None, category=None, question_text="", is_stop=True)

    return QuestionAction(
        field_name=action,
        category=_FIELD_TO_CATEGORY.get(action),
        question_text=QUESTION_TEXT[action],
    )