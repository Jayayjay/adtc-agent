"""
Task encoding for the adaptive IMCI question-asking dialogue -- converts a
DialogueState into a fixed-size numeric vector suitable for a small model's
input. This is a REAL, working encoding now (not a stub) -- see
report/REPORT_TEMPLATE_NOTES.md for the design history of how this scope was
arrived at (orchestration = adaptive question-asking, not "which symptom
module to check first", and definitely not the medical classification
itself -- see src/tools/imci_protocol.py for that).

ENCODING SCHEME (fixed-size vector, framework-agnostic -- convert to a
tensor at the model-loading layer, not here):

  [0]           age_months, normalized to [0, 1] via /60
  [1:5]         chief-complaint category one-hot (cough, diarrhea, fever, ear)
                -- "none matched" is the implicit all-zero case
  [5:5+3*21]    for each of the 21 fields in dialogue_state.ACTION_FIELDS
                (excluding the STOP sentinel, which isn't a state feature):
                a 3-value one-hot: [unknown, known_false, known_true]
                -- numeric fields (respiratory_rate_per_min, diarrhea_days,
                ear_discharge_days) use [unknown, known, known] where the
                2nd/3rd slots both fire "known" but the actual value is
                appended separately, see NUMERIC_VALUE_OFFSET below
  [tail]        3 extra scalars: normalized value (or 0 if unknown) for
                respiratory_rate_per_min (/70), diarrhea_days (/30),
                ear_discharge_days (/30), in that fixed order

Total length: 1 + 4 + 3*21 + 3 = 71
"""

from __future__ import annotations

from src.hrm.dialogue_state import ACTION_FIELDS, DialogueState, SymptomCategory
from src.hrm.expert_policy import CATEGORY_KEYWORDS

_CATEGORY_ORDER_FOR_ENCODING = [
    SymptomCategory.COUGH, SymptomCategory.DIARRHEA,
    SymptomCategory.FEVER, SymptomCategory.EAR,
]

_NUMERIC_FIELDS = {
    "respiratory_rate_per_min": 70.0,
    "diarrhea_days": 30.0,
    "ear_discharge_days": 30.0,
}

ENCODING_LENGTH = 1 + 4 + 3 * len(ACTION_FIELDS) + 3


def _chief_complaint_one_hot(chief_complaint_text: str) -> list[float]:
    text = chief_complaint_text.lower()
    return [
        1.0 if any(kw in text for kw in CATEGORY_KEYWORDS[cat]) else 0.0
        for cat in _CATEGORY_ORDER_FOR_ENCODING
    ]


def encode_task(task_description: str, context: dict) -> list[float]:
    """
    Args:
        task_description: free-text chief complaint (health worker's notes).
        context: must contain "state", a DialogueState instance -- or, for
            the very first call in a session, pass a DialogueState with
            empty known_fields/asked_fields.

    Returns:
        A fixed-length list[float] of length ENCODING_LENGTH (see module
        docstring). Convert to a tensor at the model-loading layer.
    """
    state: DialogueState = context["state"]

    vec: list[float] = [state.age_months / 60.0]
    vec.extend(_chief_complaint_one_hot(task_description or state.chief_complaint_text))

    numeric_values = []
    for field_name in ACTION_FIELDS:
        if field_name not in state.asked_fields:
            vec.extend([1.0, 0.0, 0.0])  # unknown
        elif field_name in _NUMERIC_FIELDS:
            vec.extend([0.0, 1.0, 0.0])  # known (numeric fields don't have a true/false split)
        else:
            value = state.get(field_name)
            if field_name == "danger_signs_present":
                vec.extend([0.0, 0.0, 1.0] if value else [0.0, 1.0, 0.0])
            else:
                vec.extend([0.0, 0.0, 1.0] if value else [0.0, 1.0, 0.0])

    for field_name, scale in _NUMERIC_FIELDS.items():
        if field_name in state.asked_fields:
            raw = state.get(field_name)
            numeric_values.append((raw or 0) / scale)
        else:
            numeric_values.append(0.0)
    vec.extend(numeric_values)

    assert len(vec) == ENCODING_LENGTH, f"Encoding length mismatch: {len(vec)} != {ENCODING_LENGTH}"
    return vec