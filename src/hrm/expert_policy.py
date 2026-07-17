"""
Expert policy: decides the next question to ask in an adaptive IMCI
assessment dialogue. This is BOTH:

  1. The source of HRM's training data (see scripts/generate_hrm_training_data.py,
     which rolls this policy out over many synthetic cases to produce
     (state, correct_next_question) training pairs for imitation learning).
  2. A working interim orchestrator TODAY, before HRM is trained -- see
     src/hrm/state_machine.py, which uses this policy directly until a
     trained HRM checkpoint is available to (hopefully) match or improve on
     it.

DESIGN PRINCIPLE: this policy's branch structure deliberately mirrors
src.tools.imci_protocol.assess()'s branches as closely as possible, field by
field, so there is minimal risk of the "questions we ask" drifting from the
"logic we classify with" -- see the per-category comments cross-referencing
assess()'s corresponding branch. If assess() changes, check this file too.

WHAT "SMART" MEANS HERE: this policy stops asking questions the moment
enough information is known to reach a classification -- e.g. once a
danger sign is confirmed, it doesn't bother asking about cough or diarrhea,
because the outcome (urgent referral) is already determined. This mirrors
real IMCI practice: don't waste a caretaker's time or a health worker's
attention once the answer is already clear.
"""

from __future__ import annotations

from src.hrm.dialogue_state import DialogueState, QuestionAction, SymptomCategory


CATEGORY_ORDER = [
    SymptomCategory.COUGH,
    SymptomCategory.DIARRHEA,
    SymptomCategory.FEVER,
    SymptomCategory.EAR,
]

TOP_LEVEL_FIELD = {
    SymptomCategory.COUGH: "cough_or_difficulty_breathing",
    SymptomCategory.DIARRHEA: "diarrhea",
    SymptomCategory.FEVER: "fever",
    SymptomCategory.EAR: "ear_problem",
}

# Keyword matching for chief-complaint-driven ordering -- reuses the same
# lightweight approach as src/router/rule_router.py rather than introducing
# a second, inconsistent classification mechanism.
CATEGORY_KEYWORDS = {
    SymptomCategory.COUGH: ("cough", "breathing", "breath"),
    SymptomCategory.DIARRHEA: ("diarrhea", "diarrhoea", "stool", "vomit"),
    SymptomCategory.FEVER: ("fever", "hot", "temperature"),
    SymptomCategory.EAR: ("ear",),
}

QUESTION_TEXT = {
    "danger_signs_present": (
        "Check for general danger signs: Is the child unable to drink or "
        "breastfeed? Does the child vomit everything? Has the child had "
        "convulsions, or is convulsing now? Is the child lethargic or "
        "unconscious?"
    ),
    "cough_or_difficulty_breathing": "Does the child have a cough or difficulty breathing?",
    "stridor_when_calm": "Does the child have stridor when calm?",
    "chest_indrawing": "Does the child have chest indrawing?",
    "respiratory_rate_per_min": "Count the child's breaths per minute -- what is the respiratory rate?",
    "diarrhea": "Does the child have diarrhea?",
    "child_lethargic_or_unconscious": "Is the child lethargic or unconscious? (dehydration assessment)",
    "sunken_eyes": "Are the child's eyes sunken?",
    "not_able_to_drink_or_drinking_poorly": "Is the child unable to drink, or drinking poorly?",
    "skin_pinch_goes_back_very_slowly": "Does a skin pinch go back very slowly (>2 seconds)?",
    "child_restless_or_irritable": "Is the child restless or irritable?",
    "drinking_eagerly_thirsty": "Is the child drinking eagerly, showing thirst?",
    "skin_pinch_goes_back_slowly": "Does a skin pinch go back slowly (some delay)?",
    "diarrhea_days": "For how many days has the child had diarrhea?",
    "blood_in_stool": "Is there blood in the child's stool?",
    "fever": "Does the child have a fever?",
    "stiff_neck": "Does the child have a stiff neck?",
    "ear_problem": "Does the child have an ear problem?",
    "tender_swelling_behind_ear": "Is there tender swelling behind the child's ear?",
    "ear_discharge_days": "For how many days has the child had ear discharge, if any?",
    "ear_pain": "Does the child have ear pain?",
}


def _ask(field_name: str, category: SymptomCategory | None) -> QuestionAction:
    return QuestionAction(
        field_name=field_name, category=category,
        question_text=QUESTION_TEXT[field_name],
    )


def _stop() -> QuestionAction:
    return QuestionAction(field_name=None, category=None, question_text="", is_stop=True)


def _resolve_cough(state: DialogueState) -> QuestionAction | None:
    """Mirrors imci_protocol.assess()'s cough/difficult-breathing branch.
    Returns the next question, or None if this category is fully resolved."""
    if "cough_or_difficulty_breathing" not in state.asked_fields:
        return _ask("cough_or_difficulty_breathing", SymptomCategory.COUGH)
    if not state.get("cough_or_difficulty_breathing"):
        return None  # top-level negative -- category resolved, nothing further to ask

    if "stridor_when_calm" not in state.asked_fields:
        return _ask("stridor_when_calm", SymptomCategory.COUGH)
    if state.get("stridor_when_calm"):
        return None  # severe -- no need for chest_indrawing/RR, assess() short-circuits here too

    if "chest_indrawing" not in state.asked_fields:
        return _ask("chest_indrawing", SymptomCategory.COUGH)
    if state.get("chest_indrawing"):
        return None  # moderate via chest indrawing -- assess()'s `or` means RR isn't needed

    if "respiratory_rate_per_min" not in state.asked_fields:
        return _ask("respiratory_rate_per_min", SymptomCategory.COUGH)
    return None  # all fields needed for this branch are known


def _resolve_diarrhea(state: DialogueState) -> QuestionAction | None:
    """Mirrors imci_protocol._classify_dehydration()'s any-two-of-four logic
    for both severity levels, plus the secondary dysentery/persistent-
    diarrhoea findings."""
    if "diarrhea" not in state.asked_fields:
        return _ask("diarrhea", SymptomCategory.DIARRHEA)
    if not state.get("diarrhea"):
        return None

    severe_fields = [
        "child_lethargic_or_unconscious", "sunken_eyes",
        "not_able_to_drink_or_drinking_poorly", "skin_pinch_goes_back_very_slowly",
    ]
    severe_known_true = sum(1 for f in severe_fields if state.get(f) is True)
    if severe_known_true < 2:
        for f in severe_fields:
            if f not in state.asked_fields:
                # Stop asking severe-signs questions early once 2 are already
                # confirmed True (classification is already SEVERE) --
                # recompute after each answer, so this loop naturally
                # short-circuits on the next call once the count hits 2.
                return _ask(f, SymptomCategory.DIARRHEA)
    else:
        pass  # already >=2 severe signs known -- skip remaining severe questions entirely

    if severe_known_true < 2:
        # All 4 severe-sign questions asked, still <2 -- move to "some" tier.
        some_fields = [
            "child_restless_or_irritable", "sunken_eyes",
            "drinking_eagerly_thirsty", "skin_pinch_goes_back_slowly",
        ]
        some_known_true = sum(1 for f in some_fields if state.get(f) is True)
        if some_known_true < 2:
            for f in some_fields:
                if f not in state.asked_fields:
                    return _ask(f, SymptomCategory.DIARRHEA)

    # Primary dehydration classification is now determinable. Ask secondary
    # findings (don't gate the primary classification, but IMCI still wants
    # them recorded) before resolving the category.
    if "diarrhea_days" not in state.asked_fields:
        return _ask("diarrhea_days", SymptomCategory.DIARRHEA)
    if "blood_in_stool" not in state.asked_fields:
        return _ask("blood_in_stool", SymptomCategory.DIARRHEA)
    return None


def _resolve_fever(state: DialogueState) -> QuestionAction | None:
    """Mirrors imci_protocol.assess()'s simplified fever branch. NOTE: real
    IMCI's fever branch is malaria-risk/RDT-dependent and far more involved
    than this scaffold models -- see imci_protocol.py's module docstring."""
    if "fever" not in state.asked_fields:
        return _ask("fever", SymptomCategory.FEVER)
    if not state.get("fever"):
        return None
    if "stiff_neck" not in state.asked_fields:
        return _ask("stiff_neck", SymptomCategory.FEVER)
    return None


def _resolve_ear(state: DialogueState) -> QuestionAction | None:
    """Mirrors imci_protocol.assess()'s ear-problem branch."""
    if "ear_problem" not in state.asked_fields:
        return _ask("ear_problem", SymptomCategory.EAR)
    if not state.get("ear_problem"):
        return None

    if "tender_swelling_behind_ear" not in state.asked_fields:
        return _ask("tender_swelling_behind_ear", SymptomCategory.EAR)
    if state.get("tender_swelling_behind_ear"):
        return None  # severe -- mastoiditis, no need for further ear questions

    if "ear_discharge_days" not in state.asked_fields:
        return _ask("ear_discharge_days", SymptomCategory.EAR)
    discharge_days = state.get("ear_discharge_days")
    if discharge_days is not None and discharge_days >= 14:
        return None  # chronic ear infection -- resolved

    if "ear_pain" not in state.asked_fields:
        return _ask("ear_pain", SymptomCategory.EAR)
    return None


_CATEGORY_RESOLVERS = {
    SymptomCategory.COUGH: _resolve_cough,
    SymptomCategory.DIARRHEA: _resolve_diarrhea,
    SymptomCategory.FEVER: _resolve_fever,
    SymptomCategory.EAR: _resolve_ear,
}


def _keyword_matched_category(chief_complaint_text: str) -> SymptomCategory | None:
    """
    Informational only -- NOT used for question-ordering (see next_question()'s
    docstring for why chief-complaint-based reordering was removed). Kept
    for logging/UX purposes (e.g. "you mentioned cough" acknowledgment) and
    as a feature CATEGORY_KEYWORDS feeds into src/hrm/encoders.py's state
    encoding, which the trained model can learn to use or ignore.
    """
    text = chief_complaint_text.lower()
    for cat in CATEGORY_ORDER:
        if any(kw in text for kw in CATEGORY_KEYWORDS[cat]):
            return cat
    return None


def next_question(state: DialogueState) -> QuestionAction:
    """
    Top-level orchestrator: decides the single next question to ask, or
    signals STOP once enough is known to call imci_protocol.assess().

    CORRECTED DESIGN (fixes a real bug found during model validation --
    see report/REPORT_TEMPLATE_NOTES.md): assess() evaluates categories in a
    FIXED order (cough, diarrhea, fever, ear) and returns from the FIRST one
    with a positive top-level symptom, REGARDLESS of what order a health
    worker happened to ask about things.

    Two design iterations were tried and rejected before this one:
      1. Ask in chief-complaint-prioritized order, stop as soon as ANY
         asked category resolves positive. WRONG -- if a lower-fixed-
         precedence category (e.g. fever) is asked first and is positive,
         this stops there even when a higher-precedence category (e.g.
         diarrhea) is also positive and would be what assess() actually
         used. Caught via a trained model's end-to-end validation accuracy
         being lower than its action-level accuracy.
      2. Ask ALL FOUR top-level questions unconditionally before deciding
         which to deep-dive into. Correct, but wasteful -- if cough (first
         in fixed order) is already known positive, nothing later could
         ever override it, so asking about diarrhea/fever/ear top-level is
         pure waste. Caught by a stale unit test expectation that assumed
         fewer questions were needed than this version actually asked.

    Final design: scan top-level questions in FIXED order (matching
    assess()'s exact evaluation sequence), stopping the scan the INSTANT one
    is found positive -- both correct (matches assess() exactly) and
    minimal (never asks a question that can't affect the outcome). Chief
    complaint text is no longer used to reorder top-level questions, since
    doing so is fundamentally incompatible with short-circuiting safely.
    """
    # Step 1: danger signs, always first, matching assess()'s own first check.
    if "danger_signs_present" not in state.asked_fields:
        return _ask("danger_signs_present", SymptomCategory.DANGER_SIGNS)
    if state.has_danger_sign():
        return _stop()  # matches assess()'s immediate SEVERE short-circuit

    # Step 2: scan top-level questions in FIXED order, stopping the instant
    # one is positive -- that's the category assess() will use.
    for category in CATEGORY_ORDER:  # fixed: COUGH, DIARRHEA, FEVER, EAR
        top_level_field = TOP_LEVEL_FIELD[category]
        if top_level_field not in state.asked_fields:
            return _ask(top_level_field, category)
        if state.get(top_level_field) is True:
            # Found the category assess() will use. Deep-dive into it, and
            # STOP without ever checking later categories -- they can't
            # change the outcome now.
            resolver = _CATEGORY_RESOLVERS[category]
            action = resolver(state)
            if action is not None:
                return action
            return _stop()  # this category fully resolved -- assess() would return from here
        # else: top-level was False, move to next category in fixed order.

    return _stop()  # all four top-level symptoms were False -- assess() falls through to "no_classification_matched"


def simulate_dialogue(ground_truth, chief_complaint_text: str, max_turns: int = 30) -> list[dict]:
    """
    Rolls out the expert policy against a fully-known ChildAssessment
    ("simulated caretaker" who knows all the answers), producing a full
    (state, question, answer) trajectory. This is the training-data source
    for HRM's imitation learning -- see
    scripts/generate_hrm_training_data.py.

    Args:
        ground_truth: a fully-populated src.tools.imci_protocol.ChildAssessment
        chief_complaint_text: free-text chief complaint driving question order
        max_turns: safety bound against an infinite loop from a policy bug

    Returns:
        List of trajectory steps, each:
        {"turn": int, "known_fields": {...}, "asked_fields": [...],
         "question_field": str|None, "question_text": str, "is_stop": bool,
         "answer": <value or None if is_stop>}
    """
    state = DialogueState(age_months=ground_truth.age_months, chief_complaint_text=chief_complaint_text)
    trajectory = []

    for _ in range(max_turns):
        action = next_question(state)
        step = {
            "turn": state.turn_count,
            "known_fields": dict(state.known_fields),
            "asked_fields": sorted(state.asked_fields),
            "question_field": action.field_name,
            "question_text": action.question_text,
            "is_stop": action.is_stop,
        }
        if action.is_stop:
            step["answer"] = None
            trajectory.append(step)
            break

        answer = getattr(ground_truth, action.field_name)
        step["answer"] = answer
        trajectory.append(step)
        state.record_answer(action.field_name, answer)
    else:
        raise RuntimeError(
            f"simulate_dialogue exceeded max_turns={max_turns} without stopping -- "
            "likely a bug in next_question()'s resolver logic (infinite loop)."
        )

    return trajectory