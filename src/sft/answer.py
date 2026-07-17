"""
Renders a TriageResult as the clinical answer the model is trained to produce.

TriageResult is built for programmers, not for a health worker or a judge
panel, and three things in it must never reach the weights verbatim:

  - reasoning[] contains Python list reprs -- "General danger sign(s) present:
    ['convulsions']" -- and one line that cites this repo's own source path.
  - disclaimer's default text also cites src/tools/imci_protocol.py.
  - two of the fourteen recommended_action strings describe the code ("this
    scaffold covers...", "(not modeled)") rather than telling anyone what to do.

Train on those and the submitted model recites our file paths to the judges.
So every string is dispatched through a known pattern here, and anything
unrecognised raises: if imci_protocol.assess() ever grows a new reasoning
line, generation fails loudly instead of quietly leaking it into the data.

The CLASSIFICATION: header is deliberately byte-rigid -- it is what
eval/scoring/model_sacc_scorer.py regexes and what a judge skims. Only the
body varies.
"""

from __future__ import annotations

import random
import re

from src.tools.imci_protocol import ChildAssessment, Classification, TriageResult

# Presentation for each of the 14 labels assess() can return: the human name
# and the chart booklet's colour row (pink = refer, yellow = treat, green =
# home care). Colour is derived from Classification, not hard-coded per label,
# so the two can't drift.
LABEL_TEXT = {
    "very_severe_disease": "Very severe disease",
    "severe_pneumonia_or_very_severe_disease": "Severe pneumonia or very severe disease",
    "pneumonia": "Pneumonia",
    "cough_or_cold": "Cough or cold",
    "severe_dehydration": "Severe dehydration",
    "some_dehydration": "Some dehydration",
    "no_dehydration": "Diarrhoea with no dehydration",
    "very_severe_febrile_disease": "Very severe febrile disease",
    # No em-dash: " — " is the header's own separator between severity and
    # label, and a second one makes the rigid line ambiguous to split on.
    "fever_unspecified_malaria_not_assessed": "Fever, malaria not yet assessed",
    "mastoiditis": "Mastoiditis",
    "chronic_ear_infection": "Chronic ear infection",
    "acute_ear_infection": "Acute ear infection",
    "no_ear_infection": "No ear infection",
    "no_classification_matched": "No classification from the assessed signs",
}

COLOUR_PHRASE = {
    Classification.SEVERE: "IMCI pink: refer urgently",
    Classification.MODERATE: "IMCI yellow: treat and follow up",
    Classification.MILD: "IMCI green: home care",
}

DANGER_SIGN_TEXT = {
    "convulsions": "a history of convulsions",
    "convulsing_now": "convulsing now",
    "unable_to_drink_or_breastfeed": "unable to drink or breastfeed",
    "vomits_everything": "vomiting everything",
    "lethargic_or_unconscious": "lethargic or unconscious",
}

# The two recommended_action strings that describe the code instead of the
# care. Replaced wholesale; the other twelve pass through unchanged.
ACTION_OVERRIDES = {
    # The WHY line already states the malaria limitation; this says what to do
    # about it rather than restating it.
    "fever_unspecified_malaria_not_assessed": (
        "Give paracetamol for high fever. Assess malaria risk and test per the chart "
        "booklet, then treat according to the result. Refer for assessment if the fever "
        "persists more than 7 days, and advise the mother when to return immediately."
    ),
    "no_classification_matched": (
        "No danger signs and no cough, diarrhoea, fever, or ear problem were found, so no "
        "classification applies from what was assessed. Continue with the rest of the "
        "routine visit — check nutrition, anaemia, and immunisation status per the chart "
        "booklet — and advise the mother when to return immediately."
    ),
}

DISCLAIMERS = (
    "This is decision support that follows the IMCI chart booklet, not a diagnosis. "
    "Use clinical judgment and refer when in doubt.",
    "Protocol-following guidance only — not a diagnosis. The chart booklet and your own "
    "clinical judgment come first, and referral is always the safe choice when unsure.",
    "This follows the published IMCI algorithm and does not replace a clinician. "
    "When the picture is unclear, refer.",
    "Decision support, not a diagnosis: always confirm against the IMCI chart booklet and "
    "refer if anything is in doubt.",
)


def _humanize_sign_list(raw: str, lookup: dict[str, str] | None = None) -> str:
    """Turns a Python list repr's innards into prose: "['a', 'b']" -> "a and b"."""
    items = re.findall(r"'([^']*)'", raw)
    if lookup:
        items = [lookup.get(i, i.replace("_", " ")) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# The scaffold's fever/malaria note. assess() appends it BEFORE the finding it
# qualifies, which reads backwards in prose, so render_reasoning() floats it to
# the end rather than leaving it mid-paragraph.
_FEVER_CAVEAT = re.compile(r"^NOTE: this scaffold's fever branch is simplified.*", re.S)

# (compiled pattern, handler) applied in order. Handlers take the regex match
# and return clinician-facing prose, or None to drop the line entirely.
_RULES: list[tuple[re.Pattern, object]] = [
    (
        re.compile(r"^General danger sign\(s\) present: \[(.*)\]$"),
        lambda m: (
            f"The child has a general danger sign — {_humanize_sign_list(m.group(1), DANGER_SIGN_TEXT)}. "
            f"Any general danger sign means very severe disease and urgent referral, whatever "
            f"else is found."
        ),
    ),
    (
        re.compile(r"^No general danger signs present\.$"),
        lambda m: "No general danger signs were found, so the assessment continues by symptom.",
    ),
    (
        re.compile(r"^Stridor in a calm child\.$"),
        lambda m: "There is stridor while the child is calm, which marks severe obstruction.",
    ),
    (
        re.compile(r"^Chest indrawing present\.$"),
        lambda m: "There is chest indrawing.",
    ),
    (
        re.compile(r"^Fast breathing: (\d+)/min >= threshold (\d+)/min for age (\d+)mo\.$"),
        lambda m: (
            f"The respiratory rate is {m.group(1)} breaths per minute. For a child of "
            f"{m.group(3)} months the fast-breathing threshold is {m.group(2)}, so this counts "
            f"as fast breathing."
        ),
    ),
    (
        re.compile(r"^No fast breathing, chest indrawing, or stridor -- classified as cough/cold\.$"),
        lambda m: (
            "There is no fast breathing, no chest indrawing, and no stridor, so this is a "
            "simple cough or cold rather than pneumonia."
        ),
    ),
    (
        re.compile(r"^Severe dehydration signs \(>=2 of 4 required\): \[(.*)\]$"),
        lambda m: (
            f"Signs of severe dehydration are present: {_humanize_sign_list(m.group(1))}. "
            f"Two or more of the four severe signs classify as severe dehydration."
        ),
    ),
    (
        re.compile(r"^Some dehydration signs \(>=2 of 4 required\): \[(.*)\]$"),
        lambda m: (
            f"Signs of some dehydration are present: {_humanize_sign_list(m.group(1))}. "
            f"Two or more of the four classify as some dehydration."
        ),
    ),
    (
        re.compile(r"^Not enough signs to classify as some or severe dehydration\.$"),
        lambda m: (
            "Fewer than two dehydration signs are present, so this is diarrhoea with no "
            "dehydration."
        ),
    ),
    (
        # The scaffold's self-referential fever note. The substance is real and
        # worth keeping -- the fever branch genuinely does not do malaria -- but
        # it must be said as a clinical limitation, not as a note about our code.
        _FEVER_CAVEAT,
        lambda m: (
            "Note that this assessment does not cover malaria risk or a malaria test, "
            "which the chart booklet requires for any child with fever."
        ),
    ),
    (
        re.compile(r"^Fever with stiff neck\.$"),
        lambda m: "The child has fever with a stiff neck, which points to very severe febrile disease.",
    ),
    (
        re.compile(r"^Fever without stiff neck or other danger signs\.$"),
        lambda m: "The child has fever, with no stiff neck and no general danger signs.",
    ),
    (
        re.compile(r"^Tender swelling behind ear\.$"),
        lambda m: "There is tender swelling behind the ear, which indicates mastoiditis.",
    ),
    (
        re.compile(r"^Ear discharge present for 14\+ days\.$"),
        lambda m: "Ear discharge has been present for 14 days or more, so this is a chronic ear infection.",
    ),
    (
        re.compile(r"^Ear pain or discharge <14 days\.$"),
        lambda m: "There is ear pain, or discharge for less than 14 days, so this is an acute ear infection.",
    ),
    (
        re.compile(r"^No ear pain, discharge, or tender swelling\.$"),
        lambda m: "There is no ear pain, no discharge, and no tender swelling behind the ear.",
    ),
    (
        re.compile(r"^No presenting symptoms matched a modeled classification branch\.$"),
        lambda m: "None of the assessed symptom areas turned up a problem.",
    ),
]


def humanize_reasoning(line: str) -> str | None:
    """
    Maps one assess() reasoning line to clinician-facing prose.

    Returns None for lines that should be dropped. Raises on anything
    unrecognised -- that is the point: a new reasoning line in
    imci_protocol.py must fail the generator, not slip into training data
    carrying a file path or a list repr.
    """
    for pattern, handler in _RULES:
        m = pattern.match(line)
        if m:
            return handler(m)
    raise ValueError(
        f"Unrecognised reasoning line from imci_protocol.assess():\n  {line!r}\n"
        f"src/sft/answer.py must be taught how to say this in clinical prose before it "
        f"can be trained on. Refusing to pass it through verbatim -- assess()'s raw "
        f"reasoning contains Python list reprs and source-file references."
    )


def render_reasoning(result: TriageResult) -> str:
    """Joins the humanized reasoning, floating caveats to the end so they
    qualify the findings rather than interrupting them."""
    body, caveats = [], []
    for line in result.reasoning:
        text = humanize_reasoning(line)
        if text is None:
            continue
        (caveats if _FEVER_CAVEAT.match(line) else body).append(text)
    return " ".join(body + caveats)


def render_action(result: TriageResult) -> str:
    return ACTION_OVERRIDES.get(result.condition_label, result.recommended_action)


def render_header(result: TriageResult) -> str:
    """The byte-rigid line the scorer parses. Never vary this format."""
    label = LABEL_TEXT.get(result.condition_label)
    if label is None:
        raise ValueError(
            f"No display text for condition_label {result.condition_label!r} — add it to "
            f"LABEL_TEXT in src/sft/answer.py."
        )
    return (
        f"CLASSIFICATION: {result.classification.value.upper()} — {label} "
        f"({COLOUR_PHRASE[result.classification]})"
    )


def render_answer(
    result: TriageResult,
    rng: random.Random,
    acknowledged_extra: str | None = None,
) -> str:
    """
    Full answer: rigid header, then varied body.

    `acknowledged_extra` carries the "she also has a fever, but the booklet
    takes the cough branch first" note for the ~15% of cases that deliberately
    keep a non-decisive symptom, so multi-symptom prompts don't train the model
    to silently drop what it was told.
    """
    lines = [render_header(result), ""]
    lines.append(f"WHY: {render_reasoning(result)}")
    lines.append(f"ACTION: {render_action(result)}")

    if result.secondary_findings:
        lines.append("ALSO: " + " ".join(result.secondary_findings))
    if acknowledged_extra:
        lines.append(f"NOTE: {acknowledged_extra}")

    lines.append("")
    lines.append(rng.choice(DISCLAIMERS))
    return "\n".join(lines)
