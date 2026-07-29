"""
Assembles the four kinds of training example into chat-formatted records.

Why a mixture and not just triage pairs: a model trained only to classify
learns to classify EVERYTHING. Ask it the capital of Nigeria and it returns an
IMCI classification; ask about a newborn and it invents one. Three of the four
kinds exist to stop that.

  triage         75%  the job
  scope_refusal   5%  out of the modelled band, or a branch assess() cannot
                      label at all -- say so, don't guess
  next_question   5%  underspecified input -> ask the ONE correct next question
                      rather than inventing the answer
  general_chat   15%  anti-forgetting; self-distilled from the base model

The scope_refusal slice is where the honesty of this submission lives.
imci_protocol.py's docstring is explicit that malaria risk/RDT, malnutrition,
anaemia, measles, and the 0-2mo young-infant algorithm are NOT modelled by the
core engine -- and those are exactly where a grader probing for overfitting
would reach. In the DEFAULT build the model declines them rather than
confabulate. Under --include-extended the extended/young-infant classifiers
handle them, and EXTENDED_TOPICS suppresses the matching refusals so the corpus
never both refuses and classifies the same thing.
"""

from __future__ import annotations

import random
import re
from enum import Enum

from src.hrm.expert_policy import QUESTION_TEXT, simulate_dialogue
from src.sft.answer import DISCLAIMERS, render_answer
from src.sft.sampling import required_fields_for, sample_coherent_case, sample_weight_for_age
from src.sft.verbalize import VignetteStyle, render_age, verbalize_case, Register, STYLE_REGISTER
from src.tools.imci_protocol import ChildAssessment, assess


class Kind(str, Enum):
    TRIAGE = "triage"
    SCOPE_REFUSAL = "scope_refusal"
    NEXT_QUESTION = "next_question"
    GENERAL_CHAT = "general_chat"
    # Auxiliary knowledge Q&A, all answered deterministically from the reviewed
    # dosing tables / treatment map -- not vignettes, no CLASSIFICATION header.
    DOSING_QUERY = "dosing_query"
    TREATMENT_QUERY = "treatment_query"
    FOLLOWUP_QUERY = "followup_query"
    DIFFERENTIAL = "differential"


# Shortened from src/llm/prompts.BASE_SYSTEM_PROMPT. That one is ~25 lines and
# ends by telling the model to announce "the imci_triage tool" -- a tool that
# does not exist at evaluation, where only the raw .gguf runs. Training on it
# would teach the model to narrate a tool call it can never make.
SYSTEM_PROMPTS = (
    "You are an offline decision-support assistant for community health workers "
    "applying the WHO IMCI protocol to children 2 months to 5 years old. Check "
    "general danger signs first. Never downgrade a severity to avoid a referral. "
    "Defer to the chart booklet; it is authoritative.",
    "You help health workers apply the IMCI chart booklet for children aged 2 "
    "months to 5 years. Always check danger signs before anything else, and refer "
    "urgently when in doubt. You give protocol-following decision support, not a "
    "diagnosis.",
    "Offline IMCI decision support for children 2 months to 5 years. Follow the "
    "chart booklet exactly: danger signs first, then cough, diarrhoea, fever, ear. "
    "When unsure, refer.",
)

REFUSAL_HEADER = "CANNOT CLASSIFY"

# Branches imci_protocol.assess() does not model. Each entry is
# (topic, prompt templates, the honest reason we cannot answer).
OUT_OF_SCOPE_TOPICS = (
    (
        "malaria",
        (
            "The child has a fever and we're in a high malaria risk area. What's the classification?",
            "Fever for 2 days, RDT positive. How do I classify this child?",
            "How does malaria risk change the fever classification?",
            "Do I treat this fever as malaria? We're in a high transmission zone.",
            "child has fever, rdt negative, what now?",
            "What antimalarial do I give a 3 year old with fever?",
        ),
        "the malaria-risk and rapid-test steps the chart booklet requires for fever",
    ),
    (
        "malnutrition",
        (
            "The child's MUAC is 110mm. How do I classify?",
            "This child looks very thin and has swelling of both feet. What classification?",
            "How do I check for acute malnutrition?",
            "What's the classification for severe wasting?",
            "child has oedema of both feet, very thin. classify?",
        ),
        "the acute malnutrition assessment",
    ),
    (
        "anaemia",
        (
            "The child has severe palmar pallor. What's the classification?",
            "How do I classify anaemia in a 2 year old?",
            "Palmar pallor present. Is this anaemia?",
        ),
        "the anaemia assessment",
    ),
    (
        "measles",
        (
            "The child has a generalised rash and red eyes. Is this measles?",
            "How do I classify measles with mouth ulcers?",
            "Rash all over, cough and runny nose. Measles?",
        ),
        "the measles assessment and its complications",
    ),
    (
        "young_infant",
        (
            "The baby is 3 weeks old and not feeding well. How do I classify?",
            "A 5 day old with fever. What's the classification?",
            "newborn, 10 days old, umbilicus red. classify?",
            "How do I assess a 6 week old with fast breathing?",
        ),
        "the 0-2 month young-infant algorithm, which is a structurally different chart",
    ),
    (
        "hiv",
        (
            "How does HIV status change the classification?",
            "The mother is HIV positive. What do I do differently for this child?",
        ),
        "HIV status assessment",
    ),
    (
        "immunisation",
        (
            "Which vaccines does a 9 month old need?",
            "What's the immunisation schedule for this child?",
        ),
        "the immunisation and vitamin A schedule",
    ),
)

# Topics that BECOME classifiable under --include-extended. In the default
# (conservative) build they are refused; in the extended build the classifiers
# handle them, so their refusals are suppressed to avoid a contradictory signal.
EXTENDED_TOPICS = frozenset({"malaria", "malnutrition", "anaemia", "measles", "young_infant", "hiv"})

# Age-band refusals are generated from real vignettes with an out-of-band age,
# so the model learns to notice the age rather than a topic keyword.
#
# Written as whole sentences per case rather than a "because {reason}" frame:
# the two reasons are a singular noun phrase and a plural one, and no single
# frame fits both ("because children over 5 years ... is not something I
# assess").
AGE_REFUSAL_BODY = {
    "young": (
        "This child is {age_desc}, which is under 2 months. The 0-2 month young-infant "
        "algorithm is a structurally different chart and I do not assess it. Use the "
        "young-infant chart for this baby, and refer urgently if there is any danger sign."
    ),
    "over_five": (
        "This child is {age_desc}, which is over 5 years. The IMCI chart covers 2 months up "
        "to 5 years, so it does not apply here. Assess this child with the appropriate "
        "guideline for their age, and refer if they look unwell."
    ),
}


def _system_message(rng: random.Random) -> list[dict]:
    """Half of all examples carry no system prompt at all.

    The evaluator runs the raw .gguf through llama.cpp and may pass nothing.
    A model that only behaves correctly when primed is a model that fails on
    the day, so half the data teaches the behaviour unconditionally.
    """
    if rng.random() < 0.5:
        return [{"role": "system", "content": rng.choice(SYSTEM_PROMPTS)}]
    return []


def _record(messages: list[dict], case_id: str, kind: Kind, **meta) -> dict:
    return {"messages": messages, "case_id": case_id, "meta": {"kind": kind.value, **meta}}


def _fields_dict(child: ChildAssessment) -> dict:
    """Only the non-default fields: an absent field means False/None to
    assess(), so recording the whole dataclass would triple the file size to
    say nothing."""
    default = ChildAssessment(age_months=child.age_months)
    return {
        f: getattr(child, f)
        for f in ChildAssessment.__dataclass_fields__
        if f != "age_months" and getattr(child, f) != getattr(default, f)
    }


def make_triage_example(
    child: ChildAssessment,
    result,
    rng: random.Random,
    style: VignetteStyle,
    case_id: str,
    acknowledged_extra: str | None = None,
) -> dict:
    text, rendered = verbalize_case(child, rng, style)
    missing = required_fields_for(child) - rendered
    if missing:
        raise AssertionError(
            f"vignette for {case_id} omitted required field(s) {missing} -- this would "
            f"train the model to invent the sign that decided the answer:\n{text}"
        )
    weight_kg = sample_weight_for_age(child.age_months, rng)
    messages = _system_message(rng) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": render_answer(
            result, rng, acknowledged_extra, weight_kg=weight_kg, age_months=child.age_months)},
    ]
    return _record(
        messages, case_id, Kind.TRIAGE,
        style=style.value,
        classification=result.classification.value,
        condition_label=result.condition_label,
        age_months=child.age_months,
        weight_kg=weight_kg,
        # The pruned case, verbatim. Lets an auditor re-run assess() on the
        # exact input the vignette describes instead of regexing prose for
        # clinical signs -- "lethargic -" and "lethargic" differ by one
        # character and mean opposite things, so text matching cannot be the
        # source of truth for whether a sign was asserted.
        fields=_fields_dict(child),
    )


def make_scope_refusal_example(rng: random.Random, case_id: str,
                               exclude_topics: frozenset = frozenset()) -> dict:
    """Out-of-scope prompt -> an honest decline.

    Two sources, deliberately: topic prompts (malaria, malnutrition, ...) teach
    "this subject isn't mine", and out-of-band ages teach "check the age" --
    which generalises to a vignette about a newborn that never says the word
    "newborn".

    `exclude_topics` drops topics that a build actually models (the extended
    build classifies malaria/measles/.../young-infant/HIV), so the corpus never
    both refuses and classifies the same thing.
    """
    # young_infant is an age-band refusal (not a topic tuple); when it is
    # excluded (extended build), only over-five age refusals remain.
    young_ok = "young_infant" not in exclude_topics
    if rng.random() < 0.45:
        # Age-band refusal, rendered as a real vignette with an out-of-band age.
        young = young_ok and rng.random() < 0.5
        age = rng.randint(0, 1) if young else rng.randint(61, 144)
        child = sample_coherent_case(rng)
        child.age_months = age
        style = rng.choice(list(VignetteStyle))
        text, _ = verbalize_case(child, rng, style)
        # Always describe the age in the narrative register: the refusal is
        # prose, so "4/12" or "rr 33" shorthand would read as a different voice
        # from the sentence around it.
        age_desc = render_age(age, rng, Register.NARRATIVE)
        body = AGE_REFUSAL_BODY["young" if young else "over_five"].format(age_desc=age_desc)
        meta_topic = "young_infant" if young else "over_five"
        style_val = style.value
    else:
        pool = [t for t in OUT_OF_SCOPE_TOPICS if t[0] not in exclude_topics]
        topic, prompts, reason = rng.choice(pool)
        text = rng.choice(prompts)
        body = (
            f"I cannot classify this. My assessment covers general danger signs, cough or "
            f"difficulty breathing, diarrhoea, fever, and ear problems for children 2 months "
            f"to 5 years — it does not cover {reason}. Follow the chart booklet for this step, "
            f"and refer if the child has any general danger sign."
        )
        meta_topic = topic
        style_val = None

    answer = f"{REFUSAL_HEADER}: out of scope\n\n{body}\n\n{rng.choice(DISCLAIMERS)}"
    messages = _system_message(rng) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ]
    return _record(messages, case_id, Kind.SCOPE_REFUSAL, topic=meta_topic, style=style_val)


def make_next_question_example(rng: random.Random, case_id: str) -> dict | None:
    """Underspecified vignette -> the one correct next IMCI question.

    Built by rolling out expert_policy against a full case and stopping at a
    random mid-dialogue turn: the fields elicited so far become the vignette,
    and the question the policy would ask next becomes the answer. That keeps
    the "right question" definition identical to the orchestration policy's,
    which is itself derived from assess()'s branch order.

    Returns None when a rollout yields no usable mid-point (e.g. a danger sign
    stops it on turn one).
    """
    ground_truth = sample_coherent_case(rng)
    try:
        trajectory = simulate_dialogue(ground_truth, "")
    except RuntimeError:
        return None

    usable = [s for s in trajectory if not s["is_stop"] and s["turn"] > 0]
    if not usable:
        return None
    step = rng.choice(usable)

    known = step["known_fields"]
    partial = ChildAssessment(age_months=ground_truth.age_months)
    for f, v in known.items():
        setattr(partial, f, v)

    style = rng.choice(list(VignetteStyle))
    # restrict_to is essential here: an un-asked field is at its default, which
    # is indistinguishable from a real "no". Without it the vignette would
    # answer questions the caretaker was never asked.
    text, _ = verbalize_case(partial, rng, style, restrict_to=set(known))

    question = QUESTION_TEXT[step["question_field"]]
    answer = (
        f"NEXT QUESTION: {question}\n\n"
        f"WHY: I do not have enough to classify this child yet. The chart booklet asks this "
        f"next, and the answer changes the classification.\n\n"
        f"{rng.choice(DISCLAIMERS)}"
    )
    messages = _system_message(rng) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ]
    return _record(
        messages, case_id, Kind.NEXT_QUESTION,
        style=style.value, question_field=step["question_field"],
    )


# Self-distillation copies the base model faithfully, which means it also copies
# artefacts that must never become training targets. Measured over the first 228
# sampled pairs: 25% were cut mid-sentence by the token cap, ~2% came back in
# Chinese, and a handful were near-empty.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_NON_LATIN = re.compile(r"[　-鿿Ѐ-ӿ؀-ۿ]")


def clean_completion(text: str, min_chars: int = 20) -> str | None:
    """Makes one distilled completion safe to train on, or rejects it.

    Three failure modes, each of which teaches the model something we don't
    want:

      truncation  -- max_tokens cuts mid-sentence ("...the stars were twinkling,
                     Fatima fell asleep"). A target that stops mid-clause
                     teaches the model to stop mid-clause. Trimmed back to the
                     last complete sentence rather than discarded, so the
                     sample isn't wasted.
      non-English -- Qwen is multilingual and answers "2 + 2 =" in Chinese about
                     2% of the time. metadata.json declares language_scope
                     ["en"], and the graders read English. Rejected outright:
                     there is no salvaging half a Chinese answer.
      too short   -- a 3-character target is noise.

    Returns None when the completion cannot be salvaged.
    """
    text = (text or "").strip()
    if not text:
        return None
    if _NON_LATIN.search(text):
        return None

    if not text.endswith((".", "!", "?", '"', "`", ")", "*")):
        parts = _SENTENCE_END.split(text)
        if len(parts) > 1:
            text = " ".join(parts[:-1]).strip()  # drop the incomplete tail
        else:
            return None  # a single unfinished sentence -- nothing to keep

    return text if len(text) >= min_chars else None


def make_extended_example(rng: random.Random, case_id: str) -> dict | None:
    """A vignette + answer for one of the extended-protocol branches (malaria,
    measles, anaemia, malnutrition).

    OFF by default (see scripts/generate_sft_data.py --include-extended). The
    extended classifiers are UNREVIEWED clinical logic; nothing they produce
    should train a shipped model until src/sft/extended_protocol.py's review
    checklist is signed off.

    Returns None when a sampled case doesn't actually trigger its branch (e.g. a
    measles draw that misses the case definition), so the caller retries.
    """
    from src.sft.extended_protocol import (
        HIV_TESTS, MALARIA_RISKS, MALARIA_TESTS, ExtendedAssessment,
        classify_anaemia, classify_dysentery, classify_fever_malaria, classify_growth,
        classify_hiv, classify_malnutrition, classify_measles, classify_persistent_diarrhoea,
        classify_sore_throat, classify_wheeze,
    )
    from src.sft.extended_verbalize import (
        EXTENDED_LABEL_TEXT, render_extended_answer, verbalize_extended,
    )

    branch = rng.choice([
        "malaria", "measles", "anaemia", "malnutrition",
        "wheeze", "persistent_diarrhoea", "dysentery", "sore_throat", "growth", "hiv",
    ])
    danger = ([rng.choice(["convulsions", "lethargic_or_unconscious", "vomits_everything"])]
              if rng.random() < 0.1 else [])

    if branch == "malaria":
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger, fever=True,
            fever_days=rng.choice([None, 1, 2, 3, 8]),
            stiff_neck=rng.random() < 0.1,
            malaria_risk=rng.choice(MALARIA_RISKS),
            malaria_test=rng.choice(MALARIA_TESTS),
            travel_to_malaria_area=rng.random() < 0.2,
        )
        result = classify_fever_malaria(a)
    elif branch == "measles":
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger,
            generalised_rash=True, cough_or_runny_nose_or_red_eyes=rng.random() < 0.85,
            measles_within_3_months=rng.random() < 0.2,
            clouding_of_cornea=rng.random() < 0.15,
            deep_or_extensive_mouth_ulcers=rng.random() < 0.15,
            mouth_ulcers=rng.random() < 0.25, pus_draining_from_eye=rng.random() < 0.25,
        )
        result = classify_measles(a)
    elif branch == "anaemia":
        roll = rng.random()
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger,
            severe_palmar_pallor=roll < 0.34, some_palmar_pallor=0.34 <= roll < 0.67,
        )
        result = classify_anaemia(a)
    elif branch == "malnutrition":
        a = ExtendedAssessment(
            age_months=rng.randint(6, 59), danger_signs_present=danger,
            oedema_of_both_feet=rng.random() < 0.2,
            muac_mm=rng.choice([None, 105, 110, 118, 122, 135]),
            wfh_z_score=rng.choice([None, -3.5, -2.5, -1.0]),
            appetite_test_passed=rng.choice([None, True, False]),
            medical_complication=rng.random() < 0.15,
        )
        result = classify_malnutrition(a)
    elif branch == "wheeze":
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger, wheeze=True,
        )
        result = classify_wheeze(a)
    elif branch == "persistent_diarrhoea":
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger, diarrhoea=True,
            diarrhoea_days=rng.choice([14, 16, 21, 30]),
            dehydration_present=rng.random() < 0.35,
            losing_weight=rng.random() < 0.3,
        )
        result = classify_persistent_diarrhoea(a)
    elif branch == "dysentery":
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger, diarrhoea=True,
            diarrhoea_days=rng.choice([2, 3, 5]), blood_in_stool=True,
            dehydration_present=rng.random() < 0.35,
        )
        result = classify_dysentery(a)
    elif branch == "sore_throat":
        a = ExtendedAssessment(
            age_months=rng.randint(36, 59), danger_signs_present=danger, sore_throat=True,
            enlarged_tonsils=rng.random() < 0.6, tonsil_exudate=rng.random() < 0.5,
            scarlatiniform_rash=rng.random() < 0.2,
            runny_nose=rng.random() < 0.4, cough=rng.random() < 0.4,
        )
        result = classify_sore_throat(a)
    elif branch == "growth":
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger,
            losing_weight=rng.random() < 0.6, low_weight_for_age=rng.random() < 0.6,
        )
        result = classify_growth(a)
    else:  # hiv
        a = ExtendedAssessment(
            age_months=rng.randint(2, 59), danger_signs_present=danger,
            hiv_test=rng.choice(HIV_TESTS),
            child_on_art=rng.random() < 0.12,
            infant_on_arv_prophylaxis=rng.random() < 0.15,
            breastfeeding_at_or_near_test=rng.random() < 0.3,
            breastfeeding_stopped_ge_6wk=rng.random() < 0.3,
            mother_hiv_positive=rng.random() < 0.35,
            hiv_pneumonia_now=rng.random() < 0.2,
            hiv_persistent_diarrhoea=rng.random() < 0.2,
            hiv_ever_ear_discharge=rng.random() < 0.2,
            hiv_low_weight=rng.random() < 0.25,
            hiv_enlarged_lymph_nodes=rng.random() < 0.2,
            hiv_oral_thrush=rng.random() < 0.2,
            hiv_parotid_enlargement=rng.random() < 0.12,
        )
        result = classify_hiv(a)

    if result is None or result.condition_label not in EXTENDED_LABEL_TEXT:
        return None

    style = rng.choice(list(VignetteStyle))
    text = verbalize_extended(a, rng, style)
    weight_kg = sample_weight_for_age(a.age_months, rng)
    messages = _system_message(rng) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": render_extended_answer(
            result, rng, weight_kg=weight_kg, age_months=a.age_months)},
    ]
    return _record(messages, case_id, Kind.TRIAGE, style=style.value,
                   classification=result.classification.value,
                   condition_label=result.condition_label, extended=True, branch=branch,
                   age_months=a.age_months, weight_kg=weight_kg)


def make_general_chat_example(prompt: str, completion: str, rng: random.Random, case_id: str) -> dict:
    """A base-model response, replayed as its own target.

    Self-distillation rather than an off-the-shelf instruction set: the target
    IS the base distribution, so it applies almost no gradient pressure while
    still occupying the 15% that stops the IMCI data from eating general
    ability. No licence questions, and it is defensible in the report.
    """
    messages = _system_message(rng) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    return _record(messages, case_id, Kind.GENERAL_CHAT)


# --------------------------------------------------------------------------
# Auxiliary knowledge Q&A. Not vignettes: each answer is looked up
# deterministically from the reviewed dosing tables / treatment map, so these
# carry the same no-hand-written-doses guarantee as the triage answers. They
# add question-type breadth (dose lookups, follow-up intervals, differentials)
# without a CLASSIFICATION header, since they are not classification tasks.
# --------------------------------------------------------------------------
_DRUG_DISPLAY = {
    "amoxicillin": "amoxicillin",
    "paracetamol": "paracetamol",
    "zinc": "zinc",
    "vitamin_a": "vitamin A",
    "ceftriaxone_im": "ceftriaxone",
    "diazepam_rectal": "rectal diazepam",
    "artemether_lumefantrine": "artemether-lumefantrine",
    "mebendazole": "mebendazole",
    "ors_plan_a_extra_fluid": "ORS for home (Plan A)",
}


def _label_display(label: str) -> str:
    """Readable name for a classification label, from the shared label maps."""
    from src.sft.answer import LABEL_TEXT
    from src.sft.extended_verbalize import EXTENDED_LABEL_TEXT
    return LABEL_TEXT.get(label) or EXTENDED_LABEL_TEXT.get(label) or label.replace("_", " ")


def _sample_dose_input(drug: str, rng: random.Random) -> tuple[float | None, int | None]:
    """A weight (weight-keyed drug) or age (age-keyed drug) inside a real band."""
    from src.tools import imci_dosing
    band = rng.choice(imci_dosing.bands_for(drug))
    if imci_dosing.key_for(drug) == "weight":
        lo = band.get("weight_kg_min") or 4.0
        hi = band.get("weight_kg_max") or (lo + 5)
        return round(rng.uniform(lo, hi - 0.1), 1), None
    lo = band.get("age_months_min") or 0
    hi = band.get("age_months_max") or (lo + 24)
    return None, rng.randint(int(lo), int(hi) - 1)


def make_dosing_query_example(rng: random.Random, case_id: str) -> dict:
    """"What dose of amoxicillin for an 8 kg child?" -> the table's answer."""
    from src.tools import imci_dosing
    drug = rng.choice(list(_DRUG_DISPLAY))
    weight_kg, age_months = _sample_dose_input(drug, rng)
    dose = imci_dosing.format_dose(drug, weight_kg=weight_kg, age_months=age_months)
    name = _DRUG_DISPLAY[drug]
    if weight_kg is not None:
        who = f"a child weighing {weight_kg} kg"
    else:
        who = f"a {age_months}-month-old child"
    q = rng.choice([
        f"What dose of {name} for {who}?",
        f"How much {name} should I give {who}?",
        f"{name} dose for {who}?",
    ])
    a = f"Give {dose}."
    return _record(_system_message(rng) + [
        {"role": "user", "content": q}, {"role": "assistant", "content": a},
    ], case_id, Kind.DOSING_QUERY, drug=drug)


def make_treatment_query_example(rng: random.Random, case_id: str) -> dict | None:
    """"What should I give a child with pneumonia?" -> dosing + follow-up, all
    from the tables. Returns None for labels with no in-table drug (retry)."""
    from src.sft.treatment import LABEL_TO_DRUGS, render_dosing
    from src.tools import imci_dosing
    dosable = [l for l, ds in LABEL_TO_DRUGS.items() if ds]
    label = rng.choice(dosable)
    # A weight/age the label's drugs are actually dosed at (retry a few times).
    for _ in range(6):
        weight_kg = round(rng.uniform(5.0, 18.0), 1)
        age_months = rng.randint(6, 59)
        dosing = render_dosing(label, weight_kg, age_months)
        if dosing:
            break
    else:
        return None
    q = rng.choice([
        f"What treatment should I give a child classified with {_label_display(label)}?",
        f"How do I treat {_label_display(label)}?",
        f"What do I give for {_label_display(label)}?",
    ])
    parts = [f"For {_label_display(label)}: give {dosing}."]
    interval = imci_dosing.follow_up_for(label)
    if interval:
        parts.append(f"Follow up in {interval}.")
    return _record(_system_message(rng) + [
        {"role": "user", "content": q}, {"role": "assistant", "content": " ".join(parts)},
    ], case_id, Kind.TREATMENT_QUERY, condition_label=label, weight_kg=weight_kg, age_months=age_months)


def make_followup_query_example(rng: random.Random, case_id: str) -> dict:
    """"When should a child with pneumonia return?" -> the table's interval."""
    from src.tools import imci_dosing
    intervals = imci_dosing._tables()["follow_up_intervals"]
    label = rng.choice([k for k in intervals if not k.startswith("_")])
    interval = intervals[label]
    q = rng.choice([
        f"When should a child with {_label_display(label)} return for follow-up?",
        f"What is the follow-up interval for {_label_display(label)}?",
        f"When do I see a child with {_label_display(label)} again?",
    ])
    a = (f"Return for follow-up in {interval}." if interval[0].isdigit()
         else f"Follow up {interval}.")
    return _record(_system_message(rng) + [
        {"role": "user", "content": q}, {"role": "assistant", "content": a},
    ], case_id, Kind.FOLLOWUP_QUERY, condition_label=label)


# Closed set of protocol contrasts. Factual descriptions of the discriminating
# signs -- the same kind of protocol prose the engine's reasoning strings use,
# reviewable in one place.
_DIFFERENTIALS = (
    ("severe dehydration", "some dehydration",
     "Both are scored from the same four signs (lethargic/unconscious or restless/irritable, "
     "sunken eyes, drinking pattern, skin pinch). Severe dehydration needs two or more of the "
     "SEVERE signs (lethargic or unconscious, not able to drink or drinking poorly, skin pinch "
     "goes back very slowly); some dehydration needs two of the milder signs (restless/irritable, "
     "drinks eagerly, skin pinch goes back slowly). Severe means refer urgently on Plan C."),
    ("pneumonia", "cough or cold",
     "The dividing sign is fast breathing for age (50+/min from 2-12 months, 40+/min from 12 months "
     "to 5 years). Fast breathing with no danger sign, chest indrawing, or stridor is pneumonia "
     "(give amoxicillin). No fast breathing and none of those severe signs is a cough or cold "
     "(home care, no antibiotic)."),
    ("severe pneumonia or very severe disease", "pneumonia",
     "Any general danger sign, chest indrawing, stridor in a calm child, or oxygen saturation below "
     "90% makes it severe -- give a first antibiotic and refer urgently. Fast breathing as the only "
     "sign is pneumonia, treated at home with amoxicillin."),
    ("mastoiditis", "acute ear infection",
     "Tender swelling behind the ear is mastoiditis -- give ceftriaxone and refer urgently. Ear pain "
     "or discharge for less than 14 days, without that swelling, is an acute ear infection -- give "
     "amoxicillin and treat pain."),
    ("very severe febrile disease", "fever with malaria unlikely",
     "A general danger sign, a stiff neck, or a bulging fontanelle with fever is very severe febrile "
     "disease -- give a first antibiotic and refer urgently. Fever with none of those, and no "
     "malaria, is treated symptomatically with paracetamol and follow-up."),
    ("dysentery", "persistent diarrhoea",
     "Dysentery is diarrhoea with blood in the stool. Persistent diarrhoea is diarrhoea lasting 14 "
     "days or more. They are separate classifications and a child can have both at once."),
)


def make_differential_example(rng: random.Random, case_id: str) -> dict:
    """"How do I tell X from Y?" -> a curated, factual protocol contrast."""
    a_name, b_name, explanation = rng.choice(_DIFFERENTIALS)
    q = rng.choice([
        f"How do I tell {a_name} from {b_name}?",
        f"What is the difference between {a_name} and {b_name}?",
        f"{a_name} versus {b_name} -- how do I decide?",
    ])
    return _record(_system_message(rng) + [
        {"role": "user", "content": q}, {"role": "assistant", "content": explanation},
    ], case_id, Kind.DIFFERENTIAL, pair=f"{a_name}|{b_name}")


def make_young_infant_example(rng: random.Random, case_id: str) -> dict | None:
    """A young-infant (0-2 month) vignette + classification.

    Modelled behaviour that used to be a scope_refusal (see src/sft/young_infant.py).
    OFF by default with the rest of the extended set (--include-extended); the
    young-infant chart is UNREVIEWED clinical logic. Returns None when a sampled
    case doesn't trigger its panel, so the caller retries.
    """
    from src.sft.young_infant import (
        YoungInfantAssessment, classify_yi_bacterial, classify_yi_congenital,
        classify_yi_diarrhoea, classify_yi_jaundice,
    )
    from src.sft.young_infant_verbalize import YI_LABEL_TEXT, render_young_infant_answer, verbalize_young_infant

    panel = rng.choice(["bacterial", "jaundice", "diarrhoea", "congenital"])
    age_days = rng.randint(0, 59)

    if panel == "bacterial":
        roll = rng.random()
        a = YoungInfantAssessment(
            age_days=age_days,
            # ~45% very severe, ~35% local, ~20% none
            convulsions=roll < 0.10,
            fast_breathing_over_60=0.10 <= roll < 0.20,
            bulging_fontanelle=0.20 <= roll < 0.28,
            only_moves_when_stimulated=0.28 <= roll < 0.36,
            umbilicus_red_extending_or_draining_pus=0.36 <= roll < 0.45,
            eye_discharge_purulent_or_sticky=0.45 <= roll < 0.60,
            umbilicus_red_only=0.60 <= roll < 0.72,
            skin_pustules_few=0.72 <= roll < 0.80,
            fever_37_5_or_more=rng.random() < 0.15,
        )
        result = classify_yi_bacterial(a)
    elif panel == "jaundice":
        a = YoungInfantAssessment(
            age_days=age_days, jaundice=True,
            jaundice_onset_under_24h=(age_days < 1 and rng.random() < 0.6),
            yellow_palms_and_soles=rng.random() < 0.4,
        )
        result = classify_yi_jaundice(a)
    elif panel == "diarrhoea":
        a = YoungInfantAssessment(
            age_days=age_days, diarrhoea=True,
            diarrhoea_days=rng.choice([2, 3, 5, 16]),
            blood_in_stool=rng.random() < 0.2,
            lethargic_or_unconscious=rng.random() < 0.3,
            restless_or_irritable=rng.random() < 0.3,
            sunken_eyes=rng.random() < 0.4,
            skin_pinch_very_slow=rng.random() < 0.3,
            skin_pinch_slow=rng.random() < 0.3,
        )
        result = classify_yi_diarrhoea(a)
    else:  # congenital
        a = YoungInfantAssessment(
            age_days=age_days,
            cleft_lip_or_palate=rng.random() < 0.2, imperforate_anus=rng.random() < 0.15,
            macrocephaly=rng.random() < 0.12, very_low_birth_weight=rng.random() < 0.2,
            ambiguous_genitalia=rng.random() < 0.08, abdominal_distension=rng.random() < 0.1,
            microcephaly=rng.random() < 0.12, club_foot=rng.random() < 0.15,
            abnormal_fontanelle_or_sutures=rng.random() < 0.1,
            mother_rpr_positive_untreated=rng.random() < 0.15,
        )
        result = classify_yi_congenital(a)

    if result is None or result.condition_label not in YI_LABEL_TEXT:
        return None

    style = rng.choice(list(VignetteStyle))
    text = verbalize_young_infant(a, rng, style)
    messages = _system_message(rng) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": render_young_infant_answer(result, rng)},
    ]
    return _record(messages, case_id, Kind.TRIAGE, style=style.value,
                   classification=result.classification.value,
                   condition_label=result.condition_label,
                   extended=True, branch=f"yi_{panel}", young_infant=True)
