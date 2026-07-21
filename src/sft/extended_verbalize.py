"""
Vignettes + answers for the extended-protocol branches (malaria, measles,
anaemia, malnutrition), reusing the core verbalizer's machinery.

Kept separate from src/sft/verbalize.py because ExtendedAssessment is a
different dataclass with fields the core verbalizer has never heard of (MUAC,
malaria test result, palmar pallor). Rather than fork the whole style engine,
this composes a compact per-field phrase bank into the SAME 7 styles via the
core module's helpers.

Answers are rendered from the TriageResult the extended classifiers return.
Their reasoning strings were written to be human-readable from the start (unlike
imci_protocol.assess(), whose raw reasoning leaks list reprs and file paths), so
they need no humanize_reasoning() equivalent -- but the no-leak test asserts
that rather than trusting it.
"""

from __future__ import annotations

import random

from src.sft.answer import COLOUR_PHRASE, DISCLAIMERS
from src.sft.extended_protocol import (
    ExtendedAssessment,
    MALARIA_RISK_HIGH,
    MALARIA_RISK_LOW,
    TEST_NEGATIVE,
    TEST_POSITIVE,
)
from src.sft.verbalize import (
    NAMES,
    Register,
    STYLE_REGISTER,
    VignetteStyle,
    _cap,
    _join_opener,
    _pronouns,
    render_age,
)
from src.tools.imci_protocol import Classification, TriageResult

# Display text for the labels the extended classifiers produce. The rigid
# CLASSIFICATION: header is shared with the core answer format so the scorer
# parses both with one regex.
EXTENDED_LABEL_TEXT = {
    "very_severe_febrile_disease": "Very severe febrile disease",
    "malaria": "Malaria",
    "fever_no_malaria": "Fever, malaria unlikely",
    "fever_malaria_test_required": "Fever, malaria test needed",
    "severe_complicated_measles": "Severe complicated measles",
    "measles_with_eye_or_mouth_complications": "Measles with eye or mouth complications",
    "measles": "Measles",
    "severe_anaemia": "Severe anaemia",
    "anaemia": "Anaemia",
    "no_anaemia": "No anaemia",
    "complicated_severe_acute_malnutrition": "Complicated severe acute malnutrition",
    "uncomplicated_severe_acute_malnutrition": "Uncomplicated severe acute malnutrition",
    "moderate_acute_malnutrition": "Moderate acute malnutrition",
    "no_acute_malnutrition": "No acute malnutrition",
    "wheeze": "Wheeze",
    "wheeze_with_danger_sign": "Wheeze with a general danger sign",
    "severe_persistent_diarrhoea": "Severe persistent diarrhoea",
    "persistent_diarrhoea": "Persistent diarrhoea",
    "severe_dysentery": "Severe dysentery",
    "dysentery": "Dysentery",
    "streptococcal_sore_throat": "Streptococcal sore throat",
    "sore_throat_non_streptococcal": "Sore throat, not streptococcal",
    "growth_problem": "Growth problem",
    "confirmed_hiv_infection": "Confirmed HIV infection",
    "hiv_exposed": "HIV exposed",
    "suspected_symptomatic_hiv": "Suspected symptomatic HIV",
    "possible_hiv_infection": "Possible HIV infection",
    "hiv_infection_unlikely": "HIV infection unlikely",
}

# Only POSITIVE findings are rendered, following the core verbalizer's rule:
# never omit a true sign; absent == negative to the classifiers.
_FIELD_PHRASES: dict[str, dict[Register, tuple[str, ...]]] = {
    "fever": {
        Register.NARRATIVE: ("{P} has a fever", "{P} is hot to touch", "{P} has been feverish"),
        Register.CLINICAL: ("fever +", "febrile", "c/o fever"),
        Register.SMS: ("has fever", "fever +", "hot body"),
    },
    "stiff_neck": {
        Register.NARRATIVE: ("{Pp} neck is stiff", "{P} cannot bend {Pp} neck forward"),
        Register.CLINICAL: ("stiff neck +", "neck stiffness present"),
        Register.SMS: ("stiff neck", "neck stiff"),
    },
    "generalised_rash": {
        Register.NARRATIVE: ("{P} has a rash all over {Pp} body", "there is a generalised rash",
                             "{P} has a widespread rash"),
        Register.CLINICAL: ("generalised rash +", "generalised maculopapular rash"),
        Register.SMS: ("rash all over", "generalised rash"),
    },
    "cough_or_runny_nose_or_red_eyes": {
        Register.NARRATIVE: ("{P} has red eyes and a runny nose", "{P} has a cough and runny nose",
                             "{Pp} eyes are red"),
        Register.CLINICAL: ("cough/coryza/red eyes +", "coryza and conjunctivitis"),
        Register.SMS: ("red eyes runny nose", "cough n runny nose"),
    },
    "clouding_of_cornea": {
        Register.NARRATIVE: ("there is clouding of {Pp} cornea", "{Pp} cornea looks cloudy"),
        Register.CLINICAL: ("corneal clouding +", "clouding of cornea present"),
        Register.SMS: ("cornea cloudy", "corneal clouding"),
    },
    "deep_or_extensive_mouth_ulcers": {
        Register.NARRATIVE: ("{P} has deep, extensive ulcers in {Pp} mouth",
                             "there are deep mouth ulcers"),
        Register.CLINICAL: ("deep/extensive mouth ulcers +",),
        Register.SMS: ("deep mouth ulcers", "bad mouth ulcers"),
    },
    "mouth_ulcers": {
        Register.NARRATIVE: ("{P} has some mouth ulcers", "there are mouth ulcers"),
        Register.CLINICAL: ("mouth ulcers +",),
        Register.SMS: ("mouth ulcers",),
    },
    "pus_draining_from_eye": {
        Register.NARRATIVE: ("there is pus draining from {Pp} eye", "{Pp} eye is discharging pus"),
        Register.CLINICAL: ("pus draining from eye +", "purulent eye discharge"),
        Register.SMS: ("pus from eye", "eye discharging pus"),
    },
    "severe_palmar_pallor": {
        Register.NARRATIVE: ("{Pp} palms are very pale", "there is severe palmar pallor"),
        Register.CLINICAL: ("severe palmar pallor +",),
        Register.SMS: ("palms very pale", "severe pallor"),
    },
    "some_palmar_pallor": {
        Register.NARRATIVE: ("{Pp} palms are a bit pale", "there is some palmar pallor"),
        Register.CLINICAL: ("some palmar pallor +",),
        Register.SMS: ("palms pale", "some pallor"),
    },
    "oedema_of_both_feet": {
        Register.NARRATIVE: ("both {Pp} feet are swollen", "there is swelling of both feet",
                             "{Pp} feet are puffy and swollen"),
        Register.CLINICAL: ("bilateral pedal oedema +", "oedema of both feet +"),
        Register.SMS: ("both feet swollen", "swelling both feet"),
    },
    "travel_to_malaria_area": {
        Register.NARRATIVE: ("{P} travelled to a malaria area recently",
                             "the family visited a malaria area last month"),
        Register.CLINICAL: ("recent travel to malaria area +",),
        Register.SMS: ("travelled to malaria area", "recent malaria area travel"),
    },
    "wheeze": {
        Register.NARRATIVE: ("{P} is wheezing", "there is a wheeze when {P} breathes"),
        Register.CLINICAL: ("wheeze +", "wheezing on auscultation"),
        Register.SMS: ("wheezing", "wheeze +"),
    },
    "diarrhoea": {
        Register.NARRATIVE: ("{P} has diarrhoea", "{P} has loose stools"),
        Register.CLINICAL: ("diarrhoea +", "loose stools"),
        Register.SMS: ("has diarrhoea", "diarrhoea +"),
    },
    "blood_in_stool": {
        Register.NARRATIVE: ("there is blood in {Pp} stool", "{Pp} stool has blood in it"),
        Register.CLINICAL: ("blood in stool +", "bloody stool"),
        Register.SMS: ("blood in stool", "bloody stool"),
    },
    "dehydration_present": {
        Register.NARRATIVE: ("{P} shows signs of dehydration",
                             "{P} has sunken eyes and a slow skin pinch"),
        Register.CLINICAL: ("dehydration present +", "signs of dehydration"),
        Register.SMS: ("dehydrated", "signs of dehydration"),
    },
    "losing_weight": {
        Register.NARRATIVE: ("{P} has been losing weight", "{Pp} weight is dropping"),
        Register.CLINICAL: ("losing weight +", "weight loss on the curve"),
        Register.SMS: ("losing weight", "wt dropping"),
    },
    "low_weight_for_age": {
        Register.NARRATIVE: ("{P} is underweight for {Pp} age", "{Pp} weight-for-age is low"),
        Register.CLINICAL: ("low weight-for-age +", "underweight"),
        Register.SMS: ("low weight for age", "underweight"),
    },
    "sore_throat": {
        Register.NARRATIVE: ("{P} has a sore throat", "{Pp} throat is sore"),
        Register.CLINICAL: ("sore throat +", "c/o sore throat"),
        Register.SMS: ("sore throat", "throat sore"),
    },
    "enlarged_tonsils": {
        Register.NARRATIVE: ("{Pp} tonsils are enlarged", "{Pp} tonsils look swollen"),
        Register.CLINICAL: ("enlarged tonsils +",),
        Register.SMS: ("big tonsils", "enlarged tonsils"),
    },
    "tonsil_exudate": {
        Register.NARRATIVE: ("there is white pus on {Pp} tonsils",
                             "{Pp} tonsils have a yellow-white coating"),
        Register.CLINICAL: ("tonsillar exudate +", "white/yellow exudate on tonsils"),
        Register.SMS: ("pus on tonsils", "white on tonsils"),
    },
    "scarlatiniform_rash": {
        Register.NARRATIVE: ("{P} has a fine sandpaper-like rash",
                             "there is a scarlatiniform rash"),
        Register.CLINICAL: ("scarlatiniform rash +",),
        Register.SMS: ("sandpaper rash", "scarlatiniform rash"),
    },
    "runny_nose": {
        Register.NARRATIVE: ("{P} has a runny nose",),
        Register.CLINICAL: ("runny nose +", "coryza"),
        Register.SMS: ("runny nose",),
    },
    "cough": {
        Register.NARRATIVE: ("{P} has a cough",),
        Register.CLINICAL: ("cough +",),
        Register.SMS: ("has cough", "cough +"),
    },
    "mother_hiv_positive": {
        Register.NARRATIVE: ("the mother is HIV-positive", "{Pp} mother has HIV"),
        Register.CLINICAL: ("mother HIV+ ", "maternal HIV positive"),
        Register.SMS: ("mother hiv+", "mum is hiv positive"),
    },
    "infant_on_arv_prophylaxis": {
        Register.NARRATIVE: ("{P} is on ARV prophylaxis", "{P} is taking ARV prophylaxis"),
        Register.CLINICAL: ("on ARV prophylaxis +",),
        Register.SMS: ("on arv prophylaxis", "taking arvs"),
    },
    "breastfeeding_at_or_near_test": {
        Register.NARRATIVE: ("{P} was breastfeeding around the time of the test",
                             "{P} is still breastfeeding"),
        Register.CLINICAL: ("breastfeeding at/near test +",),
        Register.SMS: ("still breastfeeding", "bf at test time"),
    },
    "breastfeeding_stopped_ge_6wk": {
        Register.NARRATIVE: ("breastfeeding stopped more than 6 weeks before the test",),
        Register.CLINICAL: ("BF stopped >=6wk before test +",),
        Register.SMS: ("stopped bf >6wk before test",),
    },
    "child_on_art": {
        Register.NARRATIVE: ("{P} is already on ART",),
        Register.CLINICAL: ("on ART +",),
        Register.SMS: ("on art", "taking art"),
    },
    "hiv_pneumonia_now": {
        Register.NARRATIVE: ("{P} has pneumonia now",),
        Register.CLINICAL: ("pneumonia now +",),
        Register.SMS: ("has pneumonia now",),
    },
    "hiv_persistent_diarrhoea": {
        Register.NARRATIVE: ("{P} has had persistent diarrhoea",),
        Register.CLINICAL: ("persistent diarrhoea +",),
        Register.SMS: ("persistent diarrhoea",),
    },
    "hiv_ever_ear_discharge": {
        Register.NARRATIVE: ("{P} has had ear discharge before",),
        Register.CLINICAL: ("h/o ear discharge +",),
        Register.SMS: ("had ear discharge before",),
    },
    "hiv_low_weight": {
        Register.NARRATIVE: ("{P} has a low weight",),
        Register.CLINICAL: ("low weight +",),
        Register.SMS: ("low weight",),
    },
    "hiv_enlarged_lymph_nodes": {
        Register.NARRATIVE: ("{P} has enlarged lymph nodes in two or more sites",
                             "there are enlarged glands in {Pp} neck and groin"),
        Register.CLINICAL: ("generalised lymphadenopathy +", "enlarged nodes >=2 sites"),
        Register.SMS: ("enlarged lymph nodes", "swollen glands"),
    },
    "hiv_oral_thrush": {
        Register.NARRATIVE: ("{P} has oral thrush",),
        Register.CLINICAL: ("oral thrush +",),
        Register.SMS: ("oral thrush",),
    },
    "hiv_parotid_enlargement": {
        Register.NARRATIVE: ("{Pp} parotid glands are enlarged",),
        Register.CLINICAL: ("parotid enlargement +",),
        Register.SMS: ("parotid swelling",),
    },
}


def _fill(text: str, subj: str, obj: str, poss: str) -> str:
    return (text.replace("{Pp}", poss).replace("{Po}", obj)
            .replace("{P}", subj).replace("{C}", "the child"))


def _numeric_fragments(a: ExtendedAssessment, register: Register) -> list[str]:
    """MUAC, weight-for-height, fever days, malaria risk/test -- the fields that
    carry a value, not just a polarity."""
    frags = []
    if a.muac_mm is not None:
        frags.append(f"MUAC {a.muac_mm}mm" if register is not Register.NARRATIVE
                     else f"the MUAC measures {a.muac_mm} millimetres")
    if a.wfh_z_score is not None:
        frags.append(f"WFH z {a.wfh_z_score}" if register is not Register.NARRATIVE
                     else f"weight-for-height is {a.wfh_z_score} z-scores")
    if a.fever_days is not None and a.fever:
        frags.append(f"fever x{a.fever_days}d" if register is not Register.NARRATIVE
                     else f"the fever has lasted {a.fever_days} days")
    if a.diarrhoea_days is not None and a.diarrhoea:
        frags.append(f"diarrhoea x{a.diarrhoea_days}d" if register is not Register.NARRATIVE
                     else f"the diarrhoea has lasted {a.diarrhoea_days} days")
    if a.appetite_test_passed is False:
        frags.append("appetite test failed" if register is not Register.NARRATIVE
                     else "the child failed the appetite test")
    if a.malaria_risk in (MALARIA_RISK_HIGH, MALARIA_RISK_LOW):
        risk = "high" if a.malaria_risk == MALARIA_RISK_HIGH else "low"
        frags.append(f"{risk} malaria risk area" if register is not Register.NARRATIVE
                     else f"we are in a {risk} malaria risk area")
    if a.malaria_test == TEST_POSITIVE:
        frags.append("RDT positive" if register is not Register.NARRATIVE
                     else "the malaria test came back positive")
    elif a.malaria_test == TEST_NEGATIVE:
        frags.append("RDT negative" if register is not Register.NARRATIVE
                     else "the malaria test came back negative")
    if a.hiv_test == TEST_POSITIVE:
        frags.append("HIV test positive" if register is not Register.NARRATIVE
                     else "the child's HIV test is positive")
    elif a.hiv_test == TEST_NEGATIVE:
        frags.append("HIV test negative" if register is not Register.NARRATIVE
                     else "the child's HIV test is negative")
    return frags


def verbalize_extended(a: ExtendedAssessment, rng: random.Random, style: VignetteStyle) -> str:
    register = STYLE_REGISTER[style]
    subj, obj, poss = _pronouns(rng)
    frags: list[str] = []

    if a.danger_signs_present:
        from src.sft.verbalize import DANGER_SIGN_PHRASES
        for sign in a.danger_signs_present:
            frags.append(_fill(rng.choice(DANGER_SIGN_PHRASES[sign][register]), subj, obj, poss))

    for field, bank in _FIELD_PHRASES.items():
        if getattr(a, field, False):
            frags.append(_fill(rng.choice(bank[register]), subj, obj, poss))

    frags += _numeric_fragments(a, register)
    rng.shuffle(frags)
    age = render_age(a.age_months, rng, register)

    if style is VignetteStyle.STRUCTURED_LIST:
        return "\n".join([f"Age: {age}"] + [f"- {f}" for f in frags])
    if style is VignetteStyle.CHW_NOTES:
        sex = "F" if subj == "she" else "M"
        return f"{age} {sex}, " + ", ".join(frags) + "."
    if style is VignetteStyle.SMS_REFERRAL:
        return f"{age} child, {', '.join(frags)}. classify?".lower()
    if style is VignetteStyle.QUESTION_FORM:
        lead = rng.choice(["what's the classification?", "how do i classify this?", "is this urgent?"])
        return f"{lead} the child, {age}, {', '.join(frags)}"
    if style is VignetteStyle.DIALOGUE_TRANSCRIPT:
        lines = [f"HW: How old is the child?", f"Caretaker: {_cap(age)}."]
        for f in frags:
            lines += ["HW: What else?", f"Caretaker: {_cap(f)}."]
        return "\n".join(lines)
    # narrative / verbose
    opener = rng.choice(["", "Doctor,", "Please help.", "Good morning."])
    closer = rng.choice(["What is the classification?", "Is this serious?", "What should I do?"])
    sentences = [f"The child is {age}."] + [f"{_cap(f)}." for f in frags]
    return _join_opener(opener, sentences, closer)


def render_extended_answer(result: TriageResult, rng: random.Random,
                           weight_kg: float | None = None, age_months: int | None = None) -> str:
    """Same rigid header + varied body as the core answer, but the extended
    classifiers' reasoning is already prose, so it renders directly.

    When weight_kg and age_months are given, a DOSING line is appended with the
    doses this classification calls for (from the reviewed tables via
    src/sft/treatment.py); most extended labels carry no in-table drug and so
    render no DOSING line."""
    label = EXTENDED_LABEL_TEXT[result.condition_label]
    header = (f"CLASSIFICATION: {result.classification.value.upper()} — {label} "
              f"({COLOUR_PHRASE[result.classification]})")
    lines = [header, "", f"WHY: {' '.join(result.reasoning)}",
             f"ACTION: {result.recommended_action}"]
    if weight_kg is not None and age_months is not None:
        from src.sft.treatment import render_dosing
        dosing = render_dosing(result.condition_label, weight_kg, age_months)
        if dosing:
            lines.append(f"DOSING: {dosing}")
    if result.secondary_findings:
        lines.append("ALSO: " + " ".join(result.secondary_findings))
    lines += ["", rng.choice(DISCLAIMERS)]
    return "\n".join(lines)
