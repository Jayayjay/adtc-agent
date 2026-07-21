"""
Vignette + answer rendering for the young-infant (0-2 month) chart.

Separate from extended_verbalize because the young-infant signs and the age
rendering (days/weeks, not months) are different. Same rigid CLASSIFICATION:
header as every other answer, so the scorer parses it identically.
"""

from __future__ import annotations

import random

from src.sft.answer import COLOUR_PHRASE, DISCLAIMERS
from src.sft.verbalize import (
    Register,
    STYLE_REGISTER,
    VignetteStyle,
    _cap,
    _join_opener,
    _pronouns,
)
from src.sft.young_infant import YoungInfantAssessment
from src.tools.imci_protocol import TriageResult

YI_LABEL_TEXT = {
    "yi_very_severe_disease": "Very severe disease (young infant)",
    "yi_local_bacterial_infection": "Local bacterial infection",
    "yi_no_bacterial_infection": "No bacterial infection",
    "yi_severe_jaundice": "Severe jaundice",
    "yi_jaundice": "Jaundice",
    "yi_severe_dehydration": "Severe dehydration (young infant)",
    "yi_some_dehydration": "Some dehydration (young infant)",
    "yi_no_dehydration": "Diarrhoea with no dehydration (young infant)",
    "yi_severe_persistent_diarrhoea": "Severe persistent diarrhoea (young infant)",
    "yi_dysentery": "Dysentery (young infant)",
    "yi_congenital_priority": "Congenital priority sign",
    "yi_congenital_abnormal_signs": "Congenital abnormal sign",
    "yi_possible_congenital_syphilis": "Possible congenital syphilis",
}


def render_age_days(age_days: int, rng: random.Random, register: Register) -> str:
    if age_days <= 1:
        return rng.choice(["a newborn", "just born", "1 day old"]) if register is Register.NARRATIVE else "1 day old"
    if age_days < 7:
        return f"{age_days} days old"
    weeks = age_days // 7
    if register is Register.CLINICAL:
        return f"{weeks}/52"
    if register is Register.SMS:
        return f"{weeks}wk old"
    return f"{weeks} weeks old"


# Positive-finding phrases per field. Only true signs are rendered.
_YI_FIELD_PHRASES = {
    "convulsions": {Register.NARRATIVE: ("{P} had convulsions",), Register.CLINICAL: ("convulsions +",), Register.SMS: ("convulsions",)},
    "apnoea": {Register.NARRATIVE: ("{P} stops breathing at times",), Register.CLINICAL: ("apnoea +",), Register.SMS: ("apnoea",)},
    "breathing_under_30": {Register.NARRATIVE: ("{P} is breathing slowly, under 30 a minute",), Register.CLINICAL: ("RR <30",), Register.SMS: ("breathing <30",)},
    "fast_breathing_over_60": {Register.NARRATIVE: ("{P} is breathing over 60 a minute",), Register.CLINICAL: ("RR >60",), Register.SMS: ("breathing >60",)},
    "severe_chest_indrawing": {Register.NARRATIVE: ("there is severe chest indrawing",), Register.CLINICAL: ("severe chest indrawing +",), Register.SMS: ("severe indrawing",)},
    "grunting": {Register.NARRATIVE: ("{P} is grunting",), Register.CLINICAL: ("grunting +",), Register.SMS: ("grunting",)},
    "bulging_fontanelle": {Register.NARRATIVE: ("{Pp} fontanelle is bulging",), Register.CLINICAL: ("bulging fontanelle +",), Register.SMS: ("bulging fontanelle",)},
    "fever_37_5_or_more": {Register.NARRATIVE: ("{P} feels hot, temperature 37.5 or more",), Register.CLINICAL: ("temp >=37.5",), Register.SMS: ("fever",)},
    "hypothermia_under_35_5": {Register.NARRATIVE: ("{P} feels cold, temperature under 35.5",), Register.CLINICAL: ("temp <35.5",), Register.SMS: ("cold, low temp",)},
    "only_moves_when_stimulated": {Register.NARRATIVE: ("{P} only moves when stimulated",), Register.CLINICAL: ("moves only on stimulation +",), Register.SMS: ("only moves when stimulated",)},
    "eye_pus_abundant_or_swollen_eyelids": {Register.NARRATIVE: ("there is abundant pus and swollen eyelids",), Register.CLINICAL: ("abundant eye pus / swollen lids +",), Register.SMS: ("lots of eye pus",)},
    "umbilicus_red_extending_or_draining_pus": {Register.NARRATIVE: ("the umbilicus is red spreading to the skin and draining pus",), Register.CLINICAL: ("umbilical redness to skin / pus +",), Register.SMS: ("umbilicus red spreading, pus",)},
    "skin_pustules_many_or_severe": {Register.NARRATIVE: ("there are many skin pustules",), Register.CLINICAL: ("many/severe skin pustules +",), Register.SMS: ("many skin pustules",)},
    "eye_discharge_purulent_or_sticky": {Register.NARRATIVE: ("there is a sticky discharge from {Pp} eye",), Register.CLINICAL: ("sticky/purulent eye discharge +",), Register.SMS: ("sticky eye discharge",)},
    "umbilicus_red_only": {Register.NARRATIVE: ("the umbilicus is red",), Register.CLINICAL: ("red umbilicus +",), Register.SMS: ("red umbilicus",)},
    "skin_pustules_few": {Register.NARRATIVE: ("there are a few skin pustules",), Register.CLINICAL: ("few skin pustules +",), Register.SMS: ("few skin pustules",)},
    "jaundice": {Register.NARRATIVE: ("{P} looks yellow (jaundiced)",), Register.CLINICAL: ("jaundice +",), Register.SMS: ("jaundiced",)},
    "yellow_palms_and_soles": {Register.NARRATIVE: ("{Pp} palms and soles are yellow",), Register.CLINICAL: ("yellow palms and soles +",), Register.SMS: ("yellow palms n soles",)},
    "diarrhoea": {Register.NARRATIVE: ("{P} has watery diarrhoea",), Register.CLINICAL: ("diarrhoea +",), Register.SMS: ("diarrhoea",)},
    "blood_in_stool": {Register.NARRATIVE: ("there is blood in the stool",), Register.CLINICAL: ("blood in stool +",), Register.SMS: ("blood in stool",)},
    "lethargic_or_unconscious": {Register.NARRATIVE: ("{P} is lethargic",), Register.CLINICAL: ("lethargic/unconscious +",), Register.SMS: ("lethargic",)},
    "restless_or_irritable": {Register.NARRATIVE: ("{P} is restless and irritable",), Register.CLINICAL: ("restless/irritable +",), Register.SMS: ("restless irritable",)},
    "sunken_eyes": {Register.NARRATIVE: ("{Pp} eyes are sunken",), Register.CLINICAL: ("sunken eyes +",), Register.SMS: ("sunken eyes",)},
    "skin_pinch_very_slow": {Register.NARRATIVE: ("the skin pinch goes back very slowly",), Register.CLINICAL: ("skin pinch >2s",), Register.SMS: ("skin pinch very slow",)},
    "skin_pinch_slow": {Register.NARRATIVE: ("the skin pinch goes back slowly",), Register.CLINICAL: ("skin pinch slow",), Register.SMS: ("skin pinch slow",)},
    "cleft_lip_or_palate": {Register.NARRATIVE: ("{P} has a cleft lip and palate",), Register.CLINICAL: ("cleft lip/palate +",), Register.SMS: ("cleft lip/palate",)},
    "imperforate_anus": {Register.NARRATIVE: ("{P} has an imperforate anus",), Register.CLINICAL: ("imperforate anus +",), Register.SMS: ("imperforate anus",)},
    "nose_not_patent": {Register.NARRATIVE: ("{Pp} nose is not patent",), Register.CLINICAL: ("nose not patent +",), Register.SMS: ("nose not patent",)},
    "macrocephaly": {Register.NARRATIVE: ("{Pp} head is very large",), Register.CLINICAL: ("macrocephaly +",), Register.SMS: ("large head",)},
    "ambiguous_genitalia": {Register.NARRATIVE: ("{P} has ambiguous genitalia",), Register.CLINICAL: ("ambiguous genitalia +",), Register.SMS: ("ambiguous genitalia",)},
    "abdominal_distension": {Register.NARRATIVE: ("{Pp} abdomen is distended",), Register.CLINICAL: ("abdominal distension +",), Register.SMS: ("abdo distension",)},
    "very_low_birth_weight": {Register.NARRATIVE: ("{P} weighs 2 kg or less",), Register.CLINICAL: ("VLBW <=2kg +",), Register.SMS: ("very low birth weight",)},
    "microcephaly": {Register.NARRATIVE: ("{Pp} head is very small",), Register.CLINICAL: ("microcephaly +",), Register.SMS: ("small head",)},
    "abnormal_fontanelle_or_sutures": {Register.NARRATIVE: ("{Pp} fontanelle and sutures look abnormal",), Register.CLINICAL: ("abnormal fontanelle/sutures +",), Register.SMS: ("abnormal fontanelle",)},
    "club_foot": {Register.NARRATIVE: ("{P} has a club foot",), Register.CLINICAL: ("club foot +",), Register.SMS: ("club foot",)},
    "mother_rpr_positive_untreated": {Register.NARRATIVE: ("the mother's RPR was positive and untreated",), Register.CLINICAL: ("maternal RPR+ untreated",), Register.SMS: ("mother rpr+ untreated",)},
}


def _fill(text: str, subj: str, obj: str, poss: str) -> str:
    return (text.replace("{Pp}", poss).replace("{Po}", obj)
            .replace("{P}", subj).replace("{C}", "the baby"))


def verbalize_young_infant(a: YoungInfantAssessment, rng: random.Random, style: VignetteStyle) -> str:
    register = STYLE_REGISTER[style]
    subj, obj, poss = _pronouns(rng)
    frags: list[str] = []
    for field_name, bank in _YI_FIELD_PHRASES.items():
        if getattr(a, field_name, False):
            frags.append(_fill(rng.choice(bank[register]), subj, obj, poss))
    if a.diarrhoea and a.diarrhoea_days is not None:
        frags.append(f"diarrhoea x{a.diarrhoea_days}d" if register is not Register.NARRATIVE
                     else f"the diarrhoea has lasted {a.diarrhoea_days} days")
    rng.shuffle(frags)
    age = render_age_days(a.age_days, rng, register)

    if style is VignetteStyle.STRUCTURED_LIST:
        return "\n".join([f"Young infant: {age}"] + [f"- {f}" for f in frags])
    if style is VignetteStyle.CHW_NOTES:
        sex = "F" if subj == "she" else "M"
        return f"Young infant {age} {sex}, " + ", ".join(frags) + "."
    if style is VignetteStyle.SMS_REFERRAL:
        return f"young infant {age}, {', '.join(frags)}. classify?".lower()
    if style is VignetteStyle.QUESTION_FORM:
        lead = rng.choice(["what's the classification?", "how do i classify this baby?", "is this urgent?"])
        return f"{lead} a young infant, {age}, {', '.join(frags)}"
    if style is VignetteStyle.DIALOGUE_TRANSCRIPT:
        lines = ["HW: How old is the baby?", f"Caretaker: {_cap(age)}."]
        for f in frags:
            lines += ["HW: What else?", f"Caretaker: {_cap(f)}."]
        return "\n".join(lines)
    opener = rng.choice(["", "Doctor,", "Please help.", "Good morning."])
    closer = rng.choice(["What is the classification?", "Is this serious?", "What should I do?"])
    sentences = [f"The baby is {age}."] + [f"{_cap(f)}." for f in frags]
    return _join_opener(opener, sentences, closer)


def render_young_infant_answer(result: TriageResult, rng: random.Random) -> str:
    """Same rigid header + body as the other answers. No DOSING line: young-infant
    treatment is IM antibiotics / referral, which are not in the reviewed dosing
    tables, so no dose is emitted."""
    label = YI_LABEL_TEXT[result.condition_label]
    header = (f"CLASSIFICATION: {result.classification.value.upper()} — {label} "
              f"({COLOUR_PHRASE[result.classification]})")
    lines = [header, "", f"WHY: {' '.join(result.reasoning)}",
             f"ACTION: {result.recommended_action}", "", rng.choice(DISCLAIMERS)]
    return "\n".join(lines)
