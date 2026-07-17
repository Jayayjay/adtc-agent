"""
Turns a pruned ChildAssessment into a natural-language vignette.

This is the part of the pipeline that decides whether the fine-tune survives
contact with the two hidden organizer prompts. Those prompts are written by
someone who has never seen our templates, explicitly to catch overfitting, and
they are 2 of the 4 prompts that decide 50% of the score. A model trained on
"N templates" learns the templates; a model trained on genuinely varied prose
learns the protocol.

So the design is multiplicative, not enumerative: 7 style families x several
independent variation axes (how age is written, how numbers are written, what
order signs appear in, whether a negative is stated or simply omitted, how much
irrelevant framing surrounds it, typos). Each axis draws independently, so the
surface forms compose out to far more than the phrase bank literally contains.

THE INVARIANT (see sampling.required_fields_for): every true sign must appear
in the text; false ones may be stated or omitted freely. assess() treats an
absent field as False, so omitting a negative is label-preserving -- and real
prompts never enumerate every negative. But a vignette that omits a POSITIVE
sign is training the model to invent that sign from nothing, which is why
generate_sft_data.py asserts this rather than trusting it.
"""

from __future__ import annotations

import random
import re
from enum import Enum

from src.sft.sampling import (
    COUGH_FIELDS,
    DIARRHEA_FIELDS,
    EAR_FIELDS,
    FEVER_FIELDS,
    required_fields_for,
)
from src.tools.imci_protocol import ChildAssessment


class VignetteStyle(str, Enum):
    CARETAKER_NARRATIVE = "caretaker_narrative"
    CHW_NOTES = "chw_notes"
    STRUCTURED_LIST = "structured_list"
    SMS_REFERRAL = "sms_referral"
    QUESTION_FORM = "question_form"
    DIALOGUE_TRANSCRIPT = "dialogue_transcript"
    VERBOSE_PARAGRAPH = "verbose_paragraph"


class Register(str, Enum):
    """Phrase banks are keyed by register rather than by style: several styles
    share a voice, and duplicating the bank per style would guarantee they drift
    apart."""

    NARRATIVE = "narrative"   # a caretaker or health worker speaking in sentences
    CLINICAL = "clinical"     # terse professional shorthand
    SMS = "sms"               # lowercase, abbreviated, sent from a phone


STYLE_REGISTER = {
    VignetteStyle.CARETAKER_NARRATIVE: Register.NARRATIVE,
    VignetteStyle.VERBOSE_PARAGRAPH: Register.NARRATIVE,
    VignetteStyle.QUESTION_FORM: Register.NARRATIVE,
    VignetteStyle.DIALOGUE_TRANSCRIPT: Register.NARRATIVE,
    VignetteStyle.CHW_NOTES: Register.CLINICAL,
    VignetteStyle.STRUCTURED_LIST: Register.CLINICAL,
    VignetteStyle.SMS_REFERRAL: Register.SMS,
}

# ---------------------------------------------------------------------------
# Phrase bank: field -> polarity -> register -> surface forms.
#
# "{P}" is substituted with a subject pronoun ("she"/"he"), "{Po}" with an
# object pronoun ("her"/"him"), "{Pp}" with a possessive ("her"/"his"), and
# "{C}" with a child noun ("the child"/"the baby"/a name). Keeping pronouns as
# slots rather than baking them in means gender varies for free across every
# phrase.
#
# The three pronoun slots are NOT interchangeable: "she" and "her" collide in
# the possessive but not in the object case, so "giving {P} water" renders as
# "giving she water" for half of all children and reads fine for the other
# half. Use {Po} after a verb or preposition.
#
# Boolean fields carry "true"/"false" banks. Numeric fields are rendered by
# dedicated functions further down, because they also vary on how the number
# itself is written.
# ---------------------------------------------------------------------------

PHRASES: dict[str, dict[str, dict[Register, tuple[str, ...]]]] = {
    "cough_or_difficulty_breathing": {
        "true": {
            Register.NARRATIVE: (
                "{P} has been coughing",
                "{C} has a cough",
                "{P}'s had a cough",
                "there's a cough that won't settle",
                "{P} is coughing a lot",
                "{P} has a bad cough",
                "{P} is having trouble breathing",
                "{Pp} breathing doesn't seem right",
                "{P} coughs and struggles to breathe",
                "the cough started a few days ago",
            ),
            Register.CLINICAL: (
                "cough present",
                "c/o cough",
                "cough +",
                "cough/difficulty breathing: yes",
                "presents with cough",
                "difficulty breathing +",
                "cough and DB",
                "resp complaint: cough",
            ),
            Register.SMS: (
                "has cough",
                "coughing",
                "cough +",
                "coughing badly",
                "cough n breathing hard",
                "child coughing",
            ),
        },
        "false": {
            Register.NARRATIVE: (
                "no cough",
                "{P} isn't coughing",
                "there's no cough or breathing trouble",
                "{P} has no cough at all",
                "breathing is fine",
                "no cough and no trouble breathing",
            ),
            Register.CLINICAL: (
                "no cough",
                "cough: no",
                "cough -",
                "no resp complaint",
                "denies cough",
                "no cough/DB",
            ),
            Register.SMS: ("no cough", "cough -", "not coughing", "no cough no DB"),
        },
    },
    "chest_indrawing": {
        "true": {
            Register.NARRATIVE: (
                "{Pp} chest pulls in when {P} breathes",
                "I can see {Pp} chest drawing in",
                "the lower chest sucks in with each breath",
                "{Pp} chest is indrawing",
                "the skin below {Pp} ribs pulls in when {P} breathes in",
                "there is chest indrawing",
                "{Pp} chest wall goes in as {P} breathes",
                "you can see the chest pulling inward",
            ),
            Register.CLINICAL: (
                "chest indrawing +",
                "lower chest wall indrawing present",
                "indrawing: yes",
                "LCWI +",
                "chest indrawing present",
                "subcostal indrawing noted",
            ),
            Register.SMS: (
                "chest indrawing",
                "chest pulling in",
                "indrawing +",
                "chest going in",
            ),
        },
        "false": {
            Register.NARRATIVE: (
                "no chest indrawing",
                "{Pp} chest isn't pulling in",
                "I didn't notice any indrawing",
                "the chest looks normal when {P} breathes",
                "no indrawing that I could see",
            ),
            Register.CLINICAL: (
                "no chest indrawing",
                "indrawing: no",
                "chest indrawing -",
                "no LCWI",
                "no indrawing",
            ),
            Register.SMS: ("no indrawing", "indrawing -", "chest ok"),
        },
    },
    "stridor_when_calm": {
        "true": {
            Register.NARRATIVE: (
                "{P} makes a harsh noise breathing in, even when {P}'s calm",
                "there's stridor when {P} is settled",
                "{P} has a rough crowing sound when {P} breathes in",
                "even resting, {Pp} breathing makes a harsh sound",
                "{P} has stridor at rest",
                "there's a noisy sound on breathing in while {P} is quiet",
            ),
            Register.CLINICAL: (
                "stridor when calm +",
                "stridor at rest present",
                "stridor (calm): yes",
                "inspiratory stridor at rest",
                "stridor present when settled",
            ),
            Register.SMS: (
                "stridor when calm",
                "harsh sound breathing in at rest",
                "stridor +",
            ),
        },
        "false": {
            Register.NARRATIVE: (
                "no stridor",
                "no harsh noise when {P} breathes",
                "{P} doesn't make any crowing sound",
                "no stridor when {P} is calm",
            ),
            Register.CLINICAL: (
                "no stridor when calm",
                "stridor: no",
                "stridor -",
                "no stridor at rest",
            ),
            Register.SMS: ("no stridor", "stridor -"),
        },
    },
    "diarrhea": {
        "true": {
            Register.NARRATIVE: (
                "{P} has diarrhoea",
                "{P}'s been passing loose stools",
                "{C} has watery stools",
                "{Pp} stomach has been running",
                "{P} keeps passing watery stool",
                "{P} has loose motions",
                "{P}'s been having runny stools",
            ),
            Register.CLINICAL: (
                "diarrhoea +",
                "c/o diarrhoea",
                "diarrhoea: yes",
                "loose stools present",
                "watery stool +",
                "presents with diarrhoea",
            ),
            Register.SMS: (
                "has diarrhea",
                "diarrhoea +",
                "loose stool",
                "watery stools",
                "stomach running",
            ),
        },
        "false": {
            Register.NARRATIVE: (
                "no diarrhoea",
                "{Pp} stools are normal",
                "{P} hasn't had diarrhoea",
                "no loose stools",
            ),
            Register.CLINICAL: (
                "no diarrhoea",
                "diarrhoea: no",
                "diarrhoea -",
                "stools normal",
            ),
            Register.SMS: ("no diarrhea", "diarrhoea -", "stool ok"),
        },
    },
    "blood_in_stool": {
        "true": {
            Register.NARRATIVE: (
                "there's blood in {Pp} stool",
                "I've seen blood when {P} passes stool",
                "{Pp} stool has blood in it",
                "the stool is bloody",
                "there is blood mixed in the stool",
            ),
            Register.CLINICAL: (
                "blood in stool +",
                "bloody stool present",
                "blood in stool: yes",
                "haematochezia +",
            ),
            Register.SMS: ("blood in stool", "bloody stool", "blood +"),
        },
        "false": {
            Register.NARRATIVE: (
                "no blood in the stool",
                "I haven't seen any blood",
                "no blood that I noticed",
            ),
            Register.CLINICAL: ("no blood in stool", "blood in stool: no", "blood -"),
            Register.SMS: ("no blood", "blood -"),
        },
    },
    "child_lethargic_or_unconscious": {
        "true": {
            Register.NARRATIVE: (
                "{P} is very drowsy and hard to wake",
                "{P}'s not responding properly",
                "{P} just lies there",
                "{P} is lethargic",
                "{P} won't wake up properly",
                "{P} seems unconscious",
                "{P} is floppy and unresponsive",
            ),
            Register.CLINICAL: (
                "lethargic +",
                "lethargic or unconscious: yes",
                "reduced consciousness",
                "lethargic/unconscious +",
                "not alert",
            ),
            Register.SMS: ("lethargic", "not waking", "unconscious", "very drowsy"),
        },
        "false": {
            Register.NARRATIVE: (
                "{P}'s alert",
                "{P} is awake and looking around",
                "{P} responds normally",
                "{P} is not drowsy",
            ),
            Register.CLINICAL: ("alert", "lethargic: no", "lethargic -", "conscious and alert"),
            Register.SMS: ("alert", "awake", "lethargic -"),
        },
    },
    "child_restless_or_irritable": {
        "true": {
            Register.NARRATIVE: (
                "{P}'s restless and irritable",
                "{P} won't settle at all",
                "{P} keeps crying and fussing",
                "{P} is very irritable",
                "{P}'s unsettled and cries whenever I put {Po} down",
            ),
            Register.CLINICAL: (
                "restless/irritable +",
                "restless or irritable: yes",
                "irritable +",
                "unsettled, irritable",
            ),
            Register.SMS: ("restless", "irritable", "restless/irritable +", "wont settle"),
        },
        "false": {
            Register.NARRATIVE: (
                "{P}'s calm",
                "{P} isn't restless",
                "{P} settles fine",
            ),
            Register.CLINICAL: ("not restless", "restless/irritable: no", "calm"),
            Register.SMS: ("calm", "restless -"),
        },
    },
    "sunken_eyes": {
        "true": {
            Register.NARRATIVE: (
                "{Pp} eyes look sunken",
                "{Pp} eyes have gone hollow",
                "{Pp} eyes are sunken in",
                "the eyes look deep-set to me",
                "{Pp} eyes look sunken compared to normal",
            ),
            Register.CLINICAL: (
                "sunken eyes +",
                "sunken eyes: yes",
                "eyes sunken",
                "sunken eyes present",
            ),
            Register.SMS: ("sunken eyes", "eyes sunken", "sunken eyes +"),
        },
        "false": {
            Register.NARRATIVE: (
                "{Pp} eyes look normal",
                "{Pp} eyes aren't sunken",
                "the eyes look fine",
            ),
            Register.CLINICAL: ("no sunken eyes", "sunken eyes: no", "eyes normal"),
            Register.SMS: ("eyes ok", "sunken eyes -"),
        },
    },
    "not_able_to_drink_or_drinking_poorly": {
        "true": {
            Register.NARRATIVE: (
                "{P} won't drink",
                "{P}'s barely drinking",
                "{P} refuses to drink anything",
                "{P} is drinking very poorly",
                "{P} takes almost nothing when I offer it",
            ),
            Register.CLINICAL: (
                "drinking poorly +",
                "not able to drink / drinking poorly: yes",
                "poor oral intake",
                "unable to drink +",
            ),
            Register.SMS: ("not drinking", "drinking poorly", "wont drink"),
        },
        "false": {
            Register.NARRATIVE: (
                "{P}'s drinking normally",
                "{P} takes fluids fine",
                "{P} drinks as usual",
            ),
            Register.CLINICAL: ("drinking normally", "oral intake normal", "drinking: normal"),
            Register.SMS: ("drinking ok", "drinks fine"),
        },
    },
    "drinking_eagerly_thirsty": {
        "true": {
            Register.NARRATIVE: (
                "{P} is drinking eagerly, like {P}'s very thirsty",
                "{P} gulps the water down and wants more",
                "{P} seems really thirsty",
                "{P} drinks eagerly whenever I offer",
                "{P} can't get enough to drink",
            ),
            Register.CLINICAL: (
                "drinks eagerly, thirsty +",
                "drinking eagerly: yes",
                "thirsty, drinks eagerly",
                "polydipsia +",
            ),
            Register.SMS: ("drinking eagerly", "very thirsty", "thirsty +"),
        },
        "false": {
            Register.NARRATIVE: (
                "{P} isn't especially thirsty",
                "{P} drinks normally, not eagerly",
            ),
            Register.CLINICAL: ("not thirsty", "drinks eagerly: no", "no excess thirst"),
            Register.SMS: ("not thirsty", "thirst -"),
        },
    },
    "skin_pinch_goes_back_very_slowly": {
        "true": {
            Register.NARRATIVE: (
                "when I pinch {Pp} skin it takes more than two seconds to go back",
                "the skin pinch goes back very slowly, over two seconds",
                "{Pp} skin stays tented for a good couple of seconds",
                "the skin takes a long time to flatten, more than 2 seconds",
                "skin pinch is very slow to return, well over two seconds",
            ),
            Register.CLINICAL: (
                "skin pinch >2s",
                "skin pinch goes back very slowly (>2s)",
                "skin turgor: very slow return >2s",
                "skin pinch very slow +",
            ),
            Register.SMS: ("skin pinch >2s", "skin pinch very slow", "pinch >2 sec"),
        },
        "false": {
            Register.NARRATIVE: (
                "the skin pinch goes straight back",
                "{Pp} skin springs back immediately",
            ),
            Register.CLINICAL: ("skin pinch immediate", "skin turgor normal", "skin pinch: normal"),
            Register.SMS: ("skin pinch ok", "pinch normal"),
        },
    },
    "skin_pinch_goes_back_slowly": {
        "true": {
            Register.NARRATIVE: (
                "the skin pinch goes back slowly",
                "when I pinch {Pp} skin there's a slight delay before it flattens",
                "{Pp} skin is a bit slow to go back, but not more than two seconds",
                "there's some delay in the skin pinch",
            ),
            Register.CLINICAL: (
                "skin pinch slow (<2s)",
                "skin pinch goes back slowly",
                "skin turgor: slow return",
                "skin pinch slow +",
            ),
            Register.SMS: ("skin pinch slow", "pinch slow", "skin slow to return"),
        },
        "false": {
            Register.NARRATIVE: (
                "the skin pinch goes back straight away",
                "no delay in the skin pinch",
            ),
            Register.CLINICAL: (
                "skin pinch immediate",
                "skin pinch: normal",
                "no delay in skin pinch",
                "skin turgor normal",
            ),
            Register.SMS: ("pinch ok", "skin pinch normal", "no delay in pinch", "pinch fine"),
        },
    },
    "fever": {
        "true": {
            Register.NARRATIVE: (
                "{P} has a fever",
                "{Pp} body is very hot",
                "{P}'s been burning up",
                "{P} feels hot to touch",
                "{P} has been feverish",
                "{Pp} temperature is up",
            ),
            Register.CLINICAL: (
                "fever +",
                "febrile",
                "fever: yes",
                "c/o fever",
                "hot to touch +",
            ),
            Register.SMS: ("has fever", "fever +", "hot body", "burning up"),
        },
        "false": {
            Register.NARRATIVE: (
                "no fever",
                "{P}'s not hot",
                "{P} hasn't had a fever",
                "{Pp} temperature is normal",
            ),
            Register.CLINICAL: ("afebrile", "no fever", "fever: no", "fever -"),
            Register.SMS: ("no fever", "fever -", "not hot"),
        },
    },
    "stiff_neck": {
        "true": {
            Register.NARRATIVE: (
                "{Pp} neck is stiff",
                "{P} can't bend {Pp} neck forward",
                "{Pp} neck seems rigid",
                "{P} cries when I try to bend {Pp} neck",
                "there's neck stiffness",
            ),
            Register.CLINICAL: (
                "stiff neck +",
                "neck stiffness present",
                "stiff neck: yes",
                "nuchal rigidity +",
            ),
            Register.SMS: ("stiff neck", "neck stiff", "stiff neck +"),
        },
        "false": {
            Register.NARRATIVE: (
                "{Pp} neck is soft",
                "no neck stiffness",
                "{P} moves {Pp} neck normally",
            ),
            Register.CLINICAL: ("no stiff neck", "stiff neck: no", "no nuchal rigidity"),
            Register.SMS: ("no stiff neck", "neck ok"),
        },
    },
    "ear_problem": {
        "true": {
            Register.NARRATIVE: (
                "there's something wrong with {Pp} ear",
                "{P} has an ear problem",
                "{P} keeps pulling at {Pp} ear",
                "{Pp} ear has been bothering {Po}",
                "{P}'s been complaining about {Pp} ear",
            ),
            Register.CLINICAL: (
                "ear problem +",
                "c/o ear problem",
                "ear problem: yes",
                "presents with ear complaint",
            ),
            Register.SMS: ("ear problem", "ear issue", "ear +"),
        },
        "false": {
            Register.NARRATIVE: (
                "no ear problem",
                "{Pp} ears are fine",
                "nothing wrong with the ears",
            ),
            Register.CLINICAL: ("no ear problem", "ear problem: no", "ears normal"),
            Register.SMS: ("no ear problem", "ear -"),
        },
    },
    "ear_pain": {
        "true": {
            Register.NARRATIVE: (
                "{Pp} ear hurts",
                "{P} says {Pp} ear is painful",
                "{P} cries and holds {Pp} ear",
                "there's ear pain",
                "{P}'s complaining of pain in {Pp} ear",
            ),
            Register.CLINICAL: (
                "ear pain +",
                "otalgia +",
                "ear pain: yes",
                "painful ear",
            ),
            Register.SMS: ("ear pain", "ear hurts", "ear pain +"),
        },
        "false": {
            Register.NARRATIVE: (
                "no ear pain",
                "{Pp} ear doesn't hurt",
                "{P} hasn't complained of pain",
            ),
            Register.CLINICAL: ("no ear pain", "ear pain: no", "no otalgia"),
            Register.SMS: ("no ear pain", "ear pain -"),
        },
    },
    "tender_swelling_behind_ear": {
        "true": {
            Register.NARRATIVE: (
                "there's a tender swelling behind {Pp} ear",
                "the area behind {Pp} ear is swollen and sore",
                "{P} has a painful lump behind {Pp} ear",
                "behind {Pp} ear is swollen and hurts when touched",
            ),
            Register.CLINICAL: (
                "tender swelling behind ear +",
                "tender mastoid swelling present",
                "tender post-auricular swelling +",
                "swelling behind ear: yes, tender",
            ),
            Register.SMS: (
                "tender swelling behind ear",
                "swelling behind ear",
                "lump behind ear tender",
            ),
        },
        "false": {
            Register.NARRATIVE: (
                "no swelling behind {Pp} ear",
                "nothing swollen behind the ear",
            ),
            Register.CLINICAL: (
                "no tender swelling behind ear",
                "no post-auricular swelling",
                "swelling behind ear: no",
            ),
            Register.SMS: ("no swelling behind ear", "no lump behind ear"),
        },
    },
}

DANGER_SIGN_PHRASES: dict[str, dict[Register, tuple[str, ...]]] = {
    "convulsions": {
        Register.NARRATIVE: (
            "{P} had a fit earlier",
            "{P} had convulsions today",
            "{Pp} body was jerking and shaking earlier",
            "{P} had a seizure this morning",
            "{P} convulsed before we came",
        ),
        Register.CLINICAL: (
            "h/o convulsions",
            "convulsions +",
            "history of convulsions this illness",
            "seizure reported",
        ),
        Register.SMS: ("had convulsions", "had a fit", "convulsions +"),
    },
    "convulsing_now": {
        Register.NARRATIVE: (
            "{P} is convulsing right now",
            "{P}'s fitting as I speak",
            "{P} is having a seizure at this moment",
            "{P}'s jerking and won't stop",
        ),
        Register.CLINICAL: (
            "convulsing now +",
            "actively convulsing",
            "seizing at presentation",
        ),
        Register.SMS: ("convulsing now", "fitting now", "seizing now"),
    },
    "unable_to_drink_or_breastfeed": {
        Register.NARRATIVE: (
            "{P} can't drink or breastfeed at all",
            "{P} won't take the breast",
            "{P} is unable to breastfeed",
            "{P} can't take anything by mouth",
        ),
        Register.CLINICAL: (
            "unable to drink/breastfeed +",
            "not feeding at all",
            "unable to breastfeed",
        ),
        Register.SMS: ("cant drink", "not breastfeeding", "unable to drink"),
    },
    "vomits_everything": {
        Register.NARRATIVE: (
            "{P} vomits everything {P} takes",
            "{P} brings up everything, even water",
            "{P} can't keep anything down",
            "everything {P} swallows comes back up",
        ),
        Register.CLINICAL: (
            "vomits everything +",
            "vomiting everything",
            "unable to retain oral intake",
        ),
        Register.SMS: ("vomits everything", "vomiting all", "cant keep anything down"),
    },
    "lethargic_or_unconscious": {
        Register.NARRATIVE: (
            "{P} is lethargic and hard to rouse",
            "{P} won't wake up properly",
            "{P} is unconscious",
            "{P} just lies there, not responding",
            "{P}'s very drowsy and floppy",
        ),
        Register.CLINICAL: (
            "lethargic/unconscious +",
            "reduced level of consciousness",
            "lethargic, not rousable",
        ),
        Register.SMS: ("lethargic", "unconscious", "not waking up"),
    },
}

# ---------------------------------------------------------------------------
# Variation axes
# ---------------------------------------------------------------------------

NAMES = (
    "Amina", "Chidi", "Ngozi", "Musa", "Fatima", "Emeka", "Zainab", "Tunde",
    "Halima", "Yusuf", "Aisha", "Obi", "Maryam", "Ibrahim", "Blessing", "Sadiq",
)

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _spell_number(n: int) -> str:
    """Spells 0..999.

    Triage cases only ever draw 2-59 months, but the scope_refusal slice
    deliberately generates out-of-band ages (up to 144 months) to teach the
    model to check the age -- so this has to cover past 99 or it raises
    IndexError on exactly the examples meant to make the model safe.
    """
    if n < 0 or n > 999:
        return str(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    hundreds, rest = divmod(n, 100)
    out = f"{_ONES[hundreds]} hundred"
    return f"{out} and {_spell_number(rest)}" if rest else out


def _article_for(n: int) -> str:
    """"an 8-month-old", "an 11-month-old", "an 18-month-old" -- but "a
    19-month-old". The article tracks the spoken form's initial vowel SOUND, not
    the digit and not the first letter.

    "one" is the trap: it begins with the letter o but is pronounced /wʌn/, so
    it takes "a" ("a one-month-old", "a 100-month-old"). Only reachable via the
    scope_refusal slice's out-of-band ages -- triage never draws 1 or >=100.
    """
    spelled = _spell_number(n)
    if spelled.startswith("one"):
        return "a"
    return "an" if spelled.startswith(("a", "e", "i", "o", "u")) else "a"


def render_age(age_months: int, rng: random.Random, register: Register) -> str:
    """Age is the most-repeated fact in every vignette, so it gets its own axis."""
    years, months = divmod(age_months, 12)
    # "1 months old" is reachable only via the scope_refusal slice's out-of-band
    # ages -- triage never draws 1 -- but it appears in every register.
    unit_full = "month" if age_months == 1 else "months"
    forms = []
    if register is Register.CLINICAL:
        forms += [f"{age_months}mo", f"{age_months} {unit_full}", f"age {age_months}mo",
                  f"{age_months}/12"]
        if years >= 1:
            forms.append(f"{years}y{months}m" if months else f"{years}y")
    elif register is Register.SMS:
        forms += [f"{age_months}mo", f"{age_months} mths", f"{age_months} {unit_full}"]
        if years >= 1:
            forms.append(f"{years}yr {months}mo" if months else f"{years}yr")
    else:
        unit = "month" if age_months == 1 else "months"
        forms += [
            f"{age_months} {unit} old",
            # "a 57-month-old", not "57-month-old": the narrative styles use age
            # as a predicate ("Halima is ...") or an appositive ("the baby, ..."),
            # and the bare adjectival form is ungrammatical in both. The article
            # follows how the number is SAID, not its first digit -- "an
            # 18-month-old" but "a 19-month-old".
            f"{_article_for(age_months)} {age_months}-month-old",
            f"about {age_months} {unit} old",
        ]
        if age_months < 20:
            forms.append(f"{_spell_number(age_months)} {unit} old")
        if years >= 1:
            if months:
                forms += [
                    f"{years} year{'s' if years > 1 else ''} and "
                    f"{months} month{'s' if months > 1 else ''} old",
                    f"just over {years}" + (" year old" if years == 1 else " years old"),
                ]
            else:
                forms.append(f"{years} year{'s' if years > 1 else ''} old")
    return rng.choice(forms)


def render_rate(rr: int, rng: random.Random, register: Register) -> str:
    """Respiratory rate: the number that decides pneumonia vs cough/cold, so
    how it's written matters. Rounding is deliberately never applied -- 'around
    50' would be a different clinical claim than '52'."""
    if register is Register.CLINICAL:
        return rng.choice([f"RR {rr}", f"RR {rr}/min", f"resp rate {rr}", f"{rr} breaths/min",
                           f"RR: {rr}"])
    if register is Register.SMS:
        return rng.choice([f"rr {rr}", f"{rr}/min", f"breathing {rr}", f"rr={rr}"])
    return rng.choice([
        f"breathing at {rr} breaths per minute",
        f"{rr} breaths a minute",
        f"I counted {rr} breaths in a minute",
        f"{rr} breaths per minute",
        f"{_spell_number(rr)} breaths per minute",
        f"{rr} breaths/minute",
    ])


def _plural_days(n: int, rng: random.Random, register: Register) -> str:
    if register is Register.CLINICAL:
        return rng.choice([f"x{n}/7" if n <= 7 else f"x{n}d", f"{n} days", f"{n}d", f"for {n} days"])
    if register is Register.SMS:
        return rng.choice([f"{n} days", f"{n}d", f"x{n} days"])
    forms = [f"for {n} days", f"{n} days now", f"going on {n} days"]
    if n < 20:
        forms.append(f"for {_spell_number(n)} days")
    if n == 1:
        forms = ["since yesterday", "for a day", "for one day"]
    return rng.choice(forms)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

# Label-neutral framing. None of it changes the classification; all of it makes
# the model read past noise instead of pattern-matching a bare sign list --
# which is what a real prompt looks like.
OPENERS_NARRATIVE = (
    "Good morning doctor.", "Please help.", "Doctor,", "Sorry to bother you.",
    "I brought my child in today.", "We walked here from the next village.",
    "The health worker told me to come.", "", "", "",
)
CLOSERS_NARRATIVE = (
    "What should I do?", "Is it serious?", "Please tell me what to do.",
    "Should I take {Po} to the hospital?", "Does {P} need medicine?",
    "What is the classification?", "How do I classify this child?",
    "", "", "",
)
SMS_OPENERS = ("chw here.", "referral q:", "pls advise.", "q for you:", "", "", "")
NOISE_NARRATIVE = (
    "{P}'s my third child.", "We came by bus this morning.",
    "{Pp} father is away working.", "{P} was fine last week.",
    "The clinic was closed yesterday.", "I've been giving {Po} water.",
    "{Pp} brother had the same thing.", "It rained all night.",
)


# A real health worker doesn't ask the same question fourteen times. These are
# in the spirit of expert_policy.QUESTION_TEXT, loosened into the way people
# actually speak -- the transcript style is meant to look like a consultation,
# not a form being read aloud.
HW_AGE_QUESTIONS = (
    "HW: How old is the child?",
    "HW: What age is {P}?",
    "HW: How many months?",
    "HW: Tell me the child's age.",
    "HW: And how old is {P}?",
)
HW_PROBES = (
    "What else have you noticed?",
    "Anything else?",
    "And what else?",
    "Go on.",
    "What else can you tell me?",
    "Have you noticed anything more?",
    "What else has been happening?",
    "Anything else you've seen?",
    "And?",
    "Tell me more.",
)


# The four "does the child have X?" gateway questions. A health worker asks all
# of them for every child, so a negative answer to any is natural in any
# vignette. The deeper signs (stridor, sunken eyes, skin pinch) are only ever
# asked once their gateway is positive -- "no stridor" in a child with no cough
# is not something anyone says.
TOP_LEVEL_FIELDS = ("cough_or_difficulty_breathing", "diarrhea", "fever", "ear_problem")


def decisive_branch_of(child: ChildAssessment) -> str:
    """Which branch assess() stopped at, inferred from an already-PRUNED case.

    Sound only because pruning has already zeroed every other branch, so the
    single positive gateway (or danger sign) identifies the branch. Mirrors
    assess()'s short-circuit order.
    """
    if child.danger_signs_present:
        return "danger"
    if child.cough_or_difficulty_breathing:
        return "cough"
    if child.diarrhea:
        return "diarrhea"
    if child.fever:
        return "fever"
    if child.ear_problem:
        return "ear"
    return "none"


BRANCH_OF_FIELD = {
    **{f: "cough" for f in COUGH_FIELDS},
    **{f: "diarrhea" for f in DIARRHEA_FIELDS},
    **{f: "fever" for f in FEVER_FIELDS},
    **{f: "ear" for f in EAR_FIELDS},
}

# Fields that are competing readings of ONE observation, not independent facts.
# assess() reads them as separate booleans, and sampling.repair_coherence stops
# both being true at once -- but the negative phrasings collide too: rendering
# skin_pinch_goes_back_slowly=False as "skin pinch is immediate" flatly
# contradicts skin_pinch_goes_back_very_slowly=True in the same vignette.
# When one member is positive, the others' falsity is implied and must not be
# stated.
MUTEX_GROUPS = (
    ("skin_pinch_goes_back_slowly", "skin_pinch_goes_back_very_slowly"),
    ("drinking_eagerly_thirsty", "not_able_to_drink_or_drinking_poorly"),
)


def _cap(text: str) -> str:
    """Capitalize without lowercasing the rest -- str.capitalize() would turn
    "RR 68" into "Rr 68" and destroy clinical shorthand."""
    return text[0].upper() + text[1:] if text else text


def _uncap(text: str) -> str:
    """Lowercase the first letter so a vocative opener runs on correctly
    ("Doctor, my child is..."), but never touch a proper noun or an acronym --
    "Doctor, halima is..." is its own kind of wrong."""
    if not text or not text[1:2].islower():
        return text  # acronym or single char: "RR 68", "I"
    first_word = text.split(maxsplit=1)[0].rstrip(".,'s")
    if first_word in NAMES:
        return text
    return text[0].lower() + text[1:]


def _join_opener(opener: str, sentences: list[str], closer: str) -> str:
    """Openers come in two shapes: full sentences ("Please help.") and
    vocatives that run on ("Doctor,"). Only the second requires the following
    clause to be lowercased."""
    if opener.endswith(",") and sentences:
        sentences = [_uncap(sentences[0])] + sentences[1:]
    return " ".join([opener] + sentences + [closer]).strip()


def _pronouns(rng: random.Random) -> tuple[str, str, str]:
    """(subject, object, possessive)."""
    return rng.choice([("she", "her", "her"), ("he", "him", "his")])


def _fill(text: str, subj: str, obj: str, poss: str, child_noun: str) -> str:
    # {Pp} and {Po} before {P}: a plain str.replace of "{P}" would corrupt the
    # longer slots into "her}"/"him}".
    return (
        text.replace("{Pp}", poss)
        .replace("{Po}", obj)
        .replace("{P}", subj)
        .replace("{C}", child_noun)
    )


def _typo(text: str, rng: random.Random) -> str:
    """Light surface noise. Real prompts are typed by tired people on phones."""
    ops = []
    if len(text) > 12:
        ops.append(lambda t: t.replace(" the ", " teh ", 1))
        ops.append(lambda t: t.replace("ing ", "ng ", 1))
        ops.append(lambda t: t.rstrip(".").rstrip())
    ops.append(lambda t: t.replace(",", "", 1))
    return rng.choice(ops)(text) if ops else text


def _phrase_for(field: str, value, rng: random.Random, register: Register,
                subj: str, obj: str, poss: str, child_noun: str) -> str | None:
    """One field -> one surface phrase, or None if the field has no bank
    (numeric fields are handled by the caller)."""
    bank = PHRASES.get(field)
    if bank is None:
        return None
    polarity = "true" if value else "false"
    forms = bank[polarity][register]
    return _fill(rng.choice(forms), subj, obj, poss, child_noun)


def _negative_is_natural(field: str, branch: str, child: ChildAssessment) -> bool:
    """Whether stating this field as absent is something a person would say,
    without contradicting something else the vignette states.

    A gateway question ("no fever") is always fair game. A deep sign is only
    reachable once its gateway is positive, so "no sunken eyes" in a child
    without diarrhoea is noise no real prompt contains -- and training on it
    teaches the model to expect an exhaustive negative checklist it will never
    see at evaluation.
    """
    for group in MUTEX_GROUPS:
        if field in group and any(getattr(child, other) for other in group if other != field):
            return False
    if field in TOP_LEVEL_FIELDS:
        return True
    return BRANCH_OF_FIELD.get(field) == branch


def _fragments(child: ChildAssessment, rng: random.Random, register: Register,
               subj: str, obj: str, poss: str, child_noun: str,
               state_negative_prob: float,
               restrict_to: set[str] | None = None) -> tuple[list[str], set[str]]:
    """
    Builds one fragment per field worth mentioning.

    Positives are always rendered (the invariant). Negatives are rendered only
    sometimes, and only where they'd be natural -- omitting them is
    label-preserving, and a vignette that enumerates every negative reads
    nothing like a real prompt.
    """
    required = required_fields_for(child)
    branch = decisive_branch_of(child)
    # One observation, one mention: if both rungs of a mutex group are false,
    # "skin turgor normal" AND "no delay in skin pinch" both describe the same
    # normal skin pinch. Stating it twice is something no real prompt does.
    mutex_spent: set[tuple[str, ...]] = set()
    frags: list[tuple[str, str]] = []   # (field, text)
    rendered: set[str] = set()

    # Danger signs are reported per-sign: only the listed ones are claims.
    if child.danger_signs_present:
        for sign in child.danger_signs_present:
            forms = DANGER_SIGN_PHRASES[sign][register]
            frags.append(("danger_signs_present", _fill(rng.choice(forms), subj, obj, poss, child_noun)))
        rendered.add("danger_signs_present")

    branch_fields = COUGH_FIELDS + DIARRHEA_FIELDS + FEVER_FIELDS + EAR_FIELDS
    for field in branch_fields:
        # Mid-dialogue, an un-asked field sits at its dataclass default, which is
        # indistinguishable from a genuine "no". Rendering it would put a claim
        # in the vignette that nobody made -- so callers holding a partial case
        # pass the set of fields actually elicited.
        if restrict_to is not None and field not in restrict_to:
            continue
        value = getattr(child, field)
        is_required = field in required

        if field == "respiratory_rate_per_min":
            if value is not None:
                frags.append((field, render_rate(value, rng, register)))
                rendered.add(field)
            continue
        if field in ("diarrhea_days", "ear_discharge_days", "fever_days"):
            if value is not None:
                noun = {"diarrhea_days": "diarrhoea", "ear_discharge_days": "ear discharge",
                        "fever_days": "fever"}[field]
                dur = _plural_days(value, rng, register)
                frags.append((field, f"{noun} {dur}" if register is not Register.NARRATIVE
                              else f"the {noun} has been going {dur}"))
                rendered.add(field)
            continue

        if is_required:
            text = _phrase_for(field, value, rng, register, subj, obj, poss, child_noun)
            if text:
                frags.append((field, text))
                rendered.add(field)
        elif not value and _negative_is_natural(field, branch, child) and rng.random() < state_negative_prob:
            group = next((g for g in MUTEX_GROUPS if field in g), None)
            if group is not None:
                if group in mutex_spent:
                    continue
                mutex_spent.add(group)
            text = _phrase_for(field, value, rng, register, subj, obj, poss, child_noun)
            if text:
                frags.append((field, text))
                rendered.add(field)

    # Order axis: real caretakers bury the lede. The protocol has to catch a
    # danger sign wherever it appears, so train it to.
    rng.shuffle(frags)
    return [t for _, t in frags], rendered


def verbalize_case(
    child: ChildAssessment,
    rng: random.Random,
    style: VignetteStyle,
    restrict_to: set[str] | None = None,
) -> tuple[str, set[str]]:
    """
    Returns (vignette_text, fields_rendered).

    The caller MUST assert required_fields_for(child) <= fields_rendered --
    see generate_sft_data.py. A vignette missing a deciding sign trains the
    model to hallucinate it.

    `restrict_to` limits which fields may be mentioned at all. Pass it when the
    case is PARTIAL (mid-dialogue, for next_question examples): un-asked fields
    look exactly like negative ones from the dataclass's side, and stating them
    would fabricate answers the caretaker never gave.
    """
    register = STYLE_REGISTER[style]
    subj, obj, poss = _pronouns(rng)
    name = rng.choice(NAMES)
    child_noun = rng.choice(["the child", "the baby", name, "my child"])

    # How often a negative gets stated rather than omitted. Structured styles
    # enumerate; a worried mother does not.
    state_neg = {
        VignetteStyle.STRUCTURED_LIST: 0.75,
        VignetteStyle.CHW_NOTES: 0.45,
        VignetteStyle.DIALOGUE_TRANSCRIPT: 0.5,
        VignetteStyle.VERBOSE_PARAGRAPH: 0.3,
        VignetteStyle.CARETAKER_NARRATIVE: 0.2,
        VignetteStyle.SMS_REFERRAL: 0.2,
        VignetteStyle.QUESTION_FORM: 0.2,
    }[style]

    frags, rendered = _fragments(child, rng, register, subj, obj, poss, child_noun,
                                 state_neg, restrict_to)
    age = render_age(child.age_months, rng, register)

    if style is VignetteStyle.STRUCTURED_LIST:
        lines = [f"Age: {age}"] + [f"- {f}" for f in frags]
        text = "\n".join(lines)

    elif style is VignetteStyle.CHW_NOTES:
        sex = "F" if subj == "she" else "M"
        text = f"{age} {sex}, " + ", ".join(frags) + "."

    elif style is VignetteStyle.SMS_REFERRAL:
        opener = rng.choice(SMS_OPENERS)
        body = ", ".join(frags)
        text = f"{opener} {age} child, {body}. advise?".strip()
        text = text.lower()
        if rng.random() < 0.5:
            text = _typo(text, rng)

    elif style is VignetteStyle.DIALOGUE_TRANSCRIPT:
        lines = [
            _fill(rng.choice(HW_AGE_QUESTIONS), subj, obj, poss, child_noun),
            f"Caretaker: {_cap(age)}.",
        ]
        for f in frags:
            lines.append("HW: " + rng.choice(HW_PROBES))
            lines.append(f"Caretaker: {_cap(f)}.")
        text = "\n".join(lines)

    elif style is VignetteStyle.QUESTION_FORM:
        body = ", ".join(frags)
        lead = rng.choice([
            "do i refer this child?", "what's the classification?",
            "how would you classify this?", "is this urgent?",
            "what does the booklet say for this?",
        ])
        text = f"{lead} {child_noun}, {age}, {body}"

    elif style is VignetteStyle.VERBOSE_PARAGRAPH:
        opener = _fill(rng.choice(OPENERS_NARRATIVE), subj, obj, poss, child_noun)
        closer = _fill(rng.choice(CLOSERS_NARRATIVE), subj, obj, poss, child_noun)
        noise = [_cap(_fill(n, subj, obj, poss, child_noun))
                 for n in rng.sample(list(NOISE_NARRATIVE), k=2)]
        sentences = [f"{_cap(child_noun)} is {age}."]
        sentences += [f"{_cap(f)}." for f in frags]
        # Irrelevant detail interleaved, not appended -- it has to be read past,
        # not skipped.
        for n in noise:
            sentences.insert(rng.randrange(1, len(sentences) + 1), n)
        text = _join_opener(opener, sentences, closer)

    else:  # CARETAKER_NARRATIVE
        opener = _fill(rng.choice(OPENERS_NARRATIVE), subj, obj, poss, child_noun)
        closer = _cap(_fill(rng.choice(CLOSERS_NARRATIVE), subj, obj, poss, child_noun))
        sentences = [f"{_cap(child_noun)} is {age}."]
        sentences += [f"{_cap(f)}." for f in frags]
        text = _join_opener(opener, sentences, closer)

    text = " ".join(text.split()) if "\n" not in text else text
    return text, rendered
