"""
Young-infant (birth up to 2 months) IMCI chart.

*** REQUIRES CLINICIAN REVIEW BEFORE IT TRAINS ANYTHING. Sign-off status:
UNREVIEWED. Gated behind scripts/generate_sft_data.py --include-extended. ***

WHY THIS IS A SEPARATE MODULE
The 0-2 month chart is structurally different from the 2 months-5 years chart
in src/tools/imci_protocol.py: different danger signs (apnoea, breathing < 30 or
> 60 per minute, bulging fontanelle, hypothermia, "only moves when stimulated"),
different thresholds, and its own classifications. It is NOT an extra branch on
ChildAssessment, so it gets its own dataclass and its own assess functions. It
reuses TriageResult only for the shared answer format.

Previously this chart was handled as a scope_refusal (see the git history of
src/sft/mixture.py). Modelling it flips young-infant vignettes from "declined"
to "classified"; the refusal fixtures were updated accordingly.

SOURCE: 2022 SA IMCI booklet pp 5-7 ("Assess and classify the sick young
infant"), cross-checked against the WHO 2014 generic young-infant chart. See
data/imci_2022/classifications.json (young_infant_0_2m).

WHAT IS MODELLED: possible bacterial infection + jaundice, young-infant
diarrhoea/dehydration, and congenital problems. NOT modelled (stay out of
scope): young-infant HIV, feeding/growth assessment, immunisation status.

REVIEW CHECKLIST (specific things for the reviewing clinician):
  1. Very-severe-disease sign list and the breathing thresholds (< 30 and > 60).
  2. Severe dehydration in a young infant is triggered by age < 1 month even
     with fewer classic signs -- confirm.
  3. Jaundice: any jaundice under 24 hours of age, or yellow palms/soles, is
     severe. Confirm the 24-hour rule and the palms/soles rule.
  4. Congenital "priority" vs "other abnormal" sign split (p7), and the
     macrocephaly (>39 cm) / very-low-birth-weight (<=2 kg) cut-offs.
  5. Congenital syphilis path (mother RPR positive and untreated / partially
     treated).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tools.imci_protocol import Classification, TriageResult


@dataclass
class YoungInfantAssessment:
    age_days: int  # 0 up to ~60

    # --- Very severe disease / danger signs (p5) ---
    convulsions: bool = False
    apnoea: bool = False
    breathing_under_30: bool = False
    fast_breathing_over_60: bool = False
    severe_chest_indrawing: bool = False
    grunting: bool = False
    bulging_fontanelle: bool = False
    fever_37_5_or_more: bool = False
    hypothermia_under_35_5: bool = False
    only_moves_when_stimulated: bool = False
    eye_pus_abundant_or_swollen_eyelids: bool = False   # severe eye finding
    umbilicus_red_extending_or_draining_pus: bool = False
    skin_pustules_many_or_severe: bool = False

    # --- Local bacterial infection (p5) ---
    eye_discharge_purulent_or_sticky: bool = False
    umbilicus_red_only: bool = False
    skin_pustules_few: bool = False

    # --- Jaundice (p5) ---
    jaundice: bool = False
    jaundice_onset_under_24h: bool = False
    yellow_palms_and_soles: bool = False

    # --- Diarrhoea / dehydration (p6) ---
    diarrhoea: bool = False
    diarrhoea_days: int | None = None
    blood_in_stool: bool = False
    lethargic_or_unconscious: bool = False
    restless_or_irritable: bool = False
    sunken_eyes: bool = False
    skin_pinch_very_slow: bool = False
    skin_pinch_slow: bool = False

    # --- Congenital problems (p7) ---
    cleft_lip_or_palate: bool = False
    imperforate_anus: bool = False
    nose_not_patent: bool = False
    macrocephaly: bool = False
    ambiguous_genitalia: bool = False
    abdominal_distension: bool = False
    very_low_birth_weight: bool = False          # <= 2 kg
    microcephaly: bool = False
    abnormal_fontanelle_or_sutures: bool = False
    club_foot: bool = False
    mother_rpr_positive_untreated: bool = False

    def __post_init__(self):
        if not isinstance(self.age_days, int):
            raise TypeError(f"age_days must be int, got {type(self.age_days).__name__}")


_VERY_SEVERE_SIGNS = (
    ("convulsions", "convulsions"),
    ("apnoea", "apnoea (breathing pauses)"),
    ("breathing_under_30", "breathing under 30 per minute"),
    ("fast_breathing_over_60", "fast breathing over 60 per minute"),
    ("severe_chest_indrawing", "severe chest indrawing"),
    ("grunting", "grunting"),
    ("bulging_fontanelle", "a bulging fontanelle"),
    ("fever_37_5_or_more", "fever 37.5 C or more"),
    ("hypothermia_under_35_5", "low body temperature under 35.5 C"),
    ("only_moves_when_stimulated", "moving only when stimulated"),
    ("eye_pus_abundant_or_swollen_eyelids", "abundant eye pus or swollen eyelids"),
    ("umbilicus_red_extending_or_draining_pus", "umbilical redness extending to the skin or draining pus"),
    ("skin_pustules_many_or_severe", "many or severe skin pustules"),
)

_LOCAL_SIGNS = (
    ("eye_discharge_purulent_or_sticky", "purulent or sticky eye discharge"),
    ("umbilicus_red_only", "a red umbilicus"),
    ("skin_pustules_few", "a few skin pustules"),
)


def _present(a: YoungInfantAssessment, signs) -> list[str]:
    return [text for field_name, text in signs if getattr(a, field_name)]


def classify_yi_bacterial(a: YoungInfantAssessment) -> TriageResult:
    """The core 'very severe disease and local bacterial infection' panel. Always
    classifies (severe / local / none)."""
    severe = _present(a, _VERY_SEVERE_SIGNS)
    if severe:
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="yi_very_severe_disease",
            reasoning=[f"The young infant has {', '.join(severe)}, which is very severe disease."],
            recommended_action=(
                "Give the first dose of an appropriate intramuscular antibiotic. Treat to prevent "
                "low blood sugar. Keep the infant warm on the way to hospital. Refer URGENTLY."
            ),
        )
    local = _present(a, _LOCAL_SIGNS)
    if local:
        return TriageResult(
            classification=Classification.MODERATE,
            condition_label="yi_local_bacterial_infection",
            reasoning=[f"The young infant has {', '.join(local)}, a local bacterial infection."],
            recommended_action=(
                "Give an appropriate oral antibiotic. If there is eye discharge, give an eye "
                "ointment. Teach the caregiver to treat the local infection at home and to give "
                "home care. Follow up in 2 days."
            ),
        )
    return TriageResult(
        classification=Classification.MILD,
        condition_label="yi_no_bacterial_infection",
        reasoning=["No signs of very severe disease or a local bacterial infection."],
        recommended_action="Counsel the caregiver on home care for the young infant.",
    )


def classify_yi_jaundice(a: YoungInfantAssessment) -> TriageResult | None:
    """Returns None when no jaundice is present."""
    if not a.jaundice:
        return None
    if (a.jaundice_onset_under_24h and a.age_days < 1) or a.yellow_palms_and_soles:
        why = ("jaundice under 24 hours of age" if a.jaundice_onset_under_24h and a.age_days < 1
               else "yellow palms and soles")
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="yi_severe_jaundice",
            reasoning=[f"There is {why}, which is severe jaundice."],
            recommended_action=(
                "Treat to prevent low blood sugar. Keep the infant warm. Refer URGENTLY to hospital."
            ),
        )
    return TriageResult(
        classification=Classification.MODERATE,
        condition_label="yi_jaundice",
        reasoning=["There is jaundice appearing after 24 hours of age, with palms and soles not "
                   "yellow."],
        recommended_action=(
            "Advise the caregiver to return immediately if the palms and soles become yellow. "
            "Follow up in 1 day. If the infant is older than 14 days, refer for assessment."
        ),
    )


def classify_yi_diarrhoea(a: YoungInfantAssessment) -> TriageResult | None:
    """Returns None when the infant has no diarrhoea."""
    if not a.diarrhoea:
        return None

    if a.blood_in_stool:
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="yi_dysentery",
            reasoning=["There is blood in the stool of a young infant."],
            recommended_action="Keep the infant warm on the way to hospital. Refer URGENTLY.",
        )
    if a.diarrhoea_days is not None and a.diarrhoea_days >= 14:
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="yi_severe_persistent_diarrhoea",
            reasoning=["Diarrhoea has lasted 14 days or more in a young infant."],
            recommended_action=(
                "Treat any dehydration before referral. Keep the infant warm. Refer URGENTLY."
            ),
        )

    severe = sum([a.lethargic_or_unconscious, a.sunken_eyes, a.skin_pinch_very_slow]) >= 2
    if severe or a.age_days < 30:
        why = ("two or more severe signs (lethargic/unconscious, sunken eyes, skin pinch goes back "
               "very slowly)" if severe else "the infant is less than 1 month old")
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="yi_severe_dehydration",
            reasoning=[f"There is diarrhoea with {why}, which is severe dehydration in a young "
                       f"infant."],
            recommended_action=(
                "Start intravenous fluids (Plan C). Give the first dose of an intramuscular "
                "antibiotic. Keep the infant warm on the way to hospital. Refer URGENTLY."
            ),
        )
    if sum([a.restless_or_irritable, a.sunken_eyes, a.skin_pinch_slow]) >= 2:
        return TriageResult(
            classification=Classification.MODERATE,
            condition_label="yi_some_dehydration",
            reasoning=["There is diarrhoea with two of restless/irritable, sunken eyes, and a slow "
                       "skin pinch, which is some dehydration."],
            recommended_action=(
                "Give fluid for some dehydration (Plan B). Advise the mother to continue "
                "breastfeeding. Give zinc for 14 days. Follow up in 2 days."
            ),
        )
    return TriageResult(
        classification=Classification.MILD,
        condition_label="yi_no_dehydration",
        reasoning=["There is diarrhoea with not enough signs for some or severe dehydration."],
        recommended_action=(
            "Give fluid and continue breastfeeding at home (Plan A). Give zinc for 14 days. "
            "Counsel the caregiver on home care. Follow up in 2 days."
        ),
    )


_CONGENITAL_PRIORITY = (
    ("cleft_lip_or_palate", "a cleft lip or palate"),
    ("imperforate_anus", "an imperforate anus"),
    ("nose_not_patent", "a nose that is not patent"),
    ("macrocephaly", "macrocephaly"),
    ("ambiguous_genitalia", "ambiguous genitalia"),
    ("abdominal_distension", "abdominal distension"),
    ("very_low_birth_weight", "a very low birth weight (2 kg or less)"),
)

_CONGENITAL_OTHER = (
    ("microcephaly", "microcephaly"),
    ("abnormal_fontanelle_or_sutures", "an abnormal fontanelle or sutures"),
    ("club_foot", "a club foot"),
)


def classify_yi_congenital(a: YoungInfantAssessment) -> TriageResult | None:
    """Returns None when no congenital sign is present."""
    priority = _present(a, _CONGENITAL_PRIORITY)
    if priority:
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="yi_congenital_priority",
            reasoning=[f"There is {', '.join(priority)}, a congenital priority sign."],
            recommended_action=(
                "Give any needed pre-referral treatment. Treat to prevent low blood sugar. Keep "
                "the infant warm. Refer URGENTLY to hospital."
            ),
        )
    other = _present(a, _CONGENITAL_OTHER)
    if other:
        return TriageResult(
            classification=Classification.MODERATE,
            condition_label="yi_congenital_abnormal_signs",
            reasoning=[f"There is {', '.join(other)}, an abnormal sign."],
            recommended_action=(
                "Keep the infant warm, skin to skin. Assess breastfeeding and support the mother. "
                "Refer for assessment."
            ),
        )
    if a.mother_rpr_positive_untreated:
        return TriageResult(
            classification=Classification.MODERATE,
            condition_label="yi_possible_congenital_syphilis",
            reasoning=["The mother's RPR is positive and she was untreated or only partially "
                       "treated, so congenital syphilis is possible."],
            recommended_action=(
                "Check for signs of congenital syphilis and refer to hospital if present. If there "
                "are no signs, give intramuscular penicillin. Ensure the mother receives full "
                "treatment."
            ),
        )
    return None


YOUNG_INFANT_LABELS = (
    "yi_very_severe_disease",
    "yi_local_bacterial_infection",
    "yi_no_bacterial_infection",
    "yi_severe_jaundice",
    "yi_jaundice",
    "yi_severe_dehydration",
    "yi_some_dehydration",
    "yi_no_dehydration",
    "yi_severe_persistent_diarrhoea",
    "yi_dysentery",
    "yi_congenital_priority",
    "yi_congenital_abnormal_signs",
    "yi_possible_congenital_syphilis",
)
