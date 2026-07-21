"""
Deterministic IMCI (Integrated Management of Childhood Illness) triage rule
engine for children 2 months to 5 years.

SOURCE: WHO IMCI Chart Booklet, March 2014 (ISBN 978-92-4-150682-3), the
standard global reference most country adaptations (including Nigeria's)
build from.
https://cdn.who.int/media/docs/default-source/mca-documents/child/imci-integrated-management-of-childhood-illness/imci-in-service-training/imci-chart-booklet.pdf
Fetched and cross-checked against this scaffold's logic on 2026-07-10.

IMPORTANT -- READ BEFORE USE:
This encodes a SUBSET of the published WHO/IMCI 2014 chart booklet -- general
danger signs, cough/difficult breathing, diarrhoea/dehydration, a simplified
fever branch, and ear problems. It is a COMPETITION SCAFFOLD, not a validated
clinical tool. Specifically NOT modeled (real IMCI covers these but this
scaffold does not):
  - Malaria risk-area-dependent fever branching and malaria testing (the real
    algorithm's fever classification depends on high/low malaria risk area
    and RDT results; this scaffold uses a simplified danger-sign/stiff-neck-
    only branch and flags this explicitly in output).
  - Measles-specific complications, acute malnutrition, anaemia, HIV status,
    wheeze/bronchodilator trial steps, persistent diarrhoea (>=14 days) and
    dysentery as SEPARATE simultaneous classifications alongside dehydration
    (real IMCI produces multiple independent classifications per symptom
    category; this scaffold returns one primary classification and notes
    secondary findings in `reasoning` only).
  - Country-specific adaptations (e.g. Nigeria's national IMCI adaptation may
    differ in specific drug choices, follow-up intervals, or malaria
    protocol from this WHO generic version).

Any real deployment needs review and sign-off by a qualified clinician
familiar with the current national IMCI adaptation before being used with
real patients, even in a pilot. This tool supports a health worker in
correctly APPLYING a known published protocol; it does not replace clinical
training, the physical chart booklet, or professional judgment.

Design rationale: classification logic is intentionally DETERMINISTIC (plain
Python, not a learned model) -- for a safety-relevant task like this,
correctness should come from code matching a published standard, not from a
27M/0.8B parameter model's learned weights. HRM's role in this system is
orchestration (deciding which symptom modules to check, sequencing
multi-symptom cases), not the final classification itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Classification(str, Enum):
    SEVERE = "severe"      # pink/red row -- urgent referral
    MODERATE = "moderate"  # yellow row -- treat + follow-up
    MILD = "mild"          # green row -- home care


# General danger signs -- checked FIRST, for every child, regardless of
# presenting complaint. Any one present -> urgent referral (Pink: VERY SEVERE
# DISEASE), skip further symptom-specific assessment.
# SOURCE: WHO IMCI Chart Booklet 2014, "CHECK FOR GENERAL DANGER SIGNS" panel.
GENERAL_DANGER_SIGNS = (
    "convulsions",                    # history of convulsions
    "unable_to_drink_or_breastfeed",
    "vomits_everything",
    "lethargic_or_unconscious",
    "convulsing_now",
)


@dataclass
class ChildAssessment:
    age_months: int
    danger_signs_present: list[str] = field(default_factory=list)

    # Cough or difficult breathing -- SOURCE: WHO IMCI 2014 "Does the child
    # have cough or difficult breathing?" panel.
    cough_or_difficulty_breathing: bool = False
    respiratory_rate_per_min: int | None = None
    chest_indrawing: bool = False
    stridor_when_calm: bool = False

    # Diarrhoea/dehydration -- SOURCE: WHO IMCI 2014 "Does the child have
    # diarrhoea?" panel. Real protocol requires ANY TWO of four signs per
    # severity level, not all four -- see _classify_dehydration().
    diarrhea: bool = False
    diarrhea_days: int | None = None
    blood_in_stool: bool = False
    child_lethargic_or_unconscious: bool = False   # shared w/ danger signs but assessed again per protocol
    child_restless_or_irritable: bool = False
    sunken_eyes: bool = False
    not_able_to_drink_or_drinking_poorly: bool = False
    drinking_eagerly_thirsty: bool = False
    skin_pinch_goes_back_very_slowly: bool = False  # >2 seconds
    skin_pinch_goes_back_slowly: bool = False        # some delay, not >2s

    # Fever -- SIMPLIFIED scaffold, see module docstring. Real IMCI fever
    # classification depends on malaria-risk area and test results, not
    # modeled here.
    fever: bool = False
    fever_days: int | None = None
    stiff_neck: bool = False
    # 2022 SA adaptation lists a bulging fontanelle alongside stiff neck as a
    # trigger for the severe fever classification. Additive OR-trigger only --
    # the label stays the 2014 `very_severe_febrile_disease` the scorer expects.
    bulging_fontanelle: bool = False

    # Ear problem -- SOURCE: WHO IMCI 2014 "Does the child have an ear
    # problem?" panel.
    ear_problem: bool = False
    ear_pain: bool = False
    ear_discharge_days: int | None = None
    tender_swelling_behind_ear: bool = False

    def __post_init__(self):
        # SAFETY-CRITICAL CHECK: a str silently satisfies "iterable of str" in
        # Python without raising, and iterating a string yields individual
        # characters -- none of which will ever match a danger-sign name.
        # That means passing "convulsions" (str) instead of ["convulsions"]
        # (list) silently and undetectably downgrades a genuine danger sign
        # to "not present". This must fail loudly, not silently produce a
        # wrong (and dangerously under-triaged) classification. (Caught by
        # this scaffold's own eval suite during development -- see
        # report/REPORT_TEMPLATE_NOTES.md for the writeup.)
        if isinstance(self.danger_signs_present, str):
            raise TypeError(
                "danger_signs_present must be a list[str], not a str. "
                f"Got {self.danger_signs_present!r} -- did you mean "
                f"[{self.danger_signs_present!r}] or a comma-split list? "
                "Silently accepting a string here previously caused a real "
                "bug where genuine danger signs were missed entirely."
            )
        if not isinstance(self.danger_signs_present, list):
            raise TypeError(
                f"danger_signs_present must be a list[str], got {type(self.danger_signs_present).__name__}."
            )


@dataclass
class TriageResult:
    classification: Classification
    condition_label: str
    reasoning: list[str]
    recommended_action: str
    secondary_findings: list[str] = field(default_factory=list)
    disclaimer: str = (
        "This is protocol-following decision support, not a diagnosis. "
        "Always follow the official IMCI chart booklet and use clinical "
        "judgment. Refer per protocol when in doubt. This scaffold models a "
        "subset of the full IMCI algorithm -- see src/tools/imci_protocol.py "
        "module docstring for what is and isn't covered."
    )


def _fast_breathing_threshold(age_months: int) -> int:
    """
    Age-banded fast-breathing threshold. SOURCE: WHO IMCI Chart Booklet 2014,
    "Does the child have cough or difficult breathing?" panel:
    "2 months up to 12 months: 50 breaths per minute or more.
     12 Months up to 5 years: 40 breaths per minute or more."
    """
    if age_months < 12:
        return 50
    return 40


def _classify_dehydration(child: ChildAssessment) -> tuple[Classification, list[str], list[str]]:
    """
    SOURCE: WHO IMCI Chart Booklet 2014, "Classify DIARRHOEA for DEHYDRATION"
    panel. Requires ANY TWO of four signs at each severity level (not all
    four) -- this is a real correction from an earlier version of this
    scaffold, which incorrectly required two SPECIFIC signs rather than any
    two from the correct set.
    """
    reasoning = []

    severe_signs = {
        "lethargic or unconscious": child.child_lethargic_or_unconscious,
        "sunken eyes": child.sunken_eyes,
        "not able to drink or drinking poorly": child.not_able_to_drink_or_drinking_poorly,
        "skin pinch goes back very slowly (>2s)": child.skin_pinch_goes_back_very_slowly,
    }
    severe_count = sum(severe_signs.values())
    if severe_count >= 2:
        present = [k for k, v in severe_signs.items() if v]
        reasoning.append(f"Severe dehydration signs (>=2 of 4 required): {present}")
        return Classification.SEVERE, ["severe_dehydration"], reasoning

    some_signs = {
        "restless or irritable": child.child_restless_or_irritable,
        "sunken eyes": child.sunken_eyes,
        "drinks eagerly, thirsty": child.drinking_eagerly_thirsty,
        "skin pinch goes back slowly": child.skin_pinch_goes_back_slowly,
    }
    some_count = sum(some_signs.values())
    if some_count >= 2:
        present = [k for k, v in some_signs.items() if v]
        reasoning.append(f"Some dehydration signs (>=2 of 4 required): {present}")
        return Classification.MODERATE, ["some_dehydration"], reasoning

    reasoning.append("Not enough signs to classify as some or severe dehydration.")
    return Classification.MILD, ["no_dehydration"], reasoning


def assess(child: ChildAssessment) -> TriageResult:
    reasoning = []
    secondary_findings = []

    # Step 1: general danger signs, checked before anything else.
    # SOURCE: WHO IMCI 2014, "CHECK FOR GENERAL DANGER SIGNS" -> Pink: VERY
    # SEVERE DISEASE.
    present_danger_signs = [s for s in child.danger_signs_present if s in GENERAL_DANGER_SIGNS]
    if present_danger_signs:
        reasoning.append(f"General danger sign(s) present: {present_danger_signs}")
        return TriageResult(
            classification=Classification.SEVERE,
            condition_label="very_severe_disease",
            reasoning=reasoning,
            recommended_action=(
                "Refer URGENTLY. Give diazepam if convulsing now, quickly complete "
                "assessment, give any pre-referral treatment immediately, treat to "
                "prevent low blood sugar, keep the child warm."
            ),
        )
    reasoning.append("No general danger signs present.")

    # Step 2: cough / difficulty breathing.
    # SOURCE: WHO IMCI 2014, "Does the child have cough or difficult breathing?"
    if child.cough_or_difficulty_breathing:
        if child.stridor_when_calm:
            reasoning.append("Stridor in a calm child.")
            return TriageResult(
                classification=Classification.SEVERE,
                condition_label="severe_pneumonia_or_very_severe_disease",
                reasoning=reasoning,
                recommended_action="Give first dose of an appropriate antibiotic. Refer URGENTLY to hospital.",
            )
        threshold = _fast_breathing_threshold(child.age_months)
        fast_breathing = bool(
            child.respiratory_rate_per_min and child.respiratory_rate_per_min >= threshold
        )
        if child.chest_indrawing or fast_breathing:
            if child.chest_indrawing:
                reasoning.append("Chest indrawing present.")
            if fast_breathing:
                reasoning.append(
                    f"Fast breathing: {child.respiratory_rate_per_min}/min >= "
                    f"threshold {threshold}/min for age {child.age_months}mo."
                )
            return TriageResult(
                classification=Classification.MODERATE,
                condition_label="pneumonia",
                reasoning=reasoning,
                recommended_action=(
                    "Give oral Amoxicillin for 5 days. Soothe throat/relieve cough. "
                    "Advise mother when to return immediately. Follow-up in 3 days."
                ),
            )
        reasoning.append("No fast breathing, chest indrawing, or stridor -- classified as cough/cold.")
        return TriageResult(
            classification=Classification.MILD,
            condition_label="cough_or_cold",
            reasoning=reasoning,
            recommended_action=(
                "Soothe the throat, relieve cough with a safe remedy. Advise mother "
                "when to return immediately. Follow-up in 5 days if not improving."
            ),
        )

    # Step 3: diarrhea assessment.
    if child.diarrhea:
        classification, labels, dehydration_reasoning = _classify_dehydration(child)
        reasoning.extend(dehydration_reasoning)

        if child.blood_in_stool:
            secondary_findings.append("dysentery (blood in stool) -- give ciprofloxacin per protocol, follow-up 3 days")
        if child.diarrhea_days is not None and child.diarrhea_days >= 14:
            secondary_findings.append(
                "persistent diarrhoea (>=14 days) -- "
                + ("refer, treat dehydration first" if classification == Classification.SEVERE
                   else "multivitamins/zinc for 14 days, follow-up 5 days")
            )

        action_map = {
            "severe_dehydration": (
                "Give fluid for severe dehydration (Plan C) if no other severe "
                "classification, else refer URGENTLY with ORS sips on the way."
            ),
            "some_dehydration": "Give fluid, zinc, and food for some dehydration (Plan B). Follow-up in 5 days if not improving.",
            "no_dehydration": "Give fluid, zinc, and food to treat diarrhoea at home (Plan A). Follow-up in 5 days if not improving.",
        }
        return TriageResult(
            classification=classification,
            condition_label=labels[0],
            reasoning=reasoning,
            recommended_action=action_map[labels[0]],
            secondary_findings=secondary_findings,
        )

    # Step 4: fever -- SIMPLIFIED, see module docstring. Real IMCI fever
    # classification depends on malaria-risk area and RDT results, which are
    # NOT modeled here.
    if child.fever:
        reasoning.append(
            "NOTE: this scaffold's fever branch is simplified and does NOT "
            "model the real IMCI malaria-risk/RDT-dependent algorithm -- "
            "see src/tools/imci_protocol.py module docstring."
        )
        if child.stiff_neck or child.bulging_fontanelle:
            severe_signs = []
            if child.stiff_neck:
                severe_signs.append("stiff neck")
            if child.bulging_fontanelle:
                severe_signs.append("bulging fontanelle")
            reasoning.append(f"Fever with {' and '.join(severe_signs)}.")
            return TriageResult(
                classification=Classification.SEVERE,
                condition_label="very_severe_febrile_disease",
                reasoning=reasoning,
                recommended_action=(
                    "Give first dose of appropriate antibiotic, treat to prevent low "
                    "blood sugar, give paracetamol for high fever. Refer URGENTLY."
                ),
            )
        reasoning.append("Fever without stiff neck or other danger signs.")
        return TriageResult(
            classification=Classification.MODERATE,
            condition_label="fever_unspecified_malaria_not_assessed",
            reasoning=reasoning,
            recommended_action=(
                "Real protocol requires malaria risk assessment and testing here "
                "(not modeled). Give paracetamol for high fever. Follow up per "
                "chart booklet; refer for assessment if fever persists >7 days."
            ),
        )

    # Step 5: ear problem.
    # SOURCE: WHO IMCI 2014, "Does the child have an ear problem?"
    if child.ear_problem:
        if child.tender_swelling_behind_ear:
            reasoning.append("Tender swelling behind ear.")
            return TriageResult(
                classification=Classification.SEVERE,
                condition_label="mastoiditis",
                reasoning=reasoning,
                recommended_action="Give first dose of antibiotic and paracetamol for pain. Refer URGENTLY.",
            )
        if child.ear_discharge_days is not None and child.ear_discharge_days >= 14:
            reasoning.append("Ear discharge present for 14+ days.")
            return TriageResult(
                classification=Classification.MODERATE,
                condition_label="chronic_ear_infection",
                reasoning=reasoning,
                recommended_action="Dry ear by wicking. Topical quinolone eardrops for 14 days. Follow-up in 5 days.",
            )
        if child.ear_pain or (child.ear_discharge_days is not None and child.ear_discharge_days < 14):
            reasoning.append("Ear pain or discharge <14 days.")
            return TriageResult(
                classification=Classification.MODERATE,
                condition_label="acute_ear_infection",
                reasoning=reasoning,
                recommended_action="Antibiotic for 5 days. Paracetamol for pain. Dry ear by wicking. Follow-up in 5 days.",
            )
        reasoning.append("No ear pain, discharge, or tender swelling.")
        return TriageResult(
            classification=Classification.MILD,
            condition_label="no_ear_infection",
            reasoning=reasoning,
            recommended_action="No treatment needed.",
        )

    reasoning.append("No presenting symptoms matched a modeled classification branch.")
    return TriageResult(
        classification=Classification.MILD,
        condition_label="no_classification_matched",
        reasoning=reasoning,
        recommended_action=(
            "No danger signs or modeled symptoms detected. This scaffold covers "
            "cough/difficulty breathing, diarrhoea, a simplified fever branch, and "
            "ear problems only -- the real IMCI chart booklet also covers acute "
            "malnutrition, anaemia, HIV status, and immunization/vitamin A status, "
            "none of which are modeled here."
        ),
    )
