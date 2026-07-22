"""
Builds eval/tasks/imci_vignettes_extended.json -- an NL eval that exercises the
branches the core Sacc vignettes never touch: the extended sick-child
classifications (malaria/measles/anaemia/malnutrition/wheeze/persistent
diarrhoea/dysentery/sore throat/growth/HIV) and the whole young-infant chart.

Discipline (same as the training corpus): the ground-truth label and severity
are NOT hand-asserted -- each entry pairs a hand-written NL prompt with a
STRUCTURED case, and the deterministic classifier is run on that case to derive
the expected severity + label. If a case doesn't trigger its branch, this script
fails loudly so the vignette gets fixed rather than shipping a wrong label.

Usage:
    python scripts/build_extended_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sft.extended_protocol import (
    ExtendedAssessment,
    classify_anaemia, classify_dysentery, classify_fever_malaria, classify_growth,
    classify_hiv, classify_malnutrition, classify_measles, classify_persistent_diarrhoea,
    classify_sore_throat, classify_wheeze,
)
from src.sft.extended_verbalize import EXTENDED_LABEL_TEXT
from src.sft.young_infant import (
    YoungInfantAssessment,
    classify_yi_bacterial, classify_yi_congenital, classify_yi_diarrhoea, classify_yi_jaundice,
)
from src.sft.young_infant_verbalize import YI_LABEL_TEXT

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval" / "tasks" / "imci_vignettes_extended.json"

LEAKS = ["ICD-10", "imci_protocol", "scaffold", ".py", "ExtendedAssessment", "YoungInfantAssessment"]

# (id, classifier, structured input, NL prompt). Severity + label are derived by
# running the classifier -- never written here.
EXTENDED = [
    ("ext_wheeze", classify_wheeze,
     dict(age_months=18, wheeze=True),
     "An 18-month-old is wheezing when she breathes. No danger signs, not fast breathing. How do I classify this?"),
    ("ext_wheeze_danger", classify_wheeze,
     dict(age_months=24, wheeze=True, danger_signs_present=["lethargic_or_unconscious"]),
     "A 2-year-old is wheezing and has become lethargic. What's the classification?"),
    ("ext_persistent_diarrhoea", classify_persistent_diarrhoea,
     dict(age_months=30, diarrhoea=True, diarrhoea_days=18),
     "A 30-month-old has had loose stools for 18 days now. No sunken eyes, skin pinch normal, not losing weight. Classify?"),
    ("ext_severe_persistent_diarrhoea", classify_persistent_diarrhoea,
     dict(age_months=30, diarrhoea=True, diarrhoea_days=20, dehydration_present=True),
     "A 30-month-old has had diarrhoea for 20 days and now shows signs of dehydration. What's the classification?"),
    ("ext_dysentery", classify_dysentery,
     dict(age_months=36, diarrhoea=True, blood_in_stool=True),
     "A 3-year-old has diarrhoea with blood in the stool, no dehydration. How do I classify?"),
    ("ext_severe_dysentery", classify_dysentery,
     dict(age_months=8, diarrhoea=True, blood_in_stool=True),
     "An 8-month-old has diarrhoea with visible blood in the stool. Classification?"),
    ("ext_strep_throat", classify_sore_throat,
     dict(age_months=48, sore_throat=True, enlarged_tonsils=True, tonsil_exudate=True),
     "A 4-year-old has a sore throat with enlarged tonsils and white exudate on them, no cough and no runny nose. Classify?"),
    ("ext_sore_throat_viral", classify_sore_throat,
     dict(age_months=48, sore_throat=True, runny_nose=True, cough=True),
     "A 4-year-old has a sore throat with a runny nose and a cough. How do I classify?"),
    ("ext_growth", classify_growth,
     dict(age_months=30, losing_weight=True),
     "A 30-month-old's weight curve shows the child has been losing weight. Classification?"),
    ("ext_malaria", classify_fever_malaria,
     dict(age_months=36, fever=True, malaria_risk="high", malaria_test="positive"),
     "A 3-year-old in a high malaria-risk area has fever, and the rapid malaria test is positive. No danger signs, no stiff neck. Classify?"),
    ("ext_fever_no_malaria", classify_fever_malaria,
     dict(age_months=36, fever=True, malaria_test="negative"),
     "A 3-year-old has fever and the malaria test came back negative. No danger signs. How do I classify?"),
    ("ext_very_severe_febrile", classify_fever_malaria,
     dict(age_months=24, fever=True, stiff_neck=True),
     "A 2-year-old has fever and a stiff neck. What's the classification?"),
    ("ext_severe_measles", classify_measles,
     dict(age_months=24, generalised_rash=True, cough_or_runny_nose_or_red_eyes=True, clouding_of_cornea=True),
     "A 2-year-old has a generalised rash with red eyes and cough, and there is clouding of the cornea. Classify?"),
    ("ext_measles", classify_measles,
     dict(age_months=24, generalised_rash=True, cough_or_runny_nose_or_red_eyes=True),
     "A 2-year-old has a generalised rash with a runny nose and red eyes, no eye or mouth complications. Classification?"),
    ("ext_severe_anaemia", classify_anaemia,
     dict(age_months=24, severe_palmar_pallor=True),
     "A 2-year-old has severe palmar pallor. How do I classify?"),
    ("ext_anaemia", classify_anaemia,
     dict(age_months=24, some_palmar_pallor=True),
     "A 2-year-old has some palmar pallor. Classification?"),
    ("ext_complicated_sam", classify_malnutrition,
     dict(age_months=24, oedema_of_both_feet=True),
     "A 2-year-old has oedema of both feet. What's the classification?"),
    ("ext_moderate_malnutrition", classify_malnutrition,
     dict(age_months=24, muac_mm=120),
     "A 2-year-old has a MUAC of 120 mm, no oedema. How do I classify?"),
    ("ext_hiv_confirmed", classify_hiv,
     dict(age_months=24, hiv_test="positive"),
     "A 2-year-old has a positive HIV test. How do I classify the HIV status?"),
    ("ext_hiv_exposed", classify_hiv,
     dict(age_months=6, infant_on_arv_prophylaxis=True),
     "A 6-month-old is on ARV prophylaxis. What is the HIV classification?"),
    ("ext_hiv_suspected", classify_hiv,
     dict(age_months=30, hiv_oral_thrush=True, hiv_low_weight=True, hiv_pneumonia_now=True),
     "A 30-month-old has oral thrush, a low weight, and pneumonia now. How do I classify for HIV?"),
    ("ext_hiv_possible", classify_hiv,
     dict(age_months=24, mother_hiv_positive=True),
     "A 2-year-old's mother is HIV-positive, and the child has no features of HIV. How do I classify?"),
    ("ext_hiv_unlikely", classify_hiv,
     dict(age_months=24, hiv_test="negative", breastfeeding_stopped_ge_6wk=True),
     "A 2-year-old has a negative HIV test and stopped breastfeeding more than 6 weeks before the test, with no features of HIV. Classify?"),
]

YOUNG_INFANT = [
    ("yi_very_severe", classify_yi_bacterial,
     dict(age_days=20, bulging_fontanelle=True),
     "A 20-day-old baby has a bulging fontanelle. How do I classify this young infant?"),
    ("yi_local_infection", classify_yi_bacterial,
     dict(age_days=20, umbilicus_red_only=True),
     "A 20-day-old baby has a red umbilicus, no danger signs. Classification?"),
    ("yi_severe_jaundice", classify_yi_jaundice,
     dict(age_days=0, jaundice=True, jaundice_onset_under_24h=True),
     "A baby less than a day old is already jaundiced. What's the classification?"),
    ("yi_jaundice", classify_yi_jaundice,
     dict(age_days=5, jaundice=True),
     "A 5-day-old baby has jaundice that appeared after the first day; palms and soles are not yellow. Classify?"),
    ("yi_severe_dehydration", classify_yi_diarrhoea,
     dict(age_days=20, diarrhoea=True),
     "A 20-day-old baby has watery diarrhoea. How do I classify this young infant's dehydration?"),
    ("yi_some_dehydration", classify_yi_diarrhoea,
     dict(age_days=45, diarrhoea=True, restless_or_irritable=True, skin_pinch_slow=True),
     "A 6-week-old baby has diarrhoea, is restless and irritable, and the skin pinch goes back slowly. Classify?"),
    ("yi_no_dehydration", classify_yi_diarrhoea,
     dict(age_days=45, diarrhoea=True),
     "A 6-week-old baby has diarrhoea but no sunken eyes, normal skin pinch, and is feeding. Classification?"),
    ("yi_dysentery", classify_yi_diarrhoea,
     dict(age_days=30, diarrhoea=True, blood_in_stool=True),
     "A 1-month-old baby has diarrhoea with blood in the stool. How do I classify?"),
    ("yi_congenital_priority", classify_yi_congenital,
     dict(age_days=2, cleft_lip_or_palate=True),
     "A 2-day-old baby has a cleft lip and palate. What's the classification?"),
    ("yi_congenital_abnormal", classify_yi_congenital,
     dict(age_days=2, club_foot=True),
     "A 2-day-old baby has a club foot, no priority signs. Classification?"),
    ("yi_congenital_syphilis", classify_yi_congenital,
     dict(age_days=2, mother_rpr_positive_untreated=True),
     "A 2-day-old baby whose mother had a positive RPR that was never treated. How do I classify?"),
]


def _build(entries, assessment_cls, label_text, group):
    tasks = []
    for tid, clf, inp, prompt in entries:
        result = clf(assessment_cls(**inp))
        if result is None:
            raise SystemExit(f"{tid}: classifier returned None for input {inp} -- fix the case")
        label = result.condition_label
        display = label_text[label]
        tasks.append({
            "id": tid,
            "prompt": prompt,
            "expected_classification": result.classification.value,
            "expected_condition_label": label,
            "expected_label_text": display,
            "group": group,
            "input": inp,
            "must_not_mention": LEAKS,
        })
    return tasks


def main() -> int:
    tasks = (_build(EXTENDED, ExtendedAssessment, EXTENDED_LABEL_TEXT, "extended")
             + _build(YOUNG_INFANT, YoungInfantAssessment, YI_LABEL_TEXT, "young_infant"))
    doc = {
        "_note": ("NL eval for the --include-extended branches (extended sick-child + young-infant). "
                  "Ground truth is engine-derived: each prompt's structured `input` is run through the "
                  "deterministic classifier to fill expected_classification + expected_condition_label. "
                  "UNREVIEWED clinical logic -- this measures the model against the engine, not the "
                  "engine against a clinician."),
        "_scorer": "eval/scoring/model_sacc_scorer.py --tasks eval/tasks/imci_vignettes_extended.json --check-label",
        "examples": tasks,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    by_sev = {}
    for t in tasks:
        by_sev[t["expected_classification"]] = by_sev.get(t["expected_classification"], 0) + 1
    print(f"wrote {OUT}: {len(tasks)} vignettes ({by_sev})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
