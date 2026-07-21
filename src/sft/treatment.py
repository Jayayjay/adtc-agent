"""
Maps a core IMCI classification to the specific drugs to dose, and renders the
DOSING line for the training answer.

Every dose comes from src/tools/imci_dosing (which reads the reviewed dosing
tables) -- never hand-written here. This module only decides WHICH drugs a
classification calls for; the numbers come from the tables. Like answer.py, it
RAISES on an unknown label so a new imci_protocol classification cannot get a
silently-empty or wrong treatment.

Covers the 14 core labels assess() returns. Extended classifications
(malaria/measles/anaemia/malnutrition/...) render through
src/sft/extended_verbalize.py and are dosed there.
"""

from __future__ import annotations

from src.tools import imci_dosing
from src.tools.imci_dosing import DoseNotApplicable

# Classification label -> ordered list of drug keys (imci_dosing) to dose.
# [] means "no specific drug dose applies" (home care / fluids / procedure only),
# which is a real answer, not a gap -- render_dosing returns None for it.
LABEL_TO_DRUGS: dict[str, list[str]] = {
    "very_severe_disease": ["ceftriaxone_im"],                       # pre-referral antibiotic
    "severe_pneumonia_or_very_severe_disease": ["ceftriaxone_im"],
    "pneumonia": ["amoxicillin"],
    "cough_or_cold": [],                                             # soothe throat only
    "severe_dehydration": [],                                        # Plan C IV fluids, not a fixed dose
    "some_dehydration": ["zinc"],                                    # Plan B + zinc
    "no_dehydration": ["ors_plan_a_extra_fluid", "zinc"],            # Plan A + zinc
    "very_severe_febrile_disease": ["ceftriaxone_im", "paracetamol"],
    "fever_unspecified_malaria_not_assessed": ["paracetamol"],
    "mastoiditis": ["ceftriaxone_im", "paracetamol"],
    "chronic_ear_infection": [],                                     # dry wicking, no systemic drug
    "acute_ear_infection": ["amoxicillin", "paracetamol"],
    "no_ear_infection": [],
    "no_classification_matched": [],

    # Extended (src/sft/extended_protocol.py) labels. Many map to [] because the
    # drug the branch calls for is not in dosing_tables.json (salbutamol,
    # ciprofloxacin, penicillin, iron) -- an empty list yields no DOSING line,
    # which is honest, not a gap. Doses are only rendered for drugs the reviewed
    # tables actually carry.
    "malaria": ["artemether_lumefantrine", "paracetamol"],
    "fever_no_malaria": ["paracetamol"],
    "fever_malaria_test_required": ["paracetamol"],
    "severe_complicated_measles": ["vitamin_a"],
    "measles_with_eye_or_mouth_complications": ["vitamin_a"],
    "measles": ["vitamin_a"],
    "severe_anaemia": [],
    "anaemia": [],
    "no_anaemia": [],
    "complicated_severe_acute_malnutrition": ["vitamin_a"],
    "uncomplicated_severe_acute_malnutrition": ["amoxicillin", "vitamin_a"],
    "moderate_acute_malnutrition": ["vitamin_a"],
    "no_acute_malnutrition": [],
    "wheeze": [],                                                    # salbutamol not in tables
    "wheeze_with_danger_sign": [],
    "severe_persistent_diarrhoea": ["vitamin_a"],
    "persistent_diarrhoea": ["zinc", "vitamin_a"],
    "severe_dysentery": [],
    "dysentery": [],                                                 # ciprofloxacin not in tables
    "streptococcal_sore_throat": [],                                 # penicillin not in tables
    "sore_throat_non_streptococcal": [],
    "growth_problem": [],
    # HIV: ART/ARV/cotrimoxazole regimens are country-specific and not in the
    # dosing tables; the ACTION prose names them, we emit no unreviewed dose.
    "confirmed_hiv_infection": [],
    "hiv_exposed": [],
    "suspected_symptomatic_hiv": [],
    "possible_hiv_infection": [],
    "hiv_infection_unlikely": [],
}


def render_dosing(label: str, weight_kg: float, age_months: int) -> str | None:
    """DOSING line for a classification, or None if no specific drug applies.

    Raises KeyError-style ValueError on an unknown label (closed set), and lets
    imci_dosing.DosingError propagate if a dose cannot be resolved -- a missing
    dose must surface, never become blank text in training data.
    """
    if label not in LABEL_TO_DRUGS:
        raise ValueError(
            f"No treatment mapping for classification {label!r} -- add it to "
            f"LABEL_TO_DRUGS in src/sft/treatment.py (do not let it render doseless)."
        )
    drugs = LABEL_TO_DRUGS[label]
    if not drugs:
        return None
    doses = []
    for d in drugs:
        try:
            doses.append(imci_dosing.format_dose(d, weight_kg=weight_kg, age_months=age_months))
        except DoseNotApplicable:
            # The drug isn't dosed at this age/weight (e.g. vitamin A < 6 months);
            # the ACTION prose still names it, we just omit a number we don't have.
            # A bare DosingError (unknown drug / missing renderer) still propagates.
            continue
    return "; ".join(doses) if doses else None
