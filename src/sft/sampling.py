"""
Case sampling for SFT data: makes _random_case()'s draws safe to turn into prose.

The HRM pipeline feeds ChildAssessments to an encoder that only ever sees a
state vector, so a clinically impossible case (sunken eyes but no diarrhoea)
is harmless there -- the model never reads it as a claim about a child.
Prose is different: every field we verbalize becomes a statement the model is
trained to reason from. Two distinct problems follow, and they need fixing at
two different points in the pipeline:

    repair_coherence  ->  assess  ->  prune_to_decisive_branch  ->  verbalize

1. BEFORE assess(): _random_case() draws every field independently, which
   produces cases that contradict themselves. Most are harmless once pruned,
   but one is not -- see repair_coherence().

2. AFTER assess(): assess() short-circuits (danger signs -> cough -> diarrhoea
   -> fever -> ear, each returning before the next), so a cough+fever case is
   labelled purely by cough. Verbalizing the fever would teach "mention of
   fever -> ignore it". See prune_to_decisive_branch().

Deliberately does NOT modify scripts/generate_hrm_training_data._random_case:
it is imported by scripts/validate_hrm.py and kept in sync with
tests/unit/test_expert_policy.py's fuzz test. We wrap it instead.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from src.tools.imci_protocol import ChildAssessment, Classification, TriageResult, assess

# Field groups, one per branch of assess(). Order mirrors assess()'s own
# branch order; if assess() changes, this must change with it.
COUGH_FIELDS = (
    "cough_or_difficulty_breathing",
    "respiratory_rate_per_min",
    "chest_indrawing",
    "stridor_when_calm",
)
DIARRHEA_FIELDS = (
    "diarrhea",
    "diarrhea_days",
    "blood_in_stool",
    "child_lethargic_or_unconscious",
    "child_restless_or_irritable",
    "sunken_eyes",
    "not_able_to_drink_or_drinking_poorly",
    "drinking_eagerly_thirsty",
    "skin_pinch_goes_back_very_slowly",
    "skin_pinch_goes_back_slowly",
)
FEVER_FIELDS = ("fever", "fever_days", "stiff_neck")
EAR_FIELDS = ("ear_problem", "ear_pain", "ear_discharge_days", "tender_swelling_behind_ear")

# condition_label -> the branch that produced it. Every label assess() can
# return must appear here; _branch_for() raises otherwise, so a new
# classification in imci_protocol.py fails loudly instead of silently
# pruning to the wrong branch.
LABEL_TO_BRANCH = {
    "very_severe_disease": "danger",
    "severe_pneumonia_or_very_severe_disease": "cough",
    "pneumonia": "cough",
    "cough_or_cold": "cough",
    "severe_dehydration": "diarrhea",
    "some_dehydration": "diarrhea",
    "no_dehydration": "diarrhea",
    "very_severe_febrile_disease": "fever",
    "fever_unspecified_malaria_not_assessed": "fever",
    "mastoiditis": "ear",
    "chronic_ear_infection": "ear",
    "acute_ear_infection": "ear",
    "no_ear_infection": "ear",
    "no_classification_matched": "none",
}
BRANCH_FIELDS = {
    "danger": (),
    "cough": COUGH_FIELDS,
    "diarrhea": DIARRHEA_FIELDS,
    "fever": FEVER_FIELDS,
    "ear": EAR_FIELDS,
    "none": (),
}

ALL_LABELS = tuple(LABEL_TO_BRANCH)


def _branch_for(result: TriageResult) -> str:
    try:
        return LABEL_TO_BRANCH[result.condition_label]
    except KeyError:
        raise ValueError(
            f"Unknown condition_label {result.condition_label!r}. imci_protocol.assess() "
            f"gained a classification that src/sft/sampling.py's LABEL_TO_BRANCH does not "
            f"map. Add it -- do not guess a branch, or vignettes will state signs the "
            f"classification never considered."
        ) from None


def repair_coherence(child: ChildAssessment, rng: random.Random) -> ChildAssessment:
    """
    Fixes contradictions that would become MISLABELLED prose.

    The one that matters: _random_case() draws child_lethargic_or_unconscious
    (a dehydration sign) independently of danger_signs_present, but "lethargic
    or unconscious" is ALSO a general danger sign. Measured over 20k draws,
    7.8% of cases are lethargic without the danger sign flagged, and in 7.0%
    assess() therefore returns severe_dehydration (or similar) instead of
    very_severe_disease. Verbalize that faithfully and the model learns
    "the child is lethargic -> not an urgent referral" -- a systematic
    UNDER-triage signal on the most safety-critical sign in IMCI. Pruning
    cannot fix it: lethargy is one of the two decisive dehydration signs, so
    it has to appear in the vignette.

    Real IMCI checks general danger signs first and would raise both
    classifications; this scaffold returns one primary, so promoting to
    very_severe_disease is both correct and the safe direction (over-triage).

    The other repairs are physical impossibilities: a child cannot both drink
    eagerly and be unable to drink, and a skin pinch cannot go back both
    "slowly" and "very slowly" (IMCI records the worst rung, so we keep the
    more severe one rather than choosing at random).
    """
    danger = list(child.danger_signs_present)

    if child.child_lethargic_or_unconscious and "lethargic_or_unconscious" not in danger:
        danger.append("lethargic_or_unconscious")

    fixes = {"danger_signs_present": danger}

    if child.drinking_eagerly_thirsty and child.not_able_to_drink_or_drinking_poorly:
        # No safe rung here -- these are opposite observations, not degrees.
        if rng.random() < 0.5:
            fixes["drinking_eagerly_thirsty"] = False
        else:
            fixes["not_able_to_drink_or_drinking_poorly"] = False

    if child.skin_pinch_goes_back_slowly and child.skin_pinch_goes_back_very_slowly:
        fixes["skin_pinch_goes_back_slowly"] = False  # ">2s" subsumes "some delay"

    return replace(child, **fixes)


def sample_coherent_case(rng: random.Random) -> ChildAssessment:
    """_random_case() + repair_coherence(). The entry point for SFT sampling."""
    from scripts.generate_hrm_training_data import _random_case

    return repair_coherence(_random_case(rng), rng)


def prune_to_decisive_branch(child: ChildAssessment, result: TriageResult) -> ChildAssessment:
    """
    Returns a copy holding only the fields assess() actually consulted to reach
    `result`, so the vignette can never mention a sign the answer ignored.

    Safe by construction: assess() returns from exactly one branch, and never
    reads a later branch's fields once it does. Zeroing those fields therefore
    cannot change the classification -- which prune_preserves_label() asserts.
    """
    branch = _branch_for(result)
    keep = set(BRANCH_FIELDS[branch])

    pruned = ChildAssessment(
        age_months=child.age_months,
        danger_signs_present=list(child.danger_signs_present),
    )
    for f in keep:
        setattr(pruned, f, getattr(child, f))
    return pruned


def required_fields_for(child: ChildAssessment) -> set[str]:
    """
    Fields the vignette MUST state, given an already-pruned case.

    The rule is asymmetric: never omit a true field, but freely omit false
    ones. assess() treats an absent field as False, so omitting a negative is
    label-preserving -- and real prompts never enumerate every negative. A
    vignette that omits a POSITIVE sign, by contrast, trains the model to
    invent that sign from nothing.

    `danger_signs_present` is reported per-sign rather than as a field name,
    since only the listed signs are claims about the child.
    """
    required = set()
    if child.danger_signs_present:
        required.add("danger_signs_present")
    for f in COUGH_FIELDS + DIARRHEA_FIELDS + FEVER_FIELDS + EAR_FIELDS:
        if getattr(child, f):  # truthy: True, or a non-None non-zero number
            required.add(f)
    return required


def prune_preserves_label(child: ChildAssessment, result: TriageResult) -> bool:
    """Guard: pruning must not change the classification. Used in tests and
    asserted during generation -- a violation means BRANCH_FIELDS has drifted
    from assess()."""
    pruned = prune_to_decisive_branch(child, result)
    after = assess(pruned)
    return (
        after.classification == result.classification
        and after.condition_label == result.condition_label
    )


def sample_stratified(
    rng: random.Random,
    target_per_label: dict[str, int],
    max_attempts: int | None = None,
) -> list[tuple[ChildAssessment, TriageResult]]:
    """
    Rejection-samples sample_coherent_case() to a target distribution over
    condition_label, returning PRUNED cases paired with their result.

    Necessary because _random_case()'s hand-set Bernoullis are wildly skewed:
    cough_or_cold/no_dehydration/pneumonia each land ~16%, while mastoiditis
    is 0.4% and chronic_ear_infection / very_severe_febrile_disease ~1%. A
    model trained on that never learns the rare-but-urgent branches, which are
    exactly the ones that matter clinically.

    Rare labels dominate the runtime (mastoiditis needs ~250 draws each), so
    the attempt budget scales with the rarest target rather than the total.
    """
    for label in target_per_label:
        if label not in LABEL_TO_BRANCH:
            raise ValueError(f"Unknown condition_label in target: {label!r}")

    if max_attempts is None:
        max_attempts = max(500_000, 400 * sum(target_per_label.values()))

    remaining = Counter(target_per_label)
    out: list[tuple[ChildAssessment, TriageResult]] = []
    attempts = 0

    while attempts < max_attempts and any(v > 0 for v in remaining.values()):
        attempts += 1
        child = sample_coherent_case(rng)
        result = assess(child)
        label = result.condition_label
        if remaining.get(label, 0) <= 0:
            continue

        if not prune_preserves_label(child, result):
            raise AssertionError(
                f"Pruning changed the classification for {label!r}. BRANCH_FIELDS in "
                f"src/sft/sampling.py has drifted from imci_protocol.assess()'s branches."
            )
        out.append((prune_to_decisive_branch(child, result), result))
        remaining[label] -= 1

    unfilled = {k: v for k, v in remaining.items() if v > 0}
    if unfilled:
        raise RuntimeError(
            f"sample_stratified exhausted {max_attempts} attempts with quotas unfilled: "
            f"{unfilled}. These labels are rarer than the attempt budget allows -- raise "
            f"max_attempts or lower their targets."
        )

    rng.shuffle(out)
    return out
