"""
The critical correctness property for the expert policy: the PARTIAL set of
fields it chooses to ask about must always be sufficient to reproduce the
SAME classification assess() would give if every field had been asked.
This is what would break silently if expert_policy.py's branch mirroring
ever drifted from imci_protocol.assess()'s actual logic.
"""

import random

import pytest

from src.tools.imci_protocol import ChildAssessment, assess
from src.hrm.expert_policy import simulate_dialogue


def _reconstruct_from_trajectory(trajectory: list[dict], age_months: int) -> ChildAssessment:
    """Builds a ChildAssessment using ONLY the fields the dialogue actually
    asked about, leaving everything else at dataclass defaults -- this is
    the real test of whether the policy asked enough."""
    known = {}
    for step in trajectory:
        if step["question_field"] is not None:
            known[step["question_field"]] = step["answer"]
    return ChildAssessment(age_months=age_months, **known)


def _assert_dialogue_matches_full_assessment(ground_truth: ChildAssessment, chief_complaint: str = ""):
    trajectory = simulate_dialogue(ground_truth, chief_complaint)
    reconstructed = _reconstruct_from_trajectory(trajectory, ground_truth.age_months)

    full_result = assess(ground_truth)
    partial_result = assess(reconstructed)

    assert partial_result.classification == full_result.classification, (
        f"Dialogue-gathered fields produced a DIFFERENT classification than "
        f"full assessment. Full: {full_result.classification}, "
        f"Partial: {partial_result.classification}. "
        f"Fields asked: {[s['question_field'] for s in trajectory if s['question_field']]}"
    )
    assert partial_result.condition_label == full_result.condition_label


class TestExpertPolicyConsistency:
    def test_danger_sign_case(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=18, danger_signs_present=["convulsions"])
        )

    def test_stridor_case(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=12, cough_or_difficulty_breathing=True, stridor_when_calm=True)
        )

    def test_mild_cough_case(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=36, cough_or_difficulty_breathing=True, respiratory_rate_per_min=25)
        )

    def test_pneumonia_via_chest_indrawing(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=24, cough_or_difficulty_breathing=True, chest_indrawing=True)
        )

    def test_pneumonia_via_fast_breathing_only(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=8, cough_or_difficulty_breathing=True, respiratory_rate_per_min=55)
        )

    def test_severe_dehydration_case(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(
                age_months=24, diarrhea=True,
                sunken_eyes=True, skin_pinch_goes_back_very_slowly=True,
            )
        )

    def test_some_dehydration_case(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(
                age_months=36, diarrhea=True,
                child_restless_or_irritable=True, drinking_eagerly_thirsty=True,
            )
        )

    def test_no_dehydration_single_sign(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=24, diarrhea=True, sunken_eyes=True)
        )

    def test_fever_with_stiff_neck(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=30, fever=True, stiff_neck=True)
        )

    def test_fever_without_stiff_neck(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=30, fever=True)
        )

    def test_mastoiditis(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=40, ear_problem=True, tender_swelling_behind_ear=True)
        )

    def test_chronic_ear_infection(self):
        _assert_dialogue_matches_full_assessment(
            ChildAssessment(age_months=40, ear_problem=True, ear_discharge_days=20)
        )

    def test_no_symptoms_at_all(self):
        _assert_dialogue_matches_full_assessment(ChildAssessment(age_months=24))

    def test_chief_complaint_ordering_does_not_change_outcome(self):
        # Same ground truth, different chief-complaint text (changes which
        # of the 4 top-level questions gets asked FIRST) -- the final
        # classification must be IDENTICAL either way, because assess()
        # always evaluates categories in a FIXED order (cough, diarrhea,
        # fever, ear) regardless of what order a health worker asked about
        # things. An earlier version of this policy got this wrong (stopped
        # based on asking order, not assess()'s fixed precedence) -- see
        # report/REPORT_TEMPLATE_NOTES.md for how that was caught (a trained
        # model's end-to-end validation accuracy was lower than its action-
        # level accuracy, which led back to this bug in the ground-truth
        # policy itself, not just the model's approximation of it).
        gt = ChildAssessment(age_months=24, diarrhea=True, fever=True, stiff_neck=True)
        true_result = assess(gt)  # cough=False, diarrhea=True -> assess() returns from diarrhea branch

        result_diarrhea_first = assess(
            _reconstruct_from_trajectory(simulate_dialogue(gt, "diarrhea"), gt.age_months)
        )
        result_fever_first = assess(
            _reconstruct_from_trajectory(simulate_dialogue(gt, "fever"), gt.age_months)
        )
        assert result_diarrhea_first.condition_label == true_result.condition_label
        assert result_fever_first.condition_label == true_result.condition_label
        assert true_result.condition_label == "no_dehydration"  # diarrhea checked before fever in assess()'s fixed order

    @pytest.mark.parametrize("seed", range(150))
    def test_randomized_fuzz(self, seed):
        """Random combinations of fields AND chief-complaint text (varying
        asking order), checking the consistency property holds broadly.
        An earlier version of this test fixed chief_complaint="" for every
        case, which meant asking order always matched assess()'s fixed
        order and could never have caught the precedence bug described in
        test_chief_complaint_ordering_does_not_change_outcome -- fixed by
        randomizing chief_complaint here too."""
        rng = random.Random(seed)
        gt = ChildAssessment(
            age_months=rng.randint(2, 59),
            danger_signs_present=(
                [rng.choice(["convulsions", "unable_to_drink_or_breastfeed", "vomits_everything",
                             "lethargic_or_unconscious", "convulsing_now"])]
                if rng.random() < 0.15 else []
            ),
            cough_or_difficulty_breathing=rng.random() < 0.4,
            respiratory_rate_per_min=rng.randint(15, 70) if rng.random() < 0.7 else None,
            chest_indrawing=rng.random() < 0.2,
            stridor_when_calm=rng.random() < 0.1,
            diarrhea=rng.random() < 0.4,
            diarrhea_days=rng.randint(1, 20) if rng.random() < 0.5 else None,
            blood_in_stool=rng.random() < 0.15,
            child_lethargic_or_unconscious=rng.random() < 0.1,
            child_restless_or_irritable=rng.random() < 0.2,
            sunken_eyes=rng.random() < 0.2,
            not_able_to_drink_or_drinking_poorly=rng.random() < 0.15,
            drinking_eagerly_thirsty=rng.random() < 0.2,
            skin_pinch_goes_back_very_slowly=rng.random() < 0.15,
            skin_pinch_goes_back_slowly=rng.random() < 0.2,
            fever=rng.random() < 0.4,
            stiff_neck=rng.random() < 0.1,
            ear_problem=rng.random() < 0.3,
            ear_pain=rng.random() < 0.3,
            ear_discharge_days=rng.randint(1, 25) if rng.random() < 0.4 else None,
            tender_swelling_behind_ear=rng.random() < 0.1,
        )
        chief_complaint = rng.choice([
            "", "child has a cough", "child has diarrhea", "child has a fever",
            "child's ear hurts", "having trouble breathing", "loose stools", "feels hot",
        ])
        _assert_dialogue_matches_full_assessment(gt, chief_complaint=chief_complaint)