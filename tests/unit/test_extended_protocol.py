"""
The extended protocol encodes the branches imci_protocol.py declares unmodelled:
malaria-risk fever, measles, anaemia, acute malnutrition.

These classifications are safety-critical and UNREVIEWED by a clinician. The
tests below pin the logic against the WHO IMCI 2014 chart so that (a) review has
something concrete to check line by line, and (b) a later edit cannot silently
change a threshold. They are not a substitute for clinical sign-off -- they
verify the code does what THIS MODULE CLAIMS, not that the claim is correct.

The direction that matters: over-triage is tolerable, under-triage is not. The
severity-ordering tests exist to catch a refactor that lets a danger sign be
missed.
"""

import pytest

from src.sft.extended_protocol import (
    EXTENDED_LABELS,
    MALARIA_RISK_HIGH,
    MALARIA_RISK_LOW,
    MALARIA_RISK_NONE,
    TEST_NEGATIVE,
    TEST_NOT_DONE,
    TEST_POSITIVE,
    ExtendedAssessment,
    HIV_TEST_NEGATIVE,
    HIV_TEST_POSITIVE,
    classify_anaemia,
    classify_dysentery,
    classify_fever_malaria,
    classify_growth,
    classify_hiv,
    classify_malnutrition,
    classify_measles,
    classify_persistent_diarrhoea,
    classify_sore_throat,
    classify_wheeze,
    has_measles,
)
from src.tools.imci_protocol import Classification


def A(**kw):
    kw.setdefault("age_months", 24)
    return ExtendedAssessment(**kw)


class TestGuards:
    def test_danger_signs_str_rejected(self):
        with pytest.raises(TypeError, match="list"):
            ExtendedAssessment(age_months=24, danger_signs_present="convulsions")

    def test_bad_malaria_risk_rejected(self):
        with pytest.raises(ValueError, match="malaria_risk"):
            ExtendedAssessment(age_months=24, malaria_risk="maybe")

    def test_bad_malaria_test_rejected(self):
        with pytest.raises(ValueError, match="malaria_test"):
            ExtendedAssessment(age_months=24, malaria_test="pending")


class TestFeverMalaria:
    def test_no_fever_returns_none(self):
        assert classify_fever_malaria(A(fever=False)) is None

    def test_danger_sign_is_very_severe_regardless_of_test(self):
        r = classify_fever_malaria(A(fever=True, malaria_test=TEST_NEGATIVE,
                                     danger_signs_present=["lethargic_or_unconscious"]))
        assert r.classification is Classification.SEVERE
        assert r.condition_label == "very_severe_febrile_disease"

    def test_stiff_neck_is_very_severe(self):
        r = classify_fever_malaria(A(fever=True, stiff_neck=True))
        assert r.classification is Classification.SEVERE

    def test_positive_test_is_malaria(self):
        r = classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_HIGH,
                                     malaria_test=TEST_POSITIVE))
        assert r.condition_label == "malaria"
        assert r.classification is Classification.MODERATE

    def test_negative_test_is_not_malaria(self):
        r = classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_HIGH,
                                     malaria_test=TEST_NEGATIVE))
        assert r.condition_label == "fever_no_malaria"
        assert r.classification is Classification.MILD

    def test_no_risk_no_travel_no_test_needed(self):
        r = classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_NONE,
                                     malaria_test=TEST_NOT_DONE))
        assert r.condition_label == "fever_no_malaria"

    def test_risk_but_untested_demands_a_test(self):
        """The one case that is NOT a classification: the chart requires a test
        first, so the honest output is 'do the test', not a made-up label."""
        r = classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_HIGH,
                                     malaria_test=TEST_NOT_DONE))
        assert r.condition_label == "fever_malaria_test_required"
        assert "test" in r.recommended_action.lower()

    def test_travel_history_makes_it_testable(self):
        r = classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_NONE,
                                     travel_to_malaria_area=True, malaria_test=TEST_NOT_DONE))
        assert r.condition_label == "fever_malaria_test_required"

    def test_low_risk_untested_demands_a_test(self):
        r = classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_LOW,
                                     malaria_test=TEST_NOT_DONE))
        assert r.condition_label == "fever_malaria_test_required"

    def test_long_fever_flagged_as_secondary(self):
        r = classify_fever_malaria(A(fever=True, fever_days=8, malaria_risk=MALARIA_RISK_NONE))
        assert any("7 day" in s for s in r.secondary_findings)


class TestMeasles:
    def test_case_definition(self):
        assert has_measles(A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True))
        assert has_measles(A(measles_within_3_months=True))
        assert not has_measles(A(generalised_rash=True))  # rash alone is not measles
        assert not has_measles(A(cough_or_runny_nose_or_red_eyes=True))

    def test_no_measles_returns_none(self):
        assert classify_measles(A()) is None

    def test_corneal_clouding_is_severe(self):
        r = classify_measles(A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True,
                               clouding_of_cornea=True))
        assert r.classification is Classification.SEVERE
        assert r.condition_label == "severe_complicated_measles"

    def test_deep_mouth_ulcers_is_severe(self):
        r = classify_measles(A(measles_within_3_months=True, deep_or_extensive_mouth_ulcers=True))
        assert r.classification is Classification.SEVERE

    def test_danger_sign_is_severe(self):
        r = classify_measles(A(measles_within_3_months=True,
                               danger_signs_present=["convulsions"]))
        assert r.classification is Classification.SEVERE

    def test_eye_pus_is_moderate(self):
        r = classify_measles(A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True,
                               pus_draining_from_eye=True))
        assert r.classification is Classification.MODERATE
        assert r.condition_label == "measles_with_eye_or_mouth_complications"

    def test_uncomplicated_is_mild(self):
        r = classify_measles(A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True))
        assert r.classification is Classification.MILD
        assert r.condition_label == "measles"
        assert "vitamin a" in r.recommended_action.lower()


class TestAnaemia:
    def test_severe_pallor(self):
        r = classify_anaemia(A(severe_palmar_pallor=True))
        assert r.classification is Classification.SEVERE
        assert r.condition_label == "severe_anaemia"

    def test_some_pallor(self):
        r = classify_anaemia(A(some_palmar_pallor=True))
        assert r.classification is Classification.MODERATE
        assert "iron" in r.recommended_action.lower()

    def test_no_pallor(self):
        r = classify_anaemia(A())
        assert r.classification is Classification.MILD
        assert r.condition_label == "no_anaemia"

    def test_severe_beats_some_when_both_flagged(self):
        r = classify_anaemia(A(severe_palmar_pallor=True, some_palmar_pallor=True))
        assert r.classification is Classification.SEVERE


class TestMalnutrition:
    def test_oedema_is_always_severe(self):
        r = classify_malnutrition(A(oedema_of_both_feet=True))
        assert r.classification is Classification.SEVERE
        assert r.condition_label == "complicated_severe_acute_malnutrition"

    def test_muac_below_115_is_severe(self):
        r = classify_malnutrition(A(age_months=24, muac_mm=110, appetite_test_passed=True))
        assert r.condition_label == "uncomplicated_severe_acute_malnutrition"

    def test_muac_115_to_125_is_moderate(self):
        r = classify_malnutrition(A(age_months=24, muac_mm=120))
        assert r.condition_label == "moderate_acute_malnutrition"
        assert r.classification is Classification.MODERATE

    def test_muac_ignored_under_6_months(self):
        """MUAC is not valid below 6 months -- a low reading there must not
        drive a classification."""
        r = classify_malnutrition(A(age_months=4, muac_mm=100))
        assert r.condition_label == "no_acute_malnutrition"

    def test_wfh_below_minus3_is_severe(self):
        r = classify_malnutrition(A(wfh_z_score=-3.5, appetite_test_passed=True))
        assert "severe_acute_malnutrition" in r.condition_label

    def test_severe_wasting_with_failed_appetite_is_complicated(self):
        r = classify_malnutrition(A(muac_mm=110, appetite_test_passed=False))
        assert r.condition_label == "complicated_severe_acute_malnutrition"
        assert r.classification is Classification.SEVERE

    def test_severe_wasting_with_danger_sign_is_complicated(self):
        r = classify_malnutrition(A(muac_mm=110, appetite_test_passed=True,
                                    danger_signs_present=["vomits_everything"]))
        assert r.condition_label == "complicated_severe_acute_malnutrition"

    def test_uncomplicated_severe_is_moderate_severity(self):
        """Uncomplicated SAM is managed at home (OTP) -- moderate severity, not
        a referral."""
        r = classify_malnutrition(A(muac_mm=110, appetite_test_passed=True))
        assert r.classification is Classification.MODERATE
        assert "rutf" in r.recommended_action.lower()

    def test_wellnourished(self):
        r = classify_malnutrition(A(muac_mm=140, wfh_z_score=0.0))
        assert r.condition_label == "no_acute_malnutrition"
        assert r.classification is Classification.MILD


class TestWheeze:
    def test_no_wheeze_returns_none(self):
        assert classify_wheeze(A()) is None

    def test_wheeze_is_moderate(self):
        assert classify_wheeze(A(wheeze=True)).condition_label == "wheeze"

    def test_wheeze_with_danger_sign_is_severe(self):
        r = classify_wheeze(A(wheeze=True, danger_signs_present=["lethargic_or_unconscious"]))
        assert r.classification is Classification.SEVERE


class TestPersistentDiarrhoea:
    def test_under_14_days_is_none(self):
        assert classify_persistent_diarrhoea(A(diarrhoea=True, diarrhoea_days=7)) is None

    def test_14_days_no_dehydration_is_moderate(self):
        r = classify_persistent_diarrhoea(A(diarrhoea=True, diarrhoea_days=14))
        assert r.condition_label == "persistent_diarrhoea"

    def test_14_days_with_dehydration_is_severe(self):
        r = classify_persistent_diarrhoea(A(diarrhoea=True, diarrhoea_days=21, dehydration_present=True))
        assert r.classification is Classification.SEVERE
        assert r.condition_label == "severe_persistent_diarrhoea"

    def test_losing_weight_makes_it_severe(self):
        r = classify_persistent_diarrhoea(A(diarrhoea=True, diarrhoea_days=16, losing_weight=True))
        assert r.classification is Classification.SEVERE


class TestDysentery:
    def test_no_blood_is_none(self):
        assert classify_dysentery(A(diarrhoea=True)) is None

    def test_blood_over_12mo_no_dehydration_is_moderate(self):
        r = classify_dysentery(A(age_months=24, diarrhoea=True, blood_in_stool=True))
        assert r.condition_label == "dysentery"

    def test_blood_under_12mo_is_severe(self):
        r = classify_dysentery(A(age_months=8, diarrhoea=True, blood_in_stool=True))
        assert r.classification is Classification.SEVERE
        assert r.condition_label == "severe_dysentery"

    def test_blood_with_dehydration_is_severe(self):
        r = classify_dysentery(A(age_months=30, diarrhoea=True, blood_in_stool=True, dehydration_present=True))
        assert r.classification is Classification.SEVERE


class TestSoreThroat:
    def test_under_3_years_not_assessed(self):
        assert classify_sore_throat(A(age_months=24, sore_throat=True, tonsil_exudate=True)) is None

    def test_strep_signs_without_cold_is_streptococcal(self):
        r = classify_sore_throat(A(age_months=48, sore_throat=True, tonsil_exudate=True))
        assert r.condition_label == "streptococcal_sore_throat"

    def test_cough_or_runny_nose_makes_it_non_streptococcal(self):
        r = classify_sore_throat(A(age_months=48, sore_throat=True, tonsil_exudate=True, cough=True))
        assert r.condition_label == "sore_throat_non_streptococcal"

    def test_no_strep_signs_is_non_streptococcal(self):
        r = classify_sore_throat(A(age_months=48, sore_throat=True))
        assert r.condition_label == "sore_throat_non_streptococcal"


class TestGrowth:
    def test_no_problem_is_none(self):
        assert classify_growth(A()) is None

    def test_losing_weight_is_growth_problem(self):
        assert classify_growth(A(losing_weight=True)).condition_label == "growth_problem"

    def test_low_weight_for_age_is_growth_problem(self):
        assert classify_growth(A(low_weight_for_age=True)).condition_label == "growth_problem"


class TestHIV:
    def test_no_signals_returns_none(self):
        assert classify_hiv(A()) is None

    def test_positive_test_is_confirmed(self):
        assert classify_hiv(A(hiv_test=HIV_TEST_POSITIVE)).condition_label == "confirmed_hiv_infection"

    def test_on_art_is_confirmed(self):
        assert classify_hiv(A(child_on_art=True)).condition_label == "confirmed_hiv_infection"

    def test_arv_prophylaxis_is_exposed(self):
        assert classify_hiv(A(infant_on_arv_prophylaxis=True)).condition_label == "hiv_exposed"

    def test_negative_while_breastfeeding_is_exposed(self):
        r = classify_hiv(A(hiv_test=HIV_TEST_NEGATIVE, breastfeeding_at_or_near_test=True))
        assert r.condition_label == "hiv_exposed"

    def test_three_features_is_suspected(self):
        r = classify_hiv(A(hiv_oral_thrush=True, hiv_low_weight=True, hiv_pneumonia_now=True))
        assert r.condition_label == "suspected_symptomatic_hiv"

    def test_one_feature_is_possible(self):
        assert classify_hiv(A(hiv_oral_thrush=True)).condition_label == "possible_hiv_infection"

    def test_mother_positive_alone_is_possible(self):
        assert classify_hiv(A(mother_hiv_positive=True)).condition_label == "possible_hiv_infection"

    def test_negative_off_breastfeeding_no_features_is_unlikely(self):
        r = classify_hiv(A(hiv_test=HIV_TEST_NEGATIVE, breastfeeding_stopped_ge_6wk=True))
        assert r.condition_label == "hiv_infection_unlikely"
        assert r.classification is Classification.MILD

    def test_confirmed_beats_feature_count(self):
        """A positive test wins even with many features -- ordering matters."""
        r = classify_hiv(A(hiv_test=HIV_TEST_POSITIVE, hiv_oral_thrush=True,
                           hiv_low_weight=True, hiv_pneumonia_now=True))
        assert r.condition_label == "confirmed_hiv_infection"

    def test_no_hiv_tier_is_pink(self):
        """ART is not urgent -- no HIV classification should be SEVERE/pink."""
        for kw in (dict(hiv_test=HIV_TEST_POSITIVE), dict(infant_on_arv_prophylaxis=True),
                   dict(hiv_oral_thrush=True, hiv_low_weight=True, hiv_pneumonia_now=True),
                   dict(mother_hiv_positive=True)):
            assert classify_hiv(A(**kw)).classification is not Classification.SEVERE


class TestLabelInventory:
    def test_every_reachable_label_is_registered(self):
        """EXTENDED_LABELS drives the SFT stratifier; a label the functions can
        return but the tuple omits would be silently un-sampleable."""
        seen = set()
        # exhaustive enough: exercise each branch above and collect labels
        cases = [
            classify_fever_malaria(A(fever=True, danger_signs_present=["convulsions"])),
            classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_HIGH, malaria_test=TEST_POSITIVE)),
            classify_fever_malaria(A(fever=True, malaria_test=TEST_NEGATIVE)),
            classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_HIGH, malaria_test=TEST_NOT_DONE)),
            classify_measles(A(measles_within_3_months=True, clouding_of_cornea=True)),
            classify_measles(A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True, pus_draining_from_eye=True)),
            classify_measles(A(measles_within_3_months=True)),
            classify_anaemia(A(severe_palmar_pallor=True)),
            classify_anaemia(A(some_palmar_pallor=True)),
            classify_anaemia(A()),
            classify_malnutrition(A(oedema_of_both_feet=True)),
            classify_malnutrition(A(muac_mm=110, appetite_test_passed=True)),
            classify_malnutrition(A(muac_mm=120)),
            classify_malnutrition(A(muac_mm=140)),
            classify_wheeze(A(wheeze=True)),
            classify_wheeze(A(wheeze=True, danger_signs_present=["convulsions"])),
            classify_persistent_diarrhoea(A(diarrhoea=True, diarrhoea_days=21, dehydration_present=True)),
            classify_persistent_diarrhoea(A(diarrhoea=True, diarrhoea_days=21)),
            classify_dysentery(A(age_months=8, diarrhoea=True, blood_in_stool=True)),
            classify_dysentery(A(age_months=24, diarrhoea=True, blood_in_stool=True)),
            classify_sore_throat(A(age_months=48, sore_throat=True, tonsil_exudate=True)),
            classify_sore_throat(A(age_months=48, sore_throat=True, cough=True)),
            classify_growth(A(losing_weight=True)),
            classify_hiv(A(hiv_test=HIV_TEST_POSITIVE)),
            classify_hiv(A(infant_on_arv_prophylaxis=True)),
            classify_hiv(A(hiv_oral_thrush=True, hiv_low_weight=True, hiv_pneumonia_now=True)),
            classify_hiv(A(hiv_oral_thrush=True)),
            classify_hiv(A(hiv_test=HIV_TEST_NEGATIVE, breastfeeding_stopped_ge_6wk=True)),
        ]
        for r in cases:
            if r is not None:
                seen.add(r.condition_label)
        assert seen <= set(EXTENDED_LABELS), f"unregistered: {seen - set(EXTENDED_LABELS)}"
        # and the fever_no_malaria label (no-risk path) is registered too
        assert "fever_no_malaria" in EXTENDED_LABELS
