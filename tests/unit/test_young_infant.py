"""
Young-infant (0-2 month) chart. Safety-critical and UNREVIEWED; these tests pin
the logic against the chart so review has something concrete to check and a
later edit cannot silently move a threshold. Over-triage is tolerable,
under-triage is not.
"""

from src.sft.young_infant import (
    YOUNG_INFANT_LABELS,
    YoungInfantAssessment,
    classify_yi_bacterial,
    classify_yi_congenital,
    classify_yi_diarrhoea,
    classify_yi_jaundice,
)
from src.sft.young_infant_verbalize import YI_LABEL_TEXT
from src.tools.imci_protocol import Classification


def A(**kw):
    kw.setdefault("age_days", 20)
    return YoungInfantAssessment(**kw)


class TestBacterial:
    def test_danger_sign_is_very_severe(self):
        for kw in (dict(convulsions=True), dict(apnoea=True), dict(fast_breathing_over_60=True),
                   dict(breathing_under_30=True), dict(bulging_fontanelle=True),
                   dict(hypothermia_under_35_5=True), dict(only_moves_when_stimulated=True)):
            r = classify_yi_bacterial(A(**kw))
            assert r.classification is Classification.SEVERE
            assert r.condition_label == "yi_very_severe_disease"

    def test_local_signs_are_moderate(self):
        for kw in (dict(eye_discharge_purulent_or_sticky=True), dict(umbilicus_red_only=True),
                   dict(skin_pustules_few=True)):
            assert classify_yi_bacterial(A(**kw)).condition_label == "yi_local_bacterial_infection"

    def test_severe_beats_local(self):
        r = classify_yi_bacterial(A(convulsions=True, umbilicus_red_only=True))
        assert r.condition_label == "yi_very_severe_disease"

    def test_no_signs_is_none_label(self):
        assert classify_yi_bacterial(A()).condition_label == "yi_no_bacterial_infection"


class TestJaundice:
    def test_no_jaundice_returns_none(self):
        assert classify_yi_jaundice(A()) is None

    def test_under_24h_is_severe(self):
        r = classify_yi_jaundice(A(age_days=0, jaundice=True, jaundice_onset_under_24h=True))
        assert r.condition_label == "yi_severe_jaundice"

    def test_yellow_palms_is_severe(self):
        r = classify_yi_jaundice(A(jaundice=True, yellow_palms_and_soles=True))
        assert r.condition_label == "yi_severe_jaundice"

    def test_after_24h_no_palms_is_moderate(self):
        r = classify_yi_jaundice(A(age_days=5, jaundice=True))
        assert r.condition_label == "yi_jaundice"


class TestDiarrhoea:
    def test_no_diarrhoea_returns_none(self):
        assert classify_yi_diarrhoea(A()) is None

    def test_blood_is_severe_dysentery(self):
        r = classify_yi_diarrhoea(A(diarrhoea=True, blood_in_stool=True))
        assert r.condition_label == "yi_dysentery"
        assert r.classification is Classification.SEVERE

    def test_14_days_is_severe_persistent(self):
        r = classify_yi_diarrhoea(A(age_days=40, diarrhoea=True, diarrhoea_days=16))
        assert r.condition_label == "yi_severe_persistent_diarrhoea"

    def test_under_one_month_is_always_severe(self):
        """A young infant under 1 month with diarrhoea is severe dehydration."""
        r = classify_yi_diarrhoea(A(age_days=20, diarrhoea=True))
        assert r.condition_label == "yi_severe_dehydration"

    def test_two_severe_signs_is_severe(self):
        r = classify_yi_diarrhoea(A(age_days=45, diarrhoea=True,
                                    lethargic_or_unconscious=True, sunken_eyes=True))
        assert r.condition_label == "yi_severe_dehydration"

    def test_some_dehydration(self):
        r = classify_yi_diarrhoea(A(age_days=45, diarrhoea=True,
                                    restless_or_irritable=True, skin_pinch_slow=True))
        assert r.condition_label == "yi_some_dehydration"

    def test_no_dehydration(self):
        r = classify_yi_diarrhoea(A(age_days=45, diarrhoea=True))
        assert r.condition_label == "yi_no_dehydration"


class TestCongenital:
    def test_no_signs_returns_none(self):
        assert classify_yi_congenital(A()) is None

    def test_priority_signs_are_severe(self):
        for kw in (dict(cleft_lip_or_palate=True), dict(imperforate_anus=True),
                   dict(macrocephaly=True), dict(very_low_birth_weight=True)):
            r = classify_yi_congenital(A(**kw))
            assert r.classification is Classification.SEVERE
            assert r.condition_label == "yi_congenital_priority"

    def test_other_abnormal_is_moderate(self):
        assert classify_yi_congenital(A(club_foot=True)).condition_label == "yi_congenital_abnormal_signs"

    def test_maternal_syphilis_path(self):
        r = classify_yi_congenital(A(mother_rpr_positive_untreated=True))
        assert r.condition_label == "yi_possible_congenital_syphilis"

    def test_priority_beats_other(self):
        r = classify_yi_congenital(A(cleft_lip_or_palate=True, club_foot=True))
        assert r.condition_label == "yi_congenital_priority"


class TestLabelInventory:
    def test_every_label_has_display_text(self):
        assert all(l in YI_LABEL_TEXT for l in YOUNG_INFANT_LABELS)

    def test_no_label_leaks_untested(self):
        seen = set()
        for r in (
            classify_yi_bacterial(A(convulsions=True)),
            classify_yi_bacterial(A(umbilicus_red_only=True)),
            classify_yi_bacterial(A()),
            classify_yi_jaundice(A(age_days=0, jaundice=True, jaundice_onset_under_24h=True)),
            classify_yi_jaundice(A(age_days=5, jaundice=True)),
            classify_yi_diarrhoea(A(diarrhoea=True, blood_in_stool=True)),
            classify_yi_diarrhoea(A(age_days=40, diarrhoea=True, diarrhoea_days=16)),
            classify_yi_diarrhoea(A(age_days=20, diarrhoea=True)),
            classify_yi_diarrhoea(A(age_days=45, diarrhoea=True, restless_or_irritable=True, skin_pinch_slow=True)),
            classify_yi_diarrhoea(A(age_days=45, diarrhoea=True)),
            classify_yi_congenital(A(cleft_lip_or_palate=True)),
            classify_yi_congenital(A(club_foot=True)),
            classify_yi_congenital(A(mother_rpr_positive_untreated=True)),
        ):
            if r:
                seen.add(r.condition_label)
        assert seen <= set(YOUNG_INFANT_LABELS)
