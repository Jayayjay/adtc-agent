"""
Vignette + answer rendering for the extended branches.

Same two properties as the core verbalizer, plus one specific to this module:
the extended classifiers' reasoning was written as prose, so render_extended_
answer emits it directly rather than through humanize_reasoning(). The no-leak
test verifies that trust was warranted -- an unreviewed reasoning string that
named a source file would otherwise ship straight to the weights.
"""

import random
import re

import pytest

from src.sft.extended_protocol import (
    EXTENDED_LABELS,
    ExtendedAssessment,
    MALARIA_RISK_HIGH,
    TEST_POSITIVE,
    classify_anaemia,
    classify_fever_malaria,
    classify_malnutrition,
    classify_measles,
)
from src.sft.extended_verbalize import (
    EXTENDED_LABEL_TEXT,
    render_extended_answer,
    verbalize_extended,
)
from src.sft.verbalize import VignetteStyle

LEAK = re.compile(r"imci_protocol|extended_protocol|\.py\b|scaffold|\['|_[a-z]+_[a-z]+", re.I)
HDR = re.compile(r"^CLASSIFICATION: (SEVERE|MODERATE|MILD) — .+ \(IMCI (pink|yellow|green): [^)]+\)$", re.M)


def _all_labels_have_display_text():
    return set(EXTENDED_LABELS) <= set(EXTENDED_LABEL_TEXT)


class TestLabelText:
    def test_every_label_is_displayable(self):
        assert _all_labels_have_display_text(), set(EXTENDED_LABELS) - set(EXTENDED_LABEL_TEXT)

    def test_no_label_embeds_the_header_separator(self):
        for text in EXTENDED_LABEL_TEXT.values():
            assert " — " not in text


class TestAnswerRendering:
    def _sample_results(self):
        A = lambda **k: ExtendedAssessment(age_months=k.pop("age", 24), **k)
        return [
            classify_fever_malaria(A(fever=True, danger_signs_present=["convulsions"])),
            classify_fever_malaria(A(fever=True, malaria_risk=MALARIA_RISK_HIGH, malaria_test=TEST_POSITIVE)),
            classify_measles(A(measles_within_3_months=True, clouding_of_cornea=True)),
            classify_measles(A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True)),
            classify_anaemia(A(severe_palmar_pallor=True)),
            classify_malnutrition(A(oedema_of_both_feet=True)),
            classify_malnutrition(A(muac_mm=120)),
        ]

    def test_header_is_rigid_and_parseable(self):
        rng = random.Random(0)
        for r in self._sample_results():
            ans = render_extended_answer(r, rng)
            assert HDR.search(ans), f"unparseable: {ans.splitlines()[0]}"

    def test_header_severity_matches_classification(self):
        rng = random.Random(0)
        for r in self._sample_results():
            ans = render_extended_answer(r, rng)
            assert ans.startswith(f"CLASSIFICATION: {r.classification.value.upper()} ")

    def test_no_leaks_in_answers(self):
        rng = random.Random(0)
        for r in self._sample_results():
            ans = render_extended_answer(r, rng)
            m = LEAK.search(ans)
            assert m is None, f"leak {m.group(0)!r} in:\n{ans}"


class TestVignettes:
    def _cases(self):
        A = lambda **k: ExtendedAssessment(age_months=k.pop("age", 24), **k)
        return [
            A(fever=True, malaria_risk=MALARIA_RISK_HIGH, malaria_test=TEST_POSITIVE, fever_days=3),
            A(generalised_rash=True, cough_or_runny_nose_or_red_eyes=True, pus_draining_from_eye=True),
            A(severe_palmar_pallor=True),
            A(muac_mm=108, appetite_test_passed=False),
            A(oedema_of_both_feet=True, danger_signs_present=["lethargic_or_unconscious"]),
        ]

    def test_every_style_renders_nonempty(self):
        rng = random.Random(0)
        for a in self._cases():
            for style in VignetteStyle:
                text = verbalize_extended(a, rng, style)
                assert text.strip(), f"empty vignette for {style.value}"

    def test_no_unfilled_slots(self):
        rng = random.Random(0)
        for a in self._cases():
            for style in VignetteStyle:
                text = verbalize_extended(a, rng, style)
                for slot in ("{P}", "{Po}", "{Pp}", "{C}"):
                    assert slot not in text, f"unfilled {slot} in {style.value}: {text}"

    def test_positive_findings_are_mentioned(self):
        """A malaria-positive case must state the positive test; a MUAC case
        must state the number. These are what the classification hinges on."""
        rng = random.Random(0)
        a = ExtendedAssessment(age_months=24, fever=True, malaria_risk=MALARIA_RISK_HIGH,
                               malaria_test=TEST_POSITIVE)
        texts = [verbalize_extended(a, rng, s).lower() for s in VignetteStyle]
        assert all("positive" in t or "rdt" in t for t in texts)

        muac = ExtendedAssessment(age_months=24, muac_mm=108)
        texts = [verbalize_extended(muac, rng, s) for s in VignetteStyle]
        assert all("108" in t for t in texts)

    def test_pronouns_are_consistent(self):
        rng = random.Random(0)
        for a in self._cases():
            for style in VignetteStyle:
                if style is VignetteStyle.SMS_REFERRAL:
                    continue
                text = verbalize_extended(a, rng, style)
                fem = re.search(r"\b(she|her)\b", text, re.I)
                masc = re.search(r"\b(he|him|his)\b", text, re.I)
                assert not (fem and masc), f"mixed pronouns in {style.value}: {text}"
