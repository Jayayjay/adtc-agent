"""
Tests for src/sft/treatment.py -- the classification -> drugs mapping and the
DOSING line renderer. Doses themselves are tested in test_imci_dosing; here we
pin that (a) every core label is mapped (closed set), and (b) render_dosing
raises on an unknown label rather than emitting a doseless answer.
"""

import pytest

from src.sft.answer import LABEL_TEXT
from src.sft.treatment import LABEL_TO_DRUGS, render_dosing


def test_every_core_label_is_mapped():
    """A core classification with no entry would render doseless silently."""
    missing = [l for l in LABEL_TEXT if l not in LABEL_TO_DRUGS]
    assert not missing, f"core labels missing a treatment mapping: {missing}"


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        render_dosing("malaria_not_a_core_label", 10.0, 24)


def test_empty_mapping_returns_none():
    # cough_or_cold has no specific drug dose -> None, not a crash or blank
    assert render_dosing("cough_or_cold", 10.0, 24) is None
    assert render_dosing("no_ear_infection", 10.0, 24) is None


def test_pneumonia_doses_amoxicillin():
    out = render_dosing("pneumonia", 8.0, 8)
    assert out.startswith("amoxicillin 1 x 250 mg tablet")


def test_severe_fever_doses_ceftriaxone_and_paracetamol():
    out = render_dosing("very_severe_febrile_disease", 8.0, 8)
    assert "ceftriaxone" in out and "paracetamol" in out
    assert ";" in out  # two drugs joined


def test_no_dehydration_doses_ors_and_zinc_by_age():
    out = render_dosing("no_dehydration", 12.0, 30)
    assert "ORS" in out and "zinc" in out
