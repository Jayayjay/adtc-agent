"""
Boundary tests for src/tools/imci_dosing.py.

The contract that matters: a resolvable dose is correct and stable, and an
UNresolvable one RAISES (never a blank or a guess). These tests pin the band
edges, because an off-by-one on a weight band is a dosing error.
"""

import json
from pathlib import Path

import pytest

from src.tools import imci_dosing as d
from src.tools.imci_dosing import DosingError, dose_for, format_dose, follow_up_for, drugs

TABLES = json.loads((Path(__file__).resolve().parents[2] / "data" / "imci_2022" / "dosing_tables.json").read_text())


def _resolve_first_band(name):
    """dose_for on a value inside a drug's first band, dispatching on its key."""
    entry = TABLES["drugs"][name]
    first = entry["bands"][0]
    if entry["key"] == "weight":
        return dose_for(name, weight_kg=(first.get("weight_kg_min") or 0) + 0.1)
    return dose_for(name, age_months=(first.get("age_months_min") or 0) + 0.1)


def test_tables_load_and_every_drug_resolves_somewhere():
    assert drugs()  # non-empty
    for name in drugs():
        assert _resolve_first_band(name)["_drug"] == name


def test_bands_are_disjoint_on_their_key():
    """No value may match two bands of the same drug (dose_for raises if so)."""
    for name in drugs():
        key = TABLES["drugs"][name]["key"]
        for v in [round(x * 0.5, 1) for x in range(0, 130)]:  # 0 .. 64.5
            try:
                if key == "weight":
                    dose_for(name, weight_kg=v)
                else:
                    dose_for(name, age_months=v)
            except DosingError as e:
                assert "Overlapping" not in str(e), f"{name}: {e}"


def test_band_edge_is_exclusive_upper():
    # amoxicillin bands: [4,10), [10,14), [14,19)
    assert format_dose("amoxicillin", 9.9).startswith("amoxicillin 1")
    assert format_dose("amoxicillin", 10).startswith("amoxicillin 2")   # 10 -> next band
    assert format_dose("amoxicillin", 13.9).startswith("amoxicillin 2")
    assert format_dose("amoxicillin", 14).startswith("amoxicillin 3")   # 14 -> next band
    with pytest.raises(DosingError):
        dose_for("amoxicillin", 19)  # top band max is exclusive -> raises, not a wrong dose


def test_ceftriaxone_bands():
    assert format_dose("ceftriaxone_im", 8) == "ceftriaxone 625 mg (2.5 ml) IM"
    assert format_dose("ceftriaxone_im", 3.5) == "ceftriaxone 312 mg (1.25 ml) IM"
    assert format_dose("ceftriaxone_im", 20) == "ceftriaxone 1500 mg (5.5 ml) IM"  # open-ended top band
    with pytest.raises(DosingError):
        dose_for("ceftriaxone_im", 3.0)  # below the lowest band


def test_unknown_drug_raises():
    with pytest.raises(DosingError):
        dose_for("aspirin", 10)
    with pytest.raises(DosingError):
        format_dose("aspirin", 10)


def test_every_drug_has_a_renderer_or_is_intentionally_excluded():
    # Renderers are the only path into training data; if a drug has no renderer,
    # format_dose must raise rather than silently drop it.
    for name in drugs():
        entry = TABLES["drugs"][name]
        first = entry["bands"][0]
        if entry["key"] == "weight":
            kw = {"weight_kg": (first.get("weight_kg_min") or 0) + 0.1}
        else:
            kw = {"age_months": (first.get("age_months_min") or 0) + 0.1}
        if name in d._RENDERERS:
            assert isinstance(format_dose(name, **kw), str)
        else:
            with pytest.raises(DosingError):
                format_dose(name, **kw)


def test_follow_up_lookup():
    assert follow_up_for("pneumonia") == "3 days"
    assert follow_up_for("moderate_acute_malnutrition") == "30 days"
    assert follow_up_for("not_a_label") is None
