"""
Deterministic IMCI drug/fluid dosing lookup.

This is the SINGLE source of every dose the fine-tuned model is trained to say.
The whole submission rests on "correctness comes from code matching a published
standard, not from a model's recollection" -- dosing is the most dangerous place
to violate that, so no dose is ever hand-written into training prose. It is
looked up here from data/imci_2022/dosing_tables.json and rendered verbatim.

Like imci_protocol.assess() and src/sft/answer.py, this RAISES on anything it
cannot resolve (unknown drug, no matching weight band) rather than guessing --
a silent wrong dose is the failure mode we will not allow.

SAFETY: the dosing tables are UNREVIEWED and gated behind
scripts/generate_sft_data.py --include-extended. Dosing is not graded by the
Sacc scorer; it needs clinician sign-off before it trains anything. See
data/imci_2022/dosing_tables.json "_meta.review_checklist".
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_TABLES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "imci_2022" / "dosing_tables.json"


class DosingError(ValueError):
    """Raised when a dose cannot be resolved. Never swallow this -- a missing
    dose must surface, not become a blank or a guess in training data."""


class DoseNotApplicable(DosingError):
    """The drug is known but simply not dosed at this age/weight (e.g. vitamin A
    under 6 months, artemether-lumefantrine under 5 kg). This is a clinical fact,
    not a bug -- a caller may legitimately skip the drug. Distinct from a bare
    DosingError (unknown drug / missing renderer / bad table), which must not be
    swallowed."""


@lru_cache(maxsize=1)
def _tables() -> dict:
    return json.loads(_TABLES_PATH.read_text())


def drugs() -> tuple[str, ...]:
    return tuple(_tables()["drugs"])


def key_for(drug: str) -> str:
    """'weight' or 'age' -- which dimension the drug is dosed by."""
    return _tables()["drugs"][drug]["key"]


def bands_for(drug: str) -> list[dict]:
    return _tables()["drugs"][drug]["bands"]


def _in_range(value: float, lo, hi) -> bool:
    """Half-open [lo, hi): lo inclusive, hi exclusive; null bound = open."""
    if lo is not None and value < lo:
        return False
    if hi is not None and value >= hi:
        return False
    return True


def _band_matches(band: dict, key: str, weight_kg, age_months) -> bool:
    if key == "weight":
        if weight_kg is None:
            return False
        return _in_range(weight_kg, band.get("weight_kg_min"), band.get("weight_kg_max"))
    if key == "age":
        if age_months is None:
            return False
        return _in_range(age_months, band.get("age_months_min"), band.get("age_months_max"))
    raise DosingError(f"Unknown band key {key!r} -- expected 'weight' or 'age'.")


def dose_for(drug: str, weight_kg: float | None = None, age_months: int | None = None) -> dict:
    """Return the matching dosing band for `drug`.

    Each drug is keyed by 'weight' or 'age' (see dosing_tables.json). Pass the
    dimension the drug needs. Raises DosingError if the drug is unknown, the
    keyed argument is missing, or no band covers the value.
    """
    table = _tables()["drugs"]
    if drug not in table:
        raise DosingError(
            f"Unknown drug {drug!r}. Known: {sorted(table)}. Add it to "
            f"dosing_tables.json (with a source) rather than inventing a dose."
        )
    entry = table[drug]
    key = entry["key"]
    if key == "weight" and weight_kg is None:
        raise DosingError(f"{drug!r} is weight-keyed; pass weight_kg.")
    if key == "age" and age_months is None:
        raise DosingError(f"{drug!r} is age-keyed; pass age_months.")
    matches = [b for b in entry["bands"] if _band_matches(b, key, weight_kg, age_months)]
    if not matches:
        val = weight_kg if key == "weight" else age_months
        raise DoseNotApplicable(f"{drug!r} is not dosed at {key}={val} (outside its bands).")
    # Bands are disjoint on their key by construction; if two match, the table is wrong.
    if len(matches) > 1:
        raise DosingError(f"Overlapping {drug!r} bands match {key}: {matches}. Fix the table.")
    band = dict(matches[0])
    band["_drug"] = drug
    band["_route"] = entry.get("route")
    band["_frequency"] = entry.get("frequency")
    band["_indication"] = entry.get("indication")
    return band


# Per-drug rendering. Each returns a single clinician-facing dose sentence built
# ONLY from table fields -- no free text about the number. A drug without a
# renderer raises, so a new table entry can't slip into data unrendered.
def _render_amoxicillin(b: dict) -> str:
    return (f"amoxicillin {b['tablet_250mg']} x 250 mg tablet "
            f"(or {b['syrup_250mg_per_5ml_ml']} ml of 250 mg/5 ml syrup), two times daily for 5 days")


def _render_paracetamol(b: dict) -> str:
    return f"paracetamol {b['tablet_100mg']} x 100 mg tablet, every 6 hours until fever or pain settles"


def _render_zinc(b: dict) -> str:
    return f"zinc {b['tablets_daily']} x 20 mg tablet daily for 14 days"


def _render_vitamin_a(b: dict) -> str:
    return f"vitamin A {b['iu']:,} IU"


def _render_ceftriaxone(b: dict) -> str:
    return f"ceftriaxone {b['dose_mg']} mg ({b['volume_ml']} ml) IM"


def _render_diazepam(b: dict) -> str:
    return f"diazepam {b['dose_mg']} mg ({b['volume_ml']} ml) per rectum"


def _render_al(b: dict) -> str:
    return f"artemether-lumefantrine {b['tablets_per_dose']} tablet(s) two times daily for 3 days, with food"


def _render_mebendazole(b: dict) -> str:
    return f"mebendazole {b['dose_mg']} mg as a single dose"


def _render_ors_plan_a(b: dict) -> str:
    return (f"ORS {b['ml_after_each_loose_stool_min']}-{b['ml_after_each_loose_stool_max']} ml "
            f"after each loose stool")


_RENDERERS = {
    "amoxicillin": _render_amoxicillin,
    "paracetamol": _render_paracetamol,
    "zinc": _render_zinc,
    "vitamin_a": _render_vitamin_a,
    "ceftriaxone_im": _render_ceftriaxone,
    "diazepam_rectal": _render_diazepam,
    "artemether_lumefantrine": _render_al,
    "mebendazole": _render_mebendazole,
    "ors_plan_a_extra_fluid": _render_ors_plan_a,
}


def format_dose(drug: str, weight_kg: float | None = None, age_months: int | None = None) -> str:
    """Human-readable dose sentence for the training answer. Raises on unknown
    drug or missing renderer -- the same loud-failure contract as answer.py."""
    band = dose_for(drug, weight_kg=weight_kg, age_months=age_months)
    if drug not in _RENDERERS:
        raise DosingError(
            f"No renderer for {drug!r}. Add one to imci_dosing._RENDERERS rather "
            f"than letting an unrendered dose reach training data."
        )
    return _RENDERERS[drug](band)


def follow_up_for(label: str) -> str | None:
    """Follow-up interval for a classification label, or None if not listed."""
    return _tables()["follow_up_intervals"].get(label)
