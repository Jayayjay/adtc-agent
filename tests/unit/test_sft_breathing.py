"""
Targeted borderline fast-breathing cases exist to fix a measured weakness of the
first fine-tune: it under-triaged pneumonia when a raised respiratory rate was
the ONLY sign (called it cough/cold). These tests pin the properties that make
the bucket useful — correct labels, both sides of the cutoff, both age bands,
and no other confounding sign.
"""

import collections
import random

from src.sft.sampling import sample_breathing_threshold_cases
from src.tools.imci_protocol import _fast_breathing_threshold, assess


def _cases(n=2000, seed=0):
    return sample_breathing_threshold_cases(random.Random(seed), n)


def test_labels_are_exactly_the_threshold_rule():
    for child, result in _cases():
        thr = _fast_breathing_threshold(child.age_months)
        expected = "pneumonia" if child.respiratory_rate_per_min >= thr else "cough_or_cold"
        assert result.condition_label == expected, (
            f"age {child.age_months}mo RR {child.respiratory_rate_per_min} "
            f"(thr {thr}) -> {result.condition_label}, expected {expected}"
        )


def test_rate_is_the_only_pneumonia_sign():
    """If indrawing or stridor were present, the case wouldn't isolate the
    threshold — which is the whole point of this bucket."""
    for child, _ in _cases():
        assert not child.chest_indrawing
        assert not child.stridor_when_calm
        assert child.cough_or_difficulty_breathing
        assert child.respiratory_rate_per_min is not None


def test_both_sides_of_the_cutoff_are_present():
    labels = collections.Counter(r.condition_label for _, r in _cases())
    assert set(labels) == {"pneumonia", "cough_or_cold"}
    # ~50/50 so the model learns the boundary, not "high rate -> pneumonia"
    lo, hi = min(labels.values()), max(labels.values())
    assert hi - lo <= 2, f"unbalanced sides: {dict(labels)}"


def test_both_age_bands_are_well_represented():
    """The <12mo threshold (50) and >=12mo threshold (40) are different rules;
    a random age would starve the young band."""
    band = collections.Counter(
        "<12" if c.age_months < 12 else ">=12" for c, _ in _cases()
    )
    assert band["<12"] > 700 and band[">=12"] > 700, dict(band)


def test_rates_straddle_the_threshold_tightly():
    """Just over / just under — borderline is where the model fails, not RR 90."""
    for child, result in _cases():
        delta = child.respiratory_rate_per_min - _fast_breathing_threshold(child.age_months)
        assert -12 <= delta <= 12


def test_rates_are_physiologically_sane():
    for child, _ in _cases():
        assert child.respiratory_rate_per_min >= 20


def test_pruning_preserved_the_label():
    """Cases come back pruned; re-running assess() must give the same answer."""
    for child, result in _cases(500):
        assert assess(child).condition_label == result.condition_label
