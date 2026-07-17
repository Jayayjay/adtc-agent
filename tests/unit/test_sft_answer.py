"""
The critical property for answer rendering: nothing that describes THIS REPO
may reach the training data.

assess()'s reasoning carries Python list reprs and a line citing
src/tools/imci_protocol.py; TriageResult.disclaimer's default cites it too;
and two recommended_action strings talk about "this scaffold". A model trained
on those recites our source paths to the judge panel. The exhaustive fuzz test
below is the real guarantee -- it renders every reachable classification and
greps the output.
"""

import random
import re

import pytest

from src.sft.answer import (
    DISCLAIMERS,
    LABEL_TEXT,
    humanize_reasoning,
    render_answer,
    render_header,
    render_reasoning,
)
from src.sft.sampling import ALL_LABELS, sample_coherent_case, sample_stratified
from src.tools.imci_protocol import Classification, assess

# Anything that betrays the implementation rather than describing the child.
LEAK = re.compile(
    r"imci_protocol|\.py\b|scaffold|module docstring|not modeled|\['|'\]|_[a-z]+_[a-z]+",
    re.I,
)


class TestNoLeaks:
    def test_every_reachable_answer_is_clean(self):
        """Renders all 14 classifications many times over and greps."""
        rng = random.Random(0)
        target = {label: 40 for label in ALL_LABELS}
        for child, result in sample_stratified(rng, target):
            answer = render_answer(result, rng)
            m = LEAK.search(answer)
            assert m is None, (
                f"leak {m.group(0)!r} in answer for {result.condition_label}:\n{answer}"
            )

    def test_danger_sign_names_are_humanized(self):
        """The raw list is "['lethargic_or_unconscious']" -- snake_case field
        names must never survive into prose."""
        line = "General danger sign(s) present: ['convulsions', 'lethargic_or_unconscious']"
        out = humanize_reasoning(line)
        assert "lethargic_or_unconscious" not in out
        assert "[" not in out and "'" not in out
        assert "lethargic or unconscious" in out
        assert "convulsions" in out

    def test_fever_scaffold_note_keeps_substance_drops_the_path(self):
        line = (
            "NOTE: this scaffold's fever branch is simplified and does NOT model the real "
            "IMCI malaria-risk/RDT-dependent algorithm -- see src/tools/imci_protocol.py "
            "module docstring."
        )
        out = humanize_reasoning(line)
        assert "imci_protocol" not in out and "scaffold" not in out
        assert "malaria" in out.lower()  # the honest limitation survives

    def test_dehydration_list_is_prose(self):
        line = "Some dehydration signs (>=2 of 4 required): ['sunken eyes', 'restless or irritable']"
        out = humanize_reasoning(line)
        assert "[" not in out and "'" not in out
        assert "sunken eyes" in out and "restless or irritable" in out

    def test_leaky_actions_are_overridden(self):
        rng = random.Random(0)
        for label in ("fever_unspecified_malaria_not_assessed", "no_classification_matched"):
            (_, result), = sample_stratified(random.Random(1), {label: 1})
            answer = render_answer(result, rng)
            assert "not modeled" not in answer and "scaffold" not in answer

    def test_default_disclaimer_never_used(self):
        """TriageResult.disclaimer's default cites the source file; we must
        substitute our own."""
        (_, result), = sample_stratified(random.Random(2), {"pneumonia": 1})
        assert "imci_protocol.py" in result.disclaimer  # the trap is still there
        answer = render_answer(result, random.Random(0))
        assert "imci_protocol.py" not in answer
        assert any(d in answer for d in DISCLAIMERS)


class TestHeader:
    def test_format_is_rigid_and_parseable(self):
        rng = random.Random(0)
        pattern = re.compile(r"^CLASSIFICATION: (SEVERE|MODERATE|MILD) — .+ \(IMCI (pink|yellow|green): .+\)$")
        for child, result in sample_stratified(rng, {l: 5 for l in ALL_LABELS}):
            header = render_header(result)
            assert pattern.match(header), f"unparseable header: {header!r}"

    def test_colour_matches_severity(self):
        rng = random.Random(0)
        expect = {
            Classification.SEVERE: "pink",
            Classification.MODERATE: "yellow",
            Classification.MILD: "green",
        }
        for child, result in sample_stratified(rng, {l: 5 for l in ALL_LABELS}):
            assert expect[result.classification] in render_header(result)

    def test_header_is_first_line_of_answer(self):
        (_, result), = sample_stratified(random.Random(0), {"pneumonia": 1})
        answer = render_answer(result, random.Random(0))
        assert answer.splitlines()[0] == render_header(result)

    def test_every_label_has_display_text(self):
        assert set(LABEL_TEXT) == set(ALL_LABELS)

    def test_no_label_contains_the_field_separator(self):
        """" — " separates severity from label in the rigid header. A label
        containing one makes the line ambiguous to split on."""
        for label, text in LABEL_TEXT.items():
            assert " — " not in text, f"{label!r} display text embeds the header separator"

    def test_header_splits_unambiguously(self):
        rng = random.Random(0)
        for child, result in sample_stratified(rng, {l: 3 for l in ALL_LABELS}):
            severity, _, rest = render_header(result).removeprefix("CLASSIFICATION: ").partition(" — ")
            assert severity in ("SEVERE", "MODERATE", "MILD")
            assert rest.endswith(")")


class TestUnknownInputFailsLoudly:
    def test_unknown_reasoning_raises(self):
        with pytest.raises(ValueError, match="Unrecognised reasoning line"):
            humanize_reasoning("Some brand new reasoning assess() started emitting.")

    def test_unknown_label_raises(self):
        class Fake:
            condition_label = "brand_new_label"
            classification = Classification.MILD

        with pytest.raises(ValueError, match="LABEL_TEXT"):
            render_header(Fake())


class TestBody:
    def test_secondary_findings_rendered(self):
        rng = random.Random(0)
        found = False
        for child, result in sample_stratified(rng, {"no_dehydration": 200}):
            if result.secondary_findings:
                assert "ALSO:" in render_answer(result, rng)
                found = True
                break
        assert found, "no case with secondary_findings — dysentery/persistent diarrhoea unreachable?"

    def test_acknowledged_extra_appears(self):
        (_, result), = sample_stratified(random.Random(0), {"pneumonia": 1})
        answer = render_answer(result, random.Random(0), acknowledged_extra="She also has a fever.")
        assert "NOTE: She also has a fever." in answer

    def test_disclaimer_varies_across_renders(self):
        (_, result), = sample_stratified(random.Random(0), {"pneumonia": 1})
        rng = random.Random(0)
        seen = {render_answer(result, rng).splitlines()[-1] for _ in range(50)}
        assert len(seen) > 1, "disclaimer never varies — it will be memorised as a blob"
