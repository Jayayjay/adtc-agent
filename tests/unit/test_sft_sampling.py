"""
The critical correctness property for SFT sampling: a vignette must never be
able to state a sign that the classification it is paired with ignored, and
must never state a danger sign alongside a non-severe classification.

The second one is the dangerous direction. _random_case() draws
child_lethargic_or_unconscious independently of danger_signs_present, so 7.0%
of raw draws describe a lethargic child that assess() classifies as something
other than very_severe_disease. Training on that teaches under-triage on the
single most safety-critical sign in IMCI. These tests exist to keep that at
zero.
"""

import random

import pytest

from src.sft.sampling import (
    ALL_LABELS,
    DIARRHEA_FIELDS,
    EAR_FIELDS,
    FEVER_FIELDS,
    prune_preserves_label,
    prune_to_decisive_branch,
    repair_coherence,
    required_fields_for,
    sample_coherent_case,
    sample_stratified,
)
from src.tools.imci_protocol import ChildAssessment, Classification, assess


class TestRepairCoherence:
    def test_lethargy_becomes_a_danger_sign(self):
        """The bug this module exists for: lethargic child, no danger sign
        flagged -> assess() would say severe_dehydration, not urgent referral."""
        child = ChildAssessment(
            age_months=18,
            diarrhea=True,
            child_lethargic_or_unconscious=True,
            sunken_eyes=True,  # 2-of-4 -> severe_dehydration without the repair
        )
        assert assess(child).condition_label == "severe_dehydration"

        repaired = repair_coherence(child, random.Random(0))
        assert "lethargic_or_unconscious" in repaired.danger_signs_present
        assert assess(repaired).condition_label == "very_severe_disease"
        assert assess(repaired).classification is Classification.SEVERE

    def test_lethargy_promotes_even_without_diarrhoea(self):
        """A lethargic child is very severe regardless of which branch would
        otherwise have caught them."""
        child = ChildAssessment(age_months=30, child_lethargic_or_unconscious=True)
        repaired = repair_coherence(child, random.Random(0))
        assert assess(repaired).condition_label == "very_severe_disease"

    def test_does_not_duplicate_an_existing_danger_sign(self):
        child = ChildAssessment(
            age_months=10,
            danger_signs_present=["lethargic_or_unconscious"],
            child_lethargic_or_unconscious=True,
        )
        repaired = repair_coherence(child, random.Random(0))
        assert repaired.danger_signs_present.count("lethargic_or_unconscious") == 1

    def test_does_not_invent_a_danger_sign(self):
        child = ChildAssessment(age_months=10, diarrhea=True, sunken_eyes=True)
        repaired = repair_coherence(child, random.Random(0))
        assert repaired.danger_signs_present == []

    def test_drink_contradiction_resolved(self):
        child = ChildAssessment(
            age_months=10,
            diarrhea=True,
            drinking_eagerly_thirsty=True,
            not_able_to_drink_or_drinking_poorly=True,
        )
        repaired = repair_coherence(child, random.Random(1))
        assert not (
            repaired.drinking_eagerly_thirsty and repaired.not_able_to_drink_or_drinking_poorly
        )

    def test_skin_pinch_contradiction_keeps_the_worse_rung(self):
        child = ChildAssessment(
            age_months=10,
            diarrhea=True,
            skin_pinch_goes_back_slowly=True,
            skin_pinch_goes_back_very_slowly=True,
        )
        repaired = repair_coherence(child, random.Random(0))
        assert repaired.skin_pinch_goes_back_very_slowly
        assert not repaired.skin_pinch_goes_back_slowly

    def test_does_not_mutate_the_input(self):
        child = ChildAssessment(age_months=10, child_lethargic_or_unconscious=True)
        repair_coherence(child, random.Random(0))
        assert child.danger_signs_present == []


class TestPruning:
    def test_cough_case_drops_later_branch_fields(self):
        """assess() returns from the cough branch without ever reading fever or
        ear fields -- so a vignette must not mention them."""
        child = ChildAssessment(
            age_months=6,
            cough_or_difficulty_breathing=True,
            respiratory_rate_per_min=56,
            fever=True,
            stiff_neck=True,
            ear_problem=True,
        )
        result = assess(child)
        assert result.condition_label == "pneumonia"

        pruned = prune_to_decisive_branch(child, result)
        assert pruned.cough_or_difficulty_breathing
        assert pruned.respiratory_rate_per_min == 56
        assert not pruned.fever
        assert not pruned.stiff_neck
        assert not pruned.ear_problem
        assert assess(pruned).condition_label == "pneumonia"

    def test_danger_sign_case_keeps_only_danger_signs(self):
        child = ChildAssessment(
            age_months=6,
            danger_signs_present=["convulsions"],
            cough_or_difficulty_breathing=True,
            diarrhea=True,
        )
        pruned = prune_to_decisive_branch(child, assess(child))
        assert pruned.danger_signs_present == ["convulsions"]
        assert not pruned.cough_or_difficulty_breathing
        assert not pruned.diarrhea

    def test_unknown_label_raises_rather_than_guessing(self):
        class FakeResult:
            condition_label = "some_new_classification"

        with pytest.raises(ValueError, match="LABEL_TO_BRANCH"):
            prune_to_decisive_branch(ChildAssessment(age_months=10), FakeResult())


class TestRequiredFields:
    def test_positive_signs_are_required_negatives_are_not(self):
        child = ChildAssessment(
            age_months=6,
            cough_or_difficulty_breathing=True,
            respiratory_rate_per_min=56,
            chest_indrawing=False,
        )
        required = required_fields_for(child)
        assert "cough_or_difficulty_breathing" in required
        assert "respiratory_rate_per_min" in required
        assert "chest_indrawing" not in required

    def test_danger_signs_required_when_present(self):
        child = ChildAssessment(age_months=6, danger_signs_present=["convulsions"])
        assert "danger_signs_present" in required_fields_for(child)
        assert required_fields_for(ChildAssessment(age_months=6)) == set()

    def test_stating_only_required_fields_preserves_the_label(self):
        """The guard invariant, from the other side: a vignette that states
        every true field and omits the false ones must round-trip."""
        rng = random.Random(11)
        for _ in range(2000):
            child = sample_coherent_case(rng)
            result = assess(child)
            pruned = prune_to_decisive_branch(child, result)

            required = required_fields_for(pruned)
            reconstructed = ChildAssessment(
                age_months=pruned.age_months,
                danger_signs_present=(
                    list(pruned.danger_signs_present)
                    if "danger_signs_present" in required
                    else []
                ),
            )
            for f in required - {"danger_signs_present"}:
                setattr(reconstructed, f, getattr(pruned, f))

            assert assess(reconstructed).condition_label == result.condition_label, (
                f"Stating only the required fields changed the classification: "
                f"{assess(reconstructed).condition_label} != {result.condition_label}"
            )


class TestFuzzInvariants:
    def test_no_undertriaged_lethargy_after_repair(self):
        """THE test. Raw _random_case() puts this at ~7.0%; it must be 0."""
        rng = random.Random(42)
        offenders = 0
        for _ in range(20_000):
            child = sample_coherent_case(rng)
            result = assess(child)
            if (
                child.child_lethargic_or_unconscious
                and result.classification is not Classification.SEVERE
            ):
                offenders += 1
        assert offenders == 0, f"{offenders} lethargic children not classified SEVERE"

    def test_pruning_never_changes_the_label(self):
        rng = random.Random(7)
        for _ in range(5000):
            child = sample_coherent_case(rng)
            assert prune_preserves_label(child, assess(child))

    def test_pruned_cases_state_no_contradictions(self):
        rng = random.Random(3)
        for _ in range(5000):
            child = sample_coherent_case(rng)
            pruned = prune_to_decisive_branch(child, assess(child))
            assert not (
                pruned.drinking_eagerly_thirsty and pruned.not_able_to_drink_or_drinking_poorly
            )
            assert not (
                pruned.skin_pinch_goes_back_slowly and pruned.skin_pinch_goes_back_very_slowly
            )

    def test_pruned_case_never_states_a_sign_from_an_unreached_branch(self):
        """e.g. a cough-classified case must not carry ear_pain, which
        assess() never looked at."""
        rng = random.Random(5)
        for _ in range(5000):
            child = sample_coherent_case(rng)
            result = assess(child)
            pruned = prune_to_decisive_branch(child, result)
            if result.condition_label in ("pneumonia", "cough_or_cold"):
                for f in DIARRHEA_FIELDS + FEVER_FIELDS + EAR_FIELDS:
                    assert not getattr(pruned, f), f"cough case still carries {f}"


class TestStratification:
    def test_hits_quotas_for_rare_labels(self):
        """mastoiditis is 0.4% of raw draws; balancing is the whole point."""
        rng = random.Random(0)
        target = {"mastoiditis": 30, "chronic_ear_infection": 30, "pneumonia": 30}
        got = sample_stratified(rng, target)
        counts = {}
        for _, result in got:
            counts[result.condition_label] = counts.get(result.condition_label, 0) + 1
        assert counts == target

    def test_returns_pruned_cases(self):
        rng = random.Random(0)
        got = sample_stratified(rng, {"pneumonia": 20})
        for child, _ in got:
            assert not child.fever and not child.ear_problem

    def test_unknown_label_rejected(self):
        with pytest.raises(ValueError, match="Unknown condition_label"):
            sample_stratified(random.Random(0), {"malaria": 5})

    def test_unfillable_quota_fails_loudly(self):
        with pytest.raises(RuntimeError, match="quotas unfilled"):
            sample_stratified(random.Random(0), {"mastoiditis": 10}, max_attempts=50)
