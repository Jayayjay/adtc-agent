import pytest

from src.core.exceptions import ToolExecutionError
from src.tools.imci_protocol import ChildAssessment, Classification, assess
from src.tools.imci_triage_tool import IMCITriageTool


class TestIMCIProtocolEngine:
    def test_danger_sign_overrides_everything(self):
        # Even with signs that would otherwise be mild, a danger sign
        # forces severe classification -- this is the most safety-critical
        # rule in the whole engine and must never be bypassed.
        child = ChildAssessment(
            age_months=24,
            danger_signs_present=["convulsions"],
            cough_or_difficulty_breathing=True,
            respiratory_rate_per_min=20,  # would be "mild" cough on its own
        )
        result = assess(child)
        assert result.classification == Classification.SEVERE
        assert result.condition_label == "very_severe_disease"

    def test_stridor_is_severe_regardless_of_other_signs(self):
        child = ChildAssessment(
            age_months=12, cough_or_difficulty_breathing=True, stridor_when_calm=True
        )
        assert assess(child).classification == Classification.SEVERE

    def test_fast_breathing_infant_classifies_pneumonia(self):
        # Threshold per WHO IMCI 2014: 2mo-12mo = 50+ breaths/min
        child = ChildAssessment(
            age_months=8, cough_or_difficulty_breathing=True, respiratory_rate_per_min=55
        )
        result = assess(child)
        assert result.classification == Classification.MODERATE
        assert result.condition_label == "pneumonia"

    def test_fast_breathing_threshold_boundary_older_child(self):
        # Threshold per WHO IMCI 2014: 12mo-5yr = 40+ breaths/min
        below = ChildAssessment(age_months=36, cough_or_difficulty_breathing=True, respiratory_rate_per_min=39)
        at = ChildAssessment(age_months=36, cough_or_difficulty_breathing=True, respiratory_rate_per_min=40)
        assert assess(below).classification == Classification.MILD
        assert assess(at).classification == Classification.MODERATE

    def test_mild_cough_no_concerning_signs(self):
        child = ChildAssessment(
            age_months=36, cough_or_difficulty_breathing=True, respiratory_rate_per_min=25
        )
        result = assess(child)
        assert result.classification == Classification.MILD

    def test_severe_dehydration_requires_two_of_four_signs(self):
        # Per WHO IMCI 2014: ANY TWO of {lethargic/unconscious, sunken eyes,
        # not able to drink/drinking poorly, skin pinch very slowly}.
        child = ChildAssessment(
            age_months=24, diarrhea=True, sunken_eyes=True, skin_pinch_goes_back_very_slowly=True
        )
        result = assess(child)
        assert result.classification == Classification.SEVERE
        assert result.condition_label == "severe_dehydration"

    def test_single_dehydration_sign_is_not_enough(self):
        # This is the corrected behavior -- an earlier version of this
        # scaffold incorrectly required two SPECIFIC signs rather than any
        # two from the correct WHO-defined set, and didn't guard against
        # under-counting with only one sign present.
        child = ChildAssessment(age_months=24, diarrhea=True, sunken_eyes=True)
        result = assess(child)
        assert result.classification == Classification.MILD
        assert result.condition_label == "no_dehydration"

    def test_some_dehydration_two_of_four_signs(self):
        child = ChildAssessment(
            age_months=36, diarrhea=True,
            child_restless_or_irritable=True, drinking_eagerly_thirsty=True,
        )
        result = assess(child)
        assert result.classification == Classification.MODERATE
        assert result.condition_label == "some_dehydration"

    def test_blood_in_stool_noted_as_secondary_finding(self):
        child = ChildAssessment(age_months=24, diarrhea=True, blood_in_stool=True)
        result = assess(child)
        assert any("dysentery" in f for f in result.secondary_findings)

    def test_every_result_includes_safety_disclaimer(self):
        child = ChildAssessment(age_months=24)
        result = assess(child)
        assert "not a diagnosis" in result.disclaimer.lower()

    def test_reasoning_trail_is_populated(self):
        child = ChildAssessment(age_months=24, danger_signs_present=["convulsions"])
        result = assess(child)
        assert len(result.reasoning) > 0

    def test_danger_signs_present_rejects_string_input(self):
        # Safety-net regression test for the type-confusion bug documented
        # in report/REPORT_TEMPLATE_NOTES.md.
        with pytest.raises(TypeError):
            ChildAssessment(age_months=18, danger_signs_present="convulsions")


class TestIMCITriageTool:
    def setup_method(self):
        self.tool = IMCITriageTool()

    def test_rejects_out_of_range_age(self):
        with pytest.raises(ToolExecutionError):
            self.tool.run(age_months=1)  # below 2-month scaffold boundary
        with pytest.raises(ToolExecutionError):
            self.tool.run(age_months=72)  # above 5-year scaffold boundary

    def test_parses_comma_separated_danger_signs(self):
        result = self.tool.run(age_months=18, danger_signs_present="convulsions, vomits_everything")
        assert result["classification"] == "severe"

    def test_returns_dict_with_expected_keys(self):
        result = self.tool.run(age_months=24)
        assert "classification" in result
        assert "condition_label" in result
        assert "reasoning" in result
        assert "disclaimer" in result
        assert isinstance(result["classification"], str)  # not an Enum instance

    def test_dehydration_signs_pass_through_correctly(self):
        result = self.tool.run(
            age_months=24, diarrhea=True,
            sunken_eyes=True, skin_pinch_goes_back_very_slowly=True,
        )
        assert result["classification"] == "severe"
        assert result["condition_label"] == "severe_dehydration"
