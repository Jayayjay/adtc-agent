import pytest

from src.config import SystemConfig
from src.hrm.state_machine import HRMSession
from src.tools.imci_protocol import TriageResult


@pytest.fixture
def config():
    return SystemConfig()


class TestHRMSessionInteractiveLoop:
    def test_start_returns_first_question(self, config):
        session = HRMSession(config)
        action = session.start(age_months=24, chief_complaint_text="child has a cough")
        assert action.field_name == "danger_signs_present"
        assert not action.is_stop

    def test_full_manual_loop_reaches_classification(self, config):
        session = HRMSession(config)
        action = session.start(age_months=18, chief_complaint_text="convulsing")
        assert action.field_name == "danger_signs_present"

        action = session.answer("danger_signs_present", ["convulsions"])
        assert action.is_stop

        result = session.classify()
        assert isinstance(result, TriageResult)
        assert result.classification.value == "severe"

    def test_is_ready_to_classify_reflects_state(self, config):
        session = HRMSession(config)
        session.start(age_months=18, chief_complaint_text="convulsing")
        assert session.is_ready_to_classify() is False

        session.answer("danger_signs_present", ["convulsions"])
        assert session.is_ready_to_classify() is True

    def test_classify_before_start_raises(self, config):
        session = HRMSession(config)
        with pytest.raises(RuntimeError):
            session.classify()

    def test_answer_before_start_raises(self, config):
        session = HRMSession(config)
        with pytest.raises(RuntimeError):
            session.answer("danger_signs_present", [])


class TestHRMSessionSingleShotBridge:
    def test_no_known_answers_returns_first_question(self, config):
        session = HRMSession(config)
        result = session.solve_from_known_answers(24, "cough", known_answers={})
        assert not hasattr(result, "classification")  # QuestionAction, not TriageResult
        assert result.field_name == "danger_signs_present"

    def test_complete_known_answers_returns_classification(self, config):
        session = HRMSession(config)
        result = session.solve_from_known_answers(
            18, "convulsing", known_answers={"danger_signs_present": ["convulsions"]}
        )
        assert isinstance(result, TriageResult)
        assert result.classification.value == "severe"

    def test_partial_known_answers_asks_next_missing_field(self, config):
        session = HRMSession(config)
        result = session.solve_from_known_answers(
            12, "cough",
            known_answers={
                "danger_signs_present": [],
                "cough_or_difficulty_breathing": True,
                "stridor_when_calm": True,
            },
        )
        # stridor=True should be enough to classify (severe, short-circuits
        # before chest_indrawing/respiratory_rate are ever needed)
        assert isinstance(result, TriageResult)
        assert result.classification.value == "severe"


class TestHRMSessionTrainedModelHook:
    def test_trained_model_mode_raises_loudly_without_checkpoint(self, config, tmp_path):
        # Point at an empty directory -- no checkpoint present.
        config.hrm_checkpoint_dir = tmp_path
        session = HRMSession(config, use_trained_model=True)
        with pytest.raises(NotImplementedError):
            session.start(age_months=24, chief_complaint_text="cough")

    def test_trained_model_mode_works_with_real_checkpoint(self, config):
        # Uses the actual trained checkpoint if present (produced by
        # scripts/train_hrm.py) -- skips if not, since CI/fresh clones won't
        # have a checkpoint until someone runs the training pipeline.
        checkpoint_path = config.hrm_checkpoint_dir / "orchestration_policy.pt"
        if not checkpoint_path.exists():
            pytest.skip("No trained checkpoint present -- run scripts/train_hrm.py first.")

        session = HRMSession(config, use_trained_model=True)
        action = session.start(age_months=18, chief_complaint_text="child is convulsing")
        assert action.field_name == "danger_signs_present"

        action = session.answer("danger_signs_present", ["convulsions"])
        assert action.is_stop

        result = session.classify()
        assert result.classification.value == "severe"