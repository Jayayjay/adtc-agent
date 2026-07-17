"""
Controls one adaptive IMCI assessment session: repeatedly decides the next
question to ask, collects the answer, and eventually calls
imci_protocol.assess() once enough is known.

CURRENT STATUS: uses src.hrm.expert_policy directly as the question-asking
orchestrator -- this is a REAL, WORKING system today, not a stub. It is also
exactly the policy a trained HRM checkpoint would be imitating (see
scripts/generate_hrm_training_data.py), so swapping in a trained model later
is a drop-in replacement of _decide_next_action(), not a redesign.
"""

from __future__ import annotations

import logging

from src.config import SystemConfig
from src.hrm.dialogue_state import DialogueState, QuestionAction
from src.hrm.expert_policy import next_question as expert_next_question
from src.tools.imci_protocol import ChildAssessment, TriageResult, assess

logger = logging.getLogger(__name__)


class HRMSession:
    def __init__(self, config: SystemConfig, use_trained_model: bool = True):
        self.config = config
        self.use_trained_model = use_trained_model
        self._trained_model = None  # lazy-loaded only if use_trained_model=True
        self.state: DialogueState | None = None

    def start(self, age_months: int, chief_complaint_text: str = "") -> QuestionAction:
        """Begins a new assessment session. Returns the first question to ask."""
        self.state = DialogueState(age_months=age_months, chief_complaint_text=chief_complaint_text)
        return self._decide_next_action()

    def answer(self, field_name: str, value) -> QuestionAction:
        """Records an answer and returns the next question (or a stop signal)."""
        if self.state is None:
            raise RuntimeError("HRMSession.start() must be called before answer().")
        self.state.record_answer(field_name, value)
        return self._decide_next_action()

    def is_ready_to_classify(self) -> bool:
        action = self._decide_next_action()
        return action.is_stop

    def classify(self) -> TriageResult:
        """Call once is_ready_to_classify() is True (or after receiving a
        stop signal from start()/answer()). Builds a ChildAssessment from
        everything collected so far and runs the deterministic rule engine."""
        if self.state is None:
            raise RuntimeError("HRMSession.start() must be called before classify().")
        child = ChildAssessment(age_months=self.state.age_months, **self.state.known_fields)
        return assess(child)

    def _decide_next_action(self) -> QuestionAction:
        if self.use_trained_model:
            return self._decide_via_trained_model()
        return expert_next_question(self.state)

    def _decide_via_trained_model(self) -> QuestionAction:
        """
        Runs the trained orchestration policy checkpoint (see
        scripts/train_hrm.py, src/hrm/model.py) instead of the expert
        policy. Requires a checkpoint at
        src/hrm/trained_models/orchestration_policy.pt -- run
        scripts/generate_hrm_training_data.py then scripts/train_hrm.py to
        produce one.
        """
        if self._trained_model is None:
            self._load_trained_model()

        from src.hrm.encoders import encode_task
        from src.hrm.decoders import decode_output

        vec = encode_task(self.state.chief_complaint_text, {"state": self.state})
        action_index = self._trained_model.predict_action_index(vec)
        return decode_output(action_index)

    def _load_trained_model(self):
        import torch
        from src.hrm.model import OrchestrationPolicyNet

        checkpoint_path = self.config.hrm_checkpoint_dir / "orchestration_policy.pt"
        if not checkpoint_path.exists():
            raise NotImplementedError(
                f"No trained checkpoint at {checkpoint_path}. Run "
                "scripts/generate_hrm_training_data.py then scripts/train_hrm.py "
                "to produce one. Until then, construct HRMSession with "
                "use_trained_model=False (the default) to use the expert "
                "policy directly -- it's a fully working orchestrator today."
            )
        model = OrchestrationPolicyNet()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._trained_model = model
        logger.info(
            "Loaded trained orchestration policy (val_acc=%.4f, epoch=%d)",
            checkpoint.get("val_acc", -1), checkpoint.get("epoch", -1),
        )

    def solve_from_known_answers(
        self, age_months: int, chief_complaint_text: str, known_answers: dict
    ) -> TriageResult | QuestionAction:
        """
        Bridges the multi-turn session API to a single-shot call, for
        callers (like src/core/agent.py) that receive one message and need
        one response. `known_answers` should be whatever fields the caller
        already extracted from the message (e.g. via the LLM's tool-calling/
        structured extraction -- NOT implemented by this method itself).

        Runs the dialogue internally using only the provided answers.
        Returns a TriageResult if that was enough info to classify, or the
        next QuestionAction if more information is still needed -- in which
        case the caller should surface that question back to the user as a
        follow-up, rather than guessing or defaulting missing fields.
        """
        action = self.start(age_months, chief_complaint_text)
        while not action.is_stop:
            if action.field_name not in known_answers:
                return action  # caller needs to ask this and call again
            action = self.answer(action.field_name, known_answers[action.field_name])
        return self.classify()