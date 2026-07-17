"""
Model architecture for the trained orchestration policy.

IMPORTANT SCOPE NOTE: this is a small MLP baseline, NOT the actual Sapient
HRM architecture (hierarchical, dual-timescale recurrent modules). It exists
to make the training/validation PIPELINE (data loading, train loop,
checkpointing, evaluation, integration into HRMSession) real and testable
now, rather than blocked on porting a separate research codebase.

Swapping this for the real HRM architecture (see scripts/setup_hrm.sh --
clones github.com/sapientinc/HRM) is a separate, well-scoped next step: the
input encoding (src/hrm/encoders.py), the action space
(src/hrm/dialogue_state.ACTION_SPACE), the training data format
(scripts/generate_hrm_training_data.py), and the evaluation harness
(scripts/validate_hrm.py) all stay the same regardless of which architecture
consumes them -- only this file and the loading code in state_machine.py's
_decide_via_trained_model() would need to change.

Given the task (22-way classification over a 71-dim input, imitating a
deterministic policy), a small MLP is a reasonable and honestly-labeled
placeholder -- it is NOT being presented as "the HRM", and the report should
say so explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hrm.dialogue_state import ACTION_SPACE
from src.hrm.encoders import ENCODING_LENGTH


class OrchestrationPolicyNet(nn.Module):
    """Baseline classifier: state_vector (71-dim) -> action logits (22-way)."""

    def __init__(self, input_dim: int = ENCODING_LENGTH, hidden_dim: int = 64,
                 num_actions: int = len(ACTION_SPACE)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_action_index(self, state_vector: list[float]) -> int:
        """Convenience method for inference on a single state (not a batch)."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([state_vector], dtype=torch.float32)
            logits = self.forward(x)
            return int(torch.argmax(logits, dim=-1).item())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)