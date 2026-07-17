"""
Validates a trained orchestration policy checkpoint on TWO distinct levels:

  1. Action-level accuracy on the held-out TEST split (not val -- val was
     used for checkpoint selection during training, so it's not a clean
     estimate of generalization).
  2. END-TO-END classification accuracy: actually run the trained model as
     the live orchestrator (swap it in for the expert policy), roll out full
     dialogues on held-out ground-truth cases, and check whether the final
     classification from imci_protocol.assess() matches what the expert
     policy (and thus a full assessment) would have produced. This is the
     metric that actually matters -- a model could have high action-level
     accuracy while still occasionally asking a wrong/redundant question
     that doesn't change the final classification, or conversely a small
     number of action mismatches early in a trajectory could cascade into
     wrong classifications. Only end-to-end rollout catches the difference.

Usage:
    python scripts/validate_hrm.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.hrm.dialogue_state import ACTION_SPACE, ACTION_STOP, DialogueState
from src.hrm.decoders import decode_output
from src.hrm.encoders import encode_task
from src.hrm.expert_policy import simulate_dialogue
from src.hrm.model import OrchestrationPolicyNet
from src.tools.imci_protocol import ChildAssessment, assess
from scripts.generate_hrm_training_data import _random_case, CHIEF_COMPLAINTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "src" / "hrm" / "trained_models" / "orchestration_policy.pt"


def action_level_accuracy(model: OrchestrationPolicyNet, test_path: Path, device: str) -> float:
    correct, total = 0, 0
    with open(test_path) as f:
        for line in f:
            ex = json.loads(line)
            pred = model.predict_action_index(ex["state_vector"])
            if pred == ex["action_index"]:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def end_to_end_rollout(model: OrchestrationPolicyNet, ground_truth: ChildAssessment,
                        chief_complaint: str, max_turns: int = 30) -> dict:
    """Runs the TRAINED MODEL (not the expert policy) as the orchestrator,
    querying the ground truth for answers, and returns whether the final
    classification matches a full assessment."""
    state = DialogueState(age_months=ground_truth.age_months, chief_complaint_text=chief_complaint)
    turns = 0

    for _ in range(max_turns):
        vec = encode_task(chief_complaint, {"state": state})
        action_index = model.predict_action_index(vec)
        action = decode_output(action_index)

        if action.is_stop:
            break
        answer = getattr(ground_truth, action.field_name)
        state.record_answer(action.field_name, answer)
        turns += 1
    else:
        return {"matched": False, "turns": turns, "reason": "max_turns exceeded (model never stopped)"}

    reconstructed = ChildAssessment(age_months=ground_truth.age_months, **state.known_fields)
    model_result = assess(reconstructed)
    true_result = assess(ground_truth)

    return {
        "matched": model_result.classification == true_result.classification
                   and model_result.condition_label == true_result.condition_label,
        "turns": turns,
        "expected": true_result.condition_label,
        "actual": model_result.condition_label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--test-data", type=str, default=str(DATA_DIR / "hrm_training_data_test.jsonl"))
    parser.add_argument("--num-rollout-cases", type=int, default=300,
                         help="Fresh synthetic cases for end-to-end rollout (separate from the action-level test set)")
    parser.add_argument("--rollout-seed", type=int, default=999,
                         help="Different seed from training data generation -- genuinely unseen cases")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"No checkpoint at {checkpoint_path}. Run scripts/train_hrm.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OrchestrationPolicyNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (val_acc={checkpoint['val_acc']:.4f}, epoch={checkpoint['epoch']})")

    # Level 1: action-level accuracy on held-out TEST split.
    test_path = Path(args.test_data)
    if test_path.exists():
        acc = action_level_accuracy(model, test_path, device)
        print(f"\n[1] Action-level accuracy on TEST split: {acc:.4f}")
    else:
        print(f"\n[1] SKIPPED -- no test file at {test_path}")

    # Level 2: end-to-end classification accuracy on FRESH cases (different
    # seed from training data -- genuinely unseen, not just held out from
    # the same generation run).
    rng = random.Random(args.rollout_seed)
    matched, total = 0, 0
    mismatches = []
    for i in range(args.num_rollout_cases):
        gt = _random_case(rng)
        chief_complaint = rng.choice(CHIEF_COMPLAINTS)
        result = end_to_end_rollout(model, gt, chief_complaint)
        total += 1
        if result["matched"]:
            matched += 1
        else:
            mismatches.append({"case": i, **result})

    print(f"\n[2] End-to-end classification accuracy on {total} FRESH synthetic cases "
          f"(seed={args.rollout_seed}, distinct from training data): {matched/total:.4f}")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches found -- first 5:")
        for m in mismatches[:5]:
            print(f"  case {m['case']}: expected={m.get('expected')}, "
                  f"actual={m.get('actual')}, reason={m.get('reason', 'classification mismatch')}")
    else:
        print("\nNo mismatches -- trained model reproduces expert-policy classifications exactly "
              "on this fresh sample.")

    print(
        "\nNOTE: this validates the model against the SYNTHETIC data distribution and the "
        "SCAFFOLD's modeled IMCI subset -- it does not validate against real clinical cases "
        "or the full IMCI algorithm. See report/REPORT_TEMPLATE_NOTES.md for scope limitations."
    )


if __name__ == "__main__":
    main()