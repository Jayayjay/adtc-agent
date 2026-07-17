"""
Generates imitation-learning training data for HRM: rolls out the expert
policy (src/hrm/expert_policy.py) over many synthetic ChildAssessment cases,
encoding each (state, next_action) pair via src/hrm/encoders.py /
dialogue_state.ACTION_SPACE indices.

The expert policy IS the ground truth here -- there's no separate "correct
answer" source, because the policy's branch logic is itself derived directly
from imci_protocol.assess() (see expert_policy.py's module docstring). HRM's
training objective is to learn to reproduce this policy's question-asking
behavior from the encoded state alone, ideally generalizing to states the
expert policy wasn't explicitly enumerated for.

Output: JSONL file, one training example per line:
    {"state_vector": [...], "chief_complaint": "...", "action_index": int,
     "action_field": str|"STOP", "case_id": str, "turn": int}

Usage:
    python scripts/generate_hrm_training_data.py --num-cases 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hrm.dialogue_state import ACTION_SPACE, ACTION_STOP, DialogueState
from src.hrm.encoders import encode_task
from src.hrm.expert_policy import simulate_dialogue
from src.tools.imci_protocol import ChildAssessment

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "hrm_training_data.jsonl"

CHIEF_COMPLAINTS = [
    "child has a bad cough", "child is having trouble breathing", "child has diarrhea",
    "child has had loose stools for days", "child has a high fever", "child feels very hot",
    "child's ear hurts", "child has discharge from the ear", "just a routine checkup",
    "caretaker is worried about the child", "",
]


def _random_case(rng: random.Random) -> ChildAssessment:
    """Same distribution as tests/unit/test_expert_policy.py's fuzz test --
    kept in sync deliberately so training data and the correctness test
    exercise the same case space. If you change one, check the other."""
    return ChildAssessment(
        age_months=rng.randint(2, 59),
        danger_signs_present=(
            [rng.choice(["convulsions", "unable_to_drink_or_breastfeed", "vomits_everything",
                         "lethargic_or_unconscious", "convulsing_now"])]
            if rng.random() < 0.12 else []
        ),
        cough_or_difficulty_breathing=rng.random() < 0.4,
        respiratory_rate_per_min=rng.randint(15, 70) if rng.random() < 0.7 else None,
        chest_indrawing=rng.random() < 0.2,
        stridor_when_calm=rng.random() < 0.08,
        diarrhea=rng.random() < 0.4,
        diarrhea_days=rng.randint(1, 20) if rng.random() < 0.5 else None,
        blood_in_stool=rng.random() < 0.15,
        child_lethargic_or_unconscious=rng.random() < 0.08,
        child_restless_or_irritable=rng.random() < 0.2,
        sunken_eyes=rng.random() < 0.2,
        not_able_to_drink_or_drinking_poorly=rng.random() < 0.15,
        drinking_eagerly_thirsty=rng.random() < 0.2,
        skin_pinch_goes_back_very_slowly=rng.random() < 0.12,
        skin_pinch_goes_back_slowly=rng.random() < 0.2,
        fever=rng.random() < 0.4,
        stiff_neck=rng.random() < 0.08,
        ear_problem=rng.random() < 0.3,
        ear_pain=rng.random() < 0.3,
        ear_discharge_days=rng.randint(1, 25) if rng.random() < 0.4 else None,
        tender_swelling_behind_ear=rng.random() < 0.08,
    )


def generate(num_cases: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    skipped = 0

    for i in range(num_cases):
        ground_truth = _random_case(rng)
        chief_complaint = rng.choice(CHIEF_COMPLAINTS)

        try:
            trajectory = simulate_dialogue(ground_truth, chief_complaint)
        except RuntimeError:
            skipped += 1  # max_turns safety bound hit -- shouldn't happen, but don't crash the run
            continue

        # Replay the trajectory to reconstruct state at each turn (matches
        # exactly what encode_task would see live, turn by turn).
        state = DialogueState(age_months=ground_truth.age_months, chief_complaint_text=chief_complaint)
        for step in trajectory:
            vec = encode_task(chief_complaint, {"state": state})
            action_field = step["question_field"] if not step["is_stop"] else ACTION_STOP
            action_index = ACTION_SPACE.index(action_field)

            examples.append({
                "state_vector": vec,
                "chief_complaint": chief_complaint,
                "action_index": action_index,
                "action_field": action_field,
                "case_id": f"synthetic_{i}",
                "turn": step["turn"],
            })

            if not step["is_stop"]:
                state.record_answer(step["question_field"], step["answer"])

    if skipped:
        print(f"WARNING: {skipped} cases hit max_turns and were skipped -- investigate expert_policy.py.")

    return examples


def split_by_case(examples: list[dict], val_frac: float, test_frac: float, seed: int) -> dict:
    """
    Splits by CASE_ID, not by individual example -- critical correctness
    point. All turns from one synthetic case share heavy state overlap
    (e.g. every turn in a case has the same age_months and most of the same
    known_fields as the turn before it). Splitting at the individual-example
    level instead of the case level would leak near-duplicate states across
    train/val/test, inflating validation accuracy in a way that wouldn't
    hold up on genuinely unseen cases.
    """
    case_ids = sorted({ex["case_id"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n = len(case_ids)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_ids = set(case_ids[:n_val])
    test_ids = set(case_ids[n_val:n_val + n_test])
    train_ids = set(case_ids[n_val + n_test:])

    return {
        "train": [ex for ex in examples if ex["case_id"] in train_ids],
        "val": [ex for ex in examples if ex["case_id"] in val_ids],
        "test": [ex for ex in examples if ex["case_id"] in test_ids],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_PATH.parent))
    args = parser.parse_args()

    examples = generate(args.num_cases, args.seed)
    splits = split_by_case(examples, args.val_frac, args.test_frac, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits.items():
        out_path = out_dir / f"hrm_training_data_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_name:5s}: {len(split_examples):5d} examples -> {out_path}")

    action_counts = {}
    for ex in examples:
        action_counts[ex["action_field"]] = action_counts.get(ex["action_field"], 0) + 1

    print(f"\nTotal: {len(examples)} training examples from {args.num_cases} synthetic cases "
          f"(split by case_id, not by example -- see split_by_case() docstring).")
    print("\nAction distribution (class balance -- worth checking before training):")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:45s}: {count:5d} ({100*count/len(examples):.1f}%)")
    print(
        "\nNOTE: this is imitation-learning data -- HRM would learn to predict "
        "action_index from state_vector. class imbalance here (e.g. "
        "danger_signs_present appearing in every trajectory's first step) is "
        "expected and should be handled at training time (e.g. class weighting), "
        "not by altering the data generation."
    )


if __name__ == "__main__":
    main()