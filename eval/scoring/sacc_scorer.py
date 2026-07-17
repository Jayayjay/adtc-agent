"""
Sacc scoring: runs eval/tasks/*.json vignettes through the deterministic IMCI
engine and checks classification correctness against expected labels.

This measures whether your SYSTEM (tool + orchestration) produces correct
classifications -- it does NOT measure judge-panel-style qualitative response
quality (tone, clarity, appropriate caveats), which the actual ADTC Sacc
score also includes. Treat this as your best local proxy for the objective
half of Sacc, not a full substitute for the judge panel.

Usage:
    python eval/scoring/sacc_scorer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.tools.imci_triage_tool import IMCITriageTool

EVAL_TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_tool = IMCITriageTool()


def score_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    total = 0
    correct = 0
    details = []

    for example in data.get("examples", []):
        if example.get("label") != "reasoning" or example.get("expected_classification") is None:
            continue  # skip fast-path examples -- not a classification task

        total += 1
        # Route through IMCITriageTool.run(), NOT ChildAssessment directly --
        # this ensures eval scoring exercises the exact same input-parsing
        # path (e.g. comma-split danger_signs_present) as the live agent.
        # A previous version of this scorer bypassed the tool wrapper and
        # constructed ChildAssessment directly from raw JSON, which silently
        # passed danger_signs_present as a string instead of a list and
        # caused real danger signs to be missed. See imci_protocol.py's
        # ChildAssessment.__post_init__ for the safety-net fix on that bug.
        result = _tool.run(**example["input"])

        is_correct = result["classification"] == example["expected_classification"]
        if is_correct:
            correct += 1

        details.append({
            "id": example["id"],
            "expected": example["expected_classification"],
            "actual": result["classification"],
            "correct": is_correct,
        })

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return {"file": path.name, "total": total, "correct": correct, "sacc_proxy": accuracy, "details": details}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for path in EVAL_TASKS_DIR.glob("*.json"):
        result = score_file(path)
        all_results.append(result)
        print(f"\n{result['file']}: {result['correct']}/{result['total']} correct "
              f"({result['sacc_proxy']:.1f}%)")
        for d in result["details"]:
            mark = "PASS" if d["correct"] else "FAIL"
            print(f"  [{mark}] {d['id']}: expected={d['expected']}, actual={str(d['actual'])}")

    overall_total = sum(r["total"] for r in all_results)
    overall_correct = sum(r["correct"] for r in all_results)
    overall_pct = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0

    print(f"\n--- Overall Sacc Proxy: {overall_correct}/{overall_total} ({overall_pct:.1f}%) ---")
    print(
        "\nNOTE: this scores the deterministic rule engine's classification "
        "accuracy against your own vignettes -- it is a proxy for the "
        "objective portion of Sacc, not the full judge-panel score, which "
        "also weighs response quality/clarity/appropriate caveats."
    )

    from datetime import datetime
    out_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"overall_sacc_proxy": overall_pct, "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
