"""
Scores the MODEL, not the rule engine.

eval/scoring/sacc_scorer.py routes vignettes through IMCITriageTool -- it scores
assess() against itself and reports 100% forever. Useful as a protocol
regression check, useless as a measure of the fine-tune. This sibling loads the
actual .gguf through llama_cpp (the same runtime the ADTC evaluator uses) and
prompts it, because the .gguf is the only thing that gets graded.

What it measures, and why each one:
  - classification accuracy       -- the objective half of Sacc
  - UNDER-triage rate, separately -- predicting mild for a severe case is
                                     categorically worse than the reverse.
                                     NEVER average the two directions.
  - format-compliance rate        -- an answer the CLASSIFICATION: regex cannot
                                     parse is one a judge cannot skim either
  - temp 0.0 AND temp 0.8 x N     -- the evaluator's sampling is not ours to
                                     choose; a model that is only right greedily
                                     is not robust

Compare against the base model to get the before/after the report needs (the
base answers a triage question with a fabricated ICD-10 code).

Usage:
    python eval/scoring/model_sacc_scorer.py --model models/<ft>.gguf
    python eval/scoring/model_sacc_scorer.py --model models/<ft>.gguf --base models/Qwen3.5-0.8B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_TASKS = REPO / "eval" / "tasks" / "imci_vignettes_nl.json"

SEVERITY_ORDER = {"mild": 0, "moderate": 1, "severe": 2}
HEADER_RE = re.compile(r"CLASSIFICATION:\s*(SEVERE|MODERATE|MILD)\b", re.I)


def parse_classification(text: str) -> str | None:
    """The rigid header first; a bare-word fallback second. Unparseable -> None,
    which counts as a failure -- if the scorer can't find the classification,
    neither can a grader."""
    m = HEADER_RE.search(text)
    if m:
        return m.group(1).lower()
    # fallback: first standalone severity word
    m = re.search(r"\b(severe|moderate|mild)\b", text, re.I)
    return m.group(1).lower() if m else None


# The label text sits between the severity and the "(colour ...)" in the header:
# "CLASSIFICATION: MODERATE — Wheeze (IMCI yellow: ...)". Used only by the
# extended eval (--check-label): severity alone can't tell a wheeze answer from
# a some-dehydration one, both MODERATE.
_LABEL_RE = re.compile(r"CLASSIFICATION:\s*(?:SEVERE|MODERATE|MILD)\s*[—\-:]+\s*(.+?)\s*[\(\n]", re.I)


def label_matches(text: str, expected_label_text: str) -> bool:
    """True only if the header names EXACTLY the expected classification label.
    Strict on purpose: 'Severe jaundice' must not pass for 'Jaundice'. If the
    header isn't parseable the label can't be confirmed, so it does not match
    (the severity check already penalises the unparseable answer separately)."""
    m = _LABEL_RE.search(text)
    if not m:
        return False
    return m.group(1).strip().lower() == expected_label_text.strip().lower()


# Short domain prime for chat mode. The evaluator/judges invoke an instruct
# model through its chat template; whether they add a system prompt is unknown,
# so we score chat mode both with and without one.
SYSTEM_PROMPT = (
    "You are an offline IMCI decision-support assistant for children 2 months to "
    "5 years old. Check general danger signs first, and refer urgently when in doubt."
)


def score_model(llm, tasks: list[dict], temperature: float, n: int,
                mode: str = "chat", system: bool = True, max_tokens: int = 256,
                check_label: bool = False) -> dict:
    total = correct = parseable = 0
    under = over = 0
    label_total = label_correct = 0
    per_case = []

    for task in tasks:
        expected = task["expected_classification"]
        expected_label = task.get("expected_label_text")
        prompt = task["prompt"]
        preds = []
        label_hits = 0
        for _ in range(n):
            if mode == "raw":
                # Bare completion, no chat template. This is NOT how an instruct
                # model is meant to be run and NOT how the profiler/judges invoke
                # it — kept only to expose the train/serve mismatch, since the
                # model was trained through the chat template.
                out = llm(prompt, max_tokens=max_tokens, temperature=temperature,
                          top_k=1 if temperature == 0 else 40, seed=0)
                text = out["choices"][0]["text"]
            else:  # "chat" — the representative mode
                messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if system else []) + \
                           [{"role": "user", "content": prompt}]
                out = llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens, temperature=temperature,
                    top_k=1 if temperature == 0 else 40, seed=0)
                text = out["choices"][0]["message"]["content"]
            preds.append(parse_classification(text))
            if check_label and expected_label and label_matches(text, expected_label):
                label_hits += 1

        # majority vote over the n samples (n=1 for greedy)
        valid = [p for p in preds if p]
        pred = max(set(valid), key=valid.count) if valid else None

        total += 1
        if pred is not None:
            parseable += 1
        if pred == expected:
            correct += 1
        elif pred is not None and expected is not None:
            if SEVERITY_ORDER[pred] < SEVERITY_ORDER[expected]:
                under += 1
            else:
                over += 1

        case = {"id": task.get("id"), "expected": expected, "predicted": pred, "all_samples": preds}
        if check_label and expected_label:
            label_total += 1
            # majority: matched in more than half the samples
            hit = label_hits > n / 2
            label_correct += int(hit)
            case["expected_label"] = expected_label
            case["label_correct"] = hit
        per_case.append(case)

    out = {
        "n_cases": total,
        "mode": mode + ("+sys" if mode == "chat" and system else ""),
        "temperature": temperature,
        "samples_per_case": n,
        "accuracy": round(100 * correct / total, 1) if total else 0.0,
        "format_compliance": round(100 * parseable / total, 1) if total else 0.0,
        "undertriage_rate": round(100 * under / total, 1) if total else 0.0,
        "overtriage_rate": round(100 * over / total, 1) if total else 0.0,
        "per_case": per_case,
    }
    if check_label and label_total:
        out["label_accuracy"] = round(100 * label_correct / label_total, 1)
    return out


def load_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    tasks = data["examples"] if isinstance(data, dict) else data
    return [t for t in tasks if t.get("prompt") and t.get("expected_classification")]


def _load_llm(model_path: str):
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=1024, n_threads=4, seed=0, verbose=False)


def run(model_path: str, tasks: list[dict], check_label: bool = False) -> dict:
    llm = _load_llm(model_path)
    # chat+sys is the representative mode (how an instruct model is invoked);
    # chat/no-sys and raw expose robustness to invocation modes we don't control.
    return {
        "model": model_path,
        "chat_sys_greedy":   score_model(llm, tasks, 0.0, 1, mode="chat", system=True, check_label=check_label),
        "chat_nosys_greedy": score_model(llm, tasks, 0.0, 1, mode="chat", system=False, check_label=check_label),
        "chat_sys_sampled":  score_model(llm, tasks, 0.8, 5, mode="chat", system=True, check_label=check_label),
        "raw_greedy":        score_model(llm, tasks, 0.0, 1, mode="raw", check_label=check_label),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the fine-tuned .gguf to score")
    ap.add_argument("--base", help="optional base .gguf for an A/B baseline")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--check-label", action="store_true",
                    help="also score whether the exact classification LABEL is right, not just the "
                         "severity (needs expected_label_text on each task -- the extended eval has it)")
    ap.add_argument("--out", help="where to write the JSON report (default eval/results/model_sacc.json)")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if not tasks:
        raise SystemExit(f"no scorable tasks (need prompt + expected_classification) in {args.tasks}")
    print(f"scoring {len(tasks)} NL vignettes\n")

    report = {"fine_tuned": run(args.model, tasks, check_label=args.check_label)}
    if args.base:
        report["base"] = run(args.base, tasks, check_label=args.check_label)

    def show(label, r):
        print(f"=== {label}: {r['model']} ===")
        for key in ("chat_sys_greedy", "chat_nosys_greedy", "chat_sys_sampled", "raw_greedy"):
            m = r[key]
            lab = f"  label {m['label_accuracy']:5.1f}%" if "label_accuracy" in m else ""
            print(f"  {key:18} acc {m['accuracy']:5.1f}%  format {m['format_compliance']:5.1f}%  "
                  f"UNDER {m['undertriage_rate']:4.1f}%  over {m['overtriage_rate']:4.1f}%{lab}")

    show("FINE-TUNED", report["fine_tuned"])
    if args.base:
        show("BASE", report["base"])

    out = Path(args.out) if args.out else REPO / "eval" / "results" / "model_sacc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
