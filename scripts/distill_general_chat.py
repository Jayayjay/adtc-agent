"""
Samples the BASE model's own answers to generic prompts, to be replayed as
training targets in the general-chat slice of the SFT mixture.

Why self-distillation rather than an off-the-shelf instruction set:

  - The target IS the base distribution, so this slice applies almost no
    gradient pressure. It occupies 15% of the mixture purely to stop the IMCI
    data from eating the model's general ability -- and it does that without
    dragging the model toward some other dataset's voice.
  - No licence or attribution question. The weights are apache-2.0 and the
    text is the model's own.
  - Defensible in the report in one sentence.

Cost: ~600 prompts x ~200 tokens at CPU speed is roughly 1-2 hours. Run it in
the background while the rest of src/sft/ is being written; it does not depend
on any of it.

Usage:
    python scripts/distill_general_chat.py --num-prompts 600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sft" / "general_chat.jsonl"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"

# Deliberately spread across what a 0.8B is actually asked to do, plus the
# small talk that must NOT trigger a triage dump (cf. case_006 in
# eval/tasks/imci_triage.json). Seeded from scripts/bench_tps.DEFAULT_PROMPTS
# and widened.
SEED_PROMPTS = [
    # small talk -- the ones that matter most for not over-triaging
    "Hello!", "Hi there, how are you?", "Thanks, that helped.", "Good morning.",
    "Who are you?", "What can you help me with?", "Thank you!", "Goodbye.",
    "Are you a doctor?", "How does this work?",
    # general knowledge
    "What is the capital of Nigeria?", "What does WHO stand for?",
    "Explain photosynthesis in one sentence.", "List three primary colours.",
    "What is the largest ocean on Earth?", "Who wrote Things Fall Apart?",
    "What causes rain?", "Name three countries in West Africa.",
    "What is the difference between weather and climate?",
    "How many continents are there?", "What language is spoken in Brazil?",
    "What is the boiling point of water?", "Explain gravity simply.",
    "What is the Sahara?", "Why is the sky blue?",
    # everyday reasoning
    "If I have 12 oranges and give away 5, how many are left?",
    "What is 84 * 3 / 2?", "2 + 2 =",
    "If a bus leaves at 3pm and takes 2 hours, when does it arrive?",
    "Which is heavier, a kilogram of iron or a kilogram of feathers?",
    "How many days are in a leap year?",
    # coding
    "Explain the difference between a list and a tuple in Python.",
    "Write a short function signature for parsing a CSV file.",
    "What steps would you take to debug a failing unit test?",
    "Describe how a hash map works.",
    "Write a Python function that reverses a string.",
    "What does the git commit command do?",
    "Explain what an API is to a beginner.",
    # writing
    "Summarize the plot of a story about a lighthouse keeper in three sentences.",
    "Write a two-line poem about rain.",
    "Once upon a time, in a village near Kano,",
    "Draft a one-sentence thank-you note.",
    "Give me three tips for writing clearly.",
]

# DELIBERATELY EXCLUDED: general health prompts ("Why do people cough?", "What
# is a fever?"). Self-distillation copies the base distribution faithfully --
# errors included -- and the base is confidently wrong about medicine. Asked why
# people cough, it answered "to help prevent bronchitis and pneumonia".
#
# For a neutral prompt that costs nothing: the target is what the model already
# does, so the gradient is ~0 and the slice does its anti-forgetting job. But
# training a CHILD-HEALTH model on its own medical misinformation is a different
# trade, and a judge probing general health knowledge is a plausible thing to
# happen. Health questions belong in the IMCI slice, where assess() is the
# ground truth, or in the scope_refusal slice -- not here, where the base model
# grades its own homework.

# Light paraphrase axes so 600 prompts aren't 45 prompts repeated 13 times.
PREFIXES = ("", "", "", "Quick question: ", "Hey, ", "Please: ", "I'm curious — ")
SUFFIXES = ("", "", "", " Thanks.", " Keep it short.", " Explain simply.", " In one sentence.")


def build_prompts(n: int, rng: random.Random) -> list[str]:
    out, seen = [], set()
    while len(out) < n:
        base = rng.choice(SEED_PROMPTS)
        p = f"{rng.choice(PREFIXES)}{base}{rng.choice(SUFFIXES)}"
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(MODEL_PATH))
    ap.add_argument("--num-prompts", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="not greedy: identical targets across near-identical prompts would "
                         "teach the model to be more deterministic than it is")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from llama_cpp import Llama

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    rng = random.Random(args.seed)
    prompts = build_prompts(args.num_prompts, rng)

    llm = Llama(model_path=str(model_path), n_ctx=1024, n_threads=args.threads,
                seed=args.seed, verbose=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, prompt in enumerate(prompts):
            try:
                r = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                text = r["choices"][0]["message"]["content"].strip()
            except Exception as e:  # a single bad generation must not kill an hour of work
                print(f"  [{i}] generation failed: {e}")
                continue

            if not text:
                continue
            f.write(json.dumps({"prompt": prompt, "completion": text}) + "\n")
            f.flush()  # partial output stays usable if this is interrupted
            written += 1

            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(prompts) - i - 1) / rate / 60
                print(f"  {i+1}/{len(prompts)}  ({rate*60:.1f}/min, ~{eta:.0f} min left)")

    print(f"\nwrote {written} pairs to {out_path} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
