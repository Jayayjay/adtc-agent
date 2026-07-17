"""
Measures tokens-per-second for a GGUF model via llama-cpp-python, matching
the ADTC Sperf formula. Run on hardware matching the ADTC Standard Laptop
profile: Intel i5 10th-12th gen or Ryzen 5 3000-5000, no discrete GPU,
Ubuntu 22.04.

Usage:
    python scripts/bench_tps.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --threads 6 --runs 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_cpp import Llama
from src.config import ADTCScoring
from src.utils.benchmarking import compute_sperf
from src.utils.helpers import save_json

DEFAULT_PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Summarize the plot of a story about a lighthouse keeper in three sentences.",
    "What steps would you take to debug a failing unit test?",
    "Describe how a hash map works.",
    "Write a short function signature for parsing a CSV file.",
]


def measure_tps(llm: Llama, prompt: str, max_tokens: int = 256) -> float:
    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    result = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.4)
    elapsed = time.perf_counter() - start
    completion_tokens = result["usage"]["completion_tokens"]
    return completion_tokens / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--export", type=str, default="report/data/tps_benchmark.json")
    args = parser.parse_args()

    scoring = ADTCScoring()

    print(f"Loading model: {args.model}")
    llm = Llama(
        model_path=args.model, n_ctx=args.n_ctx, n_threads=args.threads,
        n_gpu_layers=0, verbose=False,
    )

    prompts = (DEFAULT_PROMPTS * ((args.runs // len(DEFAULT_PROMPTS)) + 1))[: args.runs]

    tps_results = []
    for i, prompt in enumerate(prompts, 1):
        tps = measure_tps(llm, prompt, max_tokens=args.max_tokens)
        tps_results.append(tps)
        print(f"  run {i}/{len(prompts)}: {tps:.2f} tokens/sec")

    mean_tps = statistics.mean(tps_results)
    stdev_tps = statistics.stdev(tps_results) if len(tps_results) > 1 else 0.0
    sperf = compute_sperf(mean_tps, scoring)

    print("\n--- Results ---")
    print(f"Mean TPS:   {mean_tps:.2f} (stdev {stdev_tps:.2f})")
    print(f"Min/Max:    {min(tps_results):.2f} / {max(tps_results):.2f}")
    print(f"TPS_REFERENCE: {scoring.tps_reference}")
    print(f"Estimated Sperf: {sperf:.1f}")
    if mean_tps < scoring.tps_reference:
        print("\nWARNING: mean TPS below TPS_REFERENCE. Sperf will be penalized.")

    save_json(args.export, {
        "model": args.model,
        "mean_tps": mean_tps,
        "stdev_tps": stdev_tps,
        "min_tps": min(tps_results),
        "max_tps": max(tps_results),
        "runs": tps_results,
        "sperf": sperf,
    })
    print(f"\nResults exported to {args.export}")


if __name__ == "__main__":
    main()
