"""
Measures peak RSS during model load + inference, matching the ADTC Seff
formula. Run this against your combined orchestrator process (Qwen + HRM,
once implemented) for the real submission number -- LLM-only is a useful
lower bound but not your true peak.

Usage:
    python scripts/bench_ram.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_cpp import Llama
from src.config import ADTCScoring
from src.utils.resource_monitor import ResourceMonitor
from src.utils.benchmarking import compute_seff
from src.utils.helpers import save_json

DEFAULT_PROMPT = (
    "Walk through, step by step, how you would plan and execute a multi-step "
    "task involving three tools with dependencies between them."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--export", type=str, default="report/data/ram_benchmark.json")
    args = parser.parse_args()

    scoring = ADTCScoring()

    with ResourceMonitor() as monitor:
        print(f"Loading model: {args.model}")
        llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=0, verbose=False)
        print(f"Peak RSS after load: {monitor.peak_ram_mb:.1f} MB")

        print("Running inference burst...")
        llm.create_chat_completion(
            messages=[{"role": "user", "content": DEFAULT_PROMPT}],
            max_tokens=args.max_tokens, temperature=0.4,
        )
        time.sleep(0.2)

    seff = compute_seff(monitor.peak_ram_mb, scoring)

    print("\n--- Results ---")
    print(f"Peak RAM (RSS): {monitor.peak_ram_mb:.1f} MB")
    print(f"RAM budget:     {scoring.ram_budget_mb} MB")
    print(f"Estimated Seff: {seff:.1f}")
    if monitor.peak_ram_mb > scoring.ram_budget_mb:
        print("\nWARNING: peak RAM exceeds the 7GB budget. Seff will be 0.")

    save_json(args.export, {
        "model": args.model,
        "peak_ram_mb": monitor.peak_ram_mb,
        "peak_cpu_percent": monitor.peak_cpu_percent,
        "seff": seff,
    })
    print(f"\nResults exported to {args.export}")


if __name__ == "__main__":
    main()
