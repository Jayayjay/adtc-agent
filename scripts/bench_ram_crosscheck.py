"""
Cross-check RAM benchmark using `/usr/bin/time -v`, which captures peak RSS
at the OS level across the whole process lifetime -- catches spikes that a
polling-based monitor (src/utils/resource_monitor.py, used by bench_ram.py)
could theoretically miss between samples.

Run this ONCE to validate bench_ram.py's numbers agree before finalizing
report figures. If they disagree by a meaningful margin, trust this one --
polling interval can miss brief peaks that OS-level tracking won't.

Usage:
    python scripts/bench_ram_crosscheck.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ADTCScoring
from src.utils.benchmarking import compute_seff
from src.utils.helpers import save_json

INFERENCE_SNIPPET = """
import sys
from llama_cpp import Llama
llm = Llama(model_path="{model_path}", n_ctx={n_ctx}, n_gpu_layers=0, verbose=False)
llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Walk through planning a multi-step task with tool dependencies."}}],
    max_tokens={max_tokens}, temperature=0.4,
)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--export", type=str, default="report/data/ram_crosscheck.json")
    args = parser.parse_args()

    scoring = ADTCScoring()

    if not subprocess.run(["which", "time"], capture_output=True).stdout and not Path("/usr/bin/time").exists():
        print("WARNING: /usr/bin/time not found. On Ubuntu: `sudo apt install time`.")
        return

    snippet = INFERENCE_SNIPPET.format(
        model_path=args.model, n_ctx=args.n_ctx, max_tokens=args.max_tokens
    )

    cmd = ["/usr/bin/time", "-v", "python3", "-c", snippet]
    print(f"Running: {' '.join(cmd[:2])} ... (inline script omitted)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr + result.stdout

    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", output)
    if not match:
        print("Could not parse peak RSS from /usr/bin/time output. Raw output:")
        print(output[-2000:])
        return

    peak_kb = int(match.group(1))
    peak_mb = peak_kb / 1024
    seff = compute_seff(peak_mb, scoring)

    print("\n--- Cross-check Results (/usr/bin/time -v) ---")
    print(f"Peak RSS: {peak_mb:.1f} MB")
    print(f"Estimated Seff: {seff:.1f}")
    print(
        "\nCompare this against report/data/ram_benchmark.json (from bench_ram.py). "
        "If they diverge by more than a few percent, investigate before reporting either number."
    )

    save_json(args.export, {"model": args.model, "peak_ram_mb": peak_mb, "seff": seff})


if __name__ == "__main__":
    main()
