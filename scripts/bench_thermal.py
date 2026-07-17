"""
Sustained-load thermal check, matching the ADTC thermal penalty condition:
-10 points if throttled or temp > 85C during operation.

Runs repeated inference for a sustained period while monitoring CPU
temperature (where the OS exposes it) and looking for signs of frequency
throttling. This needs to run for real minutes, not seconds -- throttling
is a sustained-load phenomenon, a 10-second burst won't reveal it.

Usage:
    python scripts/bench_thermal.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --duration 300
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
from src.utils.benchmarking import compute_thermal_penalty
from src.utils.helpers import save_json

PROMPT = "Write a detailed explanation of how binary search trees work, with an example."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--duration", type=int, default=300, help="Sustained load duration, seconds")
    parser.add_argument("--export", type=str, default="report/data/thermal_benchmark.json")
    args = parser.parse_args()

    scoring = ADTCScoring()

    print(f"Loading model: {args.model}")
    llm = Llama(model_path=args.model, n_ctx=8192, n_gpu_layers=0, verbose=False)

    print(f"Running sustained load for {args.duration}s -- this needs to run the "
          f"full duration to be meaningful, throttling is a sustained-load effect.")

    temp_samples = []
    tps_samples = []
    start = time.time()

    with ResourceMonitor(interval_s=1.0) as monitor:
        while time.time() - start < args.duration:
            t0 = time.perf_counter()
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=200, temperature=0.4,
            )
            elapsed = time.perf_counter() - t0
            tps = result["usage"]["completion_tokens"] / elapsed if elapsed > 0 else 0.0
            tps_samples.append(tps)

            temp = monitor.get_cpu_temp_c()
            if temp is not None:
                temp_samples.append(temp)

            print(f"  t={time.time()-start:5.0f}s  tps={tps:5.2f}  "
                  f"temp={temp if temp else 'N/A'}")

    peak_temp = max(temp_samples) if temp_samples else None
    # Simple throttling heuristic: did TPS degrade meaningfully over the run?
    # Compare first-quarter vs last-quarter mean TPS.
    q = max(1, len(tps_samples) // 4)
    early_tps = sum(tps_samples[:q]) / q
    late_tps = sum(tps_samples[-q:]) / q
    throttled = late_tps < early_tps * 0.85  # >15% degradation -- treat as throttling signal

    penalty = compute_thermal_penalty(peak_temp, throttled, scoring)

    print("\n--- Results ---")
    print(f"Peak CPU temp: {peak_temp if peak_temp is not None else 'N/A (sensor unavailable)'}")
    print(f"Early TPS avg: {early_tps:.2f}  |  Late TPS avg: {late_tps:.2f}")
    print(f"Throttling detected: {throttled}")
    print(f"Thermal penalty: {penalty}")

    if peak_temp is None:
        print(
            "\nNOTE: no temperature sensor data available (psutil.sensors_temperatures() "
            "returned nothing). This is common in containers/VMs. Run on bare-metal "
            "reference hardware, and consider `sensors` (lm-sensors) as a cross-check."
        )

    save_json(args.export, {
        "model": args.model,
        "duration_s": args.duration,
        "peak_temp_c": peak_temp,
        "early_tps_avg": early_tps,
        "late_tps_avg": late_tps,
        "throttled": throttled,
        "thermal_penalty": penalty,
    })
    print(f"\nResults exported to {args.export}")


if __name__ == "__main__":
    main()
