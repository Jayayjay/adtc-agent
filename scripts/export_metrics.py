"""
Consolidates the individual benchmark JSON outputs (tps, ram, thermal) into
one Sacc/Sperf/Seff/total-score summary for report/data/ -- the numbers you
actually paste into the report's performance section.

Run this after bench_tps.py, bench_ram.py, and bench_thermal.py have all
produced their JSON outputs. Sacc must be supplied manually (it comes from
your own eval/scoring/sacc_scorer.py run, or the judge panel -- there's no
way to benchmark it locally the way TPS/RAM can be).

Usage:
    python scripts/export_metrics.py --sacc 78.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ADTCScoring
from src.utils.benchmarking import compute_total_score
from src.utils.helpers import load_json, save_json

REPORT_DATA = Path(__file__).resolve().parent.parent / "report" / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sacc", type=float, required=True,
                         help="Sacc score (0-100) from your own eval run or judge feedback.")
    args = parser.parse_args()

    scoring = ADTCScoring()

    tps_path = REPORT_DATA / "tps_benchmark.json"
    ram_path = REPORT_DATA / "ram_benchmark.json"
    thermal_path = REPORT_DATA / "thermal_benchmark.json"

    missing = [p.name for p in (tps_path, ram_path, thermal_path) if not p.exists()]
    if missing:
        print(f"Missing benchmark outputs: {missing}")
        print("Run bench_tps.py, bench_ram.py, and bench_thermal.py first.")
        return

    tps_data = load_json(tps_path)
    ram_data = load_json(ram_path)
    thermal_data = load_json(thermal_path)

    sperf = tps_data["sperf"]
    seff = ram_data["seff"]
    thermal_penalty = thermal_data["thermal_penalty"]

    total = compute_total_score(args.sacc, sperf, seff, thermal_penalty, scoring)

    summary = {
        "sacc": args.sacc,
        "sperf": sperf,
        "seff": seff,
        "thermal_penalty": thermal_penalty,
        "total_score": total,
        "mean_tps": tps_data["mean_tps"],
        "peak_ram_mb": ram_data["peak_ram_mb"],
        "peak_temp_c": thermal_data.get("peak_temp_c"),
        "throttled": thermal_data.get("throttled"),
    }

    print("\n--- ADTC Score Summary ---")
    for k, v in summary.items():
        print(f"{k:20s}: {v}")

    save_json(REPORT_DATA / "seff_sperf_calculation.json", summary)
    print(f"\nSaved to {REPORT_DATA / 'seff_sperf_calculation.json'}")


if __name__ == "__main__":
    main()
