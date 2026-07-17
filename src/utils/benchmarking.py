"""
Shared helper functions for computing ADTC scoring components from raw
measurements. Used by scripts/bench_tps.py, bench_ram.py, bench_thermal.py,
and export_metrics.py so the formula is defined in exactly one place.
"""

from __future__ import annotations

from src.config import ADTCScoring


def compute_sperf(tps_actual: float, scoring: ADTCScoring) -> float:
    """Sperf = 100 * (TPS_actual / TPS_reference), capped at 100."""
    return min(100.0, 100.0 * (tps_actual / scoring.tps_reference))


def compute_seff(peak_ram_mb: float, scoring: ADTCScoring) -> float:
    """Seff = 100 * ((budget - peak_ram) / budget), floored at 0."""
    return max(0.0, 100.0 * ((scoring.ram_budget_mb - peak_ram_mb) / scoring.ram_budget_mb))


def compute_thermal_penalty(peak_temp_c: float | None, throttled: bool, scoring: ADTCScoring) -> float:
    """Returns 0 or the thermal penalty (negative number)."""
    if throttled:
        return scoring.thermal_penalty
    if peak_temp_c is not None and peak_temp_c > scoring.thermal_limit_c:
        return scoring.thermal_penalty
    return 0.0


def compute_total_score(sacc: float, sperf: float, seff: float, thermal_penalty: float,
                         scoring: ADTCScoring) -> float:
    """
    Stotal = 0.50*Sacc + 0.30*Sperf + 0.20*Seff - Pthermal
    (thermal_penalty should be passed as a non-positive number, e.g. -10 or 0)
    """
    return (
        scoring.weight_sacc * sacc
        + scoring.weight_sperf * sperf
        + scoring.weight_seff * seff
        + thermal_penalty  # already negative or zero
    )
