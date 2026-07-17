"""
Background resource monitor: RAM (RSS), CPU%, and temperature (where
available). Used both by the standalone benchmark scripts (scripts/bench_*.py)
and optionally live during agent operation for self-checks against the ADTC
budgets in src/config.py's ADTCScoring.
"""

from __future__ import annotations

import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak_ram_mb: float = 0.0
        self.peak_cpu_percent: float = 0.0
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        proc = psutil.Process()
        proc.cpu_percent()  # prime the internal counter (first call is always 0.0)
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    rss += child.memory_info().rss
                ram_mb = rss / (1024 * 1024)
                cpu_pct = proc.cpu_percent()

                self.peak_ram_mb = max(self.peak_ram_mb, ram_mb)
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_pct)
                self.samples.append({"t": time.time(), "ram_mb": ram_mb, "cpu_pct": cpu_pct})
            except psutil.NoSuchProcess:
                pass
            time.sleep(self.interval_s)

    def get_cpu_temp_c(self) -> float | None:
        """
        Best-effort CPU temperature read. Returns None if unavailable --
        psutil.sensors_temperatures() is Linux-only and not all systems
        expose it. On the ADTC Ubuntu 22.04 reference environment this
        should generally work via lm-sensors, but don't assume it will --
        have a fallback plan for the thermal check in bench_thermal.py.
        """
        if not hasattr(psutil, "sensors_temperatures"):
            return None
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        # Prefer a 'coretemp' or 'k10temp' entry if present (Intel/AMD respectively)
        for key in ("coretemp", "k10temp"):
            if key in temps and temps[key]:
                return temps[key][0].current
        # Fallback: first available sensor
        first_group = next(iter(temps.values()))
        return first_group[0].current if first_group else None

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
