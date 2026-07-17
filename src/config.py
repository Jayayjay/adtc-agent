"""
Central configuration for the ADTC agent. Single source of truth for paths,
model settings, and scoring-relevant constants (so benchmark scripts and the
live agent never disagree about budgets/thresholds).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"


@dataclass
class ADTCScoring:
    """Constants pulled directly from the ADTC 2026 scoring spec. Keep this in
    sync with report/REPORT_TEMPLATE_NOTES.md if the spec changes."""
    tps_reference: float = 15.0          # provisional, per challenge brief
    ram_budget_mb: float = 7000.0        # 7 GB
    thermal_limit_c: float = 85.0
    thermal_penalty: float = -10.0

    weight_sacc: float = 0.50
    weight_sperf: float = 0.30
    weight_seff: float = 0.20


@dataclass
class ModelConfig:
    name: str
    path: Path
    n_ctx: int = 8192
    n_threads: int | None = None
    n_gpu_layers: int = 0  # ADTC reference hardware: CPU-only
    temperature: float = 0.4
    max_tokens: int = 512


@dataclass
class SystemConfig:
    # Primary (fast-path) model
    primary_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="qwen3.5-0.8b",
            path=MODELS_DIR / "Qwen3.5-0.8B-Q4_K_M.gguf",
        )
    )
    # Secondary (fallback / higher-reasoning) model — only loaded on demand.
    # See src/llm/model_manager.py for load/unload strategy.
    secondary_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="minicpm5-1b",
            path=MODELS_DIR / "MiniCPM5-1B-Q4_K_M.gguf",
        )
    )

    # Whether the secondary model is enabled at all. Defaults to False --
    # Option A (Qwen-only) is the current design decision (lowest TPS/RAM
    # risk, dominant Sacc weight doesn't clearly favor the added complexity).
    # Flip to True only after Phase 1 benchmarks justify revisiting the
    # hybrid design, and only with a resolved answer on how Sperf is scored
    # across a mixed workload -- see report/REPORT_TEMPLATE_NOTES.md.
    enable_secondary_model: bool = False

    hrm_checkpoint_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "src" / "hrm" / "trained_models"
    )

    memory_db_path: Path = field(default_factory=lambda: DATA_DIR / "memory.db")
    log_dir: Path = field(default_factory=lambda: LOGS_DIR)

    scoring: ADTCScoring = field(default_factory=ADTCScoring)

    def ensure_dirs(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> SystemConfig:
    """Load config, allowing environment variable overrides (see .env.example)."""
    config = SystemConfig()
    if os.getenv("ADTC_DISABLE_SECONDARY_MODEL"):
        config.enable_secondary_model = False
    if os.getenv("ADTC_N_THREADS"):
        config.primary_model.n_threads = int(os.getenv("ADTC_N_THREADS"))
        config.secondary_model.n_threads = int(os.getenv("ADTC_N_THREADS"))
    config.ensure_dirs()
    return config
