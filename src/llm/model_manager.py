"""
Load/unload orchestration for the primary (Qwen) and dormant secondary (MiniCPM)
models.

Current design (Option A): only the primary model is ever loaded. The
secondary model's load/unload path is implemented but inert unless
config.enable_secondary_model=True -- flip that only after Phase 1 benchmarks
justify it, per src/config.py's comments.
"""

from __future__ import annotations

import logging

from src.config import SystemConfig
from src.core.exceptions import ModelLoadError
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.primary: LLMClient | None = None
        self.secondary: LLMClient | None = None

    def load_primary(self):
        logger.info("Loading primary model: %s", self.config.primary_model.name)
        self.primary = LLMClient(self.config.primary_model)

    def load_secondary(self):
        """On-demand load of the fallback model. Only called if
        config.enable_secondary_model is True AND the router/agent explicitly
        decides it needs the secondary model for a given request."""
        if not self.config.enable_secondary_model:
            raise ModelLoadError(
                "Secondary model is disabled in config (Option A: Qwen-only). "
                "Set enable_secondary_model=True to use it -- see src/config.py."
            )
        if self.secondary is None:
            logger.info("Loading secondary model on demand: %s", self.config.secondary_model.name)
            self.secondary = LLMClient(self.config.secondary_model)
        return self.secondary

    def unload_secondary(self):
        if self.secondary is not None:
            logger.info("Unloading secondary model to reclaim RAM.")
            self.secondary = None  # llama.cpp releases memory on GC

    def unload_all(self):
        self.primary = None
        self.unload_secondary()
