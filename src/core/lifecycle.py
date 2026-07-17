"""
Startup/shutdown lifecycle for the agent process.

Keeping this separate from Agent itself so resource acquisition/release is
auditable in one place -- matters for the RAM benchmark (bench_ram.py) since
you want a clean, deterministic point where "peak RAM" measurement starts.
"""

from __future__ import annotations

import logging

from src.config import SystemConfig
from src.llm.model_manager import ModelManager
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class Lifecycle:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model_manager: ModelManager | None = None
        self.memory_manager: MemoryManager | None = None

    def startup(self):
        logger.info("Starting agent: loading primary model (%s)", self.config.primary_model.name)
        self.model_manager = ModelManager(self.config)
        self.model_manager.load_primary()

        logger.info("Initializing memory store at %s", self.config.memory_db_path)
        self.memory_manager = MemoryManager(self.config.memory_db_path)
        self.memory_manager.init_schema()

        logger.info("Startup complete.")

    def shutdown(self):
        if self.model_manager:
            self.model_manager.unload_all()
        if self.memory_manager:
            self.memory_manager.close()
        logger.info("Shutdown complete.")

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, *exc):
        self.shutdown()
