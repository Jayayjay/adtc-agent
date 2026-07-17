"""
Integration tests requiring the actual Qwen GGUF model to be downloaded.
Skipped automatically if the model file isn't present -- these won't run in
CI or this sandbox, only on your dev machine after scripts/download_models.sh.
"""

import pytest

from src.config import load_config
from src.core.agent import Agent

pytestmark = pytest.mark.skipif(
    not load_config().primary_model.path.exists(),
    reason="Qwen model not downloaded -- run scripts/download_models.sh first.",
)


def test_agent_handles_simple_message():
    config = load_config()
    with Agent(config) as agent:
        response = agent.handle_message("Hello, who are you?")
        assert isinstance(response, str)
        assert len(response) > 0


def test_agent_routes_and_falls_back_gracefully_without_hrm():
    """HRM isn't wired up with real weights yet -- confirm the REASONING
    path falls back to the LLM alone rather than crashing."""
    config = load_config()
    with Agent(config) as agent:
        response = agent.handle_message(
            "Schedule three tasks avoiding conflicts, prioritizing the urgent one."
        )
        assert isinstance(response, str)
        assert len(response) > 0
