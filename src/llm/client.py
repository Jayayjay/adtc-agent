"""
Unified GGUF client. Same interface regardless of which model (Qwen or the
dormant MiniCPM fallback) is loaded, so the rest of the codebase doesn't care
which one it's talking to.
"""

from __future__ import annotations

from pathlib import Path

from src.config import ModelConfig
from src.core.exceptions import ModelLoadError

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None


class LLMClient:
    def __init__(self, config: ModelConfig):
        if Llama is None:
            raise ModelLoadError(
                "llama-cpp-python is not installed. Run `pip install -r requirements.txt`."
            )
        if not Path(config.path).exists():
            raise ModelLoadError(
                f"Model not found at {config.path}. Run scripts/download_models.sh first."
            )
        self.config = config
        self._llm = Llama(
            model_path=str(config.path),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )

    @property
    def name(self) -> str:
        return self.config.name

    def chat(self, messages: list[dict], **overrides) -> str:
        params = {
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        result = self._llm.create_chat_completion(messages=messages, **params)
        return result["choices"][0]["message"]["content"]

    def chat_with_usage(self, messages: list[dict], **overrides) -> tuple[str, dict]:
        """Same as chat(), but also returns the usage dict (for TPS logging)."""
        params = {
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        result = self._llm.create_chat_completion(messages=messages, **params)
        return result["choices"][0]["message"]["content"], result["usage"]
