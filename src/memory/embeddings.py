"""
Embedding client -- STUB, not currently used by vector_store.py.

Loading BAAI/bge-small (or any embedding model) alongside Qwen adds real RAM
and inference-time cost against your 7GB/TPS budgets. Don't wire this in
until you've confirmed (via Phase 8 eval) that keyword retrieval in
vector_store.py is a genuine bottleneck for your task domain.

If you do need it, bge-small-en-v1.5 quantized is a reasonable choice --
it's ~130MB in fp32 and much smaller quantized, small enough to coexist with
Qwen's ~530MB footprint. But it still costs a model load + forward pass per
message stored/queried, which eats into TPS. Benchmark before committing.
"""

from __future__ import annotations


class EmbeddingClient:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def load(self):
        raise NotImplementedError(
            "Embedding model loading not implemented -- see module docstring "
            "for the RAM/TPS tradeoff before wiring this in."
        )

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            self.load()
        raise NotImplementedError("Forward pass through the embedding model.")
