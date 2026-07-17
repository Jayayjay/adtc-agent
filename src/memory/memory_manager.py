"""
Memory manager: the interface src/core/agent.py actually talks to. Wraps
VectorStore (currently keyword-based, see vector_store.py docstring) and
will later wrap real embedding-based retrieval if/when that's justified.
"""

from __future__ import annotations

from pathlib import Path

from src.memory.vector_store import VectorStore


class MemoryManager:
    def __init__(self, db_path: str | Path):
        self._store = VectorStore(db_path)

    def init_schema(self):
        self._store.init_schema()

    def store(self, session_id: str, user_message: str, agent_response: str):
        self._store.insert(session_id, "user", user_message)
        self._store.insert(session_id, "agent", agent_response)

    def query(self, message: str, top_k: int = 3) -> list[dict]:
        return self._store.query_keyword(message, top_k=top_k)

    def close(self):
        self._store.close()
