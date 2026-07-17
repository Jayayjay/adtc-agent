"""
Memory store wrapper.

IMPORTANT SCOPE NOTE: this ships as a working plain-SQLite store with simple
keyword-overlap retrieval, NOT true vector similarity search. Reasons:

  1. Real vector search (sqlite-vec + an embedding model like BAAI/bge-small)
     means loading a SECOND model alongside Qwen -- another chunk of your
     7GB RAM budget and another thing that can slow down TPS if it runs on
     every message. That's a real cost, not a free upgrade.
  2. For most agent-orchestration demo tasks, a small number of recent
     turns plus simple keyword matching gets you 80% of the practical value
     with none of the RAM/complexity cost.

Upgrade to real embeddings (see embeddings.py, currently a stub) only if you
find during Phase 8 eval that keyword retrieval is actually causing missed
context in your task set -- don't add it preemptively.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.memory.schemas import SCHEMA


class VectorStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))

    def init_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert(self, session_id: str, role: str, content: str):
        self._conn.execute(
            "INSERT INTO memory (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self._conn.commit()

    def query_keyword(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Naive keyword-overlap retrieval: scores rows by how many query words
        appear in their content. Not semantic search -- see module docstring.
        """
        words = [w.lower() for w in query.split() if len(w) > 2]
        if not words:
            return []

        like_clauses = " OR ".join(["LOWER(content) LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        cursor = self._conn.execute(
            f"SELECT role, content, created_at FROM memory WHERE {like_clauses} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, top_k),
        )
        return [
            {"role": row[0], "content": row[1], "created_at": row[2]}
            for row in cursor.fetchall()
        ]

    def close(self):
        self._conn.close()
