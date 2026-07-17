"""SQLite schema for the memory store. Plain SQLite (not sqlite-vec) by
default -- see vector_store.py docstring for why."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- 'user' or 'agent'
    content TEXT NOT NULL,
    embedding BLOB,                -- populated only if embeddings.py is wired up
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at);
"""
