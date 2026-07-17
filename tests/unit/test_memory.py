from src.memory.memory_manager import MemoryManager


def test_store_and_query_roundtrip(tmp_path):
    db_path = tmp_path / "memory.db"
    mgr = MemoryManager(db_path)
    mgr.init_schema()

    mgr.store("session1", "What's the weather in Abuja?", "It's sunny in Abuja.")
    mgr.store("session1", "Schedule a meeting tomorrow", "Meeting scheduled for tomorrow.")

    results = mgr.query("weather Abuja")
    assert len(results) >= 1
    assert any("weather" in r["content"].lower() or "sunny" in r["content"].lower() for r in results)

    mgr.close()


def test_query_returns_empty_for_no_match(tmp_path):
    db_path = tmp_path / "memory.db"
    mgr = MemoryManager(db_path)
    mgr.init_schema()
    mgr.store("session1", "hello", "hi there")

    results = mgr.query("xyzxyzxyz nonexistent query terms")
    assert results == []

    mgr.close()
