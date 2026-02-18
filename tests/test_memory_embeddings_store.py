"""Tests for memory embedding sidecar storage."""

from __future__ import annotations

from pathlib import Path

from config.controller import ConfigController
from storage.memories import MemoryStore


def _reset_singletons() -> None:
    ConfigController._instance = None


def _configure(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("assistant_name: Theo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _reset_singletons()


def test_memory_embedding_upsert_fetch_and_delete(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="loves tea", tags=["prefs"], importance=3, user_id="u")

    store.upsert_memory_embedding(
        memory_id=entry.memory_id,
        model_id="local_light-v1",
        dim=4,
        vector=b"\x01\x02\x03\x04",
        vector_norm=1.0,
        status="ready",
    )

    fetched = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])
    assert entry.memory_id in fetched
    assert fetched[entry.memory_id].model_id == "local_light-v1"
    assert fetched[entry.memory_id].dim == 4

    assert store.delete_memory_embedding(memory_id=entry.memory_id) is True
    assert store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id]) == {}


def test_delete_memory_removes_embedding_row(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="plays piano", tags=["music"], importance=4, user_id="u")
    store.upsert_memory_embedding(
        memory_id=entry.memory_id,
        model_id="local_light-v1",
        dim=3,
        vector=b"abc",
    )

    assert store.delete_memory(memory_id=entry.memory_id) is True
    assert store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id]) == {}


def test_memory_embedding_schema_migration_is_additive(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    db_path = tmp_path / "memories.db"

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE memories (
            memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            user_id TEXT,
            session_id TEXT,
            content TEXT,
            tags JSON,
            importance INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE memory_embeddings (memory_id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=db_path)
    store.upsert_memory_embedding(
        memory_id=999,
        model_id="local_light-v1",
        dim=2,
        vector=b"zz",
    )
    fetched = store.fetch_embeddings_for_memories(memory_ids=[999])
    assert fetched[999].status == "ready"
