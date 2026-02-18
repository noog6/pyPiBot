"""Tests for memory embedding sidecar storage."""

from __future__ import annotations

from pathlib import Path
import sqlite3

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


def test_memory_embedding_crud_lifecycle_including_pending_and_ready(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="tracks running cadence", tags=["fitness"], importance=3, user_id="u")

    store.enqueue_memory_embedding(memory_id=entry.memory_id, updated_at=100)
    pending = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert pending.status == "pending"
    assert pending.model_id == ""
    assert pending.dim == 0

    store.upsert_memory_embedding(
        memory_id=entry.memory_id,
        model_id="local_light-v1",
        dim=3,
        vector=b"xyz",
        vector_norm=1.0,
        updated_at=120,
        status="ready",
        error=None,
    )
    ready = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert ready.status == "ready"
    assert ready.model_id == "local_light-v1"
    assert ready.error is None

    store.upsert_memory_embedding(
        memory_id=entry.memory_id,
        model_id="local_light-v1",
        dim=3,
        vector=b"xyz",
        vector_norm=1.0,
        updated_at=140,
        status="failed",
        error="provider timeout",
    )
    failed = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert failed.status == "failed"
    assert failed.error == "provider timeout"

    assert store.delete_memory_embedding(memory_id=entry.memory_id) is True
    assert store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id]) == {}


def test_memory_embedding_migration_adds_required_columns_with_defaults(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    db_path = tmp_path / "memories.db"

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
    conn.execute(
        """
        CREATE TABLE memory_embeddings (
            memory_id INTEGER PRIMARY KEY,
            model_id TEXT,
            dim INTEGER,
            vector BLOB
        )
        """
    )
    conn.execute("INSERT INTO memory_embeddings (memory_id, model_id, dim, vector) VALUES (1, 'legacy', 2, X'0102')")
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=db_path)
    fetched = store.fetch_embeddings_for_memories(memory_ids=[1])[1]
    assert fetched.model_id == "legacy"
    assert fetched.dim == 2
    assert fetched.status == "ready"
    assert fetched.error is None
