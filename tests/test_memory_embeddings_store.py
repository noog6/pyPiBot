"""Tests for memory embedding sidecar storage."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading

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


def test_memory_store_initialization_adds_expected_indexes_to_legacy_db(tmp_path: Path, monkeypatch) -> None:
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
            importance INTEGER,
            source TEXT DEFAULT 'manual_tool',
            pinned INTEGER DEFAULT 0,
            needs_review INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE memory_embeddings (
            memory_id INTEGER PRIMARY KEY,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            vector_norm REAL,
            updated_at INTEGER,
            status TEXT DEFAULT 'ready',
            error TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    MemoryStore(db_path=db_path)

    conn = sqlite3.connect(db_path)
    indexes = {
        row[1]
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    conn.close()

    assert "idx_memories_scope_review_priority" in indexes
    assert "idx_memories_scope_pinned_review_priority" in indexes
    assert "idx_memory_embeddings_status_updated_memory" in indexes


def test_memory_store_index_migration_is_idempotent_and_non_breaking(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    db_path = tmp_path / "memories.db"
    store = MemoryStore(db_path=db_path)

    entry = store.append_memory(
        content="uses startup digest",
        tags=["digest"],
        importance=5,
        user_id="u",
        pinned=True,
    )
    store.enqueue_memory_embedding(memory_id=entry.memory_id, updated_at=123)

    # Re-running initialization should preserve behavior and not fail on pre-existing indexes.
    MemoryStore(db_path=db_path)

    pending = store.fetch_pending_memories_for_embedding(limit=10)
    assert [item.memory_id for item in pending] == [entry.memory_id]


def test_memory_store_shared_connection_handles_interleaved_read_write_threads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")

    start_gate = threading.Barrier(3)
    errors: list[Exception] = []

    def writer_worker() -> None:
        try:
            start_gate.wait()
            for index in range(120):
                entry = store.append_memory(
                    content=f"note-{index}",
                    tags=["race"],
                    importance=3,
                    user_id="u",
                )
                store.enqueue_memory_embedding(memory_id=entry.memory_id)
                if index % 2 == 0:
                    store.upsert_memory_embedding(
                        memory_id=entry.memory_id,
                        model_id="local_light-v1",
                        dim=2,
                        vector=b"ok",
                        status="ready",
                    )
        except Exception as exc:  # pragma: no cover - fails test via assertion below
            errors.append(exc)

    def reader_worker() -> None:
        try:
            start_gate.wait()
            for _ in range(120):
                store.search_memories(limit=10, user_id="u")
                pending = store.fetch_pending_memories_for_embedding(limit=10)
                store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id for entry in pending])
                store.list_recent_memories_missing_embeddings(limit=10)
        except Exception as exc:  # pragma: no cover - fails test via assertion below
            errors.append(exc)

    writer = threading.Thread(target=writer_worker)
    reader = threading.Thread(target=reader_worker)
    writer.start()
    reader.start()
    start_gate.wait()
    writer.join(timeout=10)
    reader.join(timeout=10)

    assert writer.is_alive() is False
    assert reader.is_alive() is False
    assert errors == []


def test_get_embedding_coverage_counts_applies_scope_predicates(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")

    global_ready = store.append_memory(
        content="global-ready",
        tags=["scope"],
        importance=3,
        user_id="u1",
    )
    global_pending = store.append_memory(
        content="global-pending",
        tags=["scope"],
        importance=3,
        user_id="u1",
    )
    global_error = store.append_memory(
        content="global-error",
        tags=["scope"],
        importance=3,
        user_id="u1",
    )
    session_ready = store.append_memory(
        content="session-ready",
        tags=["scope"],
        importance=3,
        user_id="u1",
        session_id="s1",
    )
    session_other = store.append_memory(
        content="session-other",
        tags=["scope"],
        importance=3,
        user_id="u1",
        session_id="s2",
    )
    store.append_memory(
        content="global-reviewed",
        tags=["scope"],
        importance=3,
        user_id="u1",
        needs_review=True,
    )
    store.append_memory(
        content="other-user-global",
        tags=["scope"],
        importance=3,
        user_id="u2",
    )

    store.upsert_memory_embedding(
        memory_id=global_ready.memory_id,
        model_id="m",
        dim=2,
        vector=b"ok",
        status="ready",
    )
    store.enqueue_memory_embedding(memory_id=global_pending.memory_id)
    store.upsert_memory_embedding(
        memory_id=global_error.memory_id,
        model_id="m",
        dim=2,
        vector=b"ok",
        status="error",
    )
    store.upsert_memory_embedding(
        memory_id=session_ready.memory_id,
        model_id="m",
        dim=2,
        vector=b"ok",
        status="ready",
    )
    store.enqueue_memory_embedding(memory_id=session_other.memory_id)

    assert store.get_embedding_coverage_counts(user_id="u1", scope="user_global") == (3, 1)
    assert store.get_embedding_coverage_counts(user_id="u1", scope="session_local", session_id="s1") == (1, 1)
    assert store.get_embedding_coverage_counts(user_id="u1", scope="session_local", session_id="s2") == (1, 0)
    assert store.get_embedding_coverage_counts(user_id="u1", scope="session_local", session_id=None) == (0, 0)
    assert store.get_embedding_backlog_counts(user_id="u1", scope="user_global") == (1, 0, 1)
    assert store.get_embedding_backlog_counts(user_id="u1", scope="session_local", session_id="s1") == (0, 0, 0)
    assert store.get_embedding_backlog_counts(user_id="u1", scope="session_local", session_id="s2") == (1, 0, 0)
    assert store.get_embedding_backlog_counts(user_id="u1", scope="session_local", session_id=None) == (0, 0, 0)
