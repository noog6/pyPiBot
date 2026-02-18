"""SQLite-backed storage for memory entries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterable

from config import ConfigController

USER_GLOBAL_SCOPE = "user_global"
SESSION_LOCAL_SCOPE = "session_local"


def _now_millis() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class MemoryEntry:
    memory_id: int
    timestamp: int
    user_id: str | None
    session_id: str | None
    content: str
    tags: list[str]
    importance: int
    source: str
    pinned: bool
    needs_review: bool


@dataclass(frozen=True)
class MemoryEmbedding:
    memory_id: int
    model_id: str
    dim: int
    vector: bytes
    vector_norm: float | None
    updated_at: int | None
    status: str
    error: str | None


class MemoryStore:
    """Manage persisted memory entries."""

    def __init__(self, db_path: Path | None = None) -> None:
        config = ConfigController.get_instance().get_config()
        if db_path is None:
            var_dir = Path(config.get("var_dir", "./var/")).expanduser()
            db_path = var_dir / "memories.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._initialize_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _initialize_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
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
        self._ensure_table_columns(
            table_name="memories",
            columns={
                "source": "TEXT DEFAULT 'manual_tool'",
                "pinned": "INTEGER DEFAULT 0",
                "needs_review": "INTEGER DEFAULT 0",
            },
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
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
        self._ensure_table_columns(
            table_name="memory_embeddings",
            columns={
                "model_id": "TEXT NOT NULL DEFAULT ''",
                "dim": "INTEGER NOT NULL DEFAULT 0",
                "vector": "BLOB NOT NULL DEFAULT X''",
                "vector_norm": "REAL",
                "updated_at": "INTEGER",
                "status": "TEXT DEFAULT 'ready'",
                "error": "TEXT",
            },
        )
        self._conn.commit()

    def _ensure_table_columns(self, *, table_name: str, columns: dict[str, str]) -> None:
        cursor = self._conn.cursor()
        existing = {
            row[1]
            for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for name, sql_type in columns.items():
            if name in existing:
                continue
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {sql_type}")

    def append_memory(
        self,
        *,
        content: str,
        tags: Iterable[str],
        importance: int,
        user_id: str | None = None,
        session_id: str | None = None,
        timestamp: int | None = None,
        source: str = "manual_tool",
        pinned: bool = False,
        needs_review: bool = False,
    ) -> MemoryEntry:
        entry_timestamp = timestamp if timestamp is not None else _now_millis()
        entry_tags = list(tags)
        entry_source = source.strip() if isinstance(source, str) and source.strip() else "manual_tool"
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO memories (
                    timestamp, user_id, session_id, content, tags, importance, source, pinned, needs_review
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_timestamp,
                    user_id,
                    session_id,
                    content,
                    json.dumps(entry_tags),
                    importance,
                    entry_source,
                    int(bool(pinned)),
                    int(bool(needs_review)),
                ),
            )
            self._conn.commit()
            memory_id = int(cursor.lastrowid)
        return MemoryEntry(
            memory_id=memory_id,
            timestamp=entry_timestamp,
            user_id=user_id,
            session_id=session_id,
            content=content,
            tags=entry_tags,
            importance=importance,
            source=entry_source,
            pinned=bool(pinned),
            needs_review=bool(needs_review),
        )

    def search_memories(
        self,
        *,
        query: str | None = None,
        limit: int = 5,
        user_id: str | None = None,
        scope: str = USER_GLOBAL_SCOPE,
        session_id: str | None = None,
        pinned_only: bool = False,
        review_state: str | None = None,
    ) -> list[MemoryEntry]:
        query_parts = [
            (
                "SELECT memory_id, timestamp, user_id, session_id, content, tags, "
                "importance, source, pinned, needs_review"
            ),
            "FROM memories",
            "WHERE 1 = 1",
        ]
        params: list[object] = []
        if user_id is not None:
            query_parts.append("AND user_id = ?")
            params.append(user_id)

        scope_value = (scope or USER_GLOBAL_SCOPE).strip().lower()
        if scope_value == SESSION_LOCAL_SCOPE:
            if not session_id:
                return []
            query_parts.append("AND session_id = ?")
            params.append(session_id)
        else:
            # User-global reads must explicitly exclude session-local rows.
            query_parts.append("AND session_id IS NULL")

        if pinned_only:
            query_parts.append("AND pinned = 1")
        if review_state == "approved":
            query_parts.append("AND needs_review = 0")
        elif review_state == "needs_review":
            query_parts.append("AND needs_review = 1")

        if query:
            query_parts.append("AND (content LIKE ? OR tags LIKE ?)")
            like_term = f"%{query}%"
            params.extend([like_term, like_term])
        query_parts.append("ORDER BY importance DESC, timestamp DESC")
        query_parts.append("LIMIT ?")
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(" ".join(query_parts), params)
        results: list[MemoryEntry] = []
        for row in cursor.fetchall():
            (
                memory_id,
                timestamp,
                user_id_value,
                session_id_value,
                content,
                tags_json,
                importance,
                source,
                pinned,
                needs_review,
            ) = row
            tags = json.loads(tags_json) if tags_json else []
            results.append(
                MemoryEntry(
                    memory_id=int(memory_id),
                    timestamp=int(timestamp),
                    user_id=user_id_value,
                    session_id=session_id_value,
                    content=content,
                    tags=tags,
                    importance=int(importance),
                    source=(source or "manual_tool"),
                    pinned=bool(pinned),
                    needs_review=bool(needs_review),
                )
            )
        return results

    def upsert_memory_embedding(
        self,
        *,
        memory_id: int,
        model_id: str,
        dim: int,
        vector: bytes,
        vector_norm: float | None = None,
        updated_at: int | None = None,
        status: str = "ready",
        error: str | None = None,
    ) -> None:
        entry_updated_at = updated_at if updated_at is not None else _now_millis()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_embeddings (
                    memory_id, model_id, dim, vector, vector_norm, updated_at, status, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    model_id = excluded.model_id,
                    dim = excluded.dim,
                    vector = excluded.vector,
                    vector_norm = excluded.vector_norm,
                    updated_at = excluded.updated_at,
                    status = excluded.status,
                    error = excluded.error
                """,
                (
                    memory_id,
                    model_id,
                    dim,
                    sqlite3.Binary(vector),
                    vector_norm,
                    entry_updated_at,
                    status,
                    error,
                ),
            )
            self._conn.commit()

    def fetch_embeddings_for_memories(self, *, memory_ids: Iterable[int]) -> dict[int, MemoryEmbedding]:
        requested_ids = [int(memory_id) for memory_id in memory_ids]
        if not requested_ids:
            return {}
        placeholders = ", ".join("?" for _ in requested_ids)
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT memory_id, model_id, dim, vector, vector_norm, updated_at, status, error
            FROM memory_embeddings
            WHERE memory_id IN ({placeholders})
            """,
            requested_ids,
        )
        return {
            int(memory_id): MemoryEmbedding(
                memory_id=int(memory_id),
                model_id=model_id,
                dim=int(dim),
                vector=bytes(vector),
                vector_norm=float(vector_norm) if vector_norm is not None else None,
                updated_at=int(updated_at) if updated_at is not None else None,
                status=(status or "ready"),
                error=error,
            )
            for (
                memory_id,
                model_id,
                dim,
                vector,
                vector_norm,
                updated_at,
                status,
                error,
            ) in cursor.fetchall()
        }

    def enqueue_memory_embedding(self, *, memory_id: int, updated_at: int | None = None) -> None:
        """Mark a memory embedding row as pending without blocking memory writes."""

        self.upsert_memory_embedding(
            memory_id=memory_id,
            model_id="",
            dim=0,
            vector=b"",
            vector_norm=None,
            updated_at=updated_at,
            status="pending",
            error=None,
        )

    def list_recent_memories_missing_embeddings(self, *, limit: int) -> list[int]:
        bounded_limit = max(1, int(limit))
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT m.memory_id
            FROM memories AS m
            LEFT JOIN memory_embeddings AS e ON e.memory_id = m.memory_id
            WHERE e.memory_id IS NULL
            ORDER BY m.timestamp DESC, m.memory_id DESC
            LIMIT ?
            """,
            (bounded_limit,),
        )
        return [int(memory_id) for (memory_id,) in cursor.fetchall()]

    def fetch_pending_memories_for_embedding(self, *, limit: int) -> list[MemoryEntry]:
        """Return pending memory rows that still need embeddings."""

        bounded_limit = max(1, int(limit))
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                m.memory_id,
                m.timestamp,
                m.user_id,
                m.session_id,
                m.content,
                m.tags,
                m.importance,
                m.source,
                m.pinned,
                m.needs_review
            FROM memories AS m
            INNER JOIN memory_embeddings AS e ON e.memory_id = m.memory_id
            WHERE e.status = 'pending'
            ORDER BY COALESCE(e.updated_at, m.timestamp) ASC, m.memory_id ASC
            LIMIT ?
            """,
            (bounded_limit,),
        )

        entries: list[MemoryEntry] = []
        for row in cursor.fetchall():
            (
                memory_id,
                timestamp,
                user_id_value,
                session_id_value,
                content,
                tags_json,
                importance,
                source,
                pinned,
                needs_review,
            ) = row
            tags = json.loads(tags_json) if tags_json else []
            entries.append(
                MemoryEntry(
                    memory_id=int(memory_id),
                    timestamp=int(timestamp),
                    user_id=user_id_value,
                    session_id=session_id_value,
                    content=content,
                    tags=tags,
                    importance=int(importance),
                    source=(source or "manual_tool"),
                    pinned=bool(pinned),
                    needs_review=bool(needs_review),
                )
            )
        return entries

    def delete_memory_embedding(self, *, memory_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self, *, memory_id: int) -> bool:
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            )
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE memory_id = ?",
                (memory_id,),
            )
            self._conn.commit()
        return cursor.rowcount > 0
