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
        columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "source" not in columns:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'manual_tool'"
            )
        if "pinned" not in columns:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0"
            )
        if "needs_review" not in columns:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN needs_review INTEGER DEFAULT 0"
            )
        self._conn.commit()

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

    def delete_memory(self, *, memory_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE memory_id = ?",
                (memory_id,),
            )
            self._conn.commit()
        return cursor.rowcount > 0
