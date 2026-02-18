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
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._retention_policy = self._load_retention_policy(config.get("memory") or {})
        self._initialize_db()

    def _load_retention_policy(self, memory_cfg: dict[str, object]) -> dict[str, object]:
        retention_cfg = memory_cfg.get("retention")
        if not isinstance(retention_cfg, dict):
            retention_cfg = {}

        tiers_cfg = retention_cfg.get("importance_tiers")
        if not isinstance(tiers_cfg, dict):
            tiers_cfg = {"low": [1, 2], "medium": [3, 4], "high": [5]}

        normalized_tiers: dict[str, tuple[int, ...]] = {}
        for tier_name, tier_values in tiers_cfg.items():
            if not isinstance(tier_name, str) or not isinstance(tier_values, list):
                continue
            bounded_values = sorted(
                {
                    int(value)
                    for value in tier_values
                    if isinstance(value, int) and 1 <= int(value) <= 5
                }
            )
            if bounded_values:
                normalized_tiers[tier_name.strip().lower()] = tuple(bounded_values)

        if not normalized_tiers:
            normalized_tiers = {"low": (1, 2), "medium": (3, 4), "high": (5,)}

        by_source_cfg = retention_cfg.get("max_age_days_by_source")
        if not isinstance(by_source_cfg, dict):
            by_source_cfg = {
                "manual_tool": {"low": 365, "medium": None, "high": None},
                "auto_reflection": {"low": 60, "medium": 180, "high": None},
            }

        normalized_by_source: dict[str, dict[str, int]] = {}
        for source, source_cfg in by_source_cfg.items():
            if not isinstance(source, str) or not isinstance(source_cfg, dict):
                continue
            normalized_source = source.strip().lower()
            if not normalized_source:
                continue
            tier_days: dict[str, int] = {}
            for tier_name, days in source_cfg.items():
                normalized_tier = str(tier_name).strip().lower()
                if normalized_tier not in normalized_tiers:
                    continue
                if days is None:
                    continue
                if isinstance(days, bool):
                    continue
                if isinstance(days, (int, float)) and int(days) >= 0:
                    tier_days[normalized_tier] = int(days)
            if tier_days:
                normalized_by_source[normalized_source] = tier_days

        return {
            "importance_tiers": normalized_tiers,
            "max_age_days_by_source": normalized_by_source,
            "protect_review_approved_strategic": bool(
                retention_cfg.get("protect_review_approved_strategic", True)
            ),
            "optimize_min_deleted_rows": max(
                0,
                int(retention_cfg.get("optimize_min_deleted_rows", 200)),
            ),
        }

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

        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

    def get_embedding_coverage_counts(
        self,
        *,
        user_id: str | None,
        scope: str = USER_GLOBAL_SCOPE,
        session_id: str | None = None,
    ) -> tuple[int, int]:
        """Return `(total_memories, ready_embeddings)` for approved memories in scope."""

        normalized_scope = (scope or USER_GLOBAL_SCOPE).strip().lower()
        if normalized_scope == SESSION_LOCAL_SCOPE:
            if session_id is None:
                return (0, 0)
            scope_predicate = "m.session_id = ?"
            scope_params: tuple[object, ...] = (session_id,)
        else:
            scope_predicate = "m.session_id IS NULL"
            scope_params = ()

        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_memories,
                    COALESCE(SUM(CASE WHEN e.status = 'ready' THEN 1 ELSE 0 END), 0) AS ready_embeddings
                FROM memories AS m
                LEFT JOIN memory_embeddings AS e ON e.memory_id = m.memory_id
                WHERE m.user_id IS ?
                  AND m.needs_review = 0
                  AND {scope_predicate}
                """,
                (user_id, *scope_params),
            ).fetchone()
        if row is None:
            return (0, 0)
        return (int(row[0] or 0), int(row[1] or 0))

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

    def prune_memories_by_retention_policy(
        self,
        *,
        now_ms: int | None = None,
        force: bool = False,
    ) -> int:
        """Delete memories that exceed source/tier retention policy."""

        policy = self._retention_policy
        source_policies = policy.get("max_age_days_by_source")
        importance_tiers = policy.get("importance_tiers")
        if not isinstance(source_policies, dict) or not isinstance(importance_tiers, dict):
            return 0

        now_value = now_ms if now_ms is not None else _now_millis()
        deleted_rows = 0

        with self._lock:
            for source_name, tier_policies in source_policies.items():
                if not isinstance(source_name, str) or not isinstance(tier_policies, dict):
                    continue

                normalized_source = source_name.strip()
                if not normalized_source:
                    continue

                for tier_name, max_days in tier_policies.items():
                    if not isinstance(tier_name, str) or not isinstance(max_days, int):
                        continue

                    importance_values = importance_tiers.get(tier_name)
                    if not isinstance(importance_values, tuple) or not importance_values:
                        continue

                    cutoff_ms = now_value - (max_days * 86_400_000)
                    placeholders = ", ".join("?" for _ in importance_values)
                    query_parts = [
                        "DELETE FROM memories",
                        "WHERE source = ?",
                        f"AND importance IN ({placeholders})",
                        "AND timestamp < ?",
                    ]

                    params: list[object] = [normalized_source, *importance_values, cutoff_ms]

                    if not force:
                        query_parts.append("AND pinned = 0")
                        if bool(policy.get("protect_review_approved_strategic", True)):
                            query_parts.append(
                                "AND NOT (needs_review = 0 AND tags LIKE ?)"
                            )
                            params.append('%"strategic"%')

                    cursor = self._conn.execute(" ".join(query_parts), params)
                    deleted_rows += cursor.rowcount

            if deleted_rows:
                self._conn.commit()

        return deleted_rows

    def purge_orphan_embeddings(self) -> int:
        """Delete embedding rows whose source memory no longer exists."""

        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM memory_embeddings
                WHERE memory_id NOT IN (SELECT memory_id FROM memories)
                """
            )
            deleted_rows = int(cursor.rowcount)
            if deleted_rows:
                self._conn.commit()
        return deleted_rows

    def maybe_optimize_storage(self, *, deleted_rows: int, force: bool = False) -> bool:
        """Run SQLite optimize/vacuum only when enough rows were deleted."""

        threshold = int(self._retention_policy.get("optimize_min_deleted_rows", 200))
        should_run = force or deleted_rows >= threshold
        if not should_run:
            return False

        with self._lock:
            self._conn.execute("PRAGMA optimize")
            self._conn.execute("VACUUM")
            self._conn.commit()
        return True
