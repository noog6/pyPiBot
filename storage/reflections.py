"""SQLite-backed storage for reflection entries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterable

from config import ConfigController


def _now_millis() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class ReflectionEntry:
    timestamp: int
    user_id: str | None
    session_id: str | None
    summary: str
    lessons: list[str]


class ReflectionStore:
    """Manage persisted reflection entries."""

    def __init__(self, db_path: Path | None = None) -> None:
        config = ConfigController.get_instance().get_config()
        if db_path is None:
            var_dir = Path(config.get("var_dir", "./var/")).expanduser()
            db_path = var_dir / "reflections.db"
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
            CREATE TABLE IF NOT EXISTS reflections (
                reflection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                user_id TEXT,
                session_id TEXT,
                summary TEXT,
                lessons JSON
            )
            """
        )
        self._conn.commit()

    def append_reflection(
        self,
        *,
        summary: str,
        lessons: Iterable[str],
        user_id: str | None = None,
        session_id: str | None = None,
        timestamp: int | None = None,
    ) -> ReflectionEntry:
        entry = ReflectionEntry(
            timestamp=timestamp if timestamp is not None else _now_millis(),
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            lessons=list(lessons),
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reflections (timestamp, user_id, session_id, summary, lessons)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entry.timestamp,
                    entry.user_id,
                    entry.session_id,
                    entry.summary,
                    json.dumps(entry.lessons),
                ),
            )
            self._conn.commit()
        return entry

    def get_recent_lessons(
        self,
        *,
        limit: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[str]:
        query = [
            "SELECT lessons",
            "FROM reflections",
            "WHERE 1 = 1",
        ]
        params: list[object] = []
        if user_id is not None:
            query.append("AND user_id = ?")
            params.append(user_id)
        if session_id is not None:
            query.append("AND session_id = ?")
            params.append(session_id)
        query.append("ORDER BY timestamp DESC")
        query.append("LIMIT ?")
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(" ".join(query), params)
        lessons: list[str] = []
        for (lessons_json,) in cursor.fetchall():
            if lessons_json:
                lessons.extend(json.loads(lessons_json))
        return lessons
