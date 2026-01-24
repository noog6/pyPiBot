"""SQLite-backed storage for user profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from config import ConfigController


def _now_millis() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    name: str | None
    preferences: dict[str, Any]
    last_seen: int | None
    favorites: list[str]


class UserProfileStore:
    """Manage persisted user profiles."""

    def __init__(self, db_path: Path | None = None) -> None:
        config = ConfigController.get_instance().get_config()
        if db_path is None:
            var_dir = Path(config.get("var_dir", "./var/")).expanduser()
            db_path = var_dir / "user_profiles.db"
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
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                preferences JSON,
                last_seen INTEGER,
                favorites JSON
            )
            """
        )
        self._conn.commit()

    def get_profile(self, user_id: str) -> UserProfile | None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT user_id, name, preferences, last_seen, favorites
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        preferences = json.loads(row[2]) if row[2] else {}
        favorites = json.loads(row[4]) if row[4] else []
        return UserProfile(
            user_id=row[0],
            name=row[1],
            preferences=preferences,
            last_seen=row[3],
            favorites=favorites,
        )

    def upsert_profile(self, profile: UserProfile) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO user_profiles (user_id, name, preferences, last_seen, favorites)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    preferences = excluded.preferences,
                    last_seen = excluded.last_seen,
                    favorites = excluded.favorites
                """,
                (
                    profile.user_id,
                    profile.name,
                    json.dumps(profile.preferences),
                    profile.last_seen,
                    json.dumps(profile.favorites),
                ),
            )
            self._conn.commit()

    def update_profile_fields(
        self,
        user_id: str,
        *,
        name: str | None = None,
        preferences: dict[str, Any] | None = None,
        favorites: list[str] | None = None,
        last_seen: int | None = None,
    ) -> UserProfile:
        existing = self.get_profile(user_id)
        profile = UserProfile(
            user_id=user_id,
            name=name if name is not None else (existing.name if existing else None),
            preferences=(
                preferences if preferences is not None else (existing.preferences if existing else {})
            ),
            last_seen=last_seen if last_seen is not None else (existing.last_seen if existing else None),
            favorites=favorites if favorites is not None else (existing.favorites if existing else []),
        )
        self.upsert_profile(profile)
        return profile

    def touch_last_seen(self, user_id: str) -> UserProfile:
        return self.update_profile_fields(user_id, last_seen=_now_millis())

