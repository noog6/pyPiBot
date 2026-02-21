"""SQLite-backed storage for research budget state and usage audit records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import threading

from storage.controller import StorageController


@dataclass(frozen=True)
class ResearchBudgetState:
    """Current budget state for a named research budget key."""

    key: str
    date_utc: str
    remaining: int
    limit: int
    updated_at_ts: int


class ResearchBudgetStorage:
    """Persist research budget state and daily spend audit rows."""

    def __init__(self) -> None:
        storage_controller = StorageController.get_instance()
        self._conn = storage_controller.conn
        self._lock = storage_controller._lock
        self._initialize_db()

    def _initialize_db(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS research_budget_state (
                        key TEXT PRIMARY KEY,
                        date_utc TEXT NOT NULL,
                        remaining INTEGER NOT NULL,
                        "limit" INTEGER NOT NULL,
                        updated_at_ts INTEGER NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS research_budget_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date_utc TEXT NOT NULL,
                        spent_at_ts INTEGER NOT NULL,
                        units INTEGER NOT NULL,
                        request_fingerprint TEXT,
                        research_id TEXT,
                        source TEXT,
                        prompt_preview TEXT,
                        provider TEXT,
                        metadata_json TEXT
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_research_budget_usage_date
                    ON research_budget_usage (date_utc)
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_research_budget_usage_spent_at
                    ON research_budget_usage (spent_at_ts)
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_research_budget_usage_fingerprint
                    ON research_budget_usage (request_fingerprint)
                    """
                )

    def upsert_state(
        self,
        *,
        key: str,
        date_utc: str,
        remaining: int,
        limit: int,
        updated_at_ts: int,
    ) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO research_budget_state (key, date_utc, remaining, "limit", updated_at_ts)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        date_utc = excluded.date_utc,
                        remaining = excluded.remaining,
                        "limit" = excluded."limit",
                        updated_at_ts = excluded.updated_at_ts
                    """,
                    (key, date_utc, int(remaining), int(limit), int(updated_at_ts)),
                )

    def append_usage(
        self,
        *,
        date_utc: str,
        spent_at_ts: int,
        units: int,
        request_fingerprint: str | None = None,
        research_id: str | None = None,
        source: str | None = None,
        prompt_preview: str | None = None,
        provider: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        metadata_json = json.dumps(metadata) if metadata is not None else None
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    """
                    INSERT INTO research_budget_usage (
                        date_utc,
                        spent_at_ts,
                        units,
                        request_fingerprint,
                        research_id,
                        source,
                        prompt_preview,
                        provider,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_utc,
                        int(spent_at_ts),
                        int(units),
                        request_fingerprint,
                        research_id,
                        source,
                        prompt_preview,
                        provider,
                        metadata_json,
                    ),
                )
                return int(cursor.lastrowid)

    def get_state(self, key: str) -> ResearchBudgetState | None:
        cursor = self._conn.cursor()
        row = cursor.execute(
            """
            SELECT key, date_utc, remaining, "limit", updated_at_ts
            FROM research_budget_state
            WHERE key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        return ResearchBudgetState(
            key=str(row[0]),
            date_utc=str(row[1]),
            remaining=int(row[2]),
            limit=int(row[3]),
            updated_at_ts=int(row[4]),
        )
