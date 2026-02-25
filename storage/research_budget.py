"""SQLite-backed storage for research budget state and usage audit records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import time

from storage.controller import StorageController


@dataclass(frozen=True)
class ResearchBudgetState:
    """Current budget state for a named research budget key."""

    key: str
    date_utc: str
    remaining: int
    limit: int
    updated_at_ts: int


BudgetState = ResearchBudgetState


@dataclass(frozen=True)
class UsageEvent:
    """Incoming spend request metadata persisted on allowed usage rows."""

    spent_at_ts: int
    request_fingerprint: str | None = None
    research_id: str | None = None
    source: str | None = None
    prompt_preview: str | None = None
    provider: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class SpendResult:
    """Outcome of a budget spend attempt."""

    allowed: bool
    remaining: int
    limit: int
    date_utc: str


@dataclass(frozen=True)
class UsageRow:
    """Usage audit row returned to callers for reporting."""

    id: int
    date_utc: str
    spent_at_ts: int
    units: int
    request_fingerprint: str | None
    research_id: str | None
    source: str | None
    prompt_preview: str | None
    provider: str | None
    metadata: dict[str, object] | None


@dataclass(frozen=True)
class UsageReservation:
    """Two-phase reservation handle for an in-flight execution."""

    usage_id: int
    date_utc: str
    spent_at_ts: int
    units: int
    key: str
    charged_to_budget: bool


class ResearchBudgetStorage:
    """Persist research budget state and daily spend audit rows."""

    @staticmethod
    def _date_utc_from_ts(ts: int) -> str:
        """Map timestamps to UTC day boundaries using `time.gmtime()`.

        We intentionally normalize by UTC date so daily limits are deterministic and
        unaffected by host-local timezone configuration or DST transitions. Input
        timestamps are accepted in either seconds or milliseconds since epoch.
        """

        ts_int = int(ts)
        ts_seconds = ts_int // 1000 if ts_int > 10**11 else ts_int
        return time.strftime("%Y-%m-%d", time.gmtime(ts_seconds))

    def __init__(self, *, storage_controller: StorageController | None = None) -> None:
        storage_controller = storage_controller or StorageController.get_instance()
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
        metadata_json = json.dumps(self.build_usage_metadata(metadata)) if metadata is not None else None
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

    def get_or_init_daily_state(
        self,
        key: str,
        daily_limit: int,
        now_ts: int,
        date_utc: str,
    ) -> BudgetState:
        """Get current state, initializing or resetting when day rolls over."""

        with self._lock:
            with self._conn:
                return self._get_or_init_daily_state_txn(
                    key=key,
                    daily_limit=daily_limit,
                    now_ts=now_ts,
                    date_utc=date_utc,
                )

    def _get_or_init_daily_state_txn(
        self,
        *,
        key: str,
        daily_limit: int,
        now_ts: int,
        date_utc: str,
    ) -> BudgetState:
        row = self._conn.execute(
            """
            SELECT key, date_utc, remaining, "limit", updated_at_ts
            FROM research_budget_state
            WHERE key = ?
            """,
            (key,),
        ).fetchone()

        if row is None or str(row[1]) != date_utc:
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
                (key, date_utc, int(daily_limit), int(daily_limit), int(now_ts)),
            )
            return BudgetState(
                key=key,
                date_utc=date_utc,
                remaining=int(daily_limit),
                limit=int(daily_limit),
                updated_at_ts=int(now_ts),
            )

        return BudgetState(
            key=str(row[0]),
            date_utc=str(row[1]),
            remaining=int(row[2]),
            limit=int(row[3]),
            updated_at_ts=int(row[4]),
        )

    def spend_budget(
        self,
        key: str,
        units: int,
        daily_limit: int,
        usage_event: UsageEvent,
    ) -> SpendResult:
        """Spend budget units in a single transaction with usage audit on success."""

        spent_at_ts = int(usage_event.spent_at_ts)
        current_date_utc = self._date_utc_from_ts(spent_at_ts)
        with self._lock:
            with self._conn:
                state = self._get_or_init_daily_state_txn(
                    key=key,
                    daily_limit=daily_limit,
                    now_ts=spent_at_ts,
                    date_utc=current_date_utc,
                )
                if int(units) <= 0:
                    return SpendResult(
                        allowed=True,
                        remaining=state.remaining,
                        limit=state.limit,
                        date_utc=state.date_utc,
                    )

                if state.remaining < int(units):
                    return SpendResult(
                        allowed=False,
                        remaining=state.remaining,
                        limit=state.limit,
                        date_utc=state.date_utc,
                    )

                new_remaining = state.remaining - int(units)
                self._conn.execute(
                    """
                    UPDATE research_budget_state
                    SET remaining = ?, updated_at_ts = ?
                    WHERE key = ?
                    """,
                    (new_remaining, spent_at_ts, key),
                )
                metadata_json = (
                    json.dumps(self.build_usage_metadata(usage_event.metadata))
                    if usage_event.metadata is not None
                    else None
                )
                self._conn.execute(
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
                        state.date_utc,
                        spent_at_ts,
                        int(units),
                        usage_event.request_fingerprint,
                        usage_event.research_id,
                        usage_event.source,
                        usage_event.prompt_preview,
                        usage_event.provider,
                        metadata_json,
                    ),
                )
                return SpendResult(
                    allowed=True,
                    remaining=new_remaining,
                    limit=state.limit,
                    date_utc=state.date_utc,
                )

    def reserve_budget_usage(
        self,
        *,
        key: str,
        units: int,
        daily_limit: int,
        usage_event: UsageEvent,
    ) -> UsageReservation | None:
        """Atomically reserve units and write a started usage row."""

        reserved_at_ts = int(usage_event.spent_at_ts)
        current_date_utc = self._date_utc_from_ts(reserved_at_ts)
        with self._lock:
            with self._conn:
                state = self._get_or_init_daily_state_txn(
                    key=key,
                    daily_limit=daily_limit,
                    now_ts=reserved_at_ts,
                    date_utc=current_date_utc,
                )
                amount = max(1, int(units))
                if state.remaining < amount:
                    return None
                new_remaining = state.remaining - amount
                self._conn.execute(
                    """
                    UPDATE research_budget_state
                    SET remaining = ?, updated_at_ts = ?
                    WHERE key = ?
                    """,
                    (new_remaining, reserved_at_ts, key),
                )
                metadata_json = json.dumps(self.build_usage_metadata(usage_event.metadata))
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
                        state.date_utc,
                        reserved_at_ts,
                        amount,
                        usage_event.request_fingerprint,
                        usage_event.research_id,
                        usage_event.source,
                        usage_event.prompt_preview,
                        usage_event.provider,
                        metadata_json,
                    ),
                )
                return UsageReservation(
                    usage_id=int(cursor.lastrowid),
                    date_utc=state.date_utc,
                    spent_at_ts=reserved_at_ts,
                    units=amount,
                    key=key,
                    charged_to_budget=True,
                )

    def finalize_budget_usage(
        self,
        *,
        reservation: UsageReservation,
        status: str,
        refund: bool,
    ) -> bool:
        """Finalize reservation status and optionally refund reserved units."""

        normalized_status = str(status).strip().lower() or "committed"
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    """
                    SELECT metadata_json
                    FROM research_budget_usage
                    WHERE id = ?
                    """,
                    (reservation.usage_id,),
                ).fetchone()
                if row is None:
                    return False
                metadata = json.loads(row[0]) if row[0] else {}
                metadata = self.build_usage_metadata(metadata) or {}
                metadata["execution_status"] = normalized_status
                self._conn.execute(
                    """
                    UPDATE research_budget_usage
                    SET metadata_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(metadata), reservation.usage_id),
                )

                if refund and reservation.charged_to_budget:
                    self._conn.execute(
                        """
                        UPDATE research_budget_state
                        SET remaining = remaining + ?, updated_at_ts = ?
                        WHERE key = ? AND date_utc = ?
                        """,
                        (int(reservation.units), int(time.time()), reservation.key, reservation.date_utc),
                    )
                return True

    def get_usage_for_date(self, date_utc: str) -> list[UsageRow]:
        """Fetch usage audit rows for a UTC day, ordered by event timestamp."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    id,
                    date_utc,
                    spent_at_ts,
                    units,
                    request_fingerprint,
                    research_id,
                    source,
                    prompt_preview,
                    provider,
                    metadata_json
                FROM research_budget_usage
                WHERE date_utc = ?
                ORDER BY spent_at_ts ASC, id ASC
                """,
                (date_utc,),
            ).fetchall()
        usage_rows: list[UsageRow] = []
        for row in rows:
            metadata_raw = row[9]
            metadata = json.loads(metadata_raw) if metadata_raw else None
            usage_rows.append(
                UsageRow(
                    id=int(row[0]),
                    date_utc=str(row[1]),
                    spent_at_ts=int(row[2]),
                    units=int(row[3]),
                    request_fingerprint=str(row[4]) if row[4] is not None else None,
                    research_id=str(row[5]) if row[5] is not None else None,
                    source=str(row[6]) if row[6] is not None else None,
                    prompt_preview=str(row[7]) if row[7] is not None else None,
                    provider=str(row[8]) if row[8] is not None else None,
                    metadata=metadata,
                )
            )
        return usage_rows

    @staticmethod
    def build_usage_metadata(
        metadata: dict[str, object] | None = None,
        *,
        over_budget_approved: bool = False,
        decision_source: str | None = None,
    ) -> dict[str, object] | None:
        """Normalize usage metadata and optionally stamp override authorization details."""

        normalized: dict[str, object] = dict(metadata or {})
        if over_budget_approved:
            normalized["over_budget_approved"] = True
            normalized["over_budget_decision_source"] = (
                str(decision_source).strip() if decision_source else "operator_confirmation"
            )
        return normalized or None
