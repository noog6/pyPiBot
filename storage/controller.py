"""SQLite-backed storage controller for runtime data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
import traceback
from typing import Any

from config import ConfigController
from core.logging import logger as LOGGER


def millis() -> int:
    """Return current time in milliseconds."""

    return int(time.time() * 1000)


@dataclass(frozen=True)
class StorageInfo:
    """Metadata about the current storage run."""

    run_id: int
    run_id_file: Path
    run_dir: Path
    log_dir: Path
    log_file: Path
    image_dir: Path
    db_path: Path


@dataclass(frozen=True)
class SessionLedgerRecord:
    """Durable lifecycle record for a single runtime session."""

    canonical_run_id: str
    run_number: int | None
    started_at: int
    ready_at: int | None
    last_seen_at: int | None
    shutdown_completed_at: int | None
    lifecycle_state: str

    @property
    def shutdown_clean(self) -> bool:
        """Return whether this run has a durable clean-shutdown marker."""

        return (
            self.lifecycle_state == "shutdown_completed"
            and self.shutdown_completed_at is not None
        )


class StorageController:
    """Singleton controller for persistent storage."""

    _instance: "StorageController | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if StorageController._instance is not None:
            raise RuntimeError("You cannot create another StorageController class")

        self.config_controller = ConfigController.get_instance()
        self.config = self.config_controller.get_config()

        var_dir, log_dir = self._resolve_storage_dirs()
        self.var_dir = var_dir
        self.log_dir = log_dir

        self.run_id_file = var_dir / "current_run"
        self.run_id = self.get_next_run_number(var_dir)

        self.db_filename = f"app_data_{self.run_id}.db"
        self.run_dir = log_dir / str(self.run_id)
        self.db_full_file_path = self.run_dir / self.db_filename

        self.conn = self.connect(log_dir)
        self.initialize_db()
        self.session_ledger_db_path = self.var_dir / "session_ledger.db"
        self.session_ledger_conn = self._connect_session_ledger_db()
        self._initialize_session_ledger_db()
        StorageController._instance = self

    @classmethod
    def get_instance(cls) -> "StorageController":
        """Return the singleton instance of the controller."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def connect(self, log_dir: Path) -> sqlite3.Connection:
        """Connect to the SQLite database for the current run."""

        log_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_full_file_path, check_same_thread=False)

    def initialize_db(self) -> None:
        """Initialize tables for logs and data."""

        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_millis INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                log_level TEXT,
                message TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS data (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_millis INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT,
                data JSON
            )
            """
        )

        self.conn.commit()

    def _connect_session_ledger_db(self) -> sqlite3.Connection:
        """Connect to the shared session-ledger SQLite database."""

        self.var_dir.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.session_ledger_db_path, check_same_thread=False)

    def _initialize_session_ledger_db(self) -> None:
        """Initialize the durable session-ledger schema."""

        cursor = self.session_ledger_conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_ledger (
                canonical_run_id TEXT PRIMARY KEY,
                run_number INTEGER,
                started_at INTEGER NOT NULL,
                ready_at INTEGER,
                last_seen_at INTEGER,
                shutdown_completed_at INTEGER,
                lifecycle_state TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_ledger_run_number
            ON session_ledger(run_number)
            """
        )
        self.session_ledger_conn.commit()

    def close(self) -> None:
        """Close the storage connection."""

        if self.conn:
            self.conn.close()
        if getattr(self, "session_ledger_conn", None):
            self.session_ledger_conn.close()

    def add_log(self, log_level: str, message: str) -> None:
        """Insert a log record into the database."""

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO logs (run_millis, log_level, message)
            VALUES (?, ?, ?)
            """,
            (millis(), log_level, message),
        )
        self.conn.commit()

    def get_next_run_number(self, var_dir: Path) -> int:
        """Return the next run number, persisting to the run-id file."""

        var_dir.mkdir(parents=True, exist_ok=True)

        next_run_number = 0
        if self.run_id_file.is_file():
            current_run_number = self.run_id_file.read_text(encoding="utf-8").strip()
            if current_run_number:
                next_run_number = int(current_run_number) + 1
        self.run_id_file.write_text(str(next_run_number), encoding="utf-8")
        return next_run_number

    def get_current_run_number(self) -> int:
        """Return the current run id."""

        return int(self.run_id)

    def get_canonical_run_id(self) -> str:
        """Return canonical runtime id used for durable session-ledger rows."""

        return f"run-{self.get_current_run_number()}"

    def get_log_file_path(self) -> Path:
        """Return the log file path for the current run."""

        return self.get_run_dir() / f"run_{self.run_id}.log"

    def get_run_dir(self) -> Path:
        """Return the artifact directory for the current run."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def get_run_image_dir(self) -> Path:
        """Return the image artifact directory for the current run."""

        image_dir = self.get_run_dir() / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        return image_dir

    def add_data(self, data_type: str, data: str) -> None:
        """Insert structured data into the database."""

        with self._lock:
            try:
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO data (run_millis, type, data)
                        VALUES (?, ?, ?)
                        """,
                        (millis(), data_type, data),
                    )
            except sqlite3.OperationalError as exc:
                self.log_exception(exc)

    def mark_session_boot_started(self, canonical_run_id: str, run_number: int | None = None) -> None:
        """Create or refresh the lifecycle row for a run entering startup."""

        run_id = str(canonical_run_id or "").strip()
        if not run_id:
            raise ValueError("canonical_run_id is required")
        resolved_run_number = self.get_current_run_number() if run_number is None else int(run_number)
        timestamp = millis()
        with self._lock:
            with self.session_ledger_conn:
                self.session_ledger_conn.execute(
                    """
                    INSERT INTO session_ledger (
                        canonical_run_id,
                        run_number,
                        started_at,
                        last_seen_at,
                        lifecycle_state
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(canonical_run_id) DO UPDATE SET
                        run_number=excluded.run_number,
                        started_at=excluded.started_at,
                        lifecycle_state=excluded.lifecycle_state
                    """,
                    (run_id, resolved_run_number, timestamp, timestamp, "boot_started"),
                )

    def mark_session_running(self, canonical_run_id: str) -> None:
        """Mark a run as active once realtime is ready."""

        run_id = str(canonical_run_id or "").strip()
        if not run_id:
            raise ValueError("canonical_run_id is required")
        timestamp = millis()
        with self._lock:
            with self.session_ledger_conn:
                self.session_ledger_conn.execute(
                    """
                    UPDATE session_ledger
                    SET lifecycle_state='running',
                        ready_at=COALESCE(ready_at, ?),
                        last_seen_at=COALESCE(last_seen_at, ?)
                    WHERE canonical_run_id=?
                    """,
                    (timestamp, timestamp, run_id),
                )

    def touch_session_last_seen(self, canonical_run_id: str) -> None:
        """Persist periodic liveness for the active run."""

        run_id = str(canonical_run_id or "").strip()
        if not run_id:
            raise ValueError("canonical_run_id is required")
        timestamp = millis()
        with self._lock:
            with self.session_ledger_conn:
                self.session_ledger_conn.execute(
                    """
                    UPDATE session_ledger
                    SET last_seen_at=?,
                        lifecycle_state=CASE
                            WHEN lifecycle_state='boot_started' THEN 'running'
                            ELSE lifecycle_state
                        END
                    WHERE canonical_run_id=?
                    """,
                    (timestamp, run_id),
                )

    def mark_session_shutdown_completed(self, canonical_run_id: str) -> None:
        """Mark a run as cleanly shutdown."""

        run_id = str(canonical_run_id or "").strip()
        if not run_id:
            raise ValueError("canonical_run_id is required")
        timestamp = millis()
        with self._lock:
            with self.session_ledger_conn:
                self.session_ledger_conn.execute(
                    """
                    UPDATE session_ledger
                    SET lifecycle_state='shutdown_completed',
                        shutdown_completed_at=?
                    WHERE canonical_run_id=?
                    """,
                    (timestamp, run_id),
                )

    def get_previous_session_record(self) -> SessionLedgerRecord | None:
        """Return the most recent run preceding the current run."""

        cursor = self.session_ledger_conn.cursor()
        cursor.execute(
            """
            SELECT canonical_run_id, run_number, started_at, ready_at, last_seen_at,
                   shutdown_completed_at, lifecycle_state
            FROM session_ledger
            WHERE run_number < ?
            ORDER BY run_number DESC
            LIMIT 1
            """,
            (self.get_current_run_number(),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._session_ledger_record_from_row(row)

    def get_recent_session_records(
        self,
        *,
        lookback_runs: int = 1,
        include_current: bool = False,
    ) -> list[SessionLedgerRecord]:
        """Return recent session-ledger rows in newest-first run order."""

        resolved_lookback = max(1, int(lookback_runs))
        current_run_number = self.get_current_run_number()
        cursor = self.session_ledger_conn.cursor()

        comparator = "<=" if include_current else "<"
        cursor.execute(
            f"""
            SELECT canonical_run_id, run_number, started_at, ready_at, last_seen_at,
                   shutdown_completed_at, lifecycle_state
            FROM session_ledger
            WHERE run_number {comparator} ?
            ORDER BY run_number DESC
            LIMIT ?
            """,
            (current_run_number, resolved_lookback),
        )
        rows = cursor.fetchall()
        return [self._session_ledger_record_from_row(row) for row in rows]

    def previous_run_was_unclean(self) -> tuple[bool, SessionLedgerRecord | None]:
        """Classify whether the prior run appears unclean from durable state."""

        previous = self.get_previous_session_record()
        if previous is None:
            return False, None
        is_clean = previous.shutdown_clean
        return (not is_clean), previous

    def add_user_input(self, data: Any) -> None:
        """Persist user input payload."""

        self.add_data("user_input", json.dumps(data))

    def add_data_sample(self, data: Any) -> None:
        """Persist a data sample payload."""

        self.add_data("data_sample", json.dumps(data))

    def log_exception(self, exc: BaseException) -> None:
        """Log an exception to stderr and the database."""

        message = traceback.format_exc()
        LOGGER.exception("Found error (%s)", exc)
        self.add_log("ERROR", message)

    def add_movement_frame(self, frame: Any) -> None:
        """Persist a movement frame object."""

        try:
            self.add_data("movement_frame", json.dumps(frame.to_dict()))
        except Exception as exc:  # noqa: BLE001 - log and store unexpected errors
            self.log_exception(exc)

    def fetch_movement_frames(self) -> list[dict[str, Any]]:
        """Fetch all persisted movement frames."""

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT data FROM data WHERE type = 'movement_frame'
            """
        )
        rows = cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_storage_info(self) -> StorageInfo:
        """Return metadata about the current run storage."""

        return StorageInfo(
            run_id=self.run_id,
            run_id_file=self.run_id_file,
            run_dir=self.get_run_dir(),
            log_dir=self.log_dir,
            log_file=self.get_log_file_path(),
            image_dir=self.get_run_image_dir(),
            db_path=self.db_full_file_path,
        )

    def _resolve_storage_dirs(self) -> tuple[Path, Path]:
        """Resolve storage directories from configuration."""

        storage_config = self.config.get("storage", {})
        var_dir = storage_config.get("var_dir", self.config.get("var_dir", "./var/"))
        log_dir = storage_config.get("log_dir", self.config.get("log_dir", "./log/"))

        return Path(var_dir).expanduser(), Path(log_dir).expanduser()

    def _session_ledger_record_from_row(self, row: tuple[Any, ...]) -> SessionLedgerRecord:
        """Build a typed session-ledger record from a query row."""

        return SessionLedgerRecord(
            canonical_run_id=str(row[0]),
            run_number=int(row[1]) if row[1] is not None else None,
            started_at=int(row[2]),
            ready_at=int(row[3]) if row[3] is not None else None,
            last_seen_at=int(row[4]) if row[4] is not None else None,
            shutdown_completed_at=int(row[5]) if row[5] is not None else None,
            lifecycle_state=str(row[6]),
        )
