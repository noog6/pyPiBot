"""SQLite-backed storage controller for runtime data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
import traceback
from typing import Any

from config import ConfigController


LOGGER = logging.getLogger(__name__)


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
        self.log_dir = log_dir

        self.run_id_file = var_dir / "current_run"
        self.run_id = self.get_next_run_number(var_dir)

        self.db_filename = f"app_data_{self.run_id}.db"
        self.db_full_file_path = log_dir / self.db_filename

        self.conn = self.connect(log_dir)
        self.initialize_db()
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

    def close(self) -> None:
        """Close the storage connection."""

        if self.conn:
            self.conn.close()

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

    def get_log_file_path(self) -> Path:
        """Return the log file path for the current run."""

        return self.log_dir / f"run_{self.run_id}.log"

    def get_run_dir(self) -> Path:
        """Return the artifact directory for the current run."""

        run_dir = self.log_dir / f"run_{self.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

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
