"""Diagnostics routines for the storage subsystem."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from diagnostics.models import DiagnosticResult, DiagnosticStatus


def probe(base_dir: Path | None = None) -> DiagnosticResult:
    """Run a storage probe to validate filesystem and SQLite access.

    Args:
        base_dir: Optional base directory for offline testing.

    Returns:
        Diagnostic result indicating storage readiness.
    """

    name = "storage"
    test_db = None
    try:
        if base_dir is None:
            from config import ConfigController

            config = ConfigController.get_instance().get_config()
            storage_config = config.get("storage", {})
            var_dir = Path(
                storage_config.get("var_dir", config.get("var_dir", "./var/"))
            ).expanduser()
            log_dir = Path(
                storage_config.get("log_dir", config.get("log_dir", "./log/"))
            ).expanduser()
        else:
            var_dir = base_dir / "var"
            log_dir = base_dir / "log"

        var_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        sentinel = var_dir / "diagnostics_probe.txt"
        sentinel.write_text("ok", encoding="utf-8")
        sentinel.unlink(missing_ok=True)

        test_db = log_dir / "diagnostics_probe.db"
        with sqlite3.connect(test_db) as conn:
            conn.execute("SELECT 1")

        details = f"Storage directories writable at {log_dir}"
        return DiagnosticResult(name=name, status=DiagnosticStatus.PASS, details=details)
    except OSError as exc:
        details = f"Filesystem access failed: {exc}"
        return DiagnosticResult(name=name, status=DiagnosticStatus.FAIL, details=details)
    except sqlite3.Error as exc:
        details = f"SQLite probe failed: {exc}"
        return DiagnosticResult(name=name, status=DiagnosticStatus.FAIL, details=details)
    finally:
        if test_db and test_db.exists():
            test_db.unlink(missing_ok=True)
