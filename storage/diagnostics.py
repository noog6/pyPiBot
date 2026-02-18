"""Diagnostics routines for the storage subsystem."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from diagnostics.models import DiagnosticResult, DiagnosticStatus


def inspect_memory_embeddings(
    config: dict[str, Any], *, base_dir: Path | None = None
) -> dict[str, Any]:
    """Inspect semantic-memory embedding table state.

    Args:
        config: Loaded configuration dictionary.
        base_dir: Optional base directory for offline testing.

    Returns:
        Dictionary containing semantic flag state, table presence, and status counts.
    """

    memory_semantic = config.get("memory_semantic", {})
    enabled = bool(memory_semantic.get("enabled", False))
    if base_dir is None:
        var_dir = Path(config.get("var_dir", "./var/")).expanduser()
    else:
        var_dir = base_dir / "var"

    db_path = var_dir / "memories.db"
    result: dict[str, Any] = {
        "enabled": enabled,
        "db_path": db_path,
        "table_exists": False,
        "ready_count": None,
        "non_ready_count": None,
    }

    if not db_path.exists():
        return result

    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='memory_embeddings'
            """
        ).fetchone()
        result["table_exists"] = bool(table_exists)
        if not result["table_exists"]:
            return result

        ready_count, non_ready_count = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN status != 'ready' OR status IS NULL THEN 1 ELSE 0 END), 0)
            FROM memory_embeddings
            """
        ).fetchone()
        result["ready_count"] = int(ready_count)
        result["non_ready_count"] = int(non_ready_count)

    return result


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
            config = {}
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

        embedding_state = inspect_memory_embeddings(config, base_dir=base_dir)
        semantic_enabled = embedding_state["enabled"]
        table_exists = embedding_state["table_exists"]
        ready_count = embedding_state["ready_count"]
        non_ready_count = embedding_state["non_ready_count"]

        semantic_details = (
            "memory_semantic.enabled="
            f"{semantic_enabled}; "
            f"memory_embeddings.table_exists={table_exists}; "
            f"memory_embeddings.ready={ready_count if ready_count is not None else 'n/a'}; "
            "memory_embeddings.non_ready="
            f"{non_ready_count if non_ready_count is not None else 'n/a'}"
        )

        if semantic_enabled and not table_exists:
            details = (
                f"Storage directories writable at {log_dir}; "
                "semantic memory is enabled but memory_embeddings table is missing; "
                f"{semantic_details}"
            )
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details=details,
            )

        details = f"Storage directories writable at {log_dir}; {semantic_details}"
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
