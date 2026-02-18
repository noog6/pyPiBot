"""Tests for storage diagnostics."""

from __future__ import annotations

import sqlite3

from diagnostics.models import DiagnosticStatus
from storage.diagnostics import inspect_memory_embeddings, probe


class _FakeConfigController:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_config(self) -> dict[str, object]:
        return dict(self._config)


def test_storage_probe_offline(tmp_path) -> None:
    """Storage probe should pass when using a temp directory."""

    result = probe(base_dir=tmp_path)
    assert result.status is DiagnosticStatus.PASS


def test_inspect_memory_embeddings_counts_ready_vs_non_ready(tmp_path) -> None:
    """Embedding inspection should report table presence and status counts."""

    var_dir = tmp_path / "var"
    var_dir.mkdir(parents=True, exist_ok=True)
    db_path = var_dir / "memories.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE memory_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO memory_embeddings (memory_id, status) VALUES (?, ?)",
            [
                (1, "ready"),
                (2, "ready"),
                (3, "pending"),
                (4, "error"),
                (5, None),
            ],
        )
        conn.commit()

    state = inspect_memory_embeddings(
        {"memory_semantic": {"enabled": True}, "var_dir": str(var_dir)}
    )

    assert state["enabled"] is True
    assert state["table_exists"] is True
    assert state["ready_count"] == 2
    assert state["non_ready_count"] == 3


def test_storage_probe_semantic_disabled_is_informational(monkeypatch, tmp_path) -> None:
    """Disabled semantic memory should keep diagnostics informational only."""

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    var_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    import config.controller as config_controller
    import storage.diagnostics as storage_diagnostics

    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController(
            {
                "var_dir": str(var_dir),
                "log_dir": str(log_dir),
                "memory_semantic": {"enabled": False},
            }
        ),
    )

    result = storage_diagnostics.probe()

    assert result.status is DiagnosticStatus.PASS
    assert "memory_semantic.enabled=False" in result.details
    assert "memory_embeddings.table_exists=False" in result.details


def test_storage_probe_fails_when_semantic_enabled_and_table_missing(
    monkeypatch, tmp_path
) -> None:
    """Enabled semantic memory should fail diagnostics if table is missing."""

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    var_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    db_path = var_dir / "memories.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE memories (memory_id INTEGER PRIMARY KEY AUTOINCREMENT)")
        conn.commit()

    import config.controller as config_controller
    import storage.diagnostics as storage_diagnostics

    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController(
            {
                "var_dir": str(var_dir),
                "log_dir": str(log_dir),
                "memory_semantic": {"enabled": True},
            }
        ),
    )

    result = storage_diagnostics.probe()

    assert result.status is DiagnosticStatus.FAIL
    assert "semantic memory is enabled" in result.details
