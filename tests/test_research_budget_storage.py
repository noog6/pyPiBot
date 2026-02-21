"""Tests for SQLite-backed research budget storage."""

from __future__ import annotations

import json


class _FakeConfigController:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_config(self) -> dict[str, object]:
        return dict(self._config)


def test_research_budget_storage_creates_tables_indexes_and_writes(monkeypatch, tmp_path) -> None:
    import config.controller as config_controller
    from storage.controller import StorageController
    from storage.research_budget import ResearchBudgetStorage

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController({"var_dir": str(var_dir), "log_dir": str(log_dir)}),
    )

    StorageController._instance = None
    try:
        store = ResearchBudgetStorage()
        store.upsert_state(
            key="research_daily_fetch",
            date_utc="2026-01-11",
            remaining=9,
            limit=10,
            updated_at_ts=1736553989000,
        )
        usage_id = store.append_usage(
            date_utc="2026-01-11",
            spent_at_ts=1736553999000,
            units=1,
            request_fingerprint="abc123",
            research_id="r-42",
            source="text_message",
            prompt_preview="what changed today",
            provider="openai",
            metadata={"round": 1},
        )

        state = store.get_state("research_daily_fetch")
        assert state is not None
        assert state.remaining == 9
        assert state.limit == 10
        assert usage_id > 0

        conn = StorageController.get_instance().conn
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(research_budget_usage)").fetchall()
        }
        assert "idx_research_budget_usage_date" in indexes
        assert "idx_research_budget_usage_spent_at" in indexes
        assert "idx_research_budget_usage_fingerprint" in indexes

        usage_row = conn.execute(
            """
            SELECT metadata_json FROM research_budget_usage
            WHERE id = ?
            """,
            (usage_id,),
        ).fetchone()
        assert usage_row is not None
        assert json.loads(usage_row[0]) == {"round": 1}
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None
