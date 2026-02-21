"""Tests for SQLite-backed research budget storage."""

from __future__ import annotations

import json


class _FakeConfigController:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_config(self) -> dict[str, object]:
        return dict(self._config)


def _init_store(monkeypatch, tmp_path):
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
    store = ResearchBudgetStorage()
    return store, StorageController


def test_research_budget_storage_creates_tables_indexes_and_writes(monkeypatch, tmp_path) -> None:
    store, StorageController = _init_store(monkeypatch, tmp_path)

    try:
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


def test_get_or_init_daily_state_resets_when_date_changes(monkeypatch, tmp_path) -> None:
    store, StorageController = _init_store(monkeypatch, tmp_path)

    try:
        state_1 = store.get_or_init_daily_state(
            key="research_daily_fetch",
            daily_limit=5,
            now_ts=1736553989000,
            date_utc="2025-01-10",
        )
        assert state_1.remaining == 5
        assert state_1.limit == 5

        store.upsert_state(
            key="research_daily_fetch",
            date_utc="2025-01-10",
            remaining=2,
            limit=5,
            updated_at_ts=1736553999000,
        )
        state_2 = store.get_or_init_daily_state(
            key="research_daily_fetch",
            daily_limit=7,
            now_ts=1736640400000,
            date_utc="2025-01-11",
        )
        assert state_2.date_utc == "2025-01-11"
        assert state_2.remaining == 7
        assert state_2.limit == 7
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None


def test_spend_budget_transaction_and_denials_are_non_mutating(monkeypatch, tmp_path) -> None:
    from storage.research_budget import UsageEvent

    store, StorageController = _init_store(monkeypatch, tmp_path)

    try:
        success = store.spend_budget(
            key="research_daily_fetch",
            units=2,
            daily_limit=3,
            usage_event=UsageEvent(spent_at_ts=1736553999000, metadata={"ok": True}),
        )
        assert success.allowed is True
        assert success.remaining == 1
        assert success.limit == 3
        assert success.date_utc == "2025-01-11"

        denied = store.spend_budget(
            key="research_daily_fetch",
            units=2,
            daily_limit=3,
            usage_event=UsageEvent(spent_at_ts=1736554999000),
        )
        assert denied.allowed is False
        assert denied.remaining == 1

        state = store.get_state("research_daily_fetch")
        assert state is not None
        assert state.remaining == 1

        usage_rows = store.get_usage_for_date("2025-01-11")
        assert len(usage_rows) == 1
        assert usage_rows[0].units == 2
        assert usage_rows[0].metadata == {"ok": True}

        conn = StorageController.get_instance().conn
        usage_count = conn.execute(
            "SELECT COUNT(*) FROM research_budget_usage WHERE date_utc = ?",
            ("2025-01-11",),
        ).fetchone()
        assert usage_count is not None
        assert int(usage_count[0]) == 1
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None
