"""Tests for research budget and cache stores."""

from __future__ import annotations

import json
import time

import pytest

from services.research.budget_manager import ResearchBudgetManager
from services.research.stores import ResearchBudgetTracker, ResearchCacheStore


class _FakeConfigController:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_config(self) -> dict[str, object]:
        return dict(self._config)


def _init_budget_manager(monkeypatch, tmp_path, daily_limit: int, legacy_state_file: str = "unused.json"):
    import config.controller as config_controller
    from storage.controller import StorageController

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController({"var_dir": str(var_dir), "log_dir": str(log_dir)}),
    )

    StorageController._instance = None
    manager = ResearchBudgetManager(legacy_state_file, daily_limit=daily_limit)
    return manager, StorageController


def test_budget_tracker_is_read_only_compatibility_layer(tmp_path) -> None:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    state_file = tmp_path / "budget.json"
    state_file.write_text(json.dumps({"date": today, "count": 1}), encoding="utf-8")

    with pytest.deprecated_call(match="ResearchBudgetTracker is deprecated"):
        tracker = ResearchBudgetTracker(str(state_file), daily_limit=2)

    assert tracker.can_spend()
    assert tracker.get_remaining() == 1
    with pytest.raises(RuntimeError, match="read-only"):
        tracker.spend()


def test_cache_store_round_trip(tmp_path) -> None:
    store = ResearchCacheStore(str(tmp_path / "cache"), ttl_hours=24)
    assert store.get("query", "abc") is None
    store.set("query", "abc", {"status": "ok", "answer_summary": "cached"})
    payload = store.get("query", "abc")
    assert payload is not None
    assert payload["answer_summary"] == "cached"


def test_budget_manager_spend_if_allowed_persists_audit(monkeypatch, tmp_path) -> None:
    manager, StorageController = _init_budget_manager(monkeypatch, tmp_path, daily_limit=2)
    try:
        assert manager.can_spend(1)
        assert manager.spend_if_allowed(
            1,
            audit_payload={
                "request_fingerprint": "fp-1",
                "research_id": "r-1",
                "source": "user",
                "prompt_preview": "find xyz",
                "provider": "openai_responses_web_search",
            },
        )

        state = manager.current_state()
        assert state["remaining"] == 1
        assert state["last_audit"]["request_fingerprint"] == "fp-1"
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None


def test_budget_manager_denied_spend_does_not_mutate_state(monkeypatch, tmp_path) -> None:
    manager, StorageController = _init_budget_manager(monkeypatch, tmp_path, daily_limit=1)
    try:
        assert manager.spend_if_allowed(1)
        state_before = manager.current_state()

        assert manager.spend_if_allowed(1) is False
        assert manager.current_state() == state_before
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None


def test_budget_manager_migrates_legacy_json_once(monkeypatch, tmp_path) -> None:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    legacy_state_file = tmp_path / "research_budget.json"
    legacy_state_file.write_text(json.dumps({"date": today, "count": 2}), encoding="utf-8")

    manager, StorageController = _init_budget_manager(
        monkeypatch,
        tmp_path,
        daily_limit=5,
        legacy_state_file=str(legacy_state_file),
    )
    try:
        state = manager.current_state()
        assert state["remaining"] == 3
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None
