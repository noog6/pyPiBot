"""Tests for research budget and cache stores."""

from __future__ import annotations

from services.research.budget_manager import ResearchBudgetManager
from services.research.stores import ResearchBudgetTracker, ResearchCacheStore


def test_budget_tracker_persists_daily_count(tmp_path) -> None:
    tracker = ResearchBudgetTracker(str(tmp_path / "budget.json"), daily_limit=2)
    assert tracker.can_spend()
    remaining = tracker.spend()
    assert remaining == 1
    tracker2 = ResearchBudgetTracker(str(tmp_path / "budget.json"), daily_limit=2)
    assert tracker2.get_remaining() == 1


def test_cache_store_round_trip(tmp_path) -> None:
    store = ResearchCacheStore(str(tmp_path / "cache"), ttl_hours=24)
    assert store.get("query", "abc") is None
    store.set("query", "abc", {"status": "ok", "answer_summary": "cached"})
    payload = store.get("query", "abc")
    assert payload is not None
    assert payload["answer_summary"] == "cached"


def test_budget_manager_spend_if_allowed_persists_audit(tmp_path) -> None:
    manager = ResearchBudgetManager(str(tmp_path / "budget-manager.json"), daily_limit=2)
    assert manager.can_spend(1)
    assert manager.spend_if_allowed(1, audit_payload={
        "request_fingerprint": "fp-1",
        "research_id": "r-1",
        "source": "user",
        "prompt_preview": "find xyz",
        "provider": "openai_responses_web_search",
    })

    state = manager.current_state()
    assert state["remaining"] == 1
    assert state["last_audit"]["request_fingerprint"] == "fp-1"


def test_budget_manager_denied_spend_does_not_mutate_state(tmp_path) -> None:
    manager = ResearchBudgetManager(str(tmp_path / "budget-manager.json"), daily_limit=1)
    assert manager.spend_if_allowed(1)
    state_before = manager.current_state()

    assert manager.spend_if_allowed(1) is False
    assert manager.current_state() == state_before
