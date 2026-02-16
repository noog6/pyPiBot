"""Tests for research budget and cache stores."""

from __future__ import annotations

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
