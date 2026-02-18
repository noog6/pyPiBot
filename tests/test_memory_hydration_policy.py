"""Tests for two-tier memory hydration and pinning policy."""

from __future__ import annotations

from types import SimpleNamespace

from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore
from ai.realtime_api import RealtimeAPI


def _make_manager(store: MemoryStore) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = "default"
    manager._active_session_id = None
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._store = store
    manager._embedding_worker = None
    manager._semantic_config = SimpleNamespace(enabled=False, background_embedding_enabled=False)
    manager._last_turn_retrieval_at = {}
    manager._auto_pin_min_importance = 5
    manager._auto_pin_requires_review = True
    manager._auto_reflection_semantic_dedupe_enabled = False
    manager._auto_reflection_dedupe_recent_limit = 24
    manager._auto_reflection_dedupe_high_risk_cosine = 0.9
    manager._auto_reflection_dedupe_policy = "skip_write"
    manager._auto_reflection_dedupe_importance = 2
    manager._auto_reflection_dedupe_clear_pin = True
    manager._auto_reflection_dedupe_needs_review = True
    manager._auto_reflection_dedupe_apply_to_manual_tool = False
    return manager


def test_startup_digest_includes_only_pinned_reviewed(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store)

    manager.remember_memory(
        content="Preferred name pronunciation: TEE-oh.",
        importance=5,
        pinned=True,
        needs_review=False,
    )
    manager.remember_memory(
        content="Auto-pinned candidate pending review.",
        importance=5,
        pinned=True,
        needs_review=True,
    )

    digest = manager.retrieve_startup_digest(max_items=2, max_chars=300)

    assert digest is not None
    assert [item.content for item in digest.items] == ["Preferred name pronunciation: TEE-oh."]


def test_auto_reflection_high_importance_auto_pins_with_review(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store)

    entry = manager.remember_memory(
        content="Long-term preference: concise bullet summaries.",
        importance=5,
        source="auto_reflection",
    )

    assert entry.pinned is True
    assert entry.needs_review is True


def test_realtime_skips_turn_retrieval_for_short_or_noisy_input() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._memory_retrieval_min_user_chars = 12
    api._memory_retrieval_min_user_tokens = 3

    assert api._should_skip_turn_memory_retrieval("ok") is True
    assert api._should_skip_turn_memory_retrieval("thanks") is True
    assert api._should_skip_turn_memory_retrieval("battery?") is True
    assert api._should_skip_turn_memory_retrieval("Please remember my favorite tea is jasmine") is False
