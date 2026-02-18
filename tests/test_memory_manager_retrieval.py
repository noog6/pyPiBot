"""Tests for turn-time memory retrieval budget, ranking, and dedupe."""

from __future__ import annotations

import time
from types import SimpleNamespace

from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


def _make_memory_manager(store: MemoryStore) -> MemoryManager:
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
    return manager


def _now_ms() -> int:
    return int(time.time() * 1000)


def test_retrieve_for_turn_enforces_item_and_char_budget(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content=(
            "Theo likes jazz records and rainy evenings by the window while cataloging "
            "albums, sketching ideas, and planning tomorrow morning tea rituals."
        ),
        tags=["music"],
        importance=5,
        user_id="default",
        timestamp=now_ms,
    )
    store.append_memory(
        content="Favorite tea is chamomile with honey.",
        tags=["drink"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="Theo",
        user_id="default",
        max_memories=2,
        max_chars=80,
    )

    assert brief is not None
    assert len(brief.items) <= 2
    assert brief.total_chars <= 80
    assert brief.items[0].content.endswith("…")


def test_retrieve_for_turn_prefers_lexical_relevance_then_recency(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="Remember project alpha milestones.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 2000,
    )
    store.append_memory(
        content="Remember project alpha budget.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )
    store.append_memory(
        content="General weather preference for cloudy days.",
        tags=["weather"],
        importance=5,
        user_id="default",
        timestamp=now_ms,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
    )

    assert brief is not None
    assert [item.content for item in brief.items] == [
        "Remember project alpha budget.",
        "Remember project alpha milestones.",
    ]


def test_retrieve_for_turn_dedupes_near_identical_content(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="User loves Ethiopian coffee beans and manual pour-over brewing.",
        tags=["coffee"],
        importance=4,
        user_id="default",
        timestamp=now_ms,
    )
    store.append_memory(
        content="User loves Ethiopian coffee beans and manual pourover brewing.",
        tags=["coffee"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 100,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="coffee beans",
        user_id="default",
        max_memories=3,
        max_chars=400,
    )

    assert brief is not None
    assert len(brief.items) == 1


def test_retrieve_for_turn_suppresses_stale_entries(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()
    two_years_ms = 1000 * 60 * 60 * 24 * 365 * 2

    store.append_memory(
        content="Legacy preference from years ago about old hardware.",
        tags=["legacy"],
        importance=5,
        user_id="default",
        timestamp=now_ms - two_years_ms,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="old hardware",
        user_id="default",
        max_memories=2,
        max_chars=200,
    )

    assert brief is None


def test_retrieve_for_turn_respects_cooldown(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="Remember project alpha budget.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms,
    )

    first = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=4,
        max_chars=300,
        cooldown_s=20.0,
        now_monotonic=50.0,
    )
    second = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=4,
        max_chars=300,
        cooldown_s=20.0,
        now_monotonic=60.0,
    )

    assert first is not None
    assert second is None
