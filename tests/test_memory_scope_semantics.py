"""Tests for explicit memory scope semantics."""

from __future__ import annotations

from types import SimpleNamespace

from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


def _make_manager(
    store: MemoryStore,
    *,
    user_id: str = "default",
    session_id: str | None = None,
) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = user_id
    manager._active_session_id = session_id
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


def test_user_global_scope_recalls_across_sessions(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")

    writer = _make_manager(store, session_id="run-1")
    writer.remember_memory(
        content="User likes mountain biking.",
        importance=4,
        scope=MemoryScope.USER_GLOBAL,
    )

    reader_other_session = _make_manager(store, session_id="run-2")
    recalled = reader_other_session.recall_memories(
        query="biking",
        scope=MemoryScope.USER_GLOBAL,
    )

    assert [item.content for item in recalled] == ["User likes mountain biking."]


def test_session_local_scope_is_isolated_by_session_id(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")

    run_one = _make_manager(store, session_id="run-1")
    run_one.remember_memory(
        content="Temporary calibration token.",
        importance=3,
        scope=MemoryScope.SESSION_LOCAL,
    )

    run_two = _make_manager(store, session_id="run-2")
    run_two.remember_memory(
        content="Temporary run-two note.",
        importance=3,
        scope=MemoryScope.SESSION_LOCAL,
    )

    run_one_recall = run_one.recall_memories(
        query="Temporary",
        scope=MemoryScope.SESSION_LOCAL,
    )
    run_two_recall = run_two.recall_memories(
        query="Temporary",
        scope=MemoryScope.SESSION_LOCAL,
    )

    assert [item.content for item in run_one_recall] == ["Temporary calibration token."]
    assert [item.content for item in run_two_recall] == ["Temporary run-two note."]


def test_user_global_scope_excludes_session_local_entries(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")

    manager = _make_manager(store, session_id="run-1")
    manager.remember_memory(
        content="Stable cross-run preference.",
        importance=4,
        scope=MemoryScope.USER_GLOBAL,
    )
    manager.remember_memory(
        content="Run-local scratch note.",
        importance=5,
        scope=MemoryScope.SESSION_LOCAL,
    )

    recalled = manager.recall_memories(
        query="note",
        scope=MemoryScope.USER_GLOBAL,
    )

    assert recalled == []

    recalled_global = manager.recall_memories(
        query="preference",
        scope=MemoryScope.USER_GLOBAL,
    )
    assert [item.content for item in recalled_global] == ["Stable cross-run preference."]


def test_session_local_scope_requires_session_id_for_write(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, session_id=None)

    try:
        manager.remember_memory(
            content="Session-only item",
            scope=MemoryScope.SESSION_LOCAL,
        )
    except ValueError as exc:
        assert "requires an active session id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for session_local without session id")


def test_session_local_scope_requires_session_id_for_recall(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, session_id=None)

    try:
        manager.recall_memories(
            query="Session-only item",
            scope=MemoryScope.SESSION_LOCAL,
        )
    except ValueError as exc:
        assert "requires an active session id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for session_local recall without session id")


def test_forget_memory_allows_cross_user_delete_attempt(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")

    owner = _make_manager(store, user_id="user-a", session_id="run-1")
    memory = owner.remember_memory(
        content="Owner note.",
        importance=3,
        scope=MemoryScope.USER_GLOBAL,
    )

    attacker = _make_manager(store, user_id="user-b", session_id="run-1")
    assert attacker.forget_memory(memory_id=memory.memory_id) is True

    owner_recall = owner.recall_memories(query="Owner", scope=MemoryScope.USER_GLOBAL)
    assert owner_recall == []


def test_forget_memory_allows_cross_session_delete_for_session_local_rows(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")

    owner = _make_manager(store, user_id="default", session_id="run-1")
    memory = owner.remember_memory(
        content="Run-scoped note.",
        importance=4,
        scope=MemoryScope.SESSION_LOCAL,
    )

    same_user_other_session = _make_manager(store, user_id="default", session_id="run-2")
    assert same_user_other_session.forget_memory(memory_id=memory.memory_id) is True

    owner_recall = owner.recall_memories(query="Run-scoped", scope=MemoryScope.SESSION_LOCAL)
    assert owner_recall == []


def test_forget_memory_reports_false_for_missing_id(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, user_id="default", session_id="run-1")

    assert manager.forget_memory(memory_id=999_999) is False
