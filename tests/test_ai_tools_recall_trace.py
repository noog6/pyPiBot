"""Tests for explainable trace output on recall_memories tool calls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ai.tools as ai_tools
from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


def _make_manager(store: MemoryStore, *, trace_enabled: bool) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = "default"
    manager._active_session_id = None
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._store = store
    manager._embedding_worker = None
    manager._semantic_config = SimpleNamespace(enabled=False, rerank_enabled=False)
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
    manager._recall_trace_enabled = trace_enabled
    manager._recall_trace_level = "info"
    return manager


def test_recall_tool_omits_trace_when_disabled(monkeypatch, tmp_path, caplog) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=False)
    manager.remember_memory(content="VAD threshold note", importance=4, scope=MemoryScope.USER_GLOBAL)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)

    with caplog.at_level("INFO"):
        result = asyncio.run(ai_tools.recall_memories(query="VAD", limit=5))

    assert "trace" not in result
    assert len(result["memories"]) == 1
    assert "memory_recall_trace" not in caplog.text


def test_recall_tool_trace_contains_expected_fields_when_enabled(monkeypatch, tmp_path, caplog) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="VAD profile is set to short confirmations", importance=4)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)

    with caplog.at_level("INFO"):
        result = asyncio.run(ai_tools.recall_memories(query="VAD", limit=5))

    assert len(result["memories"]) >= 1
    trace = result["trace"]
    assert trace["mode_used"] == "lexical"
    assert trace["query_original"] == "VAD"
    assert trace["candidate_counts"]["selected"] >= 1
    assert trace["selected"]
    assert trace["selected"][0]["selected_reason"] == "top_ranked_lexical_within_limit"
    assert "memory_recall_trace" in caplog.text


def test_recall_tool_trace_reports_empty_with_fallback(monkeypatch, tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="Voice activity detection helps with VAD.", importance=4)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)
    result = asyncio.run(ai_tools.recall_memories(query="memory ID 22", limit=5))

    assert result["memories"] == []
    trace = result["trace"]
    assert trace["candidate_counts"]["selected"] == 0
    assert trace["excluded_summary"]
    assert trace["fallback_suggestion"]["suggested_query"]
    assert "ID lookup" in trace["fallback_suggestion"]["reason"]


def test_recall_trace_counts_deduped_and_truncated_candidates(monkeypatch, tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="duplicate content", importance=5)
    manager.remember_memory(content="duplicate content", importance=4)
    manager.remember_memory(content="duplicate content", importance=3)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)
    result = asyncio.run(ai_tools.recall_memories(query="duplicate", limit=1))

    trace = result["trace"]
    assert trace["excluded_summary"]["deduped"] >= 1
    assert trace["excluded_summary"]["truncated"] >= 1


def test_recall_memories_callers_keep_list_contract(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="caller compatibility", importance=3)

    recalls = manager.recall_memories(query="caller", limit=1)

    assert isinstance(recalls, list)
    assert recalls[0].content == "caller compatibility"
