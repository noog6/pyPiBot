"""Tests for explainable trace output on recall_memories tool calls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ai.tools as ai_tools
from services.memory_manager import MemoryManager, MemoryScope, normalize_recall_query_with_fallback
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


def test_recall_query_rewrite_rules() -> None:
    assert normalize_recall_query_with_fallback("memory id 22") == ("22", "strip_memory_id_wrapper")
    assert normalize_recall_query_with_fallback("recall memory with an ID of number 22") == (
        "22",
        "strip_memory_id_wrapper",
    )
    assert normalize_recall_query_with_fallback("memory video 22") == ("memory video 22", "none")
    assert normalize_recall_query_with_fallback("VAD settings") == ("VAD settings", "none")


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

    trace = result["trace"]
    assert trace["query_original"] == "VAD"
    assert trace["query_used"] == "VAD"
    assert trace["retrieval_mode"] == "lexical"
    assert trace["rewrite_applied"] == "none"
    assert "candidate_counts" in trace
    assert "thresholds_used" in trace
    assert "ranking_summary" in trace
    assert trace["ranking_summary"]
    assert any(item.get("selected") for item in trace["ranking_summary"])
    assert "memory_recall_trace" in caplog.text


def test_recall_tool_fallback_rewrite_can_recover_results(monkeypatch, tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="Threshold was tuned to 22 for VAD sensitivity.", importance=4)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)
    result = asyncio.run(ai_tools.recall_memories(query="memory ID: 22", limit=5))

    assert len(result["memories"]) == 1
    trace = result["trace"]
    assert trace["rewrite_applied"] == "strip_memory_id_wrapper"
    assert trace["query_used"] == "22"
    assert trace["rewrite_helped"] is True
    assert len(trace["attempts"]) == 2


def test_recall_trace_includes_exclusion_reason(monkeypatch, tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="duplicate content", importance=5)
    manager.remember_memory(content="duplicate content", importance=4)
    manager.remember_memory(content="duplicate content", importance=3)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)
    result = asyncio.run(ai_tools.recall_memories(query="duplicate", limit=1))

    ranking_summary = result["trace"]["ranking_summary"]
    assert any(not item["selected"] for item in ranking_summary)
    assert any(item.get("exclusion_reason") for item in ranking_summary if not item["selected"])


def test_recall_cards_do_not_leak_memory_id(monkeypatch, tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="VAD padding is 700ms.", importance=5)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)
    result = asyncio.run(ai_tools.recall_memories(query="VAD", limit=5))

    output = result["memory_cards_text"]
    assert "memory_id" not in output.lower()
    assert "id " not in output.lower()
    assert "#" not in output
    assert "memory_id" not in str(result["memory_cards"])


def test_recall_memories_callers_keep_list_contract(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="caller compatibility", importance=3)

    recalls = manager.recall_memories(query="caller", limit=1)

    assert isinstance(recalls, list)
    assert recalls[0].content == "caller compatibility"


def test_recall_cards_exact_lexical_hit_not_low(monkeypatch, tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_manager(store, trace_enabled=True)
    manager.remember_memory(content="User's favorite editor is Vim.", importance=3)

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)
    result = asyncio.run(ai_tools.recall_memories(query="Vim", limit=5))

    assert result["memory_cards"]
    card = result["memory_cards"][0]
    assert card["confidence"] in {"High", "Medium"}
    assert card["confidence"] != "Low"
    assert "lexical exact match" in card["why_relevant"]


def test_recall_cards_semantic_only_stays_medium_or_low() -> None:
    cards = ai_tools.build_recall_memory_cards(
        query="coding style",
        memories=[
            {
                "content": "Prefers keyboard-first workflows.",
                "tags": [],
                "importance": 3,
                "source": "manual_tool",
                "pinned": False,
                "needs_review": False,
            }
        ],
        trace={
            "rewrite_applied": "none",
            "thresholds_used": {"influence_threshold": 0.0},
            "ranking_summary": [
                {
                    "selected": True,
                    "score_lexical": 0.0,
                    "score_semantic": 0.62,
                    "influence_score": 0.62,
                    "exclusion_reason": None,
                }
            ],
        },
    )

    assert cards
    assert cards[0]["confidence"] in {"Medium", "Low"}
    assert "semantic similarity=0.62" in cards[0]["why_relevant"]
