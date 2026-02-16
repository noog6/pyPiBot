"""Tests for auto-memory capture guardrails and source tagging."""

from __future__ import annotations

import asyncio

from ai.realtime_api import RealtimeAPI
from ai.tools import function_map
from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


def _make_memory_manager(store: MemoryStore) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = "default"
    manager._active_session_id = None
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._store = store
    manager._last_turn_retrieval_at = {}
    manager._auto_pin_min_importance = 5
    manager._auto_pin_requires_review = True
    return manager


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.api_key = "test"
    api._allow_ai_call = lambda *args, **kwargs: True
    api._record_ai_call = lambda: None
    api._tool_call_records = []
    api._last_response_metadata = {}
    api._last_user_input_source = "text_message"
    api._last_user_input_time = 123.0
    api._last_user_input_text = "Please remember that I prefer quiet coffee shops for work."
    api._assistant_reply_accum = "Understood, I'll keep that in mind."
    api._auto_memory_enabled = True
    api._require_confirmation_for_auto_memory = False
    api._auto_memory_min_confidence = 0.75
    return api


def test_auto_memory_guardrail_blocks_low_signal_or_low_confidence() -> None:
    api = _make_api_stub()
    api._last_user_input_text = "ok"
    assert api._should_store_auto_memory(confidence=0.95, content="prefers tea") is False

    api._last_user_input_text = "Please remember I like dark roast coffee in the morning."
    assert api._should_store_auto_memory(confidence=0.40, content="prefers dark roast") is False


def test_response_done_auto_memory_honors_confirmation_guardrail() -> None:
    api = _make_api_stub()
    api._require_confirmation_for_auto_memory = True
    api._call_openai_prompt = lambda prompt: (
        '{"summary":"x","remember_memory":{"content":"Prefers libraries","importance":3,"confidence":0.95}}'
    )

    calls: list[dict[str, object]] = []

    async def _fake_remember_memory(**kwargs):
        calls.append(kwargs)
        return {"memory_id": 1}

    original = function_map.get("remember_memory")
    function_map["remember_memory"] = _fake_remember_memory
    try:
        asyncio.run(api._run_response_done_reflection("response done"))
    finally:
        if original is not None:
            function_map["remember_memory"] = original

    assert calls == []


def test_memory_source_metadata_persists_for_audit(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)

    manager.remember_memory(
        content="User enjoys ambient instrumental playlists while coding.",
        importance=4,
        scope=MemoryScope.USER_GLOBAL,
        source="auto_reflection",
    )

    memories = manager.recall_memories(query="ambient", scope=MemoryScope.USER_GLOBAL)
    assert memories
    assert memories[0].source == "auto_reflection"
