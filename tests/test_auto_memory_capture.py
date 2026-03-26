"""Tests for auto-memory capture guardrails and source tagging."""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace

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
    manager._embedding_worker = None
    manager._last_turn_retrieval_at = {}
    manager._last_semantic_dedupe_debug = {}
    manager._semantic_config = SimpleNamespace(
        enabled=False,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
        inline_embedding_on_write_when_background_disabled=False,
    )
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


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.api_key = "test"
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn-test"
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
    api._auto_memory_repeat_min_confidence_floor = 0.55
    api._auto_memory_repeat_min_confidence_delta = 0.20
    api._auto_memory_repeat_threshold = 2
    api._auto_memory_repeat_window_s = 30.0
    api._auto_memory_candidate_history_max_entries = 3
    api._auto_memory_candidate_history = {}
    return api


def test_auto_memory_guardrail_blocks_low_signal_or_low_confidence() -> None:
    api = _make_api_stub()
    api._last_user_input_text = "ok"
    assert api._should_store_auto_memory(confidence=0.95, content="prefers tea") is False

    api._last_user_input_text = "Please remember I like dark roast coffee in the morning."
    assert api._should_store_auto_memory(confidence=0.40, content="prefers dark roast") is False


def test_auto_memory_single_shot_low_confidence_rejected() -> None:
    api = _make_api_stub()

    assert (
        api._should_store_auto_memory(
            confidence=0.62,
            content="User prefers keyboard-only navigation in IDE workflows.",
            tags=["workflow"],
        )
        is False
    )


def test_auto_memory_repeated_low_confidence_within_window_accepted() -> None:
    api = _make_api_stub()

    first = api._should_store_auto_memory(
        confidence=0.62,
        content="User prefers keyboard-only navigation in IDE workflows.",
        tags=["workflow"],
    )
    second = api._should_store_auto_memory(
        confidence=0.62,
        content="  user prefers keyboard-only navigation in ide workflows.  ",
        tags=["workflow"],
    )

    assert first is False
    assert second is True


def test_auto_memory_high_confidence_still_allows_immediately() -> None:
    api = _make_api_stub()

    assert (
        api._should_store_auto_memory(
            confidence=0.92,
            content="User consistently prefers short, bulleted summaries.",
            tags=["communication"],
        )
        is True
    )


def test_auto_memory_repetition_window_expiry_does_not_promote() -> None:
    api = _make_api_stub()

    assert (
        api._should_store_auto_memory(
            confidence=0.62,
            content="User prefers concise diffs with explicit rationale.",
            tags=["workflow"],
        )
        is False
    )

    fingerprint = api._fingerprint_auto_memory_candidate(
        normalized_candidate=api._normalize_auto_memory_candidate(
            content="User prefers concise diffs with explicit rationale.",
            tags=["workflow"],
        )
    )
    entry = api._auto_memory_candidate_history[fingerprint]
    entry["last_seen"] = float(entry["last_seen"]) - (api._auto_memory_repeat_window_s + 1.0)

    assert (
        api._should_store_auto_memory(
            confidence=0.62,
            content="User prefers concise diffs with explicit rationale.",
            tags=["workflow"],
        )
        is False
    )


def test_auto_memory_candidate_history_prunes_deterministically() -> None:
    api = _make_api_stub()
    api._auto_memory_repeat_window_s = 10_000.0
    contents = [
        "User likes dark mode themes.",
        "User likes low-latency terminals.",
        "User likes deterministic test fixtures.",
        "User likes compact commit messages.",
    ]

    for content in contents:
        api._should_store_auto_memory(confidence=0.62, content=content, tags=["pref"])

    assert len(api._auto_memory_candidate_history) == api._auto_memory_candidate_history_max_entries
    evicted = api._fingerprint_auto_memory_candidate(
        normalized_candidate=api._normalize_auto_memory_candidate(content=contents[0], tags=["pref"])
    )
    retained = api._fingerprint_auto_memory_candidate(
        normalized_candidate=api._normalize_auto_memory_candidate(content=contents[-1], tags=["pref"])
    )
    assert evicted not in api._auto_memory_candidate_history
    assert retained in api._auto_memory_candidate_history


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


def _encode_vector(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def test_auto_reflection_semantic_duplicate_skip_write_policy(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        dedupe_strong_match_cosine=0.85,
        background_embedding_enabled=False,
        inline_embedding_on_write_when_background_disabled=False,
    )
    manager._auto_reflection_semantic_dedupe_enabled = True

    existing = store.append_memory(
        content="User likes jazz while coding.",
        tags=["music"],
        importance=4,
        user_id="default",
        needs_review=False,
    )
    store.upsert_memory_embedding(
        memory_id=existing.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _ReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _ReadyProvider()

    result = manager.remember_memory(
        content="User prefers jazz while coding sessions.",
        source="auto_reflection",
        importance=4,
    )

    assert result.memory_id == existing.memory_id
    assert len(store.search_memories(user_id="default", scope=MemoryScope.USER_GLOBAL, limit=10, review_state="all")) == 1


def test_manual_tool_write_remains_authoritative_under_default_policy(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        dedupe_strong_match_cosine=0.85,
        background_embedding_enabled=False,
        inline_embedding_on_write_when_background_disabled=False,
    )
    manager._auto_reflection_semantic_dedupe_enabled = True

    existing = store.append_memory(
        content="User likes jazz while coding.",
        tags=["music"],
        importance=4,
        user_id="default",
        needs_review=False,
    )
    store.upsert_memory_embedding(
        memory_id=existing.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _ReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _ReadyProvider()

    result = manager.remember_memory(
        content="User prefers jazz while coding sessions.",
        source="manual_tool",
        importance=4,
    )

    assert result.memory_id != existing.memory_id
    assert len(store.search_memories(user_id="default", scope=MemoryScope.USER_GLOBAL, limit=10, review_state="all")) == 2


def test_auto_reflection_write_survives_semantic_dedupe_embedding_exception(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        dedupe_strong_match_cosine=0.85,
        background_embedding_enabled=False,
        inline_embedding_on_write_when_background_disabled=False,
    )
    manager._auto_reflection_semantic_dedupe_enabled = True

    store.append_memory(
        content="User likes jazz while coding.",
        tags=["music"],
        importance=4,
        user_id="default",
        needs_review=False,
    )

    class _FailingProvider:
        def embed_text(self, text: str):
            raise RuntimeError("provider unavailable")

    manager._embedding_provider = _FailingProvider()

    result = manager.remember_memory(
        content="User prefers jazz while coding sessions.",
        source="auto_reflection",
        importance=4,
    )

    assert result.content == "User prefers jazz while coding sessions."
    assert len(store.search_memories(user_id="default", scope=MemoryScope.USER_GLOBAL, limit=10, review_state="all")) == 2
    assert manager.get_last_semantic_dedupe_debug_metadata()["error_code"] == "embedding_provider_exception"
