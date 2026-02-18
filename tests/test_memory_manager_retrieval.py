"""Tests for turn-time memory retrieval budget, ranking, and dedupe."""

from __future__ import annotations

import struct
import time
from types import SimpleNamespace

from services.memory_manager import (
    MemoryManager,
    MemoryScope,
    estimate_realtime_memory_brief_item_chars,
    render_realtime_memory_brief_item,
)
from storage.memories import MemoryStore


def _make_memory_manager(store: MemoryStore) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = "default"
    manager._active_session_id = None
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._store = store
    manager._embedding_worker = None
    manager._semantic_config = SimpleNamespace(
        enabled=False,
        rerank_enabled=False,
        max_candidates_for_semantic=64,
        min_similarity=0.25,
        rerank_influence_min_cosine=0.25,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
        write_timeout_ms=75,
        query_timeout_ms=40,
        max_writes_per_minute=120,
        max_queries_per_minute=240,
    )
    manager._last_turn_retrieval_at = {}
    manager._last_turn_retrieval_debug = {}
    manager._retrieval_total_count = 0
    manager._retrieval_total_latency_ms = 0.0
    manager._retrieval_semantic_attempt_count = 0
    manager._retrieval_semantic_error_count = 0
    manager._embedding_coverage_cache_ttl_s = 60.0
    manager._embedding_coverage_cache = {}
    manager._embedding_backlog_last = {}
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
    manager._embedding_provider = None
    return manager


def _now_ms() -> int:
    return int(time.time() * 1000)


def _encode_vector(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


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


def test_retrieve_for_turn_uses_injected_item_budget_with_tags_and_numbering(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="Daily standup before coffee.",
        tags=["productivity", "habits", "daily-ritual"],
        importance=5,
        user_id="default",
        timestamp=now_ms,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="daily summary",
        user_id="default",
        max_memories=1,
        max_chars=90,
    )

    assert brief is not None
    assert len(brief.items) == 1
    assert brief.total_chars <= 90

    injected_line = render_realtime_memory_brief_item(index=1, item=brief.items[0])
    estimated_chars = estimate_realtime_memory_brief_item_chars(index=1, item=brief.items[0])

    assert len(injected_line) == estimated_chars == brief.total_chars


def test_retrieve_for_turn_budget_matches_rendered_injected_note_chars(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="Prefers early morning planning blocks and clean TODO lists.",
        tags=["workflow", "planning"],
        importance=4,
        user_id="default",
        timestamp=now_ms,
    )
    store.append_memory(
        content="Keeps a shortlist of priority bugs for end-of-day review.",
        tags=["engineering", "triage"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="planning and priority bugs",
        user_id="default",
        max_memories=2,
        max_chars=140,
    )

    assert brief is not None

    rendered_item_chars = sum(
        estimate_realtime_memory_brief_item_chars(index=index, item=item)
        for index, item in enumerate(brief.items, start=1)
    )

    assert rendered_item_chars == brief.total_chars
    assert brief.total_chars <= 140

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


def test_retrieve_for_turn_recovers_low_importance_like_match_from_query_slice(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    for idx in range(20):
        store.append_memory(
            content=f"General system note {idx}.",
            tags=["ops"],
            importance=5,
            user_id="default",
            timestamp=now_ms - idx,
        )

    store.append_memory(
        content="Remember to buy saffron for paella night.",
        tags=["cooking"],
        importance=1,
        user_id="default",
        timestamp=now_ms - 50_000,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="saffron for paella",
        user_id="default",
        max_memories=2,
        max_chars=300,
    )

    assert brief is not None
    assert any("saffron" in item.content.lower() for item in brief.items)


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


def test_retrieve_for_turn_semantic_reranks_within_lexical_pool(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=2,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
    now_ms = _now_ms()

    alpha = store.append_memory(
        content="Project alpha budget notes.",
        tags=["work"],
        importance=3,
        user_id="default",
        timestamp=now_ms,
    )
    beta = store.append_memory(
        content="Project alpha milestones checklist.",
        tags=["work"],
        importance=3,
        user_id="default",
        timestamp=now_ms - 1000,
    )
    gamma = store.append_memory(
        content="Project alpha release plan.",
        tags=["work"],
        importance=3,
        user_id="default",
        timestamp=now_ms - 2000,
    )

    # Keep gamma strongly semantic but outside the bounded semantic pool.
    store.upsert_memory_embedding(
        memory_id=alpha.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([0.9, 0.1]),
        vector_norm=0.9055,
    )
    store.upsert_memory_embedding(
        memory_id=beta.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )
    store.upsert_memory_embedding(
        memory_id=gamma.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _ReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _ReadyProvider()

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=3,
        max_chars=400,
    )

    assert brief is not None
    assert [item.content for item in brief.items] == [
        "Project alpha milestones checklist.",
        "Project alpha budget notes.",
        "Project alpha release plan.",
    ]
    assert manager.get_last_turn_retrieval_debug_metadata()["mode"] == "hybrid"


def test_retrieve_for_turn_semantic_unavailable_falls_back_to_lexical(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
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

    class _FailingProvider:
        def embed_text(self, text: str):
            raise RuntimeError("no provider")

    manager._embedding_provider = _FailingProvider()

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
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["mode"] == "hybrid_fallback_lexical"
    assert metadata["semantic_error"] == "exception"


def test_retrieve_for_turn_rerank_influence_min_cosine_gate(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=2,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.95,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
    now_ms = _now_ms()

    alpha = store.append_memory(
        content="Project alpha budget notes.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms,
    )
    beta = store.append_memory(
        content="Project alpha milestones checklist.",
        tags=["work"],
        importance=3,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    store.upsert_memory_embedding(
        memory_id=alpha.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([0.6, 0.0]),
        vector_norm=0.6,
    )
    store.upsert_memory_embedding(
        memory_id=beta.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([0.8, 0.0]),
        vector_norm=0.8,
    )

    class _ReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _ReadyProvider()

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
    )

    assert brief is not None
    assert [item.content for item in brief.items] == [
        "Project alpha budget notes.",
        "Project alpha milestones checklist.",
    ]


def test_retrieve_for_turn_reports_structured_debug_fields(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="User likes oat milk in coffee.",
        tags=["coffee"],
        importance=4,
        user_id="default",
        timestamp=now_ms,
    )
    store.append_memory(
        content="User likes oat milk with coffee.",
        tags=["coffee"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 10,
    )

    brief = manager.retrieve_for_turn(
        latest_user_utterance="coffee",
        user_id="default",
        max_memories=2,
        max_chars=80,
    )

    assert brief is not None
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["mode"] == "lexical"
    assert metadata["lexical_candidate_count"] >= 1
    assert metadata["semantic_candidate_count"] == 0
    assert metadata["semantic_scored_count"] == 0
    assert metadata["latency_ms"] >= 0
    assert metadata["dedupe_count"] >= 1
    assert metadata["truncation_count"] >= 0


def test_retrieval_health_metrics_include_coverage_error_rate_and_latency(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    first = store.append_memory(
        content="Alpha memory.",
        tags=["alpha"],
        importance=3,
        user_id="default",
        timestamp=now_ms,
    )
    store.append_memory(
        content="Beta memory.",
        tags=["beta"],
        importance=3,
        user_id="default",
        timestamp=now_ms - 1,
    )
    store.upsert_memory_embedding(
        memory_id=first.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
        status="ready",
    )

    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=4,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )

    class _FailingProvider:
        def embed_text(self, text: str):
            raise RuntimeError("provider down")

    manager._embedding_provider = _FailingProvider()
    _ = manager.retrieve_for_turn(
        latest_user_utterance="alpha",
        user_id="default",
        max_memories=2,
        max_chars=120,
    )

    metrics = manager.get_retrieval_health_metrics()
    assert metrics["embedding_coverage_pct"] == 50.0
    assert metrics["semantic_provider_error_rate_pct"] == 100.0
    assert metrics["average_retrieval_latency_ms"] >= 0.0
    assert metrics["retrieval_count"] == 1


def test_retrieve_for_turn_semantic_disabled_keeps_lexical_path(tmp_path) -> None:
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

    class _ShouldNotBeCalledProvider:
        def embed_text(self, text: str):
            raise AssertionError("semantic provider should not be called when disabled")

    manager._embedding_provider = _ShouldNotBeCalledProvider()

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
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["mode"] == "lexical"
    assert metadata["semantic_attempted"] is False


def test_retrieve_for_turn_top_level_semantic_enabled_but_provider_disabled_skips_semantic_attempt(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        provider="openai",
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
        write_timeout_ms=75,
        query_timeout_ms=40,
        max_writes_per_minute=120,
        max_queries_per_minute=240,
    )
    manager._semantic_provider_enabled = {"openai": False}
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

    class _ShouldNotBeCalledProvider:
        def embed_text(self, text: str):
            raise AssertionError("semantic provider should not be called when provider is disabled")

    manager._embedding_provider = _ShouldNotBeCalledProvider()

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
    )

    assert brief is not None
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["semantic_enabled"] is False
    assert metadata["semantic_provider_ready"] is False
    assert metadata["semantic_attempted"] is False
    assert metadata["fallback_reason"] == "openai_provider_disabled"

    metrics = manager.get_retrieval_health_metrics()
    assert metrics["semantic_provider_attempts"] == 0


def test_retrieve_for_turn_semantic_fallback_when_query_embedding_not_ready(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
    now_ms = _now_ms()

    first = store.append_memory(
        content="Remember project alpha milestones.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 2000,
    )
    second = store.append_memory(
        content="Remember project alpha budget.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )
    store.upsert_memory_embedding(
        memory_id=first.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )
    store.upsert_memory_embedding(
        memory_id=second.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _NotReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="pending", dimension=2, vector=None, vector_norm=None)

    manager._embedding_provider = _NotReadyProvider()

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
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["mode"] == "lexical"
    assert metadata["fallback_reason"] == "query_embedding_not_ready"


def test_retrieve_for_turn_session_local_cooldown_is_keyed_by_session_id(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    now_ms = _now_ms()

    store.append_memory(
        content="Session local memory for alpha project.",
        tags=["work"],
        importance=4,
        user_id="default",
        session_id="session-a",
        timestamp=now_ms,
    )
    store.append_memory(
        content="Session local memory for beta project.",
        tags=["work"],
        importance=4,
        user_id="default",
        session_id="session-b",
        timestamp=now_ms,
    )

    first = manager.retrieve_for_turn(
        latest_user_utterance="project",
        user_id="default",
        scope=MemoryScope.SESSION_LOCAL,
        session_id="session-a",
        max_memories=2,
        max_chars=300,
        cooldown_s=20.0,
        now_monotonic=100.0,
    )
    second = manager.retrieve_for_turn(
        latest_user_utterance="project",
        user_id="default",
        scope=MemoryScope.SESSION_LOCAL,
        session_id="session-b",
        max_memories=2,
        max_chars=300,
        cooldown_s=20.0,
        now_monotonic=105.0,
    )

    assert first is not None
    assert second is not None
    assert [item.content for item in first.items] == ["Session local memory for alpha project."]
    assert [item.content for item in second.items] == ["Session local memory for beta project."]


def test_retrieve_for_turn_semantic_enabled_applies_hybrid_reranking(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=2,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
    now_ms = _now_ms()

    alpha = store.append_memory(
        content="Project alpha budget notes.",
        tags=["work"],
        importance=3,
        user_id="default",
        timestamp=now_ms,
    )
    beta = store.append_memory(
        content="Project alpha milestones checklist.",
        tags=["work"],
        importance=3,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    store.upsert_memory_embedding(
        memory_id=alpha.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([0.8, 0.2]),
        vector_norm=0.824,
    )
    store.upsert_memory_embedding(
        memory_id=beta.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _ReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _ReadyProvider()

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
    )

    assert brief is not None
    assert [item.content for item in brief.items] == [
        "Project alpha milestones checklist.",
        "Project alpha budget notes.",
    ]
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["mode"] == "hybrid"
    assert metadata["semantic_applied"] is True


def test_retrieve_for_turn_semantic_provider_exception_falls_back_to_lexical(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
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

    class _FailingProvider:
        def embed_text(self, text: str):
            raise RuntimeError("provider unavailable")

    manager._embedding_provider = _FailingProvider()

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
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["mode"] == "hybrid_fallback_lexical"
    assert metadata["semantic_error"] == "exception"


def test_retrieve_for_turn_semantic_query_timeout_respected(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
        write_timeout_ms=75,
        query_timeout_ms=1,
        max_writes_per_minute=120,
        max_queries_per_minute=240,
    )
    now_ms = _now_ms()

    store.append_memory(
        content="Remember project alpha milestones.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    class _SlowProvider:
        def embed_text(self, text: str):
            time.sleep(0.02)
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _SlowProvider()

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
        cooldown_s=0.0,
    )

    assert brief is not None
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["fallback_reason"] == "query_embedding_not_ready"
    assert metadata["semantic_error_code"] == "timeout"


def test_retrieve_for_turn_semantic_query_rate_limit_respected(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
        write_timeout_ms=75,
        query_timeout_ms=40,
        max_writes_per_minute=120,
        max_queries_per_minute=1,
    )
    now_ms = _now_ms()

    store.append_memory(
        content="Remember project alpha milestones.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    class _ReadyProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _ReadyProvider()

    first = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
        cooldown_s=0.0,
    )
    second = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
        cooldown_s=0.0,
    )

    assert first is not None
    assert second is not None
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["fallback_reason"] == "query_embedding_not_ready"
    assert metadata["semantic_error_code"] == "rate_limited"


def test_retrieve_for_turn_semantic_query_non_ready_status_sets_error_code(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=True,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
        write_timeout_ms=75,
        query_timeout_ms=40,
        max_writes_per_minute=120,
        max_queries_per_minute=240,
    )
    now_ms = _now_ms()
    store.append_memory(
        content="Remember project alpha milestones.",
        tags=["work"],
        importance=4,
        user_id="default",
        timestamp=now_ms - 1000,
    )

    class _UnavailableProvider:
        def embed_text(self, text: str):
            return SimpleNamespace(
                status="unavailable",
                dimension=0,
                vector=b"",
                vector_norm=None,
            )

    manager._embedding_provider = _UnavailableProvider()

    brief = manager.retrieve_for_turn(
        latest_user_utterance="project alpha",
        user_id="default",
        max_memories=2,
        max_chars=300,
        cooldown_s=0.0,
    )

    assert brief is not None
    metadata = manager.get_last_turn_retrieval_debug_metadata()
    assert metadata["fallback_reason"] == "query_embedding_not_ready"
    assert metadata["semantic_error_code"] == "unavailable"


def test_find_semantic_duplicate_respects_write_timeout(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=False,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=0.9,
        background_embedding_enabled=False,
        write_timeout_ms=1,
        query_timeout_ms=40,
        max_writes_per_minute=120,
        max_queries_per_minute=240,
    )
    manager._auto_reflection_semantic_dedupe_enabled = True
    now_ms = _now_ms()
    prior = store.append_memory(
        content="Project alpha budget notes.",
        tags=["work"],
        importance=4,
        user_id="default",
        needs_review=False,
        timestamp=now_ms,
    )
    store.upsert_memory_embedding(
        memory_id=prior.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _SlowProvider:
        def embed_text(self, text: str):
            time.sleep(0.02)
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    manager._embedding_provider = _SlowProvider()

    duplicate = manager._find_semantic_duplicate(
        content="Project alpha budget notes.",
        user_id="default",
        scope=MemoryScope.USER_GLOBAL,
        session_id=None,
        source="auto_reflection",
    )

    assert duplicate is None


def test_find_semantic_duplicate_respects_write_rate_limit(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager._semantic_config = SimpleNamespace(
        enabled=True,
        rerank_enabled=False,
        max_candidates_for_semantic=8,
        min_similarity=0.0,
        rerank_influence_min_cosine=0.0,
        dedupe_strong_match_cosine=0.9,
        background_embedding_enabled=False,
        write_timeout_ms=75,
        query_timeout_ms=40,
        max_writes_per_minute=1,
        max_queries_per_minute=240,
    )
    manager._auto_reflection_semantic_dedupe_enabled = True
    now_ms = _now_ms()
    prior = store.append_memory(
        content="Project alpha budget notes.",
        tags=["work"],
        importance=4,
        user_id="default",
        needs_review=False,
        timestamp=now_ms,
    )
    store.upsert_memory_embedding(
        memory_id=prior.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
    )

    class _ReadyProvider:
        def __init__(self) -> None:
            self.calls = 0

        def embed_text(self, text: str):
            self.calls += 1
            return SimpleNamespace(status="ready", dimension=2, vector=_encode_vector([1.0, 0.0]), vector_norm=1.0)

    provider = _ReadyProvider()
    manager._embedding_provider = provider

    first = manager._find_semantic_duplicate(
        content="Project alpha budget notes.",
        user_id="default",
        scope=MemoryScope.USER_GLOBAL,
        session_id=None,
        source="auto_reflection",
    )
    second = manager._find_semantic_duplicate(
        content="Project alpha budget notes.",
        user_id="default",
        scope=MemoryScope.USER_GLOBAL,
        session_id=None,
        source="auto_reflection",
    )

    assert first is not None
    assert second is None
    assert provider.calls == 1


def test_retrieval_health_metrics_accept_scope_and_session_overrides(tmp_path) -> None:
    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)

    global_entry = store.append_memory(
        content="Global memory.",
        tags=["global"],
        importance=3,
        user_id="default",
    )
    session_entry = store.append_memory(
        content="Session memory.",
        tags=["session"],
        importance=3,
        user_id="default",
        session_id="session-1",
    )
    store.upsert_memory_embedding(
        memory_id=global_entry.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([1.0, 0.0]),
        vector_norm=1.0,
        status="ready",
    )
    store.upsert_memory_embedding(
        memory_id=session_entry.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([0.0, 1.0]),
        vector_norm=1.0,
        status="pending",
    )

    default_metrics = manager.get_retrieval_health_metrics()
    assert default_metrics["embedding_total_memories"] == 1
    assert default_metrics["embedding_coverage_pct"] == 100.0
    assert default_metrics["embedding_pending_memories"] == 0
    assert default_metrics["embedding_missing_memories"] == 0
    assert default_metrics["embedding_backlog_memories"] == 0
    assert default_metrics["embedding_backlog_delta_since_last"] == 0

    session_metrics = manager.get_retrieval_health_metrics(
        scope=MemoryScope.SESSION_LOCAL,
        session_id="session-1",
    )
    assert session_metrics["embedding_total_memories"] == 1
    assert session_metrics["embedding_ready_memories"] == 0
    assert session_metrics["embedding_coverage_pct"] == 0.0
    assert session_metrics["embedding_pending_memories"] == 1
    assert session_metrics["embedding_missing_memories"] == 0
    assert session_metrics["embedding_backlog_memories"] == 1
    assert session_metrics["embedding_backlog_delta_since_last"] == 0

    store.upsert_memory_embedding(
        memory_id=session_entry.memory_id,
        model_id="unit",
        dim=2,
        vector=_encode_vector([0.0, 1.0]),
        vector_norm=1.0,
        status="ready",
    )
    updated_session_metrics = manager.get_retrieval_health_metrics(
        scope=MemoryScope.SESSION_LOCAL,
        session_id="session-1",
    )
    assert updated_session_metrics["embedding_backlog_memories"] == 0
    assert updated_session_metrics["embedding_backlog_delta_since_last"] == -1
