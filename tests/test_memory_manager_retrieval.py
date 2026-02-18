"""Tests for turn-time memory retrieval budget, ranking, and dedupe."""

from __future__ import annotations

import struct
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
    manager._semantic_config = SimpleNamespace(
        enabled=False,
        rerank_enabled=False,
        max_candidates_for_semantic=64,
        min_similarity=0.25,
        rerank_influence_min_cosine=0.25,
        dedupe_strong_match_cosine=None,
        background_embedding_enabled=False,
    )
    manager._last_turn_retrieval_at = {}
    manager._last_turn_retrieval_debug = {}
    manager._retrieval_total_count = 0
    manager._retrieval_total_latency_ms = 0.0
    manager._retrieval_semantic_attempt_count = 0
    manager._retrieval_semantic_error_count = 0
    manager._embedding_coverage_cache_ttl_s = 60.0
    manager._embedding_coverage_cache_at = 0.0
    manager._embedding_coverage_cache = {
        "total_memories": 0,
        "ready_embeddings": 0,
        "coverage_pct": 0.0,
    }
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
