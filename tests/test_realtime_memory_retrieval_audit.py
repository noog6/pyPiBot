"""Tests for turn memory retrieval audit enrichment."""

from __future__ import annotations


from ai.realtime_api import RealtimeAPI


class _FakeManager:
    def retrieve_for_turn(self, **kwargs):
        return None

    def get_active_user_id(self):
        return "default"

    def get_active_session_id(self):
        return None

    def get_last_turn_retrieval_debug_metadata(self):
        return {
            "mode": "lexical",
            "lexical_candidate_count": 1,
            "semantic_candidate_count": 1,
            "semantic_scored_count": 0,
            "candidates_without_ready_embedding": 1,
            "candidates_below_influence_threshold": 0,
            "candidates_semantic_applied": 0,
            "selected_count": 1,
            "fallback_reason": "query_embedding_not_ready",
            "latency_ms": 4,
            "truncated": False,
            "truncation_count": 0,
            "dedupe_count": 0,
            "semantic_provider": "openai",
            "semantic_model": "text-embedding-3-small",
            "semantic_query_timeout_ms": 40,
            "semantic_query_duration_ms": 12,
            "semantic_timeout_source": "wrapper",
            "semantic_error_code": "timeout_wrapper",
            "semantic_error_class": "TimeoutError",
            "semantic_failure_class": "timeout",
            "canary_last_error_code": "timeout",
            "canary_last_latency_ms": 48,
            "canary_last_checked_age_ms": 730,
            "semantic_provider_last_error_code": "timeout_backoff",
            "timeout_backoff_until_remaining_ms": 120,
            "semantic_scoring_skipped_reason": "query_embedding_timeout",
            "query_fingerprint_hash": "abcdef0123456789",
            "query_fingerprint_length": 19,
        }

    def get_semantic_runtime_health(self):
        return {
            "ready": False,
            "query_embedding_not_ready_streak": 5,
            "last_error_code": "timeout_backoff",
            "readiness_last_transition_at": 123.45,
            "readiness_age_ms": 987,
            "readiness_transition_count": 4,
        }


def test_prepare_turn_memory_brief_logs_semantic_runtime_health_when_streak_non_zero(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._pending_turn_memory_brief = None
    api._memory_retrieval_enabled = True
    api._memory_retrieval_max_memories = 2
    api._memory_retrieval_max_chars = 400
    api._memory_retrieval_cooldown_s = 0.0
    api._memory_retrieval_scope = "user_global"
    api._memory_retrieval_min_user_chars = 1
    api._memory_retrieval_min_user_tokens = 1
    api._memory_manager = _FakeManager()
    api._memory_retrieval_error_throttle_s = 60.0
    api._memory_retrieval_last_error_log_at = 0.0
    api._memory_retrieval_suppressed_errors = 0

    captured = {}

    def _capture(message, *args):
        captured["message"] = message % args

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)
    api._prepare_turn_memory_brief("please remember this", source="unit-test")

    message = captured["message"]
    assert "semantic_runtime_streak=5" in message
    assert "semantic_runtime_last_error=timeout_backoff" in message
    assert "semantic_runtime_readiness_last_transition_at=123.45" in message
    assert "semantic_runtime_readiness_age_ms=987" in message
    assert "semantic_runtime_readiness_transition_count=4" in message
    assert "semantic_provider=openai" in message
    assert "semantic_model=text-embedding-3-small" in message
    assert "semantic_query_timeout_ms=40" in message
    assert "semantic_query_duration_ms=12" in message
    assert "semantic_timeout_source=wrapper" in message
    assert "semantic_error_code=timeout_wrapper" in message
    assert "semantic_error_class=TimeoutError" in message
    assert "semantic_failure_class=timeout" in message
    assert "canary_last_error_code=timeout" in message
    assert "canary_last_latency_ms=48" in message
    assert "canary_last_checked_age_ms=730" in message
    assert "semantic_provider_last_error_code=timeout_backoff" in message
    assert "timeout_backoff_until_remaining_ms=120" in message
    assert "semantic_scoring_skipped_reason=query_embedding_timeout" in message
    assert "query_fingerprint_hash=abcdef0123456789" in message
    assert "query_fingerprint_length=19" in message
