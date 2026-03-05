"""Tests for realtime session health payload composition."""

from __future__ import annotations

from threading import Event

from ai.realtime_api import RealtimeAPI


class _FakeMemoryManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None]] = []

    def get_active_session_id(self) -> str | None:
        return "session-abc"

    def get_retrieval_health_metrics(
        self,
        *,
        scope: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, float | int]:
        self.calls.append((scope, session_id))
        return {
            "retrieval_count": 3,
            "average_retrieval_latency_ms": 8.5,
            "semantic_provider_attempts": 2,
            "semantic_provider_errors": 0,
            "semantic_provider_error_rate_pct": 0.0,
            "embedding_total_memories": 6,
            "embedding_ready_memories": 5,
            "embedding_coverage_pct": 83.33,
            "pending_count": 4,
            "retry_blocked_count": 2,
            "consecutive_failures": 1,
            "oldest_pending_age_ms": 1250,
        }


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._memory_manager = _FakeMemoryManager()
    api._memory_retrieval_scope = "session_local"
    api._session_connected = True
    api.is_ready_for_injections = lambda: True
    api.ready_event = Event()
    api.ready_event.set()
    api._session_connection_attempts = 4
    api._session_connections = 2
    api._session_reconnects = 1
    api._session_failures = 0
    api._last_connect_time = 123.0
    api._last_disconnect_reason = None
    api._last_failure_reason = None
    api._silent_turn_incident_count = 2
    api._sensor_event_aggregation_metrics = {}
    api.rate_limits_supports_tokens = False
    api.rate_limits_supports_requests = False
    api.rate_limits_last_present_names = set()
    api.rate_limits_last_event_id = ""
    return api


def test_get_session_health_passes_scope_and_active_session_to_memory_metrics() -> None:
    api = _make_api_stub()

    health = api.get_session_health()

    assert api._memory_manager.calls == [("session_local", "session-abc")]
    assert health["memory_retrieval"]["embedding_coverage_pct"] == 83.33
    assert health["memory_retrieval"]["retrieval_count"] == 3
    assert health["memory_retrieval"]["pending_count"] == 4
    assert health["memory_retrieval"]["oldest_pending_age_ms"] == 1250


def test_get_session_health_includes_silent_turn_incident_counter() -> None:
    api = _make_api_stub()

    health = api.get_session_health()

    assert health["silent_turn_incidents"] == 2


def test_get_session_health_ready_uses_injection_readiness_semantics() -> None:
    api = _make_api_stub()
    api.is_ready_for_injections = lambda: False

    health = api.get_session_health()

    assert health["ready"] is False
    assert health["injection_ready"] is False
    assert health["session_ready"] is True
