"""Tests for realtime session health payload composition."""

from __future__ import annotations

from ai.realtime_api import RealtimeAPI


class _FakeReadyEvent:
    def __init__(self, ready: bool) -> None:
        self._ready = ready

    def is_set(self) -> bool:
        return self._ready


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
        }


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._memory_manager = _FakeMemoryManager()
    api._memory_retrieval_scope = "session_local"
    api._session_connected = True
    api.ready_event = _FakeReadyEvent(True)
    api._session_connection_attempts = 4
    api._session_connections = 2
    api._session_reconnects = 1
    api._session_failures = 0
    api._last_connect_time = 123.0
    api._last_disconnect_reason = None
    api._last_failure_reason = None
    return api


def test_get_session_health_passes_scope_and_active_session_to_memory_metrics() -> None:
    api = _make_api_stub()

    health = api.get_session_health()

    assert api._memory_manager.calls == [("session_local", "session-abc")]
    assert health["memory_retrieval"]["embedding_coverage_pct"] == 83.33
    assert health["memory_retrieval"]["retrieval_count"] == 3
