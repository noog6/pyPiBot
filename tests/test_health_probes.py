"""Tests for health probe detail enrichment."""

from services.health_probes import probe_realtime_session


class _FakeRealtime:
    def get_session_health(self):
        return {
            "connected": True,
            "ready": True,
            "failures": 0,
            "reconnects": 1,
            "memory_retrieval": {
                "embedding_coverage_pct": 87.5,
                "semantic_provider_error_rate_pct": 2.0,
                "average_retrieval_latency_ms": 14.2,
                "retrieval_count": 10,
                "semantic_provider_attempts": 8,
                "semantic_provider_errors": 1,
                "query_embedding_latency_p90_ms": 120,
                "query_embedding_latency_bucket_gt_1000ms": 2,
                "canary_refresh_latency_p50_ms": 40,
                "canary_refresh_latency_bucket_le_100ms": 4,
            },
        }

    def is_ready_for_injections(self):
        return True


def test_probe_realtime_session_includes_memory_retrieval_details() -> None:
    result = probe_realtime_session(_FakeRealtime())

    assert result.details["memory_embedding_coverage_pct"] == 87.5
    assert result.details["memory_semantic_provider_error_rate_pct"] == 2.0
    assert result.details["memory_average_retrieval_latency_ms"] == 14.2
    assert result.details["memory_retrieval_count"] == 10
    assert result.details["memory_semantic_provider_attempts"] == 8
    assert result.details["memory_semantic_provider_errors"] == 1
    assert result.details["memory_query_embedding_latency_p90_ms"] == 120
    assert result.details["memory_query_embedding_latency_bucket_gt_1000ms"] == 2
    assert result.details["memory_canary_refresh_latency_p50_ms"] == 40
    assert result.details["memory_canary_refresh_latency_bucket_le_100ms"] == 4
    assert result.details["ready_source"] == "is_ready_for_injections()"
    assert result.details["connected_source"] == "session_health.connected"


class _CallableReadyEmptyHealth:
    def get_session_health(self):
        return {}

    def is_ready_for_injections(self):
        return True


def test_probe_realtime_session_uses_callable_injection_readiness_first() -> None:
    result = probe_realtime_session(_CallableReadyEmptyHealth())

    assert result.details["ready"] is True
    assert result.details["ready_source"] == "is_ready_for_injections()"


class _PropertyReadyRealtime:
    is_ready_for_injections = True

    def get_session_health(self):
        return {"connected": True, "ready": False}


def test_probe_realtime_session_supports_boolean_readiness_property() -> None:
    result = probe_realtime_session(_PropertyReadyRealtime())

    assert result.details["ready"] is True
    assert result.details["ready_source"] == "is_ready_for_injections_property"
    assert result.details["connected_source"] == "session_health.connected"


class _ReadyMismatchRealtime:
    def get_session_health(self):
        return {"connected": True, "ready": False, "session_ready": False}

    def is_ready_for_injections(self):
        return True


def test_probe_realtime_session_uses_injection_ready_for_status() -> None:
    result = probe_realtime_session(_ReadyMismatchRealtime())

    assert result.status.value == "ok"
    assert result.summary == "Realtime session connected"
    assert result.details["injection_ready"] is True
    assert result.details["session_ready"] is False
    assert result.details["session_ready_source"] == "session_health.session_ready"


class _ConnectedNotReadyRealtime:
    def get_session_health(self):
        return {
            "connected": True,
            "ready": False,
            "injection_ready": False,
            "injection_ready_reason": "ready_event_not_set",
        }


class _ConnectedFirstTurnRealtime:
    def get_session_health(self):
        return {
            "connected": True,
            "ready": False,
            "injection_ready": False,
            "injection_ready_reason": "response_in_progress",
        }


class _DisconnectedRealtime:
    def get_session_health(self):
        return {
            "connected": False,
            "ready": False,
            "failures": 2,
        }


class _ReadyTransitionRealtime:
    def __init__(self):
        self.ready = False

    def get_session_health(self):
        return {
            "connected": True,
            "ready": self.ready,
            "injection_ready": self.ready,
            "injection_ready_reason": "ready" if self.ready else "ready_event_not_set",
        }


def test_probe_realtime_session_connected_not_ready_is_warmup() -> None:
    result = probe_realtime_session(_ConnectedNotReadyRealtime())

    assert result.status.value == "warmup"
    assert result.summary == "Realtime connected (awaiting injection readiness)"


def test_probe_realtime_session_first_turn_unsettled_is_ok_note() -> None:
    result = probe_realtime_session(_ConnectedFirstTurnRealtime())

    assert result.status.value == "ok"
    assert result.summary == "Realtime session connected (response in progress)"


def test_probe_realtime_session_disconnected_remains_failing() -> None:
    result = probe_realtime_session(_DisconnectedRealtime())

    assert result.status.value == "failing"
    assert result.summary == "Realtime session disconnected"


def test_probe_realtime_session_transitions_to_ok_after_readiness() -> None:
    realtime = _ReadyTransitionRealtime()

    initial = probe_realtime_session(realtime)
    realtime.ready = True
    ready = probe_realtime_session(realtime)

    assert initial.status.value == "warmup"
    assert ready.status.value == "ok"
    assert ready.summary == "Realtime session connected"
