"""Tests for ops orchestrator health summary enrichment."""

from __future__ import annotations

from core.ops_models import HealthStatus
from services.health_probes import HealthProbeResult
from services.ops_orchestrator import OpsOrchestrator


def _new_orchestrator() -> OpsOrchestrator:
    OpsOrchestrator._instance = None
    orchestrator = OpsOrchestrator()
    OpsOrchestrator._instance = None
    return orchestrator


def test_summarize_health_includes_probe_reasons_for_warmup() -> None:
    orchestrator = _new_orchestrator()

    summary = orchestrator._summarize_health(
        HealthStatus.DEGRADED,
        [
            HealthProbeResult("battery", HealthStatus.DEGRADED, "Battery monitor warming up"),
            HealthProbeResult("realtime", HealthStatus.DEGRADED, "Realtime API not initialized"),
        ],
    )

    assert summary.startswith("Degraded:")
    assert "battery (warming up)" in summary
    assert "realtime (not initialized)" in summary


def test_summarize_health_distinguishes_warmup_from_hard_failure() -> None:
    orchestrator = _new_orchestrator()

    summary = orchestrator._summarize_health(
        HealthStatus.FAILING,
        [
            HealthProbeResult("battery", HealthStatus.DEGRADED, "Battery monitor warming up"),
            HealthProbeResult("realtime", HealthStatus.FAILING, "Realtime session disconnected"),
        ],
    )

    assert summary.startswith("Critical issues:")
    assert "battery (warming up)" in summary
    assert "realtime (disconnected)" in summary
    assert "realtime (warming up)" not in summary


def test_summarize_health_truncates_to_safe_maximum() -> None:
    orchestrator = _new_orchestrator()
    long_reason = "startup synchronization delay due to repeated calibration cycle " * 20

    summary = orchestrator._summarize_health(
        HealthStatus.DEGRADED,
        [HealthProbeResult("audio", HealthStatus.DEGRADED, long_reason)],
    )

    assert len(summary) <= orchestrator._MAX_HEALTH_SUMMARY_CHARS
    assert summary.endswith("...")


def test_probe_memory_semantic_runtime_degrades_when_streak_exceeds_threshold(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._semantic_offline_streak_threshold = 2

    class _Mgr:
        def get_semantic_runtime_health(self):
            return {
                "ready": False,
                "query_embedding_not_ready_streak": 3,
                "last_error_code": "timeout_backoff",
            }

    monkeypatch.setattr("services.ops_orchestrator.MemoryManager.get_instance", lambda: _Mgr())

    result = orchestrator._probe_memory_semantic_runtime()

    assert result.name == "memory_semantic_runtime"
    assert result.status == HealthStatus.DEGRADED
    assert "offline" in result.summary
    assert result.details["query_embedding_not_ready_streak"] == 3


def test_summarize_health_includes_semantic_runtime_probe_reason() -> None:
    orchestrator = _new_orchestrator()

    summary = orchestrator._summarize_health(
        HealthStatus.DEGRADED,
        [
            HealthProbeResult(
                "memory_semantic_runtime",
                HealthStatus.DEGRADED,
                "Semantic retrieval offline (streak=24, code=timeout_backoff)",
            )
        ],
    )

    assert summary.startswith("Degraded:")
    assert "memory_semantic_runtime (offline)" in summary



def test_emit_canonical_snapshot_uses_versioned_stable_fields() -> None:
    orchestrator = _new_orchestrator()

    orchestrator._emit_canonical_snapshot(123.0, reason="startup")

    event = orchestrator._recent_events[-1]
    assert event.event_type == "ops_snapshot"
    assert set(event.metadata.keys()) == {
        "schema_version",
        "emitted_at",
        "reason",
        "mode",
        "loop_phase",
        "active_probe",
        "ticks",
        "heartbeats",
        "errors",
        "health_status",
        "health_summary",
        "loop_period_s",
        "heartbeat_period_s",
    }
    assert event.metadata["schema_version"] == "ops_snapshot.v1"
    assert event.metadata["reason"] == "startup"


def test_start_loop_emits_startup_snapshot(monkeypatch) -> None:
    orchestrator = _new_orchestrator()

    class FakeThread:
        def __init__(self, target=None, daemon=None) -> None:
            self._alive = False
            self.name = "fake"
            self.ident = 1

        def start(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout=None) -> None:
            self._alive = False

    reasons: list[str] = []

    monkeypatch.setattr("services.ops_orchestrator.threading.Thread", FakeThread)
    monkeypatch.setattr(orchestrator, "_load_probe_config", lambda: None)
    monkeypatch.setattr(
        orchestrator,
        "_emit_canonical_snapshot",
        lambda timestamp, reason: reasons.append(reason),
    )

    orchestrator.start_loop(loop_period_s=0.1, heartbeat_period_s=0.2)

    assert reasons == ["startup"]
