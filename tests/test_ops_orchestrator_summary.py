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
