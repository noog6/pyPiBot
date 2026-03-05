"""Tests for ops orchestrator health summary enrichment."""

from __future__ import annotations

import time
import types
import sys

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")


if "ai.realtime_api" not in sys.modules:
    realtime_api_stub = types.ModuleType("ai.realtime_api")
    realtime_api_stub.RealtimeAPI = object
    sys.modules["ai.realtime_api"] = realtime_api_stub

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


def test_tick_reports_warmup_for_transient_startup_degradations(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._health_debounce_s = 0.0
    captured = []

    monkeypatch.setattr(
        orchestrator,
        "_run_health_probes",
        lambda: [
            HealthProbeResult("audio", HealthStatus.DEGRADED, "Audio partially available"),
            HealthProbeResult(
                "realtime",
                HealthStatus.DEGRADED,
                "Realtime session offline",
                details={"connected": 0, "ready": 0},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ],
    )
    monkeypatch.setattr(orchestrator, "_maybe_run_micro_presence", lambda now: None)
    monkeypatch.setattr(orchestrator, "_maybe_run_memory_maintenance", lambda now: None)
    monkeypatch.setattr(orchestrator, "_emit_health_snapshot", lambda snapshot: captured.append(snapshot))

    orchestrator._tick()

    assert captured[-1].status == HealthStatus.WARMUP
    assert captured[-1].summary.startswith("Warming up:")


def test_tick_exits_warmup_with_criteria_met_for_boolean_realtime_details(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._health_debounce_s = 0.0
    captured = []

    monkeypatch.setattr(
        orchestrator,
        "_run_health_probes",
        lambda: [
            HealthProbeResult("audio", HealthStatus.OK, "Audio input/output ready"),
            HealthProbeResult(
                "realtime",
                HealthStatus.OK,
                "Realtime session connected",
                details={"connected": True, "ready": True},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ],
    )
    monkeypatch.setattr(orchestrator, "_maybe_run_micro_presence", lambda now: None)
    monkeypatch.setattr(orchestrator, "_maybe_run_memory_maintenance", lambda now: None)
    monkeypatch.setattr(orchestrator, "_emit_health_snapshot", lambda snapshot: captured.append(snapshot))

    orchestrator._tick()

    assert captured[-1].status == HealthStatus.OK
    assert orchestrator._warmup_active is False
    assert captured[-1].details.get("warmup_exit_reason") == "criteria_met"


def test_warmup_summary_does_not_report_realtime_offline_when_connected() -> None:
    orchestrator = _new_orchestrator()

    summary = orchestrator._summarize_warmup(
        [
            HealthProbeResult("audio", HealthStatus.DEGRADED, "Audio partially available"),
            HealthProbeResult(
                "realtime",
                HealthStatus.DEGRADED,
                "Realtime session connected (not ready)",
                details={"connected": True, "ready": False},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ]
    )

    assert "realtime (offline)" not in summary
    assert "Realtime session connected" in summary

def test_tick_exits_warmup_to_ok_when_startup_criteria_met(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._health_debounce_s = 0.0
    captured = []

    monkeypatch.setattr(
        orchestrator,
        "_run_health_probes",
        lambda: [
            HealthProbeResult("audio", HealthStatus.OK, "Audio input/output ready"),
            HealthProbeResult(
                "realtime",
                HealthStatus.OK,
                "Realtime session connected",
                details={"connected": 1, "ready": 1},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ],
    )
    monkeypatch.setattr(orchestrator, "_maybe_run_micro_presence", lambda now: None)
    monkeypatch.setattr(orchestrator, "_maybe_run_memory_maintenance", lambda now: None)
    monkeypatch.setattr(orchestrator, "_emit_health_snapshot", lambda snapshot: captured.append(snapshot))

    orchestrator._tick()

    assert captured[-1].status == HealthStatus.OK
    assert captured[-1].summary == "All systems nominal"
    assert orchestrator._warmup_active is False


def test_tick_does_not_exit_warmup_until_probe_results_settle(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._health_debounce_s = 5.0
    captured = []

    monkeypatch.setattr(
        orchestrator,
        "_run_health_probes",
        lambda: [
            HealthProbeResult("audio", HealthStatus.OK, "Audio input/output ready"),
            HealthProbeResult(
                "realtime",
                HealthStatus.OK,
                "Realtime session connected",
                details={"connected": 1, "ready": 1},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ],
    )
    monkeypatch.setattr(orchestrator, "_maybe_run_micro_presence", lambda now: None)
    monkeypatch.setattr(orchestrator, "_maybe_run_memory_maintenance", lambda now: None)
    monkeypatch.setattr(orchestrator, "_emit_health_snapshot", lambda snapshot: captured.append(snapshot))

    orchestrator._health_states = {
        "audio": types.SimpleNamespace(status=HealthStatus.DEGRADED, since=0.0, last_update=0.0),
        "realtime": types.SimpleNamespace(status=HealthStatus.DEGRADED, since=0.0, last_update=0.0),
        "battery": types.SimpleNamespace(status=HealthStatus.OK, since=0.0, last_update=0.0),
    }

    orchestrator._tick()

    assert captured[-1].status == HealthStatus.WARMUP
    assert orchestrator._warmup_active is True


def test_pending_summary_is_never_treated_as_degraded_status() -> None:
    orchestrator = _new_orchestrator()

    summary = orchestrator._summarize_health(HealthStatus.DEGRADED, [])

    assert summary == "System health pending"


def test_tick_reports_degraded_after_warmup_timeout_when_issue_persists(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._health_debounce_s = 0.0
    orchestrator._warmup_grace_period_s = 1.0
    orchestrator._warmup_started_at = time.monotonic() - 2.0
    captured = []

    monkeypatch.setattr(
        orchestrator,
        "_run_health_probes",
        lambda: [
            HealthProbeResult("audio", HealthStatus.DEGRADED, "Audio partially available"),
            HealthProbeResult(
                "realtime",
                HealthStatus.DEGRADED,
                "Realtime session offline",
                details={"connected": 0, "ready": 0},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ],
    )
    monkeypatch.setattr(orchestrator, "_maybe_run_micro_presence", lambda now: None)
    monkeypatch.setattr(orchestrator, "_maybe_run_memory_maintenance", lambda now: None)
    monkeypatch.setattr(orchestrator, "_emit_health_snapshot", lambda snapshot: captured.append(snapshot))

    orchestrator._tick()

    assert captured[-1].status == HealthStatus.DEGRADED
    assert captured[-1].summary.startswith("Degraded:")


def test_tick_does_not_keep_warmup_status_after_timeout_transition(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._health_debounce_s = 60.0
    captured = []

    monkeypatch.setattr(
        orchestrator,
        "_run_health_probes",
        lambda: [
            HealthProbeResult("audio", HealthStatus.DEGRADED, "Audio partially available"),
            HealthProbeResult(
                "realtime",
                HealthStatus.DEGRADED,
                "Realtime session offline",
                details={"connected": 0, "ready": 0},
            ),
            HealthProbeResult("battery", HealthStatus.OK, "Battery nominal"),
        ],
    )
    monkeypatch.setattr(orchestrator, "_maybe_run_micro_presence", lambda now: None)
    monkeypatch.setattr(orchestrator, "_maybe_run_memory_maintenance", lambda now: None)
    monkeypatch.setattr(orchestrator, "_emit_health_snapshot", lambda snapshot: captured.append(snapshot))

    orchestrator._warmup_grace_period_s = 60.0
    orchestrator._tick()

    orchestrator._warmup_grace_period_s = 1.0
    orchestrator._warmup_started_at = time.monotonic() - 2.0
    orchestrator._tick()

    assert captured[0].status == HealthStatus.WARMUP
    assert captured[-1].status != HealthStatus.WARMUP
    assert captured[-1].status == HealthStatus.DEGRADED
    assert captured[-1].details.get("warmup_exit_reason") == "timeout"


def test_emit_health_alert_ignores_pending_summary_and_alerts_real_degraded(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    emitted = []

    monkeypatch.setattr(orchestrator, "_emit_alert", lambda alert: emitted.append(alert))

    orchestrator._emit_health_alert(
        types.SimpleNamespace(status=HealthStatus.DEGRADED, summary="System health pending")
    )
    orchestrator._emit_health_alert(
        types.SimpleNamespace(status=HealthStatus.DEGRADED, summary="Degraded: realtime (offline)")
    )

    assert len(emitted) == 1
    assert emitted[0].severity == "warning"
    assert emitted[0].key == "health_degraded"
