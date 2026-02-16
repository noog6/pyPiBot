"""Tests for ops orchestrator shutdown timeout behavior."""

from __future__ import annotations

import threading

from core.ops_models import HealthStatus
from services.health_probes import HealthProbeResult
from services.ops_orchestrator import OpsOrchestrator


def _new_orchestrator() -> OpsOrchestrator:
    OpsOrchestrator._instance = None
    orchestrator = OpsOrchestrator()
    OpsOrchestrator._instance = None
    return orchestrator


def test_stop_loop_timeout_warns_and_reports_timed_out(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    tick_blocker = threading.Event()

    def blocking_tick() -> None:
        tick_blocker.wait()

    orchestrator._tick = blocking_tick  # type: ignore[method-assign]
    orchestrator.start_loop(loop_period_s=0.2)

    warning_messages: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warning_messages.append(message % args if args else message)

    monkeypatch.setattr("services.ops_orchestrator.LOGGER.warning", capture_warning)

    try:
        status = orchestrator.stop_loop(timeout_s=0.01, grace_period_s=0.01)

        assert status == "timed_out"
        assert orchestrator.forced_shutdown_continuation() is True
        assert warning_messages
        warning_text = warning_messages[-1]
        assert "join timed out" in warning_text
        assert "thread=" in warning_text
        assert "ident=" in warning_text
        assert "elapsed=" in warning_text
        assert "timeout=" in warning_text
        assert "stop_event_set=True" in warning_text
    finally:
        tick_blocker.set()
        assert orchestrator.stop_loop(timeout_s=0.2, grace_period_s=0.2) == "stopped"
        assert orchestrator.forced_shutdown_continuation() is False


def test_stop_loop_handles_join_failure_without_raising() -> None:
    orchestrator = _new_orchestrator()

    class FakeThread:
        name = "fake-ops"
        ident = 42

        def join(self, timeout=None) -> None:
            raise RuntimeError("join exploded")

        def is_alive(self) -> bool:
            return True

    orchestrator._loop_thread = FakeThread()  # type: ignore[assignment]

    status = orchestrator.stop_loop(timeout_s=0.01, grace_period_s=0.01)

    assert status == "timed_out"
    assert orchestrator.forced_shutdown_continuation() is True


def test_stop_loop_timeout_warning_includes_probe_blocker_metadata(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    probe_blocker = threading.Event()

    def blocking_probe_audio(realtime_api):
        probe_blocker.wait()
        return HealthProbeResult(
            name="audio",
            status=HealthStatus.OK,
            summary="Audio probe completed",
        )

    warning_messages: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warning_messages.append(message % args if args else message)

    monkeypatch.setattr("services.ops_orchestrator.probe_audio", blocking_probe_audio)
    monkeypatch.setattr(
        "services.ops_orchestrator.probe_battery",
        lambda: HealthProbeResult(name="battery", status=HealthStatus.OK, summary="Battery probe"),
    )
    monkeypatch.setattr(
        "services.ops_orchestrator.probe_motion",
        lambda: HealthProbeResult(name="motion", status=HealthStatus.OK, summary="Motion probe"),
    )
    monkeypatch.setattr(
        "services.ops_orchestrator.probe_realtime_session",
        lambda realtime_api: HealthProbeResult(
            name="realtime",
            status=HealthStatus.OK,
            summary="Realtime probe",
        ),
    )
    monkeypatch.setattr("services.ops_orchestrator.LOGGER.warning", capture_warning)

    orchestrator.start_loop(loop_period_s=0.2)

    try:
        status = orchestrator.stop_loop(timeout_s=0.01, grace_period_s=0.01)

        assert status == "timed_out"
        assert warning_messages
        warning_text = warning_messages[-1]
        assert "last_probe=audio" in warning_text
        assert "probe_elapsed_s=" in warning_text
        assert "tick_started_at=" in warning_text
        assert "phase=probe:audio" in warning_text
    finally:
        probe_blocker.set()
        assert orchestrator.stop_loop(timeout_s=0.2, grace_period_s=0.2) == "stopped"
