"""Tests for ops orchestrator shutdown timeout behavior."""

from __future__ import annotations

import threading
import time

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


def test_stop_loop_handles_interrupt_while_collecting_timeout_metadata() -> None:
    orchestrator = _new_orchestrator()

    class FakeThread:
        name = "fake-ops"
        ident = 7

        def join(self, timeout=None) -> None:
            return None

        def is_alive(self) -> bool:
            return True

    class InterruptingLock:
        def acquire(self, timeout=None) -> bool:
            raise KeyboardInterrupt

        def release(self) -> None:
            return None

    orchestrator._loop_thread = FakeThread()  # type: ignore[assignment]
    orchestrator._lock = InterruptingLock()  # type: ignore[assignment]

    status = orchestrator.stop_loop(timeout_s=0.01, grace_period_s=0.01)

    assert status == "timed_out"
    assert orchestrator.forced_shutdown_continuation() is True


def test_stop_loop_lock_contention_does_not_block_timeout_path(monkeypatch) -> None:
    orchestrator = _new_orchestrator()

    class FakeThread:
        name = "fake-ops"
        ident = 9

        def join(self, timeout=None) -> None:
            return None

        def is_alive(self) -> bool:
            return True

    class ContendedLock:
        def acquire(self, timeout=None) -> bool:
            if timeout:
                time.sleep(timeout)
            return False

        def release(self) -> None:
            return None

    warning_messages: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warning_messages.append(message % args if args else message)

    orchestrator._loop_thread = FakeThread()  # type: ignore[assignment]
    orchestrator._lock = ContendedLock()  # type: ignore[assignment]
    monkeypatch.setattr("services.ops_orchestrator.LOGGER.warning", capture_warning)

    timeout_s = 0.01
    grace_period_s = 0.02
    started = time.monotonic()
    status = orchestrator.stop_loop(timeout_s=timeout_s, grace_period_s=grace_period_s)
    elapsed = time.monotonic() - started

    assert status == "timed_out"
    assert elapsed < 0.08
    assert orchestrator.forced_shutdown_continuation() is True
    assert warning_messages
    assert any("metadata unavailable due to lock contention" in msg for msg in warning_messages)


def test_stop_loop_normal_shutdown_emits_info_without_warning(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    warning_messages: list[str] = []
    info_messages: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warning_messages.append(message % args if args else message)

    def capture_info(message: str, *args) -> None:
        info_messages.append(message % args if args else message)

    monkeypatch.setattr("services.ops_orchestrator.LOGGER.warning", capture_warning)
    monkeypatch.setattr("services.ops_orchestrator.LOGGER.info", capture_info)

    orchestrator._tick = lambda: None  # type: ignore[method-assign]
    orchestrator.start_loop(loop_period_s=0.05)
    time.sleep(0.06)
    status = orchestrator.stop_loop(timeout_s=0.2, grace_period_s=0.05)

    assert status == "stopped"
    assert orchestrator.forced_shutdown_continuation() is False
    assert warning_messages == []
    assert any("Loop thread stopped cleanly" in message for message in info_messages)


def test_stop_loop_retries_join_after_explicit_wake(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    wake_calls = 0

    class FakeThread:
        name = "fake-ops"
        ident = 11

        def __init__(self) -> None:
            self.join_calls = 0

        def join(self, timeout=None) -> None:
            self.join_calls += 1

        def is_alive(self) -> bool:
            return self.join_calls < 2

    fake_thread = FakeThread()
    orchestrator._loop_thread = fake_thread  # type: ignore[assignment]

    def capture_wake() -> None:
        nonlocal wake_calls
        wake_calls += 1

    monkeypatch.setattr(orchestrator, "_wake_loop", capture_wake)

    status = orchestrator.stop_loop(timeout_s=0.01, grace_period_s=0.02)

    assert status == "stopped"
    assert wake_calls == 2
    assert orchestrator.forced_shutdown_continuation() is False


def test_loop_emits_final_heartbeat_after_stop_request() -> None:
    orchestrator = _new_orchestrator()
    orchestrator._tick = lambda: None  # type: ignore[method-assign]
    orchestrator.start_loop(loop_period_s=0.05, heartbeat_period_s=30.0)
    time.sleep(0.06)

    status = orchestrator.stop_loop(timeout_s=0.2, grace_period_s=0.05)

    assert status == "stopped"
    counters = orchestrator.get_counters()
    assert counters.heartbeats >= 1
