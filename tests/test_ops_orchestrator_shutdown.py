"""Tests for ops orchestrator shutdown timeout behavior."""

from __future__ import annotations

import threading

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
