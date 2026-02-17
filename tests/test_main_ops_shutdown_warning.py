"""Focused tests for main shutdown logging behavior."""

from __future__ import annotations

from contextlib import nullcontext

import main


class _FakeConfigController:
    def get_config(self) -> dict[str, object]:
        return {"logging_level": "INFO", "file_logging_enabled": False}


class _FakeStorageController:
    def get_current_run_number(self) -> int:
        return 1


class _FakeMemoryManager:
    def set_active_session_id(self, _: str) -> None:
        return None


class _FakeRealtimeAPI:
    def __init__(self, prompts) -> None:
        self._prompts = prompts

    def get_event_bus(self):
        return object()

    async def run(self) -> None:
        return None


class _FakeMotionController:
    def start_control_loop(self) -> None:
        return None

    def stop_control_loop(self) -> None:
        return None


class _FakeCameraController:
    def set_realtime_instance(self, _instance) -> None:
        return None

    def start_vision_loop(self, vision_loop_period_ms: int) -> None:
        _ = vision_loop_period_ms

    def stop_vision_loop(self) -> None:
        return None


class _FakeMonitor:
    def start_loop(self) -> None:
        return None

    def stop_loop(self) -> None:
        return None

    def create_event_bus_handler(self, _event_bus):
        return object()

    def register_event_handler(self, _handler) -> None:
        return None

    def unregister_event_handler(self, _handler) -> None:
        return None


class _FakeOpsOrchestrator:
    def set_realtime_api(self, _realtime_api) -> None:
        return None

    def set_event_bus(self, _event_bus) -> None:
        return None

    def start_loop(self) -> None:
        return None

    def stop_loop(self) -> str:
        return "timed_out"

    def forced_shutdown_continuation(self) -> bool:
        return True


def test_main_logs_warning_when_ops_shutdown_not_stopped(monkeypatch) -> None:
    warnings: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _FakeRealtimeAPI)
    monkeypatch.setattr(main.MotionController, "get_instance", lambda: _FakeMotionController())
    monkeypatch.setattr(main.CameraController, "get_instance", lambda: _FakeCameraController())
    monkeypatch.setattr(main.ImuMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.BatteryMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.OpsOrchestrator, "get_instance", lambda: _FakeOpsOrchestrator())
    monkeypatch.setattr(main, "suppress_noisy_stderr", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(main.logger, "warning", capture_warning)

    exit_code = main.main([])

    assert exit_code == 0
    assert (
        "Ops orchestrator shutdown incomplete "
        "(status=timed_out forced_shutdown_continuation=True)"
    ) in warnings


class _FakeInterruptingOpsOrchestrator(_FakeOpsOrchestrator):
    def stop_loop(self) -> str:
        raise KeyboardInterrupt


def test_main_handles_keyboard_interrupt_while_stopping_ops(monkeypatch) -> None:
    warnings: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _FakeRealtimeAPI)
    monkeypatch.setattr(main.MotionController, "get_instance", lambda: _FakeMotionController())
    monkeypatch.setattr(main.CameraController, "get_instance", lambda: _FakeCameraController())
    monkeypatch.setattr(main.ImuMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.BatteryMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.OpsOrchestrator, "get_instance", lambda: _FakeInterruptingOpsOrchestrator())
    monkeypatch.setattr(main, "suppress_noisy_stderr", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(main.logger, "warning", capture_warning)

    exit_code = main.main([])

    assert exit_code == 0
    assert "Interrupted while stopping ops orchestrator; continuing shutdown." in warnings
    assert (
        "Ops orchestrator shutdown incomplete "
        "(status=interrupted forced_shutdown_continuation=True)"
    ) in warnings


def test_main_stops_ops_orchestrator_before_other_teardown(monkeypatch) -> None:
    stop_order: list[str] = []

    class OrderedMotion(_FakeMotionController):
        def stop_control_loop(self) -> None:
            stop_order.append("motion")

    class OrderedCamera(_FakeCameraController):
        def stop_vision_loop(self) -> None:
            stop_order.append("camera")

    class OrderedMonitor(_FakeMonitor):
        def __init__(self, name: str) -> None:
            self._name = name

        def stop_loop(self) -> None:
            stop_order.append(self._name)

    class OrderedOps(_FakeOpsOrchestrator):
        def stop_loop(self) -> str:
            stop_order.append("ops")
            return "stopped"

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _FakeRealtimeAPI)
    monkeypatch.setattr(main.MotionController, "get_instance", lambda: OrderedMotion())
    monkeypatch.setattr(main.CameraController, "get_instance", lambda: OrderedCamera())
    monkeypatch.setattr(main.ImuMonitor, "get_instance", lambda: OrderedMonitor("imu"))
    monkeypatch.setattr(main.BatteryMonitor, "get_instance", lambda: OrderedMonitor("battery"))
    monkeypatch.setattr(main.OpsOrchestrator, "get_instance", lambda: OrderedOps())
    monkeypatch.setattr(main, "suppress_noisy_stderr", lambda *args, **kwargs: nullcontext())

    exit_code = main.main([])

    assert exit_code == 0
    assert stop_order[0] == "ops"
    assert set(stop_order[1:]) == {"camera", "motion", "imu", "battery"}
