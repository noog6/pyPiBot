"""Focused tests for main shutdown logging behavior."""

from __future__ import annotations

from contextlib import nullcontext

import main
from ai.realtime_api import RealtimeAPIStartupError, StartupDependencyOutcome


class _FakeConfigController:
    def get_config(self) -> dict[str, object]:
        return {"logging_level": "INFO", "file_logging_enabled": False}


class _FakeStorageController:
    def get_current_run_number(self) -> int:
        return 1


class _FakeMemoryManager:
    def get_semantic_startup_summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "provider": "openai",
            "provider_model": "text-embedding-3-small",
            "provider_timeout_s": 10.0,
            "query_timeout_ms": 40,
            "write_timeout_ms": 75,
            "startup_canary_timeout_ms": 60,
            "effective_timeout_budget_ms": 40,
            "rerank_enabled": True,
            "background_embedding_enabled": True,
            "provider_ready": False,
            "provider_readiness_reason": "openai_provider_disabled",
            "canary_success": False,
            "canary_latency_ms": 12,
            "canary_dimension": 0,
            "canary_error_code": "auth",
            "max_queries_per_minute": 240,
            "max_writes_per_minute": 120,
        }

    def set_active_session_id(self, _: str) -> None:
        return None

    def get_embedding_worker(self):
        return None


class _FakeRealtimeAPI:
    def __init__(self, prompts) -> None:
        self._prompts = prompts
        self._failures = 0
        self._last_failure_reason = ""

    def get_event_bus(self):
        return object()

    async def run(self) -> None:
        return None

    def get_session_health(self) -> dict[str, object]:
        return {"failures": 0, "last_failure_reason": ""}

    def is_ready_for_injections(self) -> bool:
        return False

    def get_session_health(self) -> dict[str, object]:
        return {
            "failures": self._failures,
            "last_failure_reason": self._last_failure_reason,
        }


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
    def __init__(self, loop_alive: bool = True) -> None:
        self._loop_alive = loop_alive

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

    def is_loop_alive(self) -> bool:
        return self._loop_alive

    def has_startup_snapshot_emitted(self) -> bool:
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
        "(status=timed_out forced_shutdown_continuation=True loop_alive=True)"
    ) in warnings
    assert any("Ops orchestrator timed out. Follow-up:" in message for message in warnings)


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
        "(status=interrupted forced_shutdown_continuation=True loop_alive=True)"
    ) in warnings
    assert not any("Ops orchestrator timed out. Follow-up:" in message for message in warnings)


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


def test_main_logs_semantic_memory_state_at_startup(monkeypatch) -> None:
    infos: list[str] = []

    def capture_info(message: str, *args) -> None:
        infos.append(message % args if args else message)

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
    monkeypatch.setattr(
        main,
        "inspect_memory_embeddings",
        lambda _config: {"enabled": True, "table_exists": False},
    )
    monkeypatch.setattr(main.logger, "info", capture_info)

    exit_code = main.main([])

    assert exit_code == 0
    assert "Semantic memory enabled=True" in infos
    assert "Semantic embeddings table available=False" in infos
    assert (
        "Semantic startup summary enabled=True provider=openai model=text-embedding-3-small rerank_enabled=True "
        "background_embedding_enabled=True provider_ready=False readiness_reason=openai_provider_disabled "
        "provider_timeout_s=10.0 startup_canary_timeout_ms=60 query_timeout_ms=40 write_timeout_ms=75 "
        "effective_timeout_budget_ms=40 max_queries_per_minute=240 max_writes_per_minute=120"
    ) in infos
    assert "embedding_canary success=False latency_ms=12 dimension=0 error_code=auth" in infos


def test_main_realtime_api_init_failure_is_fatal_with_required_marker(monkeypatch) -> None:
    exceptions: list[str] = []

    class _FailingRealtimeAPI:
        def __init__(self, _prompts) -> None:
            raise RuntimeError("boom")

    def capture_exception(message: str, *args) -> None:
        exceptions.append(message % args if args else message)

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _FailingRealtimeAPI)
    monkeypatch.setattr(main.logger, "exception", capture_exception)

    exit_code = main.main([])

    assert exit_code == 1
    assert any(
        "startup component=realtime_api dependency_class=required status=fatal detail=boom"
        in message
        for message in exceptions
    )


def test_main_optional_startup_failure_logs_warning_with_marker(monkeypatch) -> None:
    warnings: list[str] = []

    class _FailingMotionController(_FakeMotionController):
        def start_control_loop(self) -> None:
            raise RuntimeError("motor unavailable")

    def capture_warning(message: str, *args) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _FakeRealtimeAPI)
    monkeypatch.setattr(main.MotionController, "get_instance", lambda: _FailingMotionController())
    monkeypatch.setattr(main.CameraController, "get_instance", lambda: _FakeCameraController())
    monkeypatch.setattr(main.ImuMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.BatteryMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.OpsOrchestrator, "get_instance", lambda: _FakeOpsOrchestrator())
    monkeypatch.setattr(main, "suppress_noisy_stderr", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(main.logger, "warning", capture_warning)

    exit_code = main.main([])

    assert exit_code == 0
    assert any(
        "startup component=motion_controller dependency_class=optional status=warning detail=motor unavailable"
        in message
        for message in warnings
    )


def test_main_audio_startup_failure_is_fatal_in_strict_mode(monkeypatch) -> None:
    exceptions: list[str] = []

    class _StrictAudioFailingRealtimeAPI:
        def __init__(self, _prompts) -> None:
            raise RealtimeAPIStartupError(
                "audio input initialization failed",
                outcome=StartupDependencyOutcome(
                    component="audio_input",
                    dependency_class="required",
                    status="fatal",
                    detail="PyAudio is required for AsyncMicrophone",
                ),
            )

    def capture_exception(message: str, *args) -> None:
        exceptions.append(message % args if args else message)

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _StrictAudioFailingRealtimeAPI)
    monkeypatch.setattr(main.logger, "exception", capture_exception)

    exit_code = main.main([])

    assert exit_code == 1
    assert any(
        "startup component=audio_input dependency_class=required status=fatal "
        "detail=PyAudio is required for AsyncMicrophone" in message
        for message in exceptions
    )



def test_main_logs_audio_input_startup_phase(monkeypatch) -> None:
    infos: list[str] = []

    def capture_info(message: str, *args) -> None:
        infos.append(message % args if args else message)

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
    monkeypatch.setattr(main.logger, "info", capture_info)

    exit_code = main.main([])

    assert exit_code == 0
    assert "startup component=audio_input dependency_class=required status=starting" in infos
    assert "startup component=audio_input dependency_class=required status=ready" in infos

def test_main_returns_nonzero_when_runtime_websocket_session_fails(monkeypatch) -> None:
    class _RuntimeFailureRealtimeAPI(_FakeRealtimeAPI):
        async def run(self) -> None:
            raise RuntimeError("session connect failed")

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _RuntimeFailureRealtimeAPI)
    monkeypatch.setattr(main.MotionController, "get_instance", lambda: _FakeMotionController())
    monkeypatch.setattr(main.CameraController, "get_instance", lambda: _FakeCameraController())
    monkeypatch.setattr(main.ImuMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.BatteryMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.OpsOrchestrator, "get_instance", lambda: _FakeOpsOrchestrator())
    monkeypatch.setattr(main, "suppress_noisy_stderr", lambda *args, **kwargs: nullcontext())

    exit_code = main.main([])

    assert exit_code == 1


def test_main_returns_nonzero_when_runtime_records_session_failure(monkeypatch) -> None:
    class _RuntimeRecordedFailureRealtimeAPI(_FakeRealtimeAPI):
        async def run(self) -> None:
            return None

        def get_session_health(self) -> dict[str, object]:
            return {"failures": 1, "last_failure_reason": "session closed unexpectedly"}

    monkeypatch.setattr(main.ConfigController, "get_instance", lambda: _FakeConfigController())
    monkeypatch.setattr(main.StorageController, "get_instance", lambda: _FakeStorageController())
    monkeypatch.setattr(main.MemoryManager, "get_instance", lambda: _FakeMemoryManager())
    monkeypatch.setattr(main, "RealtimeAPI", _RuntimeRecordedFailureRealtimeAPI)
    monkeypatch.setattr(main.MotionController, "get_instance", lambda: _FakeMotionController())
    monkeypatch.setattr(main.CameraController, "get_instance", lambda: _FakeCameraController())
    monkeypatch.setattr(main.ImuMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.BatteryMonitor, "get_instance", lambda: _FakeMonitor())
    monkeypatch.setattr(main.OpsOrchestrator, "get_instance", lambda: _FakeOpsOrchestrator())
    monkeypatch.setattr(main, "suppress_noisy_stderr", lambda *args, **kwargs: nullcontext())

    exit_code = main.main([])

    assert exit_code == 1
