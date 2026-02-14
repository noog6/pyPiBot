import time

import pytest
import importlib.util
import sys
from pathlib import Path

_module_path = Path(__file__).resolve().parents[1] / "hardware" / "imx500_controller.py"
_spec = importlib.util.spec_from_file_location("imx500_controller_under_test", _module_path)
assert _spec is not None and _spec.loader is not None
_imx500 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _imx500
_spec.loader.exec_module(_imx500)
Imx500Controller = _imx500.Imx500Controller
Imx500Settings = _imx500.Imx500Settings


def _reset_singleton() -> None:
    Imx500Controller._instance = None


def test_worker_lifecycle_and_publishing(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(enabled=True, fps_cap=20, min_confidence=0.4),
    )
    monkeypatch.setattr(Imx500Controller, "_check_backend_available", lambda self: (True, ""))

    shutdown_calls = []

    def fake_create(self, model: str):
        return object(), {"model": model}, True

    def fake_read(self, camera, model_stack):
        return ([{"label": "person", "confidence": 0.95, "bbox": (0.1, 0.1, 0.2, 0.2)}], time.time())

    def fake_shutdown(self, camera, model_stack, owns_camera_stack):
        shutdown_calls.append((camera, model_stack, owns_camera_stack))

    monkeypatch.setattr(Imx500Controller, "_create_imx500_stack", fake_create)
    monkeypatch.setattr(Imx500Controller, "_read_raw_detections", fake_read)
    monkeypatch.setattr(Imx500Controller, "_shutdown_imx500_stack", fake_shutdown)

    controller = Imx500Controller.get_instance()
    received = []
    controller.subscribe(lambda dets, ts: received.append((len(dets), ts)))

    controller.start()
    time.sleep(0.16)
    controller.stop()

    latest = controller.get_latest_event()
    assert latest is not None
    assert latest.frame_id is not None and latest.frame_id >= 1
    assert latest.detections and latest.detections[0].label == "person"
    assert received
    assert shutdown_calls


def test_convert_raw_detections_filters_low_confidence(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(enabled=True, min_confidence=0.7),
    )

    controller = Imx500Controller.get_instance()
    detections, frame_id = controller._convert_raw_detections(
        [
            {"label": "person", "confidence": 0.8, "bbox": (0.0, 0.0, 0.5, 0.5)},
            {"label": "cat", "confidence": 0.3, "bbox": (0.0, 0.0, 0.5, 0.5)},
        ]
    )

    assert frame_id >= 1
    assert len(detections) == 1
    assert detections[0].label == "person"


def test_start_worker_does_not_clear_stop_when_previous_worker_alive(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=True))

    controller = Imx500Controller.get_instance()

    class AliveThread:
        def __init__(self) -> None:
            self.started = False

        def is_alive(self) -> bool:
            return True

        def start(self) -> None:
            self.started = True

    prior_worker = AliveThread()
    controller._worker_thread = prior_worker
    controller._worker_stop.set()

    controller._start_worker_locked()

    assert controller._worker_stop.is_set()
    assert controller._worker_thread is prior_worker
    assert not prior_worker.started


def test_convert_raw_detections_skips_invalid_bbox_payloads(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(enabled=True, min_confidence=0.1),
    )

    controller = Imx500Controller.get_instance()
    detections, frame_id = controller._convert_raw_detections(
        [
            {"label": "bad", "confidence": 0.9, "bbox": (None, 0.1, 0.2, 0.2)},
            {"label": "good", "confidence": 0.8, "bbox": (0.2, 0.2, 0.2, 0.2)},
        ]
    )

    assert frame_id >= 1
    assert len(detections) == 1
    assert detections[0].label == "good"


def test_runtime_status_tracks_recent_detection_metrics(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(enabled=True, min_confidence=0.1),
    )

    controller = Imx500Controller.get_instance()
    controller._publish_detections(
        [
            _imx500.Detection(
                label="person",
                confidence=0.91,
                bbox=(0.1, 0.1, 0.2, 0.2),
                metadata={},
            )
        ],
        timestamp=time.time(),
        frame_id=1,
    )

    status = controller.get_runtime_status()
    assert status["events_published"] == 1
    assert status["detection_events_published"] == 1
    assert status["last_event_age_s"] >= 0.0
    assert status["last_detection_age_s"] >= 0.0
    assert "person:0.91" in status["last_classes_confidences"]


def test_warns_when_no_detection_events_after_grace(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(enabled=True, startup_grace_s=5.0, status_log_period_s=10.0),
    )

    controller = Imx500Controller.get_instance()
    controller._available = True
    controller._start_monotonic = 10.0

    controller._maybe_log_status(now_monotonic=16.0)

    assert controller._no_detections_warning_logged is True


def test_worker_retries_after_camera_attach_failure(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(enabled=True, fps_cap=30, startup_retry_interval_s=0.01),
    )
    monkeypatch.setattr(Imx500Controller, "_check_backend_available", lambda self: (True, ""))

    attempts = {"count": 0}

    def fake_create(self, model: str):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Device or resource busy")
        return object(), None, True

    def fake_read(controller, camera, model_stack):
        controller._worker_stop.set()
        return ([], time.time())

    monkeypatch.setattr(Imx500Controller, "_create_imx500_stack", fake_create)
    monkeypatch.setattr(Imx500Controller, "_read_raw_detections", fake_read)

    controller = Imx500Controller.get_instance()
    controller.start()
    time.sleep(0.65)
    controller.stop()

    status = controller.get_runtime_status()
    assert attempts["count"] >= 2
    assert status["startup_attempts"] >= 2
    assert status["last_error"] == ""


def test_worker_stops_retrying_after_max_attach_failures(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(
        Imx500Controller,
        "_load_settings",
        lambda self: Imx500Settings(
            enabled=True,
            startup_retry_interval_s=0.01,
            startup_max_attach_retries=2,
        ),
    )
    monkeypatch.setattr(Imx500Controller, "_check_backend_available", lambda self: (True, ""))

    def always_fail(self, model: str):
        raise RuntimeError("Camera __init__ sequence did not complete.")

    monkeypatch.setattr(Imx500Controller, "_create_imx500_stack", always_fail)

    controller = Imx500Controller.get_instance()
    controller.start()
    time.sleep(1.2)

    status = controller.get_runtime_status()

    assert status["startup_attempts"] == 2
    assert status["loop_alive"] == 0
    assert status["backend_available"] == 0
    assert "did not complete" in status["last_error"]

    controller.stop()


def test_resolve_model_path_maps_known_nickname(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()

    resolved = controller._resolve_model_path("yolo11n_pp")

    assert resolved == _imx500.DEFAULT_IMX500_MODEL_PATH


def test_resolve_model_path_warns_on_unknown_nickname(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()

    warnings = []
    monkeypatch.setattr(_imx500.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    resolved = controller._resolve_model_path("custom_alias")

    assert resolved == "custom_alias"
    assert any("no known .rpk mapping" in warning for warning in warnings)


def test_create_imx500_stack_uses_positional_constructor(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()

    calls = {}

    class FakePicamera2:
        def create_preview_configuration(self):
            return {"preview": True}

        def configure(self, config):
            calls["config"] = config

        def start(self):
            calls["started"] = True

    class FakePicameraModule:
        Picamera2 = FakePicamera2

    class FakeModelStack:
        def __init__(self, model):
            calls["model_arg"] = model

        def create_preview_configuration(self):
            return {"from_model": True}

    class FakeImx500Module:
        IMX500 = FakeModelStack

    def fake_import_module(name: str):
        if name == "picamera2":
            return FakePicameraModule
        if name == "picamera2.devices.imx500":
            return FakeImx500Module
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(_imx500.importlib, "import_module", fake_import_module)

    _, model_stack, owns_camera_stack = controller._create_imx500_stack("/tmp/model.rpk")

    assert isinstance(model_stack, FakeModelStack)
    assert calls["model_arg"] == "/tmp/model.rpk"
    assert calls["config"] == {"from_model": True}
    assert calls["started"] is True
    assert owns_camera_stack is True


def test_create_imx500_stack_prefers_camera_controller_stack(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()

    shared_camera = object()
    shared_model_stack = object()
    monkeypatch.setattr(
        Imx500Controller,
        "_get_camera_controller_stack",
        lambda self: (shared_camera, shared_model_stack),
    )

    camera, model_stack, owns_camera_stack = controller._create_imx500_stack("/tmp/model.rpk")

    assert camera is shared_camera
    assert model_stack is shared_model_stack
    assert owns_camera_stack is False


def test_capture_metadata_uses_non_blocking_on_shared_camera(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()
    calls: list[bool] = []

    class FakeCamera:
        def capture_metadata(self, wait=True):
            calls.append(bool(wait))
            return {"detections": []}

    controller._using_shared_camera_stack = True
    metadata = controller._capture_metadata(FakeCamera())

    assert metadata == {"detections": []}
    assert calls == [False]


def test_capture_metadata_resolves_async_job_result(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()

    class FakeJob:
        def get_result(self, timeout=0):
            return {"detections": [1]}

    class FakeCamera:
        def capture_metadata(self, wait=True):
            assert wait is False
            return FakeJob()

    controller._using_shared_camera_stack = True
    metadata = controller._capture_metadata(FakeCamera())

    assert metadata == {"detections": [1]}


def test_read_raw_detections_skips_non_mapping_metadata_for_model_stack(monkeypatch) -> None:
    _reset_singleton()
    monkeypatch.setattr(Imx500Controller, "_load_settings", lambda self: Imx500Settings(enabled=False))

    controller = Imx500Controller.get_instance()
    warnings: list[str] = []
    monkeypatch.setattr(_imx500.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    class FakeModelStack:
        def get_outputs(self, metadata, add_batch=True):
            raise AssertionError("get_outputs should not be called for non-dict metadata")

    monkeypatch.setattr(controller, "_capture_metadata", lambda camera: object())
    detections, timestamp_s = controller._read_raw_detections(object(), FakeModelStack())

    assert detections == []
    assert isinstance(timestamp_s, float)
    assert any("expected dict" in item for item in warnings)
