import time

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
        return object(), {"model": model}

    def fake_read(self, camera, model_stack):
        return ([{"label": "person", "confidence": 0.95, "bbox": (0.1, 0.1, 0.2, 0.2)}], time.time())

    def fake_shutdown(self, camera, model_stack):
        shutdown_calls.append((camera, model_stack))

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
        return object(), None

    def fake_read(self, camera, model_stack):
        self._worker_stop.set()
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
