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
