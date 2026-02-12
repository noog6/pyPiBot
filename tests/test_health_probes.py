from __future__ import annotations

from core.ops_models import HealthStatus
from services.health_probes import probe_imx500


class _FakeImxController:
    def __init__(self, status: dict[str, int | float | str]) -> None:
        self._status = status

    def get_runtime_status(self) -> dict[str, int | float | str]:
        return dict(self._status)


def test_probe_imx500_ok(monkeypatch) -> None:
    controller = _FakeImxController(
        {
            "enabled": 1,
            "backend_available": 1,
            "loop_alive": 1,
            "events_published": 12,
            "detection_events_published": 3,
            "last_event_age_s": 0.2,
            "last_detection_age_s": 1.4,
            "last_classes_confidences": "person:0.92",
        }
    )
    monkeypatch.setattr(
        "services.health_probes.Imx500Controller.get_instance",
        lambda: controller,
    )

    result = probe_imx500()

    assert result.status is HealthStatus.OK
    assert result.details["detection_events_published"] == 3


def test_probe_imx500_failing_when_backend_unavailable(monkeypatch) -> None:
    controller = _FakeImxController(
        {
            "enabled": 1,
            "backend_available": 0,
            "loop_alive": 0,
            "events_published": 0,
            "detection_events_published": 0,
            "last_event_age_s": -1.0,
            "last_detection_age_s": -1.0,
            "last_classes_confidences": "",
        }
    )
    monkeypatch.setattr(
        "services.health_probes.Imx500Controller.get_instance",
        lambda: controller,
    )

    result = probe_imx500()

    assert result.status is HealthStatus.FAILING
    assert "backend unavailable" in result.summary.lower()
