"""Focused tests for monitor event bus handler contracts."""

from __future__ import annotations

from dataclasses import dataclass

from services.battery_monitor import BatteryMonitor, BatteryStatusEvent
from services.imu_monitor import ImuMonitor, ImuMotionEvent


@dataclass
class _PublishCall:
    event: object
    coalesce: bool


class _RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list[_PublishCall] = []

    def publish(self, event: object, *, coalesce: bool = False) -> None:
        self.calls.append(_PublishCall(event=event, coalesce=coalesce))


class _FakeBatterySensor:
    def read_battery_voltage(self) -> float:
        return 7.6


class _FakeImuSensor:
    def read_accelerometer_gyro_data(self):
        return (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)

    def read_magnetometer_data(self):
        return (0.0, 0.0, 0.0)


def test_imu_create_event_bus_handler_emits_unchanged_contract(monkeypatch) -> None:
    ImuMonitor._instance = None
    monkeypatch.setattr("services.imu_monitor.ICM20948Sensor.get_instance", lambda: _FakeImuSensor())

    monitor = ImuMonitor.get_instance()
    publisher = _RecordingPublisher()
    handler = monitor.create_event_bus_handler(publisher)

    motion = ImuMotionEvent(
        timestamp=123.0,
        event_type="sudden_rotation",
        severity="warning",
        details={"yaw_rate_dps": 300.0},
    )
    handler(motion)

    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call.coalesce is True
    event = call.event
    assert event.source == "imu"
    assert event.kind == "warning"
    assert event.priority == "high"
    assert event.dedupe_key == "imu_motion"
    assert event.metadata == {
        "event_type": "sudden_rotation",
        "severity": "warning",
        "details": {"yaw_rate_dps": 300.0},
    }


def test_battery_create_event_bus_handler_emits_unchanged_contract(tmp_path, monkeypatch) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "default.yaml").write_text(
        "\n".join(
            [
                "battery:",
                "  voltage_min: 7.0",
                "  voltage_max: 8.4",
                "  warning_percent: 50",
                "  critical_percent: 25",
                "  hysteresis_percent: 0",
                "  response:",
                "    enabled: true",
                "    cooldown_s: 60",
                "    allow_warning: true",
                "    allow_critical: true",
                "    require_transition: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    BatteryMonitor._instance = None
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeBatterySensor())

    monitor = BatteryMonitor.get_instance()
    publisher = _RecordingPublisher()
    handler = monitor.create_event_bus_handler(publisher)

    status = BatteryStatusEvent(
        timestamp=12.0,
        voltage=7.5,
        percent_of_range=0.36,
        severity="warning",
        event_type="status",
        transition="enter_warning",
        delta_percent=-9.0,
        rapid_drop=True,
    )
    handler(status)

    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call.coalesce is True
    event = call.event
    assert event.source == "battery"
    assert event.kind == "status"
    assert event.priority == "high"
    assert event.dedupe_key == "battery_status"
    assert event.metadata == {
        "voltage": 7.5,
        "percent_of_range": 0.36,
        "severity": "warning",
        "event_type": "status",
        "transition": "enter_warning",
        "delta_percent": -9.0,
        "rapid_drop": True,
    }
