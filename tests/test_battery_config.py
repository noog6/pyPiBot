"""Tests for battery config normalization and monitor behavior."""

from __future__ import annotations

from pathlib import Path

from ai.event_bus import EventBus
from config.controller import ConfigController
from services.battery_monitor import BatteryMonitor, BatteryStatusEvent


class _FakeSensor:
    def read_battery_voltage(self) -> float:
        return 7.5


def _reset_singletons() -> None:
    ConfigController._instance = None
    BatteryMonitor._instance = None


def test_config_controller_maps_legacy_battery_keys(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "battery_voltage_min: 6.8",
                "battery_voltage_max: 8.2",
                "battery_warning_percent: 45",
                "battery_critical_percent: 20",
                "battery_hysteresis_percent: 5",
                "battery_response_enabled: false",
                "battery_response_cooldown_s: 15",
                "battery_response_allow_warning: false",
                "battery_response_allow_critical: true",
                "battery_response_require_transition: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    cfg = ConfigController.get_instance().get_config()["battery"]
    assert cfg["voltage_min"] == 6.8
    assert cfg["voltage_max"] == 8.2
    assert cfg["warning_percent"] == 45.0
    assert cfg["critical_percent"] == 20.0
    assert cfg["hysteresis_percent"] == 5.0
    assert cfg["response"]["enabled"] is False
    assert cfg["response"]["cooldown_s"] == 15.0
    assert cfg["response"]["allow_warning"] is False
    assert cfg["response"]["allow_critical"] is True
    assert cfg["response"]["require_transition"] is True


def test_battery_monitor_hysteresis_prevents_jitter_flips(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "battery:",
                "  voltage_min: 6.0",
                "  voltage_max: 9.0",
                "  warning_percent: 40",
                "  critical_percent: 15",
                "  hysteresis_percent: 10",
                "  response:",
                "    enabled: true",
                "    cooldown_s: 0",
                "    allow_warning: true",
                "    allow_critical: false",
                "    require_transition: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    previous_warning = BatteryStatusEvent(
        timestamp=0.0,
        voltage=7.2,
        percent_of_range=0.39,
        severity="warning",
    )
    near_boundary = monitor._build_event(7.35, previous_warning)
    assert near_boundary.severity == "warning"
    assert near_boundary.transition == "steady_warning"


def test_unchanged_status_does_not_trigger_response_and_metadata_present(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "battery:",
                "  voltage_min: 7.0",
                "  voltage_max: 8.4",
                "  warning_percent: 50",
                "  critical_percent: 25",
                "  hysteresis_percent: 5",
                "  response:",
                "    enabled: true",
                "    cooldown_s: 0",
                "    allow_warning: true",
                "    allow_critical: true",
                "    require_transition: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    event_bus = EventBus()
    handler = monitor.create_event_bus_handler(event_bus)

    initial_warning = BatteryStatusEvent(
        timestamp=0.0,
        voltage=7.6,
        percent_of_range=0.43,
        severity="warning",
        transition="enter_warning",
        delta_percent=-8.0,
        rapid_drop=True,
    )
    steady_warning = BatteryStatusEvent(
        timestamp=1.0,
        voltage=7.58,
        percent_of_range=0.42,
        severity="warning",
        transition="steady_warning",
        delta_percent=-1.0,
        rapid_drop=False,
    )

    handler(initial_warning)
    first = event_bus.get_next(timeout=0.1)
    assert first is not None
    assert first.request_response is True
    assert first.metadata["transition"] == "enter_warning"
    assert first.metadata["rapid_drop"] is True

    handler(steady_warning)
    second = event_bus.get_next(timeout=0.1)
    assert second is not None
    assert second.request_response is False
    assert second.metadata["transition"] == "steady_warning"
    assert second.metadata["delta_percent"] == -1.0
