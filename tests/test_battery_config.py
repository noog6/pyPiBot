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



def test_two_consecutive_rising_voltage_samples_set_inferred_charger_connected(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    baseline = BatteryStatusEvent(timestamp=0.0, voltage=7.50, percent_of_range=0.36, severity="warning")

    first_rise = monitor._build_event(7.54, baseline)
    second_rise = monitor._build_event(7.58, first_rise)

    assert first_rise.inferred_charger_connected is False
    assert first_rise.inference_reason == "voltage_rising_pending"
    assert second_rise.inferred_charger_connected is True
    assert second_rise.inference_reason == "voltage_rising"


def test_first_sample_uses_no_prior_sample_reason(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()

    event = monitor._build_event(7.54, None)

    assert event.inferred_charger_connected is False
    assert event.inference_reason == "no_prior_sample"


def test_flat_sample_clears_pending_rise_before_charger_connects(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    baseline = BatteryStatusEvent(timestamp=0.0, voltage=7.50, percent_of_range=0.36, severity="warning")

    first_rise = monitor._build_event(7.54, baseline)
    flat_after_rise = monitor._build_event(7.54, first_rise)
    second_rise_after_flat = monitor._build_event(7.58, flat_after_rise)

    assert first_rise.inference_reason == "voltage_rising_pending"
    assert flat_after_rise.inferred_charger_connected is False
    assert flat_after_rise.inference_reason == "voltage_flat"
    assert second_rise_after_flat.inferred_charger_connected is False
    assert second_rise_after_flat.inference_reason == "voltage_rising_pending"


def test_flat_voltage_keeps_inferred_charger_disconnected(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    previous = BatteryStatusEvent(timestamp=0.0, voltage=7.50, percent_of_range=0.36, severity="warning")

    event = monitor._build_event(7.50, previous)

    assert event.inferred_charger_connected is False
    assert event.inference_reason == "voltage_flat"


def test_falling_voltage_keeps_inferred_charger_disconnected(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    previous = BatteryStatusEvent(timestamp=0.0, voltage=7.50, percent_of_range=0.36, severity="warning")

    event = monitor._build_event(7.46, previous)

    assert event.inferred_charger_connected is False
    assert event.inference_reason == "voltage_falling"


def test_tiny_voltage_uptick_below_threshold_does_not_set_inferred_charger_connected(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    previous = BatteryStatusEvent(timestamp=0.0, voltage=7.50, percent_of_range=0.36, severity="warning")

    event = monitor._build_event(7.52, previous)

    assert event.inferred_charger_connected is False
    assert event.inference_reason == "voltage_rise_below_threshold"


def test_battery_event_bus_metadata_includes_inferred_charger_state(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("battery: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("services.battery_monitor.ADS1015Sensor.get_instance", lambda: _FakeSensor())
    _reset_singletons()

    monitor = BatteryMonitor.get_instance()
    event_bus = EventBus()
    handler = monitor.create_event_bus_handler(event_bus)
    event = BatteryStatusEvent(
        timestamp=1.0,
        voltage=7.54,
        percent_of_range=0.39,
        severity="warning",
        transition="steady_warning",
        delta_percent=2.0,
        rapid_drop=False,
        inferred_charger_connected=True,
        inference_reason="voltage_rising",
    )

    handler(event)
    published = event_bus.get_next(timeout=0.1)

    assert published is not None
    assert published.metadata["inferred_charger_connected"] is True
    assert published.metadata["inference_reason"] == "voltage_rising"
    assert published.metadata["severity"] == "warning"
    assert published.metadata["delta_percent"] == 2.0
