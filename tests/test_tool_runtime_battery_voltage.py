"""Focused regressions for battery tool payload exposure."""

from __future__ import annotations

from types import SimpleNamespace

from services import tool_runtime


class _FakeSensor:
    def __init__(self, voltage: float, amperage: float) -> None:
        self._voltage = voltage
        self._amperage = amperage
        self.battery_voltage_read_count = 0
        self.system_amperage_read_count = 0

    def read_battery_voltage(self) -> float:
        self.battery_voltage_read_count += 1
        return self._voltage

    def read_system_amperage(self) -> float:
        self.system_amperage_read_count += 1
        return self._amperage


def test_read_battery_voltage_prefers_latest_monitor_snapshot(monkeypatch) -> None:
    sensor = _FakeSensor(8.05, 0.61)
    monitor = SimpleNamespace(
        get_latest_event=lambda: SimpleNamespace(
            voltage=8.03,
            amperage=0.57,
            power_watts=4.58,
            inferred_charger_connected=True,
            inference_reason="voltage_rising",
        )
    )
    monkeypatch.setattr(tool_runtime.ADS1015Sensor, "get_instance", classmethod(lambda cls: sensor))
    monkeypatch.setattr(tool_runtime.BatteryMonitor, "get_instance", classmethod(lambda cls: monitor))

    payload = tool_runtime.read_battery_voltage()

    assert payload == {
        "voltage": 8.03,
        "amperage": 0.57,
        "power_watts": 4.58,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
        "inferred_charger_connected": True,
        "inference_reason": "voltage_rising",
        "telemetry_source": "battery_monitor",
    }
    assert sensor.battery_voltage_read_count == 0
    assert sensor.system_amperage_read_count == 0


def test_read_battery_voltage_defaults_when_monitor_has_no_event(monkeypatch) -> None:
    sensor = _FakeSensor(7.92, 0.52)
    monitor = SimpleNamespace(get_latest_event=lambda: None)
    monkeypatch.setattr(tool_runtime.ADS1015Sensor, "get_instance", classmethod(lambda cls: sensor))
    monkeypatch.setattr(tool_runtime.BatteryMonitor, "get_instance", classmethod(lambda cls: monitor))

    payload = tool_runtime.read_battery_voltage()

    assert payload == {
        "voltage": 7.92,
        "amperage": 0.52,
        "power_watts": 4.12,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
        "inferred_charger_connected": False,
        "inference_reason": "no_prior_sample",
        "telemetry_source": "ads1015_direct_read",
    }
    assert sensor.battery_voltage_read_count == 1
    assert sensor.system_amperage_read_count == 1
