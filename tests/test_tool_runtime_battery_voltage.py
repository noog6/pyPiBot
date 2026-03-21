"""Focused regressions for battery tool payload exposure."""

from __future__ import annotations

from types import SimpleNamespace

from services import tool_runtime


class _FakeSensor:
    def __init__(self, voltage: float) -> None:
        self._voltage = voltage

    def read_battery_voltage(self) -> float:
        return self._voltage


def test_read_battery_voltage_includes_latest_monitor_inference(monkeypatch) -> None:
    sensor = _FakeSensor(8.05)
    monitor = SimpleNamespace(
        get_latest_event=lambda: SimpleNamespace(
            inferred_charger_connected=True,
            inference_reason="voltage_rising",
        )
    )
    monkeypatch.setattr(tool_runtime.ADS1015Sensor, "get_instance", classmethod(lambda cls: sensor))
    monkeypatch.setattr(tool_runtime.BatteryMonitor, "get_instance", classmethod(lambda cls: monitor))

    payload = tool_runtime.read_battery_voltage()

    assert payload == {
        "voltage": 8.05,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
        "inferred_charger_connected": True,
        "inference_reason": "voltage_rising",
    }


def test_read_battery_voltage_defaults_when_monitor_has_no_event(monkeypatch) -> None:
    sensor = _FakeSensor(7.92)
    monitor = SimpleNamespace(get_latest_event=lambda: None)
    monkeypatch.setattr(tool_runtime.ADS1015Sensor, "get_instance", classmethod(lambda cls: sensor))
    monkeypatch.setattr(tool_runtime.BatteryMonitor, "get_instance", classmethod(lambda cls: monitor))

    payload = tool_runtime.read_battery_voltage()

    assert payload == {
        "voltage": 7.92,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
        "inferred_charger_connected": False,
        "inference_reason": "no_prior_sample",
    }
