"""Regression tests for ai.tools service-module delegation."""

from __future__ import annotations

import asyncio

import ai.tools as ai_tools


def test_read_battery_voltage_delegates_and_keeps_payload_shape(monkeypatch) -> None:
    called = {"count": 0}

    def fake_read_battery_voltage() -> dict[str, float | str]:
        called["count"] += 1
        return {
            "voltage": 8.1,
            "unit": "V",
            "min_voltage": 7.0,
            "max_voltage": 8.4,
        }

    monkeypatch.setattr(ai_tools.tool_runtime, "read_battery_voltage", fake_read_battery_voltage)

    payload = asyncio.run(ai_tools.read_battery_voltage())

    assert called["count"] == 1
    assert payload == {
        "voltage": 8.1,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
    }


def test_gesture_nod_delegates_and_keeps_payload_shape(monkeypatch) -> None:
    captured: dict[str, float | int] = {}

    def fake_enqueue_nod_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, object]:
        captured["delay_ms"] = delay_ms
        captured["intensity"] = intensity
        return {
            "queued": True,
            "gesture": "nod",
            "delay_ms": delay_ms,
            "intensity": intensity,
        }

    monkeypatch.setattr(ai_tools.tool_runtime, "enqueue_nod_gesture", fake_enqueue_nod_gesture)

    payload = asyncio.run(ai_tools.enqueue_nod_gesture(delay_ms=120, intensity=1.25))

    assert captured == {"delay_ms": 120, "intensity": 1.25}
    assert payload == {
        "queued": True,
        "gesture": "nod",
        "delay_ms": 120,
        "intensity": 1.25,
    }


def test_set_output_volume_delegates_and_keeps_payload_shape(monkeypatch) -> None:
    captured: dict[str, int | bool] = {}

    def fake_set_output_volume(percent: int, emergency: bool = False) -> dict[str, int | bool]:
        captured["percent"] = percent
        captured["emergency"] = emergency
        return {
            "percent": percent,
            "muted": False,
        }

    monkeypatch.setattr(ai_tools.tool_runtime, "set_output_volume", fake_set_output_volume)

    payload = asyncio.run(ai_tools.set_output_volume(percent=42, emergency=True))

    assert captured == {"percent": 42, "emergency": True}
    assert payload == {"percent": 42, "muted": False}
