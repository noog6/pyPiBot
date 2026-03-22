"""Regression tests for ai.tools service-module delegation."""

from __future__ import annotations

import asyncio

import ai.tools as ai_tools


def test_read_battery_voltage_delegates_and_keeps_payload_shape(monkeypatch) -> None:
    called = {"count": 0}

    def fake_read_battery_voltage() -> dict[str, float | str | bool]:
        called["count"] += 1
        return {
            "voltage": 8.1,
            "unit": "V",
            "min_voltage": 7.0,
            "max_voltage": 8.4,
            "inferred_charger_connected": False,
            "inference_reason": "voltage_flat",
        }

    monkeypatch.setattr(ai_tools.tool_runtime, "read_battery_voltage", fake_read_battery_voltage)

    payload = asyncio.run(ai_tools.read_battery_voltage())

    assert called["count"] == 1
    assert payload == {
        "voltage": 8.1,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
        "inferred_charger_connected": False,
        "inference_reason": "voltage_flat",
    }


def test_read_battery_voltage_tool_description_guides_first_person_reason_aware_answers() -> None:
    battery_tool = next(tool for tool in ai_tools.tools if tool.get("name") == "read_battery_voltage")
    description = battery_tool["description"]

    assert "speak in first person" in description
    assert "Start the answer with the exact phrasing pattern" in description
    assert "space after 'at'" in description
    assert "never default to third-person phrasing" in description
    assert "I'm at {voltage:.2f} volts right now." in description
    assert "I don't have enough trend data yet to infer whether my charger is connected." in description
    assert "I may be seeing the start of charging, but I can't confirm it yet." in description
    assert (
        "My voltage looks supported in a way that could fit a weak charger, but I still can't "
        "confirm charging yet." in description
    )
    assert "I currently infer that my charger is connected." in description
    assert "I do not currently infer that my charger is connected." in description


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
