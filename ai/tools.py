"""Tool definitions for realtime API calls."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from hardware import ADS1015Sensor
from motion import (
    MotionController,
    gesture_attention_snap,
    gesture_curious_tilt,
    gesture_idle,
    gesture_look_around,
    gesture_no,
    gesture_nod,
)


ToolFn = Callable[..., Awaitable[Any]]

function_map: dict[str, ToolFn] = {}

tools: list[dict[str, Any]] = []


async def read_battery_voltage() -> dict[str, Any]:
    """Return the current LiPo battery voltage via the ADS1015 sensor."""

    sensor = ADS1015Sensor.get_instance()
    voltage = sensor.read_battery_voltage()
    return {
        "voltage": voltage,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
    }


def _enqueue_gesture(action: Any) -> None:
    controller = MotionController.get_instance()
    controller.add_action_to_queue(action)


async def enqueue_idle_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue an idle gesture action on the motion controller."""

    action = gesture_idle(delay_ms=delay_ms, intensity=float(intensity))
    _enqueue_gesture(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


async def enqueue_nod_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a nod gesture action on the motion controller."""

    action = gesture_nod(delay_ms=delay_ms, intensity=float(intensity))
    _enqueue_gesture(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


async def enqueue_no_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a head shake gesture action on the motion controller."""

    action = gesture_no(delay_ms=delay_ms, intensity=float(intensity))
    _enqueue_gesture(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


async def enqueue_look_around_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a casual look around gesture action on the motion controller."""

    action = gesture_look_around(delay_ms=delay_ms, intensity=float(intensity))
    _enqueue_gesture(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


async def enqueue_curious_tilt_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a curious tilt gesture action on the motion controller."""

    action = gesture_curious_tilt(delay_ms=delay_ms, intensity=float(intensity))
    _enqueue_gesture(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


async def enqueue_attention_snap_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a quick attention snap gesture action on the motion controller."""

    action = gesture_attention_snap(delay_ms=delay_ms, intensity=float(intensity))
    _enqueue_gesture(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


tools.append(
    {
        "type": "function",
        "name": "read_battery_voltage",
        "description": (
            "Fetch the current voltage of the onboard 2S LiPo battery. "
            "Safe operating range is 7.0V to 8.4V. If the reading is within "
            "0.5V of the minimum voltage, complain about it; being near the max is fine."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_battery_voltage"] = read_battery_voltage

tools.append(
    {
        "type": "function",
        "name": "gesture_idle",
        "description": (
            "Queue a gentle idle gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_idle"] = enqueue_idle_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_nod",
        "description": (
            "Queue a nod gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_nod"] = enqueue_nod_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_no",
        "description": (
            "Queue a head shake (no) gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_no"] = enqueue_no_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_around",
        "description": (
            "Queue a casual look around gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_around"] = enqueue_look_around_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_curious_tilt",
        "description": (
            "Queue a curious tilt gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_curious_tilt"] = enqueue_curious_tilt_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_attention_snap",
        "description": (
            "Queue a quick attention snap gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_attention_snap"] = enqueue_attention_snap_gesture
