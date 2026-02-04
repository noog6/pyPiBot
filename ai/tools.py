"""Tool definitions for realtime API calls."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from hardware import ADS1015Sensor, LPS22HBSensor
from motion import (
    MotionController,
    gesture_attention_snap,
    gesture_curious_tilt,
    gesture_idle,
    gesture_look_around,
    gesture_no,
    gesture_nod,
)
from motion.action import Action
from motion.motion_controller import millis
from services.imu_monitor import ImuMonitor
from services.memory_manager import MemoryManager
from services.output_volume import OutputVolumeController
from services.profile_manager import ProfileManager


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


async def read_environment() -> dict[str, Any]:
    """Return the current onboard air pressure and temperature."""

    sensor = LPS22HBSensor.get_instance()
    air_pressure, air_temperature = sensor.read_value()
    return {
        "air_pressure": air_pressure,
        "air_temperature": air_temperature,
        "pressure_unit": "hPa",
        "temperature_unit": "C",
    }


async def read_imu_data() -> dict[str, Any]:
    """Return the latest IMU orientation and sensor readings."""

    monitor = ImuMonitor.get_instance()
    sample = monitor.get_latest_sample()
    if sample is None:
        return {
            "status": "no_data",
            "message": "IMU has not produced a sample yet.",
        }

    events = monitor.get_recent_events(limit=5)
    event_items = [
        {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "severity": event.severity,
            "details": event.details,
        }
        for event in events
    ]
    orientation = {
        "roll_deg": sample.roll,
        "pitch_deg": sample.pitch,
        "yaw_deg": sample.yaw,
    }
    summary = (
        "IMU readings: "
        f"roll {sample.roll:.2f}°, pitch {sample.pitch:.2f}°, yaw {sample.yaw:.2f}°. "
        f"Accel {tuple(round(val, 3) for val in sample.accel)}. "
        f"Gyro {tuple(round(val, 3) for val in sample.gyro)}. "
        f"Mag {tuple(round(val, 3) for val in sample.mag)}."
    )
    return {
        "status": "ok",
        "summary": summary,
        "timestamp": sample.timestamp,
        "orientation": orientation,
        "accel": sample.accel,
        "gyro": sample.gyro,
        "mag": sample.mag,
        "recent_events": event_items,
        "units": {
            "orientation": "degrees",
            "accel": "raw",
            "gyro": "raw",
            "mag": "raw",
        },
    }


def _enqueue_gesture(action: Any) -> None:
    controller = MotionController.get_instance()
    controller.add_action_to_queue(action)


def _enqueue_servo_move(
    name: str, pan_degrees: float | None = None, tilt_degrees: float | None = None
) -> dict[str, Any]:
    controller = MotionController.get_instance()
    pan_servo = controller.servo_registry.servos["pan"]
    tilt_servo = controller.servo_registry.servos["tilt"]
    current_pan = float(pan_servo.read_value())
    current_tilt = float(tilt_servo.read_value())
    target_pan = current_pan if pan_degrees is None else float(pan_degrees)
    target_tilt = current_tilt if tilt_degrees is None else float(tilt_degrees)

    frame = controller.generate_base_keyframe(
        pan_degrees=int(target_pan), tilt_degrees=int(target_tilt)
    )
    frame.name = name
    action = Action(priority=2, timestamp=millis(), name=name, frames=frame)
    _enqueue_gesture(action)
    return {
        "queued": True,
        "action": name,
        "target": {"pan": target_pan, "tilt": target_tilt},
        "current": {"pan": current_pan, "tilt": current_tilt},
    }


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


async def set_pan(degrees: int) -> dict[str, Any]:
    """Set the pan servo to an absolute position."""

    return _enqueue_servo_move("set_pan", pan_degrees=degrees)


async def set_tilt(degrees: int) -> dict[str, Any]:
    """Set the tilt servo to an absolute position."""

    return _enqueue_servo_move("set_tilt", tilt_degrees=degrees)


async def get_servo_position(servo_name: str) -> dict[str, Any]:
    """Return the current position of the requested servo."""

    controller = MotionController.get_instance()
    servo = controller.servo_registry.servos[servo_name]
    return {
        "servo": servo_name,
        "degrees": float(servo.read_value()),
        "min_degrees": float(servo.min_angle),
        "max_degrees": float(servo.max_angle),
    }


async def update_user_profile(
    name: str | None = None,
    preferences: dict[str, Any] | None = None,
    favorites: list[str] | None = None,
) -> dict[str, Any]:
    """Update the active user profile with provided fields."""

    manager = ProfileManager.get_instance()
    profile = manager.update_active_profile_fields(
        name=name,
        preferences=preferences,
        favorites=favorites,
        last_seen=int(time.time() * 1000),
    )
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "preferences": profile.preferences,
        "favorites": profile.favorites,
        "last_seen": profile.last_seen,
    }


async def get_output_volume() -> dict[str, Any]:
    """Return the current output audio volume."""

    controller = OutputVolumeController.get_instance()
    status = controller.get_volume()
    return {
        "percent": status.percent,
        "muted": status.muted,
    }


async def set_output_volume(percent: int, emergency: bool = False) -> dict[str, Any]:
    """Set the output audio volume within safe bounds."""

    controller = OutputVolumeController.get_instance()
    status = controller.set_volume(percent=int(percent), emergency=bool(emergency))
    return {
        "percent": status.percent,
        "muted": status.muted,
    }


async def remember_memory(
    content: str,
    tags: list[str] | None = None,
    importance: int = 3,
) -> dict[str, Any]:
    """Store a memory entry for later recall."""

    manager = MemoryManager.get_instance()
    entry = manager.remember_memory(content=content, tags=tags, importance=importance)
    return {
        "memory_id": entry.memory_id,
        "content": entry.content,
        "tags": entry.tags,
        "importance": entry.importance,
    }


async def recall_memories(query: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Recall stored memories based on a query."""

    manager = MemoryManager.get_instance()
    memories = manager.recall_memories(query=query, limit=limit)
    return {"memories": [memory.__dict__ for memory in memories]}


async def forget_memory(memory_id: int) -> dict[str, Any]:
    """Delete a stored memory by id."""

    manager = MemoryManager.get_instance()
    removed = manager.forget_memory(memory_id=memory_id)
    return {"removed": removed, "memory_id": memory_id}


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
        "name": "read_environment",
        "description": (
            "Fetch Theo's internal air pressure and temperature from the LPS22HB sensor. "
            "This is Theo's onboard reading, not external weather data. "
            "Return values in hPa and Celsius."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_environment"] = read_environment

tools.append(
    {
        "type": "function",
        "name": "read_imu_data",
        "description": (
            "Fetch the latest IMU orientation (roll/pitch/yaw) and raw accel/gyro/mag readings. "
            "Return a human-readable summary string plus structured values."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_imu_data"] = read_imu_data

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

tools.append(
    {
        "type": "function",
        "name": "set_pan",
        "description": (
            "Set Theo's head pan servo to an absolute position between -90 and +90 degrees. "
            "Use this to look left/right."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {
                    "type": "integer",
                    "minimum": -90,
                    "maximum": 90,
                    "description": (
                        "Target pan position where 0 is neutral, -90 is full left, "
                        "and +90 is full right."
                    ),
                },
            },
            "required": ["degrees"],
        },
    }
)

function_map["set_pan"] = set_pan

tools.append(
    {
        "type": "function",
        "name": "set_tilt",
        "description": (
            "Set Theo's head tilt servo to an absolute position between -45 and +45 degrees. "
            "Use this to look up/down."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {
                    "type": "integer",
                    "minimum": -45,
                    "maximum": 45,
                    "description": (
                        "Target tilt position where 0 is neutral, -45 is full down, "
                        "and +45 is full up."
                    ),
                },
            },
            "required": ["degrees"],
        },
    }
)

function_map["set_tilt"] = set_tilt

tools.append(
    {
        "type": "function",
        "name": "get_servo_position",
        "description": "Read the current position of the requested servo.",
        "parameters": {
            "type": "object",
            "properties": {
                "servo_name": {
                    "type": "string",
                    "enum": ["pan", "tilt"],
                    "description": "The name of the servo to read.",
                },
            },
            "required": ["servo_name"],
        },
    }
)

function_map["get_servo_position"] = get_servo_position

tools.append(
    {
        "type": "function",
        "name": "update_user_profile",
        "description": (
            "Update the active user profile with personal details like name, "
            "preferences, or favorites."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "preferences": {"type": "object"},
                "favorites": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
    }
)

function_map["update_user_profile"] = update_user_profile

tools.append(
    {
        "type": "function",
        "name": "get_output_volume",
        "description": "Read the current output audio volume.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["get_output_volume"] = get_output_volume

tools.append(
    {
        "type": "function",
        "name": "set_output_volume",
        "description": (
            "Set the output audio volume. Volume percent must be between 1 and 100. "
            "Changes are rate-limited to once per second unless emergency is true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "percent": {"type": "integer", "minimum": 1, "maximum": 100},
                "emergency": {"type": "boolean", "default": False},
            },
            "required": ["percent"],
        },
    }
)

function_map["set_output_volume"] = set_output_volume

tools.append(
    {
        "type": "function",
        "name": "remember_memory",
        "description": (
            "Store a durable memory about the user, preferences, or facts worth reusing. "
            "Only store when the user provides stable, repeatable facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["content"],
        },
    }
)

function_map["remember_memory"] = remember_memory

tools.append(
    {
        "type": "function",
        "name": "recall_memories",
        "description": (
            "Fetch relevant stored memories when the user asks about prior facts, "
            "preferences, or context that might have been saved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": [],
        },
    }
)

function_map["recall_memories"] = recall_memories

tools.append(
    {
        "type": "function",
        "name": "forget_memory",
        "description": "Remove a stored memory when the user asks to delete or forget it.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer", "minimum": 1},
            },
            "required": ["memory_id"],
        },
    }
)

function_map["forget_memory"] = forget_memory
