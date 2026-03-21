"""Shared hardware and motion operations used by AI tool handlers."""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

from hardware import ADS1015Sensor, LPS22HBSensor
from motion import (
    MotionController,
    gesture_attention_snap,
    gesture_curious_tilt,
    gesture_idle,
    gesture_look_around,
    gesture_look_center,
    gesture_look_down,
    gesture_look_left,
    gesture_look_right,
    gesture_look_up,
    gesture_no,
    gesture_nod,
)
from services.output_volume import OutputVolumeController


logger = logging.getLogger(__name__)


_LOOK_CENTER_PAN_DEGREES = 0.0
_LOOK_CENTER_TILT_DEGREES = 0.0
_LOOK_CENTER_EPS_DEGREES = 0.5


def _is_pose_centered(controller: MotionController, *, eps_degrees: float = _LOOK_CENTER_EPS_DEGREES) -> bool:
    current_position = getattr(controller, "current_servo_position", {}) or {}
    try:
        pan = float(current_position.get("pan", 0.0))
        tilt = float(current_position.get("tilt", 0.0))
    except (TypeError, ValueError, AttributeError):
        return False
    return (
        abs(pan - _LOOK_CENTER_PAN_DEGREES) <= eps_degrees
        and abs(tilt - _LOOK_CENTER_TILT_DEGREES) <= eps_degrees
    )


def _has_queued_motion_action(controller: MotionController, *, action_name: str) -> bool:
    action_queue = getattr(controller, "action_queue", None) or ()
    return any(str(getattr(action, "name", "") or "").strip() == action_name for action in action_queue)


def _classify_look_center_request(controller: MotionController) -> str:
    current_action = getattr(controller, "current_action", None)
    current_action_name = str(getattr(current_action, "name", "") or "").strip()
    if current_action_name == "gesture_look_center" and bool(controller.is_moving()):
        return "in_progress"
    if _has_queued_motion_action(controller, action_name="gesture_look_center"):
        return "queued"
    if not bool(controller.is_moving()) and current_action is None and _is_pose_centered(controller):
        return "already_satisfied"
    return "new_request"


def read_battery_voltage() -> dict[str, Any]:
    """Return the current LiPo battery voltage via the ADS1015 sensor."""

    sensor = ADS1015Sensor.get_instance()
    voltage = sensor.read_battery_voltage()
    return {
        "voltage": voltage,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
    }


def read_environment() -> dict[str, Any]:
    """Return the current onboard air pressure and temperature."""

    sensor = LPS22HBSensor.get_instance()
    air_pressure, air_temperature = sensor.read_value()
    cpu_temperature = None
    cpu_status = "unavailable"
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", result.stdout)
        if match:
            cpu_temperature = float(match.group(1))
            cpu_status = "ok"
    return {
        "air_pressure": air_pressure,
        "air_temperature": air_temperature,
        "air_temperature_context": "LPS22HB onboard ambient sensor inside Theo",
        "cpu_temperature": cpu_temperature,
        "cpu_temperature_context": "Broadcom SoC core temperature (vcgencmd)",
        "cpu_temperature_status": cpu_status,
        "pressure_unit": "hPa",
        "temperature_unit": "C",
    }


def enqueue_idle_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue an idle gesture action on the motion controller."""

    return _enqueue_gesture(gesture_idle(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_nod_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a nod gesture action on the motion controller."""

    return _enqueue_gesture(gesture_nod(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_no_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a head shake gesture action on the motion controller."""

    return _enqueue_gesture(gesture_no(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_around_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a casual look around gesture action on the motion controller."""

    return _enqueue_gesture(
        gesture_look_around(delay_ms=delay_ms, intensity=float(intensity)),
        delay_ms,
        intensity,
    )


def enqueue_look_up_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look up gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_up(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_left_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look left gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_left(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_right_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look right gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_right(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_down_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look down gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_down(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_center_gesture(delay_ms: int = 0) -> dict[str, Any]:
    """Queue a look center gesture action when the body state actually needs it."""

    controller = MotionController.get_instance()
    motion_request_state = _classify_look_center_request(controller)
    result: dict[str, Any] = {
        "queued": False,
        "gesture": "gesture_look_center",
        "delay_ms": delay_ms,
        "motion_request_state": motion_request_state,
    }
    if motion_request_state == "already_satisfied":
        result["message"] = "Already centered."
        result["tool_result_has_distinct_info"] = True
    elif motion_request_state == "in_progress":
        result["message"] = "Already moving back to center."
        result["tool_result_has_distinct_info"] = True
    elif motion_request_state == "queued":
        result["message"] = "Center motion is already queued."
        result["tool_result_has_distinct_info"] = True
    else:
        action = gesture_look_center(delay_ms=delay_ms)
        controller.add_action_to_queue(action)
        result["queued"] = True
        result["gesture"] = action.name
        result["message"] = "Moving back to center."
    logger.info(
        "motion_request_eval tool=%s state=%s no_duplicate_motion_queued=%s delay_ms=%s",
        result["gesture"],
        motion_request_state,
        str(not bool(result.get("queued", False))).lower(),
        delay_ms,
    )
    return result


def enqueue_curious_tilt_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a curious tilt gesture action on the motion controller."""

    return _enqueue_gesture(
        gesture_curious_tilt(delay_ms=delay_ms, intensity=float(intensity)),
        delay_ms,
        intensity,
    )


def enqueue_attention_snap_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a quick attention snap gesture action on the motion controller."""

    return _enqueue_gesture(
        gesture_attention_snap(delay_ms=delay_ms, intensity=float(intensity)),
        delay_ms,
        intensity,
    )


def get_output_volume() -> dict[str, Any]:
    """Return the current output audio volume."""

    controller = OutputVolumeController.get_instance()
    status = controller.get_volume()
    return {
        "percent": status.percent,
        "muted": status.muted,
    }


def set_output_volume(percent: int, emergency: bool = False) -> dict[str, Any]:
    """Set the output audio volume within safe bounds."""

    controller = OutputVolumeController.get_instance()
    status = controller.set_volume(percent=int(percent), emergency=bool(emergency))
    return {
        "percent": status.percent,
        "muted": status.muted,
    }


def _enqueue_gesture(action: Any, delay_ms: int, intensity: float) -> dict[str, Any]:
    _add_action_to_motion_queue(action)
    return {"queued": True, "gesture": action.name, "delay_ms": delay_ms, "intensity": intensity}


def _add_action_to_motion_queue(action: Any) -> None:
    controller = MotionController.get_instance()
    controller.add_action_to_queue(action)

