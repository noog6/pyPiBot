"""Gesture builders for motion sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from motion.action import Action
from motion.keyframe import Keyframe
from motion.motion_controller import MotionController, millis


@dataclass(frozen=True)
class GestureSpec:
    """Definition for a single gesture keyframe target."""

    name: str
    pan: float
    tilt: float
    duration_ms: int


def _get_servo_limits(controller: MotionController, name: str) -> tuple[float, float]:
    servo = controller.servo_registry.servos[name]
    return float(servo.min_angle), float(servo.max_angle)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _build_keyframe_chain(
    controller: MotionController,
    specs: Iterable[GestureSpec],
) -> Keyframe:
    iterator = iter(specs)
    first_spec = next(iterator)
    first_frame = _create_keyframe(controller, first_spec)
    current = first_frame

    for spec in iterator:
        next_frame = _create_keyframe(controller, spec)
        current.next = next_frame
        current = next_frame

    return first_frame


def _create_keyframe(controller: MotionController, spec: GestureSpec) -> Keyframe:
    pan_min, pan_max = _get_servo_limits(controller, "pan")
    tilt_min, tilt_max = _get_servo_limits(controller, "tilt")
    frame = controller.generate_base_keyframe(
        pan_degrees=_clamp(spec.pan, pan_min, pan_max),
        tilt_degrees=_clamp(spec.tilt, tilt_min, tilt_max),
    )
    frame.name = spec.name
    frame.final_target_time = spec.duration_ms
    return frame


def gesture_idle(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a gentle idle gesture action."""

    controller = MotionController.get_instance()
    pan_servo = controller.servo_registry.servos["pan"]
    tilt_servo = controller.servo_registry.servos["tilt"]
    current_pan = float(pan_servo.read_value())
    current_tilt = float(tilt_servo.read_value())

    pan_delta = 4.0 * intensity
    tilt_delta = 2.5 * intensity

    specs = [
        GestureSpec(
            name="idle-left",
            pan=current_pan - pan_delta,
            tilt=current_tilt + tilt_delta,
            duration_ms=1200,
        ),
        GestureSpec(
            name="idle-right",
            pan=current_pan + pan_delta,
            tilt=current_tilt - tilt_delta,
            duration_ms=1200,
        ),
        GestureSpec(
            name="idle-center",
            pan=current_pan,
            tilt=current_tilt,
            duration_ms=1000,
        ),
    ]

    frames = _build_keyframe_chain(controller, specs)
    return Action(
        priority=1,
        timestamp=millis() + delay_ms,
        name="gesture_idle",
        frames=frames,
    )


def gesture_nod(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a nod gesture action."""

    controller = MotionController.get_instance()
    pan_servo = controller.servo_registry.servos["pan"]
    tilt_servo = controller.servo_registry.servos["tilt"]
    current_pan = float(pan_servo.read_value())
    current_tilt = float(tilt_servo.read_value())

    tilt_delta = 10.0 * intensity

    specs = [
        GestureSpec(
            name="nod-down",
            pan=current_pan,
            tilt=current_tilt - tilt_delta,
            duration_ms=350,
        ),
        GestureSpec(
            name="nod-up",
            pan=current_pan,
            tilt=current_tilt + tilt_delta,
            duration_ms=350,
        ),
        GestureSpec(
            name="nod-center",
            pan=current_pan,
            tilt=current_tilt,
            duration_ms=400,
        ),
    ]

    frames = _build_keyframe_chain(controller, specs)
    return Action(
        priority=2,
        timestamp=millis() + delay_ms,
        name="gesture_nod",
        frames=frames,
    )
