"""PandaTheo logical-to-physical body model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Any

LOGICAL_POSE_AXES: tuple[str, ...] = (
    "pan",
    "tilt",
    "roll",
    "ear_left",
    "ear_right",
)
PHYSICAL_SERVO_NAMES: tuple[str, ...] = (
    "pan",
    "tilt_left",
    "tilt_right",
    "ear_left",
    "ear_right",
)
DEFAULT_ROLL_TO_TILT_SIGN = 1.0


class ServoLimit(Protocol):
    min_angle: float
    max_angle: float


@dataclass(frozen=True)
class ClampEvent:
    """Description of one physical target clamp."""

    servo_name: str
    requested: float
    clamped: float
    minimum: float
    maximum: float


def normalize_logical_pose(pose: Mapping[str, float] | None) -> dict[str, float]:
    """Return a complete logical pose with missing axes defaulted to neutral."""

    source = pose or {}
    return {axis: float(source.get(axis, 0.0)) for axis in LOGICAL_POSE_AXES}


def _limit_tuple(limit: Any) -> tuple[float, float]:
    if isinstance(limit, tuple) or isinstance(limit, list):
        return float(limit[0]), float(limit[1])
    return float(limit.min_angle), float(limit.max_angle)


def servo_limits_from_registry(
    servos: Mapping[str, ServoLimit],
) -> dict[str, tuple[float, float]]:
    """Extract physical servo min/max limits from a servo registry mapping."""

    return {
        name: (float(servo.min_angle), float(servo.max_angle))
        for name, servo in servos.items()
    }


def compose_physical_targets(
    logical_pose: Mapping[str, float] | None,
    *,
    roll_to_tilt_sign: float = DEFAULT_ROLL_TO_TILT_SIGN,
) -> dict[str, float]:
    """Compose public logical pose axes into PandaTheo physical servo targets."""

    pose = normalize_logical_pose(logical_pose)
    pan = pose["pan"]
    tilt = pose["tilt"]
    roll = pose["roll"] * float(roll_to_tilt_sign)
    return {
        "pan": pan,
        "tilt_left": tilt + roll,
        "tilt_right": tilt - roll,
        "ear_left": pose["ear_left"],
        "ear_right": pose["ear_right"],
    }


def clamp_physical_targets(
    physical_targets: Mapping[str, float],
    servo_limits: Mapping[str, ServoLimit | tuple[float, float]],
) -> tuple[dict[str, float], tuple[ClampEvent, ...]]:
    """Clamp physical targets to per-servo limits after logical composition."""

    clamped_targets: dict[str, float] = {}
    clamp_events: list[ClampEvent] = []
    for name in PHYSICAL_SERVO_NAMES:
        requested = float(physical_targets.get(name, 0.0))
        minimum, maximum = _limit_tuple(servo_limits[name])
        clamped = max(minimum, min(maximum, requested))
        clamped_targets[name] = clamped
        if clamped != requested:
            clamp_events.append(
                ClampEvent(
                    servo_name=name,
                    requested=requested,
                    clamped=clamped,
                    minimum=minimum,
                    maximum=maximum,
                )
            )
    return clamped_targets, tuple(clamp_events)


def map_logical_pose_to_physical_targets(
    logical_pose: Mapping[str, float] | None,
    servo_limits: Mapping[str, ServoLimit | tuple[float, float]],
    *,
    roll_to_tilt_sign: float = DEFAULT_ROLL_TO_TILT_SIGN,
) -> tuple[dict[str, float], tuple[ClampEvent, ...]]:
    """Translate logical pose into clamped physical servo targets."""

    composed = compose_physical_targets(
        logical_pose,
        roll_to_tilt_sign=roll_to_tilt_sign,
    )
    return clamp_physical_targets(composed, servo_limits)
