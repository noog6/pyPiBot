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
EXPRESSIVE_TILT_SOFT_LIMIT_DEGREES = 44.0


class ServoLimit(Protocol):
    min_angle: float
    max_angle: float


@dataclass(frozen=True)
class ExpressionTiltEnvelopeResult:
    """Logical expression pose after soft-envelope tilt/roll shaping."""

    tilt: float
    roll: float
    requested_tilt: float
    requested_roll: float

    @property
    def changed(self) -> bool:
        return self.tilt != self.requested_tilt or self.roll != self.requested_roll

    @property
    def tilt_changed(self) -> bool:
        return self.tilt != self.requested_tilt

    @property
    def roll_changed(self) -> bool:
        return self.roll != self.requested_roll


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


def constrain_expression_frame_to_tilt_envelope(
    *,
    target_tilt: float,
    target_roll: float,
    soft_limit_degrees: float = EXPRESSIVE_TILT_SOFT_LIMIT_DEGREES,
    roll_to_tilt_sign: float = DEFAULT_ROLL_TO_TILT_SIGN,
) -> ExpressionTiltEnvelopeResult:
    """Shape logical tilt/roll so composed tilt servos stay in a soft envelope.

    Roll is reduced before tilt because expressive overlays should preserve gaze
    tilt whenever possible. If the tilt itself is outside the soft envelope,
    roll is neutralized and tilt is brought back inside the envelope.
    """

    requested_tilt = float(target_tilt)
    requested_roll = float(target_roll)
    limit = abs(float(soft_limit_degrees))
    if limit <= 0.0:
        return ExpressionTiltEnvelopeResult(
            tilt=0.0,
            roll=0.0,
            requested_tilt=requested_tilt,
            requested_roll=requested_roll,
        )

    signed_requested_roll = requested_roll * float(roll_to_tilt_sign)
    left = requested_tilt + signed_requested_roll
    right = requested_tilt - signed_requested_roll
    if -limit <= left <= limit and -limit <= right <= limit:
        return ExpressionTiltEnvelopeResult(
            tilt=requested_tilt,
            roll=requested_roll,
            requested_tilt=requested_tilt,
            requested_roll=requested_roll,
        )

    shaped_tilt = max(-limit, min(limit, requested_tilt))
    lower = max(-limit - shaped_tilt, shaped_tilt - limit)
    upper = min(limit - shaped_tilt, shaped_tilt + limit)
    shaped_signed_roll = max(lower, min(upper, signed_requested_roll))
    sign = float(roll_to_tilt_sign)
    shaped_roll = 0.0 if sign == 0.0 else shaped_signed_roll / sign
    return ExpressionTiltEnvelopeResult(
        tilt=shaped_tilt,
        roll=shaped_roll,
        requested_tilt=requested_tilt,
        requested_roll=requested_roll,
    )
