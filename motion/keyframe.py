"""Keyframe data structure for motion sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Keyframe:
    """Represents a runtime motion keyframe.

    Timing fields:
    - `final_target_time`: nominal per-frame target duration in milliseconds.
      Gesture builds copy authored spec time into this field.
    - `deadline_ms`: absolute monotonic timestamp (`millis()`) computed when
      the controller initializes this frame (`start_time_ms + final_target_time`).
    - `duration_ms`: runtime-normalized duration derived at init time from
      `final_target_time` for logging/introspection (`max(1, final_target_time)`).
    """

    id: int = 0
    name: str = ""
    final_target_time: int = 0
    deadline_ms: int | None = None
    is_initialized: bool = False
    start_time_ms: int | None = None
    duration_ms: int | None = None
    last_pan_log_ms: int | None = None
    start_pos: dict[str, float] = field(default_factory=lambda: {"pan": 0.0, "tilt_left": 0.0, "tilt_right": 0.0, "ear_left": 0.0, "ear_right": 0.0})
    delta_pos: dict[str, float] = field(default_factory=lambda: {"pan": 0.0, "tilt_left": 0.0, "tilt_right": 0.0, "ear_left": 0.0, "ear_right": 0.0})
    servo_destination: dict[str, float] = field(
        default_factory=lambda: {"pan": 0.0, "tilt_left": 0.0, "tilt_right": 0.0, "ear_left": 0.0, "ear_right": 0.0}
    )
    audio: Any | None = None
    next: "Keyframe | None" = None

    def has_deadline(self) -> bool:
        """Return True if the frame has a deadline."""

        return self.deadline_ms is not None

    def remaining_ms(self, now_ms: int) -> int:
        """Return remaining milliseconds until the deadline."""

        if self.deadline_ms is None:
            return 0
        return max(0, int(self.deadline_ms - now_ms))

    def __str__(self) -> str:
        return (
            f"pan       :{self.servo_destination['pan']:.3f} "
            f"tilt_left :{self.servo_destination['tilt_left']:.3f}"
            f"tilt_right:{self.servo_destination['tilt_right']:.3f}"
            f"ear_left  :{self.servo_destination['ear_left']:.3f}"
            f"ear_right :{self.servo_destination['ear_right']:.3f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the keyframe to a dictionary."""

        return {
            "id": self.id,
            "name": self.name,
            "final_target_time": self.final_target_time,
            "is_initialized": self.is_initialized,
            "servo_destination": self.servo_destination,
            "audio": self.audio,
            "next": self.next,
        }
