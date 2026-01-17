"""Keyframe data structure for motion sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Keyframe:
    """Represents a motion keyframe."""

    id: int = 0
    name: str = ""
    final_target_time: int = 0
    deadline_ms: int | None = None
    is_initialized: bool = False
    start_time_ms: int | None = None
    duration_ms: int | None = None
    start_pos: dict[str, float] = field(default_factory=lambda: {"pan": 0.0, "tilt": 0.0})
    delta_pos: dict[str, float] = field(default_factory=lambda: {"pan": 0.0, "tilt": 0.0})
    servo_destination: dict[str, float] = field(
        default_factory=lambda: {"pan": 0.0, "tilt": 0.0}
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
            f"pan :{self.servo_destination['pan']:.3f} "
            f"tilt:{self.servo_destination['tilt']:.3f}"
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
