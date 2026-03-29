"""Action sequences composed of keyframes."""

from __future__ import annotations

from motion.keyframe import Keyframe


class Action:
    """Represents a sequence of keyframes to execute."""

    def __init__(self, priority: int, timestamp: int, name: str, frames: Keyframe) -> None:
        self.priority = priority
        self.timestamp = timestamp
        self.name = name
        self.frames = frames
        self.current_frame = frames

    def __lt__(self, other: "Action") -> bool:
        if self.priority != other.priority:
            return self.priority > other.priority

        return self.timestamp < other.timestamp

    def set_frame_times(self, target_start_time: int) -> None:
        """Reset per-run timing state for a new action execution.

        This clears runtime-derived timing (`deadline_ms`, init flag) so each
        frame recomputes absolute timing from its nominal `final_target_time`
        when first processed by the motion controller.
        """

        frame_index = self.frames
        while frame_index:
            frame_index.deadline_ms = None
            frame_index.is_initialized = False
            frame_index = frame_index.next
