"""Diagnostics routines for motion logic."""

from __future__ import annotations

from dataclasses import dataclass

from diagnostics.models import DiagnosticResult, DiagnosticStatus
from motion.action import Action
from motion.keyframe import Keyframe
from motion.motion_controller import limit_step


@dataclass(frozen=True)
class MotionProbeConfig:
    """Configuration for motion diagnostics."""

    expected_servos: tuple[str, ...] = ("pan", "tilt")


def probe(servo_names: list[str] | None = None, config: MotionProbeConfig | None = None) -> DiagnosticResult:
    """Run a motion probe to validate sequencing logic.

    Args:
        servo_names: Optional list of servo names for offline testing.
        config: Optional configuration for expected names.

    Returns:
        Diagnostic result indicating motion readiness.
    """

    name = "motion"
    settings = config or MotionProbeConfig()

    frame = Keyframe(id=1, name="diagnostics")
    action = Action(priority=1, timestamp=0, name="diag_action", frames=frame)
    action.set_frame_times(0)

    next_pos = limit_step(
        current=0.0,
        target=1.0,
        v_state={"pan": 0.0},
        axis="pan",
        dt_s=0.1,
        v_max=1.0,
        a_max=1.0,
    )
    if next_pos <= 0.0:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="Motion limit_step failed to advance",
        )

    if servo_names is not None:
        missing = [servo for servo in settings.expected_servos if servo not in servo_names]
        if missing:
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.WARN,
                details=f"Missing expected servos: {', '.join(missing)}",
            )
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.PASS,
            details=f"Servo names: {', '.join(servo_names)}",
        )

    return DiagnosticResult(
        name=name,
        status=DiagnosticStatus.PASS,
        details="Motion logic checks passed",
    )
