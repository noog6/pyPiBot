"""Diagnostics routines for motion logic."""

from __future__ import annotations

from dataclasses import dataclass
import time

from diagnostics.models import DiagnosticResult, DiagnosticStatus
from motion.action import Action
from motion.body_model import PHYSICAL_SERVO_NAMES
from motion.keyframe import Keyframe
from motion.logging import log_info
from motion.motion_controller import MotionController, limit_step


@dataclass(frozen=True)
class MotionProbeConfig:
    """Configuration for motion diagnostics."""

    expected_servos: tuple[str, ...] = PHYSICAL_SERVO_NAMES


FIVE_SERVO_VALIDATION_EXPECTED_SERVOS = set(PHYSICAL_SERVO_NAMES)


FIVE_SERVO_VALIDATION_POSES: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "center",
        {"pan": 0.0, "tilt": 0.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "pan-plus-5",
        {"pan": 5.0, "tilt": 0.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "pan-minus-5",
        {"pan": -5.0, "tilt": 0.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "tilt-plus-5",
        {"pan": 0.0, "tilt": 5.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "tilt-minus-5",
        {"pan": 0.0, "tilt": -5.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "roll-plus-5",
        {"pan": 0.0, "tilt": 0.0, "roll": 5.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "roll-minus-5",
        {"pan": 0.0, "tilt": 0.0, "roll": -5.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "ear-left-plus-5",
        {"pan": 0.0, "tilt": 0.0, "roll": 0.0, "ear_left": 5.0, "ear_right": 0.0},
    ),
    (
        "ear-left-minus-5",
        {"pan": 0.0, "tilt": 0.0, "roll": 0.0, "ear_left": -5.0, "ear_right": 0.0},
    ),
    (
        "ear-right-plus-5",
        {"pan": 0.0, "tilt": 0.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 5.0},
    ),
    (
        "ear-right-minus-5",
        {"pan": 0.0, "tilt": 0.0, "roll": 0.0, "ear_left": 0.0, "ear_right": -5.0},
    ),
    (
        "tilt-5-roll-5",
        {"pan": 0.0, "tilt": 5.0, "roll": 5.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
    (
        "return-center",
        {"pan": 0.0, "tilt": 0.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    ),
)


def run_five_servo_validation(
    controller: MotionController | None = None,
    *,
    step_duration_ms: int = 350,
    settle_timeout_s: float = 3.0,
    poll_interval_s: float = 0.02,
) -> DiagnosticResult:
    """Run a temporary manual five-servo bring-up validation sequence.

    This intentionally uses the MotionController logical keyframe path so the
    validation exercises the same ramping, clamping, and physical translation
    seam used by normal motion execution.
    """

    name = "motion_five_servo_validation"
    active_controller = controller or MotionController.get_instance()

    actual_servos = set(active_controller.servo_registry.servos)
    missing = FIVE_SERVO_VALIDATION_EXPECTED_SERVOS - actual_servos
    if missing:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Missing expected servos before validation: {', '.join(sorted(missing))}",
        )

    if active_controller.is_control_loop_alive():
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="Five-servo validation requires the motion control loop to be stopped.",
        )

    if active_controller.current_action is not None or active_controller.action_queue:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="Five-servo validation requires no active or queued actions.",
        )

    log_info(
        "[MOTION][five-servo-validation] starting steps=%s step_duration_ms=%s",
        len(FIVE_SERVO_VALIDATION_POSES),
        step_duration_ms,
    )

    completed_steps: list[str] = []
    for step_name, logical_pose in FIVE_SERVO_VALIDATION_POSES:
        physical_targets = active_controller._logical_pose_to_servo_targets(
            logical_pose
        )
        log_info(
            "[MOTION][five-servo-validation] step=%s requested_logical_pose=%s translated_physical_targets=%s",
            step_name,
            logical_pose,
            physical_targets,
        )
        frame = active_controller.generate_base_keyframe(
            pan_degrees=logical_pose["pan"],
            tilt_degrees=logical_pose["tilt"],
            roll_degrees=logical_pose["roll"],
            ear_left_degrees=logical_pose["ear_left"],
            ear_right_degrees=logical_pose["ear_right"],
        )
        frame.name = f"five-servo-validation:{step_name}"
        frame.final_target_time = step_duration_ms

        deadline_s = time.monotonic() + settle_timeout_s
        while not active_controller.move_to_keyframe(frame):
            if time.monotonic() >= deadline_s:
                return DiagnosticResult(
                    name=name,
                    status=DiagnosticStatus.FAIL,
                    details=f"Timed out waiting for validation step: {step_name}",
                )
            time.sleep(poll_interval_s)
        completed_steps.append(step_name)

    log_info(
        "[MOTION][five-servo-validation] completed steps=%s",
        ",".join(completed_steps),
    )
    return DiagnosticResult(
        name=name,
        status=DiagnosticStatus.PASS,
        details=f"Completed five-servo validation steps: {', '.join(completed_steps)}",
    )


def probe(
    servo_names: list[str] | None = None, config: MotionProbeConfig | None = None
) -> DiagnosticResult:
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
        missing = [
            servo for servo in settings.expected_servos if servo not in servo_names
        ]
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
