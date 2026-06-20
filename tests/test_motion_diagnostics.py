"""Tests for motion diagnostics."""

from __future__ import annotations

import types

from diagnostics.models import DiagnosticStatus
from motion.diagnostics import MotionProbeConfig, probe


def test_motion_probe_offline_pass() -> None:
    """Motion probe should pass when expected servos exist."""

    result = probe(
        servo_names=["pan", "tilt_left", "tilt_right", "ear_left", "ear_right"],
        config=MotionProbeConfig(),
    )
    assert result.status is DiagnosticStatus.PASS


def test_motion_probe_warns_on_missing_servo() -> None:
    """Motion probe should warn when expected servos are missing."""

    result = probe(
        servo_names=["pan", "tilt_left", "ear_left", "ear_right"],
        config=MotionProbeConfig(),
    )
    assert result.status is DiagnosticStatus.WARN


def test_five_servo_validation_uses_logical_keyframe_sequence(monkeypatch) -> None:
    """Manual bring-up validation should use the controller keyframe path."""

    from motion.diagnostics import (
        FIVE_SERVO_VALIDATION_POSES,
        run_five_servo_validation,
    )

    class _Controller:
        def __init__(self) -> None:
            self.servo_registry = types.SimpleNamespace(
                servos={
                    "pan": object(),
                    "tilt_left": object(),
                    "tilt_right": object(),
                    "ear_left": object(),
                    "ear_right": object(),
                }
            )
            self.current_action = None
            self.action_queue: list[object] = []
            self.translated: list[dict[str, float]] = []
            self.generated: list[dict[str, float]] = []
            self.moved: list[object] = []

        def is_control_loop_alive(self):
            return False

        def _logical_pose_to_servo_targets(self, logical_pose):
            self.translated.append(dict(logical_pose))
            return {
                "pan": logical_pose["pan"],
                "tilt_left": logical_pose["tilt"] + logical_pose["roll"],
                "tilt_right": logical_pose["tilt"] - logical_pose["roll"],
                "ear_left": logical_pose["ear_left"],
                "ear_right": logical_pose["ear_right"],
            }

        def generate_base_keyframe(
            self,
            *,
            pan_degrees,
            tilt_degrees,
            roll_degrees,
            ear_left_degrees,
            ear_right_degrees,
        ):
            from motion.keyframe import Keyframe

            pose = {
                "pan": pan_degrees,
                "tilt": tilt_degrees,
                "roll": roll_degrees,
                "ear_left": ear_left_degrees,
                "ear_right": ear_right_degrees,
            }
            self.generated.append(pose)
            frame = Keyframe(name="manual-validation")
            frame.servo_destination.update(pose)
            return frame

        def move_to_keyframe(self, frame):
            self.moved.append(frame)
            return True

    logs: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        "motion.diagnostics.log_info",
        lambda message, *args: logs.append((message, args)),
    )
    controller = _Controller()

    result = run_five_servo_validation(
        controller, step_duration_ms=123, settle_timeout_s=0.01, poll_interval_s=0.0
    )

    assert result.status is DiagnosticStatus.PASS
    assert [name for name, _pose in FIVE_SERVO_VALIDATION_POSES] == [
        "center",
        "pan-plus-15",
        "pan-minus-15",
        "tilt-plus-15",
        "tilt-minus-15",
        "roll-plus-15",
        "roll-minus-15",
        "ear-left-plus-25",
        "ear-left-minus-25",
        "ear-right-plus-25",
        "ear-right-minus-25",
        "tilt-15-roll-15",
        "return-center",
    ]
    assert controller.generated == [pose for _name, pose in FIVE_SERVO_VALIDATION_POSES]
    assert controller.translated == controller.generated
    assert len(controller.moved) == len(FIVE_SERVO_VALIDATION_POSES)
    assert all(frame.final_target_time == 123 for frame in controller.moved)
    assert any("requested_logical_pose" in message for message, _args in logs)
    assert any("translated_physical_targets" in message for message, _args in logs)


def test_five_servo_validation_reports_timeout() -> None:
    from motion.diagnostics import run_five_servo_validation

    class _Controller:
        def __init__(self) -> None:
            self.servo_registry = types.SimpleNamespace(
                servos={
                    "pan": object(),
                    "tilt_left": object(),
                    "tilt_right": object(),
                    "ear_left": object(),
                    "ear_right": object(),
                }
            )
            self.current_action = None
            self.action_queue: list[object] = []

        def is_control_loop_alive(self):
            return False

        def _logical_pose_to_servo_targets(self, logical_pose):
            return {
                "pan": logical_pose["pan"],
                "tilt_left": logical_pose["tilt"] + logical_pose["roll"],
                "tilt_right": logical_pose["tilt"] - logical_pose["roll"],
                "ear_left": logical_pose["ear_left"],
                "ear_right": logical_pose["ear_right"],
            }

        def generate_base_keyframe(self, **kwargs):
            from motion.keyframe import Keyframe

            return Keyframe(name="never-settles")

        def move_to_keyframe(self, frame):
            return False

    result = run_five_servo_validation(
        _Controller(), settle_timeout_s=0.0, poll_interval_s=0.0
    )

    assert result.status is DiagnosticStatus.FAIL
    assert "Timed out waiting for validation step: center" in result.details


def test_five_servo_validation_fails_before_moving_when_servo_missing() -> None:
    from motion.diagnostics import run_five_servo_validation

    class _Controller:
        def __init__(self) -> None:
            self.servo_registry = types.SimpleNamespace(
                servos={
                    "pan": object(),
                    "tilt_left": object(),
                    "ear_left": object(),
                    "ear_right": object(),
                }
            )
            self.moved = False
            self.current_action = None
            self.action_queue: list[object] = []

        def is_control_loop_alive(self):
            return False

        def move_to_keyframe(self, frame):
            self.moved = True
            return True

    controller = _Controller()

    result = run_five_servo_validation(controller)

    assert result.status is DiagnosticStatus.FAIL
    assert result.details == "Missing expected servos before validation: tilt_right"
    assert controller.moved is False


def test_five_servo_validation_requires_stopped_control_loop() -> None:
    from motion.diagnostics import run_five_servo_validation

    class _Controller:
        servo_registry = types.SimpleNamespace(
            servos={
                "pan": object(),
                "tilt_left": object(),
                "tilt_right": object(),
                "ear_left": object(),
                "ear_right": object(),
            }
        )
        current_action = None
        action_queue: list[object] = []

        def is_control_loop_alive(self):
            return True

    result = run_five_servo_validation(_Controller())

    assert result.status is DiagnosticStatus.FAIL
    assert (
        result.details
        == "Five-servo validation requires the motion control loop to be stopped."
    )


def test_five_servo_validation_requires_empty_action_state() -> None:
    from motion.diagnostics import run_five_servo_validation

    class _Controller:
        servo_registry = types.SimpleNamespace(
            servos={
                "pan": object(),
                "tilt_left": object(),
                "tilt_right": object(),
                "ear_left": object(),
                "ear_right": object(),
            }
        )
        current_action = object()
        action_queue: list[object] = []

        def is_control_loop_alive(self):
            return False

    result = run_five_servo_validation(_Controller())

    assert result.status is DiagnosticStatus.FAIL
    assert (
        result.details == "Five-servo validation requires no active or queued actions."
    )
