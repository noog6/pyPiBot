"""Tests for logical motion pose to physical servo translation."""

from __future__ import annotations

import threading
import types

from motion.keyframe import Keyframe
from motion.motion_controller import LOGICAL_POSE_AXES, MotionController


class _Servo:
    def __init__(self, minimum: float = -90.0, maximum: float = 90.0) -> None:
        self.min_angle = minimum
        self.max_angle = maximum
        self.writes: list[float] = []
        self.relaxed = False

    def write_value(self, value: float) -> None:
        self.writes.append(value)

    def relax(self) -> None:
        self.relaxed = True


def _controller() -> MotionController:
    controller = MotionController.__new__(MotionController)
    controller.servo_registry = types.SimpleNamespace(
        servos={
            "pan": _Servo(-90.0, 90.0),
            "tilt_left": _Servo(-45.0, 45.0),
            "tilt_right": _Servo(-45.0, 45.0),
            "ear_left": _Servo(-90.0, 90.0),
            "ear_right": _Servo(-90.0, 90.0),
        }
    )
    controller.current_logical_pose = {axis: 0.0 for axis in LOGICAL_POSE_AXES}
    controller.current_servo_position = controller.current_logical_pose
    controller.axis_v = {axis: 0.0 for axis in LOGICAL_POSE_AXES}
    controller.axis_v_max = {axis: 0.0 for axis in LOGICAL_POSE_AXES}
    controller.transition_time = 1500
    controller.control_loop_period_ms = 20
    controller._last_update_ms = None
    controller._last_motion_debug_log_ms = None
    controller._moving_event = threading.Event()
    return controller


def _frame(**destination: float) -> Keyframe:
    frame = Keyframe(final_target_time=0, name="test")
    frame.servo_destination.update(destination)
    return frame


def test_logical_pan_maps_to_physical_pan_unchanged() -> None:
    controller = _controller()

    targets = controller._logical_pose_to_servo_targets(
        {"pan": 17.0, "tilt": 0.0, "roll": 0.0}
    )

    assert targets["pan"] == 17.0


def test_logical_tilt_with_zero_roll_maps_both_tilt_servos_equally() -> None:
    controller = _controller()

    targets = controller._logical_pose_to_servo_targets(
        {"pan": 0.0, "tilt": 10.0, "roll": 0.0}
    )

    assert targets["tilt_left"] == 10.0
    assert targets["tilt_right"] == 10.0


def test_logical_roll_maps_tilt_servos_in_opposite_directions() -> None:
    controller = _controller()

    targets = controller._logical_pose_to_servo_targets(
        {"pan": 0.0, "tilt": 0.0, "roll": 10.0}
    )

    assert targets["tilt_left"] == 10.0
    assert targets["tilt_right"] == -10.0


def test_logical_tilt_plus_roll_clamps_to_physical_servo_limits() -> None:
    controller = _controller()

    targets = controller._logical_pose_to_servo_targets(
        {"pan": 0.0, "tilt": 40.0, "roll": 20.0}
    )

    assert targets["tilt_left"] == 45.0
    assert targets["tilt_right"] == 20.0


def test_logical_target_clamp_logs_detectable_event(monkeypatch) -> None:
    controller = _controller()
    warnings: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        "motion.motion_controller.log_warning",
        lambda message, *args: warnings.append((message, args)),
    )

    targets = controller._logical_pose_to_servo_targets(
        {"pan": 0.0, "tilt": 40.0, "roll": 20.0}
    )

    assert targets["tilt_left"] == 45.0
    assert warnings
    assert warnings[0][1][0] == "tilt_left"


def test_init_frame_uses_all_logical_pose_axes() -> None:
    controller = _controller()
    controller.current_logical_pose.update(
        {"pan": 1.0, "tilt": 2.0, "roll": 3.0, "ear_left": 4.0, "ear_right": 5.0}
    )
    frame = _frame(pan=2.0, tilt=4.0, roll=6.0, ear_left=8.0, ear_right=10.0)

    controller._init_frame(frame, now_ms=1000)

    assert frame.start_pos == {
        "pan": 1.0,
        "tilt": 2.0,
        "roll": 3.0,
        "ear_left": 4.0,
        "ear_right": 5.0,
    }
    assert frame.delta_pos == {
        "pan": 1.0,
        "tilt": 2.0,
        "roll": 3.0,
        "ear_left": 4.0,
        "ear_right": 5.0,
    }


def test_move_to_keyframe_writes_five_physical_servos(monkeypatch) -> None:
    controller = _controller()
    monkeypatch.setattr("motion.motion_controller.millis", lambda: 1000)
    frame = _frame(pan=1.0, tilt=1.0, roll=1.0, ear_left=1.0, ear_right=1.0)

    controller.move_to_keyframe(frame)

    assert set(controller.servo_registry.servos) == {
        "pan",
        "tilt_left",
        "tilt_right",
        "ear_left",
        "ear_right",
    }
    assert all(servo.writes for servo in controller.servo_registry.servos.values())


def test_roll_is_ramped_over_time_not_snapped(monkeypatch) -> None:
    controller = _controller()
    monkeypatch.setattr("motion.motion_controller.millis", lambda: 1000)
    frame = _frame(roll=20.0)

    done = controller.move_to_keyframe(frame)

    assert done is False
    assert 0.0 < controller.current_logical_pose["roll"] < 20.0


def test_ears_are_ramped_over_time_not_snapped(monkeypatch) -> None:
    controller = _controller()
    monkeypatch.setattr("motion.motion_controller.millis", lambda: 1000)
    frame = _frame(ear_left=20.0, ear_right=-20.0)

    done = controller.move_to_keyframe(frame)

    assert done is False
    assert 0.0 < controller.current_logical_pose["ear_left"] < 20.0
    assert -20.0 < controller.current_logical_pose["ear_right"] < 0.0


def test_frame_completion_waits_for_roll_and_ears(monkeypatch) -> None:
    controller = _controller()
    monkeypatch.setattr("motion.motion_controller.millis", lambda: 1000)
    frame = _frame(roll=20.0, ear_left=20.0, ear_right=-20.0)

    assert controller.move_to_keyframe(frame) is False
    assert frame.has_deadline()


def test_generate_base_keyframe_defaults_to_logical_zero_roll_and_neutral_ears() -> (
    None
):
    controller = _controller()

    frame = controller.generate_base_keyframe(pan_degrees=3.0, tilt_degrees=10.0)

    assert frame.servo_destination == {
        "pan": 3.0,
        "tilt": 10.0,
        "roll": 0,
        "ear_left": 0,
        "ear_right": 0,
    }


def test_relax_all_servos_relaxes_registered_physical_servos() -> None:
    controller = _controller()

    controller.relax_all_servos()

    assert all(servo.relaxed for servo in controller.servo_registry.servos.values())
