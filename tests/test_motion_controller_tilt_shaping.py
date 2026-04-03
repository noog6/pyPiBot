"""Contract tests for seam-local pan/tilt step shaping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_motion_controller_module():
    fake_hardware = types.ModuleType("hardware")
    fake_servo_registry = types.ModuleType("hardware.servo_registry")

    class _FakeServoRegistry:
        @classmethod
        def get_instance(cls):
            return cls()

    fake_servo_registry.ServoRegistry = _FakeServoRegistry
    fake_hardware.servo_registry = fake_servo_registry
    sys.modules["hardware"] = fake_hardware
    sys.modules["hardware.servo_registry"] = fake_servo_registry

    fake_motion = types.ModuleType("motion")
    fake_action = types.ModuleType("motion.action")
    fake_keyframe = types.ModuleType("motion.keyframe")
    fake_logging = types.ModuleType("motion.logging")
    fake_action.Action = object
    fake_keyframe.Keyframe = object
    fake_logging.log_debug = lambda *args, **kwargs: None
    fake_logging.log_error = lambda *args, **kwargs: None
    fake_logging.log_info = lambda *args, **kwargs: None
    fake_logging.log_warning = lambda *args, **kwargs: None
    fake_motion.action = fake_action
    fake_motion.keyframe = fake_keyframe
    fake_motion.logging = fake_logging
    sys.modules["motion"] = fake_motion
    sys.modules["motion.action"] = fake_action
    sys.modules["motion.keyframe"] = fake_keyframe
    sys.modules["motion.logging"] = fake_logging

    module_path = Path(__file__).resolve().parents[1] / "motion" / "motion_controller.py"
    spec = importlib.util.spec_from_file_location("motion_controller_tilt_shape_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["motion_controller_tilt_shape_for_test"] = module
    spec.loader.exec_module(module)
    return module


def test_tilt_uses_distinct_min_max_scale_curve() -> None:
    motion_controller = _load_motion_controller_module()
    tuning = motion_controller.MotionTuning(
        pan_step_min_deg=0.05,
        pan_step_max_deg=0.20,
        pan_step_scale_deg=100.0,
        tilt_step_min_deg=0.50,
        tilt_step_max_deg=3.00,
        tilt_step_scale_deg=10.0,
    )

    nominal_dt_s = 0.02
    remaining_error = 5.0

    tilt_v = motion_controller.axis_step_v_max("tilt", remaining_error, nominal_dt_s, tuning)
    expected_tilt_v = (
        motion_controller.scaled_step(
            remaining_error,
            tuning.tilt_step_min_deg,
            tuning.tilt_step_max_deg,
            tuning.tilt_step_scale_deg,
        )
        / nominal_dt_s
    )
    pan_v = motion_controller.axis_step_v_max("pan", remaining_error, nominal_dt_s, tuning)

    assert tilt_v == expected_tilt_v
    assert tilt_v != pan_v


def test_tilt_can_command_larger_step_than_pan_for_same_error() -> None:
    motion_controller = _load_motion_controller_module()

    nominal_dt_s = 0.02
    representative_error = 45.0
    pan_v = motion_controller.axis_step_v_max("pan", representative_error, nominal_dt_s, motion_controller.TUNING)
    tilt_v = motion_controller.axis_step_v_max("tilt", representative_error, nominal_dt_s, motion_controller.TUNING)

    assert tilt_v > pan_v


def test_small_tilt_error_still_settles_without_oversized_move() -> None:
    motion_controller = _load_motion_controller_module()

    settled = motion_controller.limit_step(
        current=0.0,
        target=0.03,
        v_state={"tilt": 0.0},
        axis="tilt",
        dt_s=0.02,
        v_max=motion_controller.axis_step_v_max("tilt", 0.03, 0.02, motion_controller.TUNING),
        a_max=motion_controller.TUNING.tilt_a_max,
        eps=motion_controller.TUNING.position_eps_deg,
    )

    assert settled == 0.03


def test_pan_shaping_behavior_is_unchanged_contract() -> None:
    motion_controller = _load_motion_controller_module()

    nominal_dt_s = 0.02
    remaining_error = 35.0
    pan_v = motion_controller.axis_step_v_max("pan", remaining_error, nominal_dt_s, motion_controller.TUNING)
    expected_pan_v = (
        motion_controller.scaled_step(
            remaining_error,
            motion_controller.TUNING.pan_step_min_deg,
            motion_controller.TUNING.pan_step_max_deg,
            motion_controller.TUNING.pan_step_scale_deg,
        )
        / nominal_dt_s
    )

    assert pan_v == expected_pan_v
