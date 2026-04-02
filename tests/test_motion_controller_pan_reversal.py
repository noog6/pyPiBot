"""Tests for pan reversal-aware limiting behavior."""

from __future__ import annotations

import importlib.util
import pytest
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
    spec = importlib.util.spec_from_file_location("motion_controller_pan_reversal_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["motion_controller_pan_reversal_for_test"] = module
    spec.loader.exec_module(module)
    return module


def test_pan_reversal_profile_unchanged_when_not_reversing(monkeypatch) -> None:
    motion_controller = _load_motion_controller_module()
    monkeypatch.setattr(motion_controller, "TUNING", motion_controller.MotionTuning())

    v_max, a_max, reversal = motion_controller.pan_reversal_limited_profile(
        pan_err_deg=15.0,
        pan_v_dps=30.0,
        pan_v_max_dps=50.0,
        pan_a_max_dps2=165.0,
    )

    assert reversal is False
    assert v_max == 50.0
    assert a_max == 165.0


def test_pan_reversal_profile_scales_limits_when_direction_flips(monkeypatch) -> None:
    motion_controller = _load_motion_controller_module()
    monkeypatch.setattr(
        motion_controller,
        "TUNING",
        motion_controller.MotionTuning(
            pan_reversal_speed_threshold_dps=20.0,
            pan_reversal_v_max_scale=0.7,
            pan_reversal_a_max_scale=0.55,
        ),
    )

    v_max, a_max, reversal = motion_controller.pan_reversal_limited_profile(
        pan_err_deg=-10.0,
        pan_v_dps=35.0,
        pan_v_max_dps=60.0,
        pan_a_max_dps2=165.0,
    )

    assert reversal is True
    assert v_max == 42.0
    assert a_max == pytest.approx(90.75)


def test_pan_reversal_profile_ignores_low_speed_sign_flips(monkeypatch) -> None:
    motion_controller = _load_motion_controller_module()
    monkeypatch.setattr(
        motion_controller,
        "TUNING",
        motion_controller.MotionTuning(pan_reversal_speed_threshold_dps=20.0),
    )

    v_max, a_max, reversal = motion_controller.pan_reversal_limited_profile(
        pan_err_deg=-10.0,
        pan_v_dps=5.0,
        pan_v_max_dps=60.0,
        pan_a_max_dps2=165.0,
    )

    assert reversal is False
    assert v_max == 60.0
    assert a_max == 165.0
