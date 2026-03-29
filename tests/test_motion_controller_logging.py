"""Tests for motion-controller debug logging cadence."""

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
    spec = importlib.util.spec_from_file_location("motion_controller_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["motion_controller_for_test"] = module
    spec.loader.exec_module(module)
    return module


def _controller_for_debug_gate(motion_controller):
    controller = motion_controller.MotionController.__new__(motion_controller.MotionController)
    controller._last_motion_debug_log_ms = None
    return controller


def test_motion_debug_log_gate_respects_interval(monkeypatch) -> None:
    motion_controller = _load_motion_controller_module()
    controller = _controller_for_debug_gate(motion_controller)
    monkeypatch.setattr(
        motion_controller,
        "TUNING",
        motion_controller.MotionTuning(debug_motion=True, debug_motion_log_interval_ms=200),
    )

    assert controller._should_emit_motion_debug_log(1000) is True
    assert controller._should_emit_motion_debug_log(1100) is False
    assert controller._should_emit_motion_debug_log(1200) is True


def test_motion_debug_log_gate_allows_full_rate_when_interval_disabled(monkeypatch) -> None:
    motion_controller = _load_motion_controller_module()
    controller = _controller_for_debug_gate(motion_controller)
    monkeypatch.setattr(
        motion_controller,
        "TUNING",
        motion_controller.MotionTuning(debug_motion=True, debug_motion_log_interval_ms=0),
    )

    assert controller._should_emit_motion_debug_log(1000) is True
    assert controller._should_emit_motion_debug_log(1001) is True
