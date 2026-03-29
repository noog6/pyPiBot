"""Contract tests for MotionController._frame_done timing semantics."""

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
    spec = importlib.util.spec_from_file_location("motion_controller_frame_done_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["motion_controller_frame_done_for_test"] = module
    spec.loader.exec_module(module)
    return module


class _Frame:
    def __init__(self, *, deadline_ms: int | None, name: str = "frame") -> None:
        self.deadline_ms = deadline_ms
        self.name = name

    def has_deadline(self) -> bool:
        return self.deadline_ms is not None


def test_frame_done_without_deadline_is_destination_only() -> None:
    motion_controller = _load_motion_controller_module()
    controller = motion_controller.MotionController.__new__(motion_controller.MotionController)
    frame = _Frame(deadline_ms=None)

    assert controller._frame_done(frame, at_dest=False, now_ms=1_000) is False
    assert controller._frame_done(frame, at_dest=True, now_ms=1_000) is True


def test_frame_done_with_deadline_requires_at_dest_and_time_elapsed() -> None:
    motion_controller = _load_motion_controller_module()
    controller = motion_controller.MotionController.__new__(motion_controller.MotionController)
    frame = _Frame(deadline_ms=2_000)

    assert controller._frame_done(frame, at_dest=True, now_ms=1_999) is False
    assert controller._frame_done(frame, at_dest=False, now_ms=2_000) is False
    assert controller._frame_done(frame, at_dest=True, now_ms=2_000) is True
