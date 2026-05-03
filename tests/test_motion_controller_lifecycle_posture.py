"""Focused tests for lifecycle posture selection in motion controller."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_motion_controller_module(monkeypatch):
    fake_ai = types.ModuleType("ai")
    fake_embodiment_policy = types.ModuleType("ai.embodiment_policy")

    class _LifecyclePostureEvent:
        STARTUP = types.SimpleNamespace(value="startup")
        SHUTDOWN = types.SimpleNamespace(value="shutdown")

    class _EmbodimentPolicy:
        def decide_lifecycle_posture(self, *, event):
            cue = "gesture_startup_presence" if event.value == "startup" else "gesture_shutdown_rest"
            return types.SimpleNamespace(cue_name=cue, reason_codes=(f"lifecycle_{event.value}",))

    fake_embodiment_policy.EmbodimentPolicy = _EmbodimentPolicy
    fake_embodiment_policy.LifecyclePostureEvent = _LifecyclePostureEvent
    fake_ai.embodiment_policy = fake_embodiment_policy
    monkeypatch.setitem(sys.modules, "ai", fake_ai)
    monkeypatch.setitem(sys.modules, "ai.embodiment_policy", fake_embodiment_policy)

    fake_hardware = types.ModuleType("hardware")
    fake_servo_registry = types.ModuleType("hardware.servo_registry")

    class _FakeServoRegistry:
        servos = {"pan": types.SimpleNamespace(write_value=lambda *_: None), "tilt": types.SimpleNamespace(write_value=lambda *_: None)}

        @classmethod
        def get_instance(cls):
            return cls()

    fake_servo_registry.ServoRegistry = _FakeServoRegistry
    fake_hardware.servo_registry = fake_servo_registry
    monkeypatch.setitem(sys.modules, "hardware", fake_hardware)
    monkeypatch.setitem(sys.modules, "hardware.servo_registry", fake_servo_registry)

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
    monkeypatch.setitem(sys.modules, "motion", fake_motion)
    monkeypatch.setitem(sys.modules, "motion.action", fake_action)
    monkeypatch.setitem(sys.modules, "motion.keyframe", fake_keyframe)
    monkeypatch.setitem(sys.modules, "motion.logging", fake_logging)

    module_path = Path(__file__).resolve().parents[1] / "motion" / "motion_controller.py"
    spec = importlib.util.spec_from_file_location("motion_controller_lifecycle_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "motion_controller_lifecycle_for_test", module)
    spec.loader.exec_module(module)
    return module


def test_startup_and_shutdown_request_lifecycle_policy_and_dispatch_gesture(monkeypatch) -> None:
    motion_controller = _load_motion_controller_module(monkeypatch)
    calls: list[str] = []

    class _Policy:
        def decide_lifecycle_posture(self, *, event):
            cue = "gesture_startup_presence" if event.value == "startup" else "gesture_shutdown_rest"
            return types.SimpleNamespace(cue_name=cue, reason_codes=("ok",))

    monkeypatch.setattr(motion_controller, "EmbodimentPolicy", _Policy)

    controller = motion_controller.MotionController.__new__(motion_controller.MotionController)
    controller._run_lifecycle_gesture_blocking = lambda *, gesture_name: calls.append(gesture_name)

    controller._run_lifecycle_posture_blocking(event=motion_controller.LifecyclePostureEvent.STARTUP)
    controller._run_lifecycle_posture_blocking(event=motion_controller.LifecyclePostureEvent.SHUTDOWN)

    assert calls == ["gesture_startup_presence", "gesture_shutdown_rest"]
