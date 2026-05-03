"""Import smoke test for motion modules with minimal hardware stubs."""

from __future__ import annotations

import importlib
import sys
import types


def test_motion_import_path_smoke(monkeypatch) -> None:
    fake_hardware = types.ModuleType("hardware")
    fake_servo_registry = types.ModuleType("hardware.servo_registry")

    class _FakeServoRegistry:
        @classmethod
        def get_instance(cls):
            return cls()

    fake_servo_registry.ServoRegistry = _FakeServoRegistry
    fake_hardware.servo_registry = fake_servo_registry

    monkeypatch.setitem(sys.modules, "hardware", fake_hardware)
    monkeypatch.setitem(sys.modules, "hardware.servo_registry", fake_servo_registry)

    importlib.import_module("motion.motion_controller")
    importlib.import_module("motion.gesture_library")
    importlib.import_module("motion.gestures")
