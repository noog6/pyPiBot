"""Tests for distance-aware + style-aware look gesture timing policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_gesture_library_module():
    module_names = (
        "core",
        "core.logging",
        "motion",
        "motion.action",
        "motion.keyframe",
        "motion.motion_controller",
        "storage",
        "storage.controller",
        "gesture_library_timing_for_test",
    )
    original_modules = {name: sys.modules.get(name) for name in module_names}

    fake_core = types.ModuleType("core")
    fake_core_logging = types.ModuleType("core.logging")
    fake_core_logging.logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None
    )
    fake_core.logging = fake_core_logging
    sys.modules["core"] = fake_core
    sys.modules["core.logging"] = fake_core_logging

    fake_motion = types.ModuleType("motion")
    fake_action = types.ModuleType("motion.action")
    fake_keyframe = types.ModuleType("motion.keyframe")
    fake_motion_controller = types.ModuleType("motion.motion_controller")

    class _Action:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Keyframe:
        def __init__(self):
            self.servo_destination = {
                "pan": 0.0,
                "tilt": 0.0,
                "roll": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
            }
            self.name = ""
            self.final_target_time = 0
            self.next = None

    class _MotionController:
        @classmethod
        def get_instance(cls):
            return cls()

    fake_action.Action = _Action
    fake_keyframe.Keyframe = _Keyframe
    fake_motion_controller.MotionController = _MotionController
    fake_motion_controller.millis = lambda: 1_000
    fake_motion.action = fake_action
    fake_motion.keyframe = fake_keyframe
    fake_motion.motion_controller = fake_motion_controller
    sys.modules["motion"] = fake_motion
    sys.modules["motion.action"] = fake_action
    sys.modules["motion.keyframe"] = fake_keyframe
    sys.modules["motion.motion_controller"] = fake_motion_controller

    fake_storage = types.ModuleType("storage")
    fake_storage_controller = types.ModuleType("storage.controller")

    class _StorageController:
        @classmethod
        def get_instance(cls):
            return cls()

    fake_storage_controller.StorageController = _StorageController
    fake_storage.controller = fake_storage_controller
    sys.modules["storage"] = fake_storage
    sys.modules["storage.controller"] = fake_storage_controller

    module_path = Path(__file__).resolve().parents[1] / "motion" / "gesture_library.py"
    spec = importlib.util.spec_from_file_location(
        "gesture_library_timing_for_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gesture_library_timing_for_test"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def _definition(module, name: str):
    return next(item for item in module.DEFAULT_GESTURES if item.name == name)


def test_look_center_default_is_present() -> None:
    gesture_library = _load_gesture_library_module()
    definition = _definition(gesture_library, "gesture_look_center")

    assert definition.timing_style == "neutral"
    assert len(definition.frames) == 1
    assert definition.frames[0].name == "look-center"


def test_distance_and_style_adjust_look_timing() -> None:
    gesture_library = _load_gesture_library_module()
    library = gesture_library.GestureLibrary.__new__(gesture_library.GestureLibrary)
    definition = _definition(gesture_library, "gesture_look_right")
    spec = definition.frames[0]

    short_neutral = library._duration_for_frame(
        definition=definition,
        spec=spec,
        target_pan=8.0,
        target_tilt=0.0,
        transition_pan=6.0,
        transition_tilt=0.0,
        style="neutral",
    )
    long_neutral = library._duration_for_frame(
        definition=definition,
        spec=spec,
        target_pan=15.0,
        target_tilt=0.0,
        transition_pan=-15.0,
        transition_tilt=0.0,
        style="neutral",
    )
    long_snap = library._duration_for_frame(
        definition=definition,
        spec=spec,
        target_pan=15.0,
        target_tilt=0.0,
        transition_pan=-15.0,
        transition_tilt=0.0,
        style="snap",
    )
    long_solemn = library._duration_for_frame(
        definition=definition,
        spec=spec,
        target_pan=15.0,
        target_tilt=0.0,
        transition_pan=-15.0,
        transition_tilt=0.0,
        style="solemn",
    )

    assert short_neutral < long_neutral
    assert long_snap < long_neutral < long_solemn


def test_look_center_targets_canonical_pose_regardless_of_starting_pose() -> None:
    gesture_library = _load_gesture_library_module()
    library = gesture_library.GestureLibrary.__new__(gesture_library.GestureLibrary)
    definition = _definition(gesture_library, "gesture_look_center")
    spec = definition.frames[0]

    class _Servo:
        def __init__(self, minimum: float, maximum: float) -> None:
            self.min_angle = minimum
            self.max_angle = maximum

    class _Controller:
        def __init__(self) -> None:
            self.servo_registry = types.SimpleNamespace(
                servos={
                    "pan": _Servo(-90.0, 90.0),
                    "tilt": _Servo(-45.0, 45.0),
                }
            )

        def generate_base_keyframe(
            self,
            *,
            pan_degrees: float,
            tilt_degrees: float,
            roll_degrees: float = 0.0,
            ear_left_degrees: float = 0.0,
            ear_right_degrees: float = 0.0,
        ):
            frame = gesture_library.Keyframe()
            frame.servo_destination = {
                "pan": pan_degrees,
                "tilt": tilt_degrees,
                "roll": roll_degrees,
                "ear_left": ear_left_degrees,
                "ear_right": ear_right_degrees,
            }
            return frame

    frame = library._create_keyframe(
        _Controller(),
        definition=definition,
        spec=spec,
        base_pan=-55.96,
        base_tilt=25.0,
        transition_pan=-55.96,
        transition_tilt=25.0,
        intensity=1.0,
        style="neutral",
    )

    assert frame.servo_destination["pan"] == 0.0
    assert frame.servo_destination["tilt"] == 0.0


def test_non_look_gesture_keeps_authored_duration() -> None:
    gesture_library = _load_gesture_library_module()
    library = gesture_library.GestureLibrary.__new__(gesture_library.GestureLibrary)
    definition = _definition(gesture_library, "gesture_nod")
    spec = definition.frames[0]

    duration = library._duration_for_frame(
        definition=definition,
        spec=spec,
        target_pan=0.0,
        target_tilt=20.0,
        transition_pan=0.0,
        transition_tilt=-20.0,
        style="solemn",
    )

    assert duration == spec.duration_ms


def test_tilt_offsets_remain_logical_tilt_with_zero_roll() -> None:
    gesture_library = _load_gesture_library_module()
    library = gesture_library.GestureLibrary.__new__(gesture_library.GestureLibrary)
    definition = _definition(gesture_library, "gesture_curious_tilt")
    spec = definition.frames[0]

    class _Servo:
        def __init__(self, minimum: float, maximum: float) -> None:
            self.min_angle = minimum
            self.max_angle = maximum

    class _Controller:
        def __init__(self) -> None:
            self.servo_registry = types.SimpleNamespace(
                servos={
                    "pan": _Servo(-90.0, 90.0),
                    "tilt_left": _Servo(-45.0, 45.0),
                    "tilt_right": _Servo(-45.0, 45.0),
                    "ear_left": _Servo(-90.0, 90.0),
                    "ear_right": _Servo(-90.0, 90.0),
                }
            )

        def generate_base_keyframe(
            self,
            *,
            pan_degrees: float,
            tilt_degrees: float,
            roll_degrees: float = 0.0,
            ear_left_degrees: float = 0.0,
            ear_right_degrees: float = 0.0,
        ):
            frame = gesture_library.Keyframe()
            frame.servo_destination = {
                "pan": pan_degrees,
                "tilt": tilt_degrees,
                "roll": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
            }
            return frame

    frame = library._create_keyframe(
        _Controller(),
        definition=definition,
        spec=spec,
        base_pan=0.0,
        base_tilt=2.0,
        transition_pan=0.0,
        transition_tilt=2.0,
        intensity=1.0,
        style="neutral",
    )

    assert frame.servo_destination["tilt"] == 10.0
    assert frame.servo_destination["roll"] == 0.0
