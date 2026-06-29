"""Tests for attention hold/release and lifecycle gesture definitions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_gesture_library_module(monkeypatch):
    fake_motion = types.ModuleType("motion")
    fake_action = types.ModuleType("motion.action")
    fake_keyframe = types.ModuleType("motion.keyframe")
    fake_motion_controller = types.ModuleType("motion.motion_controller")

    class _FakeKeyframe:
        def __init__(self, final_target_time=1500, name="base"):
            self.final_target_time = final_target_time
            self.name = name
            self.servo_destination = {
                "pan": 0.0,
                "tilt": 0.0,
                "roll": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
            }
            self.next = None

    class _FakeController:
        current_servo_position = {
            "pan": 11.0,
            "tilt": 9.0,
            "roll": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,
        }
        transition_time = 1500
        servo_registry = types.SimpleNamespace(
            servos={
                "pan": types.SimpleNamespace(
                    min_angle=-90, max_angle=90, read_value=lambda: 11.0
                ),
                "tilt": types.SimpleNamespace(
                    min_angle=-90, max_angle=90, read_value=lambda: 9.0
                ),
            }
        )

        @classmethod
        def get_instance(cls):
            return cls()

        def get_current_logical_pose(self):
            return dict(self.current_servo_position)

        def generate_base_keyframe(
            self,
            pan_degrees,
            tilt_degrees,
            roll_degrees=0.0,
            ear_left_degrees=0.0,
            ear_right_degrees=0.0,
        ):
            frame = _FakeKeyframe(final_target_time=self.transition_time, name="base")
            frame.servo_destination["pan"] = pan_degrees
            frame.servo_destination["tilt"] = tilt_degrees
            frame.servo_destination["roll"] = roll_degrees
            frame.servo_destination["ear_left"] = ear_left_degrees
            frame.servo_destination["ear_right"] = ear_right_degrees
            return frame

    class _FakeAction:
        def __init__(self, priority, timestamp, name, frames):
            self.priority = priority
            self.timestamp = timestamp
            self.name = name
            self.frames = frames
            self.current_frame = frames

    fake_action.Action = _FakeAction
    fake_keyframe.Keyframe = _FakeKeyframe
    fake_motion_controller.MotionController = _FakeController
    fake_motion_controller.millis = lambda: 0

    monkeypatch.setitem(sys.modules, "motion", fake_motion)
    monkeypatch.setitem(sys.modules, "motion.action", fake_action)
    monkeypatch.setitem(sys.modules, "motion.keyframe", fake_keyframe)
    monkeypatch.setitem(sys.modules, "motion.motion_controller", fake_motion_controller)

    module_path = Path(__file__).resolve().parents[1] / "motion" / "gesture_library.py"
    spec = importlib.util.spec_from_file_location(
        "gesture_library_for_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gesture_library_for_test", module)
    spec.loader.exec_module(module)
    module.GestureLibrary._instance = None
    return module


def _gesture(module, name: str):
    return next(g for g in module.DEFAULT_GESTURES if g.name == name)


def _frames(action):
    frames = []
    frame = action.current_frame
    while frame is not None:
        frames.append(frame)
        frame = frame.next
    return frames


def test_attention_hold_has_no_immediate_recenter_tail(monkeypatch) -> None:
    module = _load_gesture_library_module(monkeypatch)
    hold = _gesture(module, "gesture_attention_hold")
    assert len(hold.frames) == 2
    assert hold.frames[-1].name == "attention-hold"


def test_attention_release_recenters_to_neutral(monkeypatch) -> None:
    module = _load_gesture_library_module(monkeypatch)
    release = _gesture(module, "gesture_attention_release")
    assert release.frames[-1].name == "attention-release-neutral"
    assert release.frames[-1].pan_offset == 0.0
    assert release.frames[-1].tilt_offset == 0.0


def test_attention_release_runtime_final_frame_neutralizes_held_ears(
    monkeypatch,
) -> None:
    module = _load_gesture_library_module(monkeypatch)
    module.MotionController.current_servo_position = {
        "pan": 6.0,
        "tilt": 1.5,
        "roll": 0.0,
        "ear_left": 4.0,
        "ear_right": 4.0,
    }
    library = module.GestureLibrary.get_instance()

    final_frame = _frames(library.build_action("gesture_attention_release"))[-1]

    assert final_frame.name == "attention-release-neutral"
    assert final_frame.servo_destination["pan"] == 0.0
    assert final_frame.servo_destination["tilt"] == 0.0
    assert final_frame.servo_destination["roll"] == 0.0
    assert final_frame.servo_destination["ear_left"] == 0.0
    assert final_frame.servo_destination["ear_right"] == 0.0


def test_speaking_settle_runtime_final_frame_neutralizes_posture_ears(
    monkeypatch,
) -> None:
    module = _load_gesture_library_module(monkeypatch)
    module.MotionController.current_servo_position = {
        "pan": 0.0,
        "tilt": 2.5,
        "roll": 0.0,
        "ear_left": 1.0,
        "ear_right": 1.0,
    }
    library = module.GestureLibrary.get_instance()

    final_frame = _frames(library.build_action("gesture_speaking_settle"))[-1]

    assert final_frame.name == "speaking-settle-neutral"
    assert final_frame.servo_destination["pan"] == 0.0
    assert final_frame.servo_destination["tilt"] == 0.0
    assert final_frame.servo_destination["roll"] == 0.0
    assert final_frame.servo_destination["ear_left"] == 0.0
    assert final_frame.servo_destination["ear_right"] == 0.0


def test_lifecycle_startup_and_shutdown_gestures_preserve_absolute_targets(
    monkeypatch,
) -> None:
    module = _load_gesture_library_module(monkeypatch)
    library = module.GestureLibrary.get_instance()

    startup_action = library.build_action("gesture_startup_presence")
    startup_frames = []
    frame = startup_action.current_frame
    while frame is not None:
        startup_frames.append(
            (
                frame.servo_destination["pan"],
                frame.servo_destination["tilt"],
                frame.final_target_time,
            )
        )
        frame = frame.next

    shutdown_action = library.build_action("gesture_shutdown_rest")
    shutdown_frame = shutdown_action.current_frame

    assert startup_frames == [(0.0, -40.0, 1500), (0.0, 25.0, 1500)]
    assert shutdown_frame is not None
    assert (
        shutdown_frame.servo_destination["pan"],
        shutdown_frame.servo_destination["tilt"],
        shutdown_frame.final_target_time,
    ) == (0.0, -40.0, 1000)


def test_non_lifecycle_idle_center_remains_relative(monkeypatch) -> None:
    module = _load_gesture_library_module(monkeypatch)
    idle = _gesture(module, "gesture_idle")
    assert idle.frames[-1].name == "idle-center"
    assert idle.frames[-1].absolute_target is False


def test_legacy_frame_payload_defaults_new_axes_to_zero(monkeypatch) -> None:
    module = _load_gesture_library_module(monkeypatch)

    spec = module.GestureFrameSpec.from_dict(
        {
            "name": "legacy",
            "pan_offset": 1.0,
            "tilt_offset": 2.0,
            "duration_ms": 300,
        }
    )

    assert spec.roll_offset == 0.0
    assert spec.ear_left_offset == 0.0
    assert spec.ear_right_offset == 0.0


def test_legacy_pan_tilt_gesture_builds_zero_roll_and_ears(monkeypatch) -> None:
    module = _load_gesture_library_module(monkeypatch)
    library = module.GestureLibrary.get_instance()
    library.register(
        module.GestureDefinition(
            name="gesture_legacy_pan_tilt",
            priority=1,
            frames=(
                module.GestureFrameSpec(
                    name="legacy-frame",
                    pan_offset=0.0,
                    tilt_offset=-10.0,
                    duration_ms=350,
                ),
            ),
        ),
        persist=False,
    )

    action = library.build_action("gesture_legacy_pan_tilt")
    frame = action.current_frame

    assert frame.servo_destination["pan"] == 11.0
    assert frame.servo_destination["tilt"] == -1.0
    assert frame.servo_destination["roll"] == 0.0
    assert frame.servo_destination["ear_left"] == 0.0
    assert frame.servo_destination["ear_right"] == 0.0
