"""Phase 2 PandaTheo gesture accent coverage."""

from __future__ import annotations

import motion.gestures as public_gestures
from motion.body_model import compose_physical_targets
from motion.gesture_library import DEFAULT_GESTURES, GestureDefinition, GestureLibrary

SAFE_TILT_LIMIT = 45.0
PUBLIC_GESTURE_NAMES = {
    "gesture_idle",
    "gesture_nod",
    "gesture_no",
    "gesture_look_around",
    "gesture_look_up",
    "gesture_look_left",
    "gesture_look_right",
    "gesture_look_down",
    "gesture_look_center",
    "gesture_curious_tilt",
    "gesture_speaking_posture",
    "gesture_speaking_settle",
    "gesture_attention_hold",
    "gesture_attention_release",
    "gesture_attention_snap",
    "gesture_startup_presence",
    "gesture_shutdown_rest",
}


def _definition(name: str) -> GestureDefinition:
    return next(definition for definition in DEFAULT_GESTURES if definition.name == name)


def test_attention_hold_uses_symmetric_ear_alert() -> None:
    hold = _definition("gesture_attention_hold")

    assert any(frame.ear_left_offset != 0.0 for frame in hold.frames)
    assert any(frame.ear_right_offset != 0.0 for frame in hold.frames)
    assert all(frame.ear_left_offset == frame.ear_right_offset for frame in hold.frames)
    assert all(frame.roll_offset == 0.0 for frame in hold.frames)


def test_attention_release_returns_roll_and_ears_toward_neutral() -> None:
    release = _definition("gesture_attention_release")
    neutral = release.frames[-1]

    assert neutral.name == "attention-release-neutral"
    assert neutral.roll_offset == 0.0
    assert neutral.ear_left_offset == 0.0
    assert neutral.ear_right_offset == 0.0


def test_curious_tilt_uses_roll_and_asymmetric_ears() -> None:
    curious = _definition("gesture_curious_tilt")

    assert any(frame.roll_offset != 0.0 for frame in curious.frames)
    assert any(
        frame.ear_left_offset != frame.ear_right_offset
        for frame in curious.frames
        if frame.ear_left_offset != 0.0 or frame.ear_right_offset != 0.0
    )


def test_startup_presence_composed_tilts_stay_inside_safe_limits() -> None:
    startup = _definition("gesture_startup_presence")

    for frame in startup.frames:
        physical = compose_physical_targets(
            {"tilt": frame.tilt_offset, "roll": frame.roll_offset}
        )
        assert abs(physical["tilt_left"]) < SAFE_TILT_LIMIT
        assert abs(physical["tilt_right"]) < SAFE_TILT_LIMIT


def test_look_up_and_look_down_do_not_roll_near_tilt_limits() -> None:
    for gesture_name in ("gesture_look_up", "gesture_look_down"):
        definition = _definition(gesture_name)
        assert all(frame.roll_offset == 0.0 for frame in definition.frames)


def test_look_up_and_look_down_use_absolute_safe_margin_targets() -> None:
    look_up = _definition("gesture_look_up").frames[0]
    look_down = _definition("gesture_look_down").frames[0]

    assert look_up.absolute_target is True
    assert look_up.tilt_offset == 44.0
    assert look_down.absolute_target is True
    assert look_down.tilt_offset == -44.0


def test_direct_horizontal_looks_do_not_leave_persistent_ear_offsets() -> None:
    for gesture_name in ("gesture_look_left", "gesture_look_right"):
        definition = _definition(gesture_name)
        assert all(frame.ear_left_offset == 0.0 for frame in definition.frames)
        assert all(frame.ear_right_offset == 0.0 for frame in definition.frames)


def test_look_center_returns_roll_and_ears_to_neutral() -> None:
    center = _definition("gesture_look_center")
    frame = center.frames[-1]

    assert frame.absolute_target is True
    assert frame.pan_offset == 0.0
    assert frame.tilt_offset == 0.0
    assert frame.roll_offset == 0.0
    assert frame.ear_left_offset == 0.0
    assert frame.ear_right_offset == 0.0


def test_default_gesture_frames_stay_inside_physical_tilt_safety_bounds() -> None:
    for definition in DEFAULT_GESTURES:
        if definition.name in {"gesture_look_up", "gesture_look_down"}:
            continue
        for frame in definition.frames:
            if frame.absolute_target:
                tilt = frame.tilt_offset
                roll = frame.roll_offset
            else:
                tilt = frame.tilt_offset
                roll = frame.roll_offset
            physical = compose_physical_targets({"tilt": tilt, "roll": roll})
            assert abs(physical["tilt_left"]) < SAFE_TILT_LIMIT, (
                definition.name,
                frame.name,
            )
            assert abs(physical["tilt_right"]) < SAFE_TILT_LIMIT, (
                definition.name,
                frame.name,
            )


def test_old_public_gesture_names_remain_available() -> None:
    assert {definition.name for definition in DEFAULT_GESTURES} == PUBLIC_GESTURE_NAMES
    for gesture_name in PUBLIC_GESTURE_NAMES:
        assert hasattr(public_gestures, gesture_name)


def test_legacy_pan_tilt_frame_payload_compatibility_is_preserved() -> None:
    payload = {
        "name": "legacy-pan-tilt",
        "priority": 1,
        "frames": [
            {
                "name": "legacy-frame",
                "pan_offset": 3.0,
                "tilt_offset": -2.0,
                "duration_ms": 250,
            }
        ],
    }

    definition = GestureDefinition.from_dict(payload)
    frame = definition.frames[0]

    assert frame.roll_offset == 0.0
    assert frame.ear_left_offset == 0.0
    assert frame.ear_right_offset == 0.0
    assert frame.reset_expression_axes is False


def test_roll_and_ear_fields_round_trip_through_serialization() -> None:
    curious = _definition("gesture_curious_tilt")

    restored = GestureDefinition.from_dict(curious.to_dict())

    assert restored == curious
    assert any(frame.roll_offset != 0.0 for frame in restored.frames)
    assert any(frame.ear_left_offset != 0.0 for frame in restored.frames)
    assert any(frame.ear_right_offset != 0.0 for frame in restored.frames)


def test_ensure_defaults_refreshes_persisted_default_gesture_definitions(
    tmp_path,
) -> None:
    library = GestureLibrary.__new__(GestureLibrary)
    old_hold = GestureDefinition.from_dict(
        {
            "name": "gesture_attention_hold",
            "priority": 2,
            "frames": [
                {
                    "name": "attention-hold",
                    "pan_offset": 6.0,
                    "tilt_offset": 1.5,
                    "duration_ms": 900,
                }
            ],
        }
    )
    library._definitions = {"gesture_attention_hold": old_hold}
    library._library_path = tmp_path / "gesture_library.json"

    library.ensure_defaults()

    refreshed = library.get("gesture_attention_hold")
    assert refreshed == _definition("gesture_attention_hold")
    assert any(frame.ear_left_offset != 0.0 for frame in refreshed.frames)
