"""Tests for PandaTheo logical-to-physical body mapping."""

from __future__ import annotations

from motion.body_model import (
    LOGICAL_POSE_AXES,
    PHYSICAL_SERVO_NAMES,
    compose_physical_targets,
    map_logical_pose_to_physical_targets,
    normalize_logical_pose,
)

SERVO_LIMITS = {
    "pan": (-90.0, 90.0),
    "tilt_left": (-45.0, 45.0),
    "tilt_right": (-45.0, 45.0),
    "ear_left": (-90.0, 90.0),
    "ear_right": (-90.0, 90.0),
}


def test_body_model_declares_logical_and_physical_axes() -> None:
    assert LOGICAL_POSE_AXES == ("pan", "tilt", "roll", "ear_left", "ear_right")
    assert PHYSICAL_SERVO_NAMES == (
        "pan",
        "tilt_left",
        "tilt_right",
        "ear_left",
        "ear_right",
    )


def test_normalize_logical_pose_defaults_missing_axes_to_zero() -> None:
    assert normalize_logical_pose({"pan": 12.0, "tilt": -3.0}) == {
        "pan": 12.0,
        "tilt": -3.0,
        "roll": 0.0,
        "ear_left": 0.0,
        "ear_right": 0.0,
    }


def test_logical_tilt_maps_to_both_physical_tilt_servos() -> None:
    targets, clamp_events = map_logical_pose_to_physical_targets(
        {"tilt": 5.0}, SERVO_LIMITS
    )

    assert targets["tilt_left"] == 5.0
    assert targets["tilt_right"] == 5.0
    assert clamp_events == ()


def test_logical_roll_maps_to_opposed_physical_tilt_servos() -> None:
    targets, clamp_events = map_logical_pose_to_physical_targets(
        {"roll": 5.0}, SERVO_LIMITS
    )

    assert targets["tilt_left"] == 5.0
    assert targets["tilt_right"] == -5.0
    assert clamp_events == ()


def test_tilt_and_roll_compose_before_physical_clamping() -> None:
    composed = compose_physical_targets({"tilt": 40.0, "roll": 10.0})
    targets, clamp_events = map_logical_pose_to_physical_targets(
        {"tilt": 40.0, "roll": 10.0}, SERVO_LIMITS
    )

    assert composed["tilt_left"] == 50.0
    assert composed["tilt_right"] == 30.0
    assert targets["tilt_left"] == 45.0
    assert targets["tilt_right"] == 30.0
    assert len(clamp_events) == 1
    assert clamp_events[0].servo_name == "tilt_left"
    assert clamp_events[0].requested == 50.0
    assert clamp_events[0].clamped == 45.0


def test_ears_pass_through_to_physical_targets() -> None:
    targets, clamp_events = map_logical_pose_to_physical_targets(
        {"ear_left": 12.0, "ear_right": -8.0}, SERVO_LIMITS
    )

    assert targets["ear_left"] == 12.0
    assert targets["ear_right"] == -8.0
    assert clamp_events == ()
