"""Tests for attention hold/release gesture definitions."""

from motion.gesture_library import DEFAULT_GESTURES


def _gesture(name: str):
    return next(g for g in DEFAULT_GESTURES if g.name == name)


def test_attention_hold_has_no_immediate_recenter_tail() -> None:
    hold = _gesture("gesture_attention_hold")

    assert len(hold.frames) == 2
    assert hold.frames[-1].name == "attention-hold"
    assert hold.frames[-1].pan_offset != 0.0 or hold.frames[-1].tilt_offset != 0.0


def test_attention_release_recenters_to_neutral() -> None:
    release = _gesture("gesture_attention_release")

    assert release.frames[-1].name == "attention-release-neutral"
    assert release.frames[-1].pan_offset == 0.0
    assert release.frames[-1].tilt_offset == 0.0
