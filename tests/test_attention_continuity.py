"""Unit tests for embodied attention continuity helper."""

from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.attention_continuity import AttentionContinuity


def test_attention_acquire_and_hold_refresh() -> None:
    continuity = AttentionContinuity(hold_window_s=1.0)

    acquired = continuity.acquire(now_s=10.0, reason="listening")
    assert acquired.active is True
    assert acquired.acquired_at_s == 10.0
    assert acquired.release_reason == "active:listening"

    held = continuity.mark_user_speaking(now_s=11.0, speaking=False)
    assert held.active is True
    assert held.hold_until_s == 12.0
    assert held.release_reason == "hold:speech_stopped"


def test_attention_expires_after_hold_window() -> None:
    continuity = AttentionContinuity(hold_window_s=1.0)

    continuity.acquire(now_s=10.0, reason="listening")
    continuity.refresh_hold(now_s=10.5, reason="transcript_churn")

    still_active = continuity.snapshot(now_s=11.0)
    assert still_active.active is True

    expired = continuity.snapshot(now_s=11.6)
    assert expired.active is False
    assert expired.release_reason == "hold_expired"


def test_attention_releases_on_terminal_state() -> None:
    continuity = AttentionContinuity(hold_window_s=1.0)
    continuity.acquire(now_s=2.0, reason="listening")

    released = continuity.release(now_s=3.0, reason="state_speaking")
    assert released.active is False
    assert released.user_speaking is False
    assert released.release_reason == "state_speaking"
