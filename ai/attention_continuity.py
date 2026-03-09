"""Tiny embodied-attention continuity helper for realtime interaction churn."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionSnapshot:
    """Read-only attention state for policy/logging surfaces."""

    active: bool
    user_speaking: bool
    acquired_at_s: float | None
    hold_until_s: float | None
    release_reason: str


class AttentionContinuity:
    """Tracks short-lived embodied attention continuity across ASR churn windows."""

    def __init__(self, *, hold_window_s: float = 1.25) -> None:
        self._hold_window_s = max(0.0, float(hold_window_s))
        self._active = False
        self._user_speaking = False
        self._acquired_at_s: float | None = None
        self._hold_until_s: float | None = None
        self._release_reason = "never_acquired"

    def acquire(self, *, now_s: float, reason: str) -> AttentionSnapshot:
        if not self._active:
            self._acquired_at_s = now_s
        self._active = True
        self._hold_until_s = None
        self._release_reason = f"active:{reason}"
        return self.snapshot(now_s=now_s)

    def mark_user_speaking(self, *, now_s: float, speaking: bool) -> AttentionSnapshot:
        self._user_speaking = speaking
        if speaking:
            self.acquire(now_s=now_s, reason="user_speaking")
        else:
            self.refresh_hold(now_s=now_s, reason="speech_stopped")
        return self.snapshot(now_s=now_s)

    def refresh_hold(self, *, now_s: float, reason: str) -> AttentionSnapshot:
        if not self._active:
            return self.snapshot(now_s=now_s)
        self._hold_until_s = now_s + self._hold_window_s
        self._release_reason = f"hold:{reason}"
        return self.snapshot(now_s=now_s)

    def release(self, *, now_s: float, reason: str) -> AttentionSnapshot:
        self._active = False
        self._user_speaking = False
        self._hold_until_s = None
        self._release_reason = reason
        return self.snapshot(now_s=now_s)

    def snapshot(self, *, now_s: float) -> AttentionSnapshot:
        if self._active and not self._user_speaking and self._hold_until_s is not None and now_s >= self._hold_until_s:
            self._active = False
            self._hold_until_s = None
            self._release_reason = "hold_expired"
        return AttentionSnapshot(
            active=self._active,
            user_speaking=self._user_speaking,
            acquired_at_s=self._acquired_at_s,
            hold_until_s=self._hold_until_s,
            release_reason=self._release_reason,
        )

