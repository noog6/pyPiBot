"""Camera change detection policy using EMA, hysteresis, and debounce."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CameraInterestState(str, Enum):
    """Interest state for camera change detection."""

    BORING = "boring"
    INTERESTING = "interesting"


@dataclass(frozen=True)
class CameraChangeConfig:
    """Configuration for camera change policy."""

    trigger_threshold: float = 25.0
    clear_threshold: float = 15.0
    debounce_frames: int = 3
    cooldown_seconds: float = 10.0
    ema_alpha: float = 0.3


@dataclass(frozen=True)
class CameraChangeResult:
    """Result of a single policy update."""

    mad: float
    ema_mad: float
    state: CameraInterestState
    state_changed: bool
    debounce_count: int
    promoted: bool
    cooldown_remaining: float


class CameraChangePolicy:
    """State machine for promoting meaningful camera changes."""

    def __init__(self, config: CameraChangeConfig) -> None:
        self._config = config
        self._state = CameraInterestState.BORING
        self._ema_mad: float | None = None
        self._debounce_count = 0
        self._cooldown_until = 0.0

    @property
    def state(self) -> CameraInterestState:
        return self._state

    def reset(self) -> None:
        self._state = CameraInterestState.BORING
        self._ema_mad = None
        self._debounce_count = 0
        self._cooldown_until = 0.0

    def update(self, mad: float, now_s: float) -> CameraChangeResult:
        if self._ema_mad is None:
            ema_mad = mad
        else:
            alpha = self._config.ema_alpha
            ema_mad = alpha * mad + (1.0 - alpha) * self._ema_mad
        self._ema_mad = ema_mad

        promoted = False
        state_changed = False
        debounce_frames = max(self._config.debounce_frames, 1)

        if self._state == CameraInterestState.BORING:
            if ema_mad >= self._config.trigger_threshold:
                self._debounce_count += 1
                if self._debounce_count >= debounce_frames:
                    self._state = CameraInterestState.INTERESTING
                    state_changed = True
                    self._debounce_count = 0
                    if now_s >= self._cooldown_until:
                        promoted = True
                        self._cooldown_until = now_s + self._config.cooldown_seconds
            else:
                self._debounce_count = 0
        else:
            if ema_mad <= self._config.clear_threshold:
                self._debounce_count += 1
                if self._debounce_count >= debounce_frames:
                    self._state = CameraInterestState.BORING
                    state_changed = True
                    self._debounce_count = 0
            else:
                self._debounce_count = 0

        cooldown_remaining = max(0.0, self._cooldown_until - now_s)
        return CameraChangeResult(
            mad=mad,
            ema_mad=ema_mad,
            state=self._state,
            state_changed=state_changed,
            debounce_count=self._debounce_count,
            promoted=promoted,
            cooldown_remaining=cooldown_remaining,
        )
