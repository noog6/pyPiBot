"""Interaction state management and cue hooks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Awaitable, Callable

from config import ConfigController
from core.logging import logger

StateHandler = Callable[["InteractionState"], Awaitable[None] | None]


class InteractionState(str, Enum):
    """Supported interaction states for realtime UX."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class StateCueConfig:
    """Configuration for state cues and timing thresholds."""

    cues_enabled: bool = True
    gesture_enabled: bool = True
    earcon_enabled: bool = False
    min_state_duration_ms: int = 150
    cue_delays_ms: dict[str, int] = field(
        default_factory=lambda: {
            InteractionState.IDLE.value: 0,
            InteractionState.LISTENING.value: 0,
            InteractionState.THINKING.value: 150,
            InteractionState.SPEAKING.value: 0,
        }
    )

    @classmethod
    def from_config(cls) -> "StateCueConfig":
        config = ConfigController.get_instance().get_config()
        state_config = config.get("interaction_states", {})
        cue_delays = state_config.get("cue_delays_ms", {})
        merged_delays = dict(cls().cue_delays_ms)
        for key, value in cue_delays.items():
            if isinstance(value, int):
                merged_delays[key] = value
        return cls(
            cues_enabled=bool(state_config.get("cues_enabled", True)),
            gesture_enabled=bool(state_config.get("gesture_enabled", True)),
            earcon_enabled=bool(state_config.get("earcon_enabled", False)),
            min_state_duration_ms=int(state_config.get("min_state_duration_ms", 150)),
            cue_delays_ms=merged_delays,
        )


class InteractionStateManager:
    """Track interaction state transitions and emit cue hooks."""

    def __init__(
        self,
        cue_config: StateCueConfig | None = None,
    ) -> None:
        self.cue_config = cue_config or StateCueConfig.from_config()
        self.state = InteractionState.IDLE
        self._last_transition = time.monotonic()
        self._last_cue_time = 0.0
        self._pending_task: asyncio.Task[None] | None = None
        self._gesture_handler: StateHandler | None = None
        self._earcon_handler: StateHandler | None = None

    def set_gesture_handler(self, handler: StateHandler | None) -> None:
        self._gesture_handler = handler

    def set_earcon_handler(self, handler: StateHandler | None) -> None:
        self._earcon_handler = handler

    def update_state(self, new_state: InteractionState, reason: str = "") -> bool:
        if new_state == self.state:
            return False

        now = time.monotonic()
        last_state = self.state
        self.state = new_state
        self._last_transition = now
        logger.info(
            "Interaction state transition: %s -> %s%s",
            last_state.value,
            new_state.value,
            f" ({reason})" if reason else "",
        )

        if self._pending_task:
            self._pending_task.cancel()
            self._pending_task = None

        if not self.cue_config.cues_enabled:
            return True

        elapsed_ms = (now - self._last_cue_time) * 1000.0
        if elapsed_ms < self.cue_config.min_state_duration_ms:
            return True

        delay_ms = self.cue_config.cue_delays_ms.get(new_state.value, 0)
        if delay_ms > 0:
            self._pending_task = asyncio.create_task(
                self._emit_cues_after(delay_ms / 1000.0, new_state)
            )
        else:
            self._emit_cues(new_state)

        return True

    async def _emit_cues_after(self, delay_s: float, state: InteractionState) -> None:
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return

        if state == self.state:
            self._emit_cues(state)

    def _emit_cues(self, state: InteractionState) -> None:
        self._last_cue_time = time.monotonic()
        if self.cue_config.gesture_enabled and self._gesture_handler:
            self._dispatch_handler(self._gesture_handler, state, "gesture")
        if self.cue_config.earcon_enabled and self._earcon_handler:
            self._dispatch_handler(self._earcon_handler, state, "earcon")

    def _dispatch_handler(
        self, handler: StateHandler, state: InteractionState, label: str
    ) -> None:
        try:
            result = handler(state)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception:
            logger.exception("Failed to dispatch %s cue for state %s", label, state.value)
