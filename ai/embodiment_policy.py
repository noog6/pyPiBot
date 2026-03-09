"""Embodiment policy decisions for realtime state cues."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from interaction import InteractionState


class EmbodimentActionType(str, Enum):
    """Policy-level action types for embodiment decisions."""

    NONE = "none"
    EMIT_CUE = "emit_cue"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class EmbodimentDecision:
    """Decision payload returned by the embodiment policy."""

    action: EmbodimentActionType
    reason: str
    cue_name: str | None = None
    delay_ms: int = 0


class EmbodimentPolicy:
    """Owns deterministic policy for state-driven embodiment cues."""

    def decide_state_cue(
        self,
        *,
        state: InteractionState,
        previous_state: InteractionState,
        turn_contract_blocks_gestures: bool,
        now_monotonic_s: float,
        last_gesture_time_s: float,
        gesture_global_cooldown_s: float,
        gesture_name_last_fired_s: dict[str, float],
        gesture_cooldowns_s: dict[str, float],
        random_delay_ms: Callable[[int, int], int],
    ) -> EmbodimentDecision:
        if turn_contract_blocks_gestures:
            return EmbodimentDecision(
                action=EmbodimentActionType.SUPPRESS,
                reason="turn_contract_no_gesture",
            )

        if state == InteractionState.SPEAKING:
            return EmbodimentDecision(action=EmbodimentActionType.NONE, reason="state_speaking")

        gesture_name: str | None = None
        delay_ms = 0
        if state == InteractionState.LISTENING:
            gesture_name = "gesture_attention_snap"
        elif state == InteractionState.THINKING:
            gesture_name = "gesture_curious_tilt"
            delay_ms = random_delay_ms(150, 300)
        elif state == InteractionState.IDLE and previous_state == InteractionState.SPEAKING:
            gesture_name = "gesture_nod"

        if not gesture_name:
            return EmbodimentDecision(action=EmbodimentActionType.NONE, reason="state_no_cue")

        global_elapsed = now_monotonic_s - last_gesture_time_s
        if global_elapsed < gesture_global_cooldown_s:
            return EmbodimentDecision(
                action=EmbodimentActionType.SUPPRESS,
                reason="global_cooldown_active",
                cue_name=gesture_name,
                delay_ms=delay_ms,
            )

        per_cooldown = gesture_cooldowns_s.get(gesture_name, 0.0)
        last_fired = gesture_name_last_fired_s.get(gesture_name, 0.0)
        per_elapsed = now_monotonic_s - last_fired
        if per_elapsed < per_cooldown:
            return EmbodimentDecision(
                action=EmbodimentActionType.SUPPRESS,
                reason="gesture_cooldown_active",
                cue_name=gesture_name,
                delay_ms=delay_ms,
            )

        return EmbodimentDecision(
            action=EmbodimentActionType.EMIT_CUE,
            reason="state_cue_emission",
            cue_name=gesture_name,
            delay_ms=delay_ms,
        )
