"""Embodiment policy decisions for realtime state cues."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from ai.attention_continuity import AttentionSnapshot
from ai.governance_spine import GovernanceDecision

from interaction import InteractionState


EMBODIMENT_PRIORITY_ALLOW = 30
EMBODIMENT_PRIORITY_NOOP = 40
EMBODIMENT_PRIORITY_SUPPRESS = 70


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


def embodiment_decision_to_governance(decision: EmbodimentDecision) -> GovernanceDecision:
    """Translate embodiment cue outcomes into the shared governance envelope."""

    # Adapter mapping is intentionally seam-local:
    # - emit cue -> allow (eligible to execute now)
    # - suppress -> suppress (actively blocked by policy/cooldown)
    # - none -> expire/noop (non-emission and not queued for retry)
    # GovernanceDecision.priority here is seam-local/observability metadata and
    # must not be compared across subsystem envelopes without an explicit arbiter.
    if decision.action == EmbodimentActionType.EMIT_CUE:
        envelope_decision = "allow"
        priority = EMBODIMENT_PRIORITY_ALLOW
    elif decision.action == EmbodimentActionType.SUPPRESS:
        envelope_decision = "suppress"
        priority = EMBODIMENT_PRIORITY_SUPPRESS
    elif decision.action == EmbodimentActionType.NONE:
        envelope_decision = "expire"
        priority = EMBODIMENT_PRIORITY_NOOP
    else:
        raise ValueError(f"Unsupported embodiment action for governance mapping: {decision.action}")

    metadata = {
        "action": decision.action.value,
        "result_class": "noop" if decision.action == EmbodimentActionType.NONE else envelope_decision,
        "cue_name": decision.cue_name,
        "delay_ms": decision.delay_ms,
    }
    return GovernanceDecision(
        decision=envelope_decision,
        reason_code=decision.reason,
        subsystem="embodiment",
        priority=priority,
        expires_at=None,
        ttl_s=None,
        metadata=metadata,
    )


class EmbodimentPolicy:
    """Owns deterministic policy for state-driven embodiment cues."""

    _GLOBAL_COOLDOWN_EXEMPT_GESTURES = {
        "gesture_speaking_posture",
        "gesture_speaking_settle",
    }

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
        attention: AttentionSnapshot,
    ) -> EmbodimentDecision:
        if turn_contract_blocks_gestures:
            return EmbodimentDecision(
                action=EmbodimentActionType.SUPPRESS,
                reason="turn_contract_no_gesture",
            )

        if state == InteractionState.THINKING and attention.active:
            return EmbodimentDecision(action=EmbodimentActionType.NONE, reason="attention_continuity_hold")

        gesture_name: str | None = None
        delay_ms = 0
        if state == InteractionState.LISTENING:
            gesture_name = "gesture_attention_hold"
        elif state == InteractionState.THINKING:
            gesture_name = "gesture_curious_tilt"
            delay_ms = random_delay_ms(150, 300)
        elif state == InteractionState.SPEAKING:
            gesture_name = "gesture_speaking_posture"

        if not gesture_name:
            return EmbodimentDecision(action=EmbodimentActionType.NONE, reason="state_no_cue")

        global_elapsed = now_monotonic_s - last_gesture_time_s
        if (
            gesture_name not in self._GLOBAL_COOLDOWN_EXEMPT_GESTURES
            and global_elapsed < gesture_global_cooldown_s
        ):
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
