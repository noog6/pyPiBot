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
    reason_codes: tuple[str, ...] = ()
    posture: str = "suppressed"
    allow_nonessential_gesture: bool = False


@dataclass(frozen=True)
class SituatedPostureDecision:
    """First-class, deterministic posture decision for runtime state cues."""

    posture: str
    cue_name: str | None
    allow_nonessential_gesture: bool
    suppress_reason: str | None
    reason_codes: tuple[str, ...]
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
        "posture": decision.posture,
        "allow_nonessential_gesture": decision.allow_nonessential_gesture,
        "reason_codes": list(decision.reason_codes),
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




class LifecyclePostureEvent(str, Enum):
    """Lifecycle-owned events eligible for situated embodiment cue selection."""

    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class SituationalCueEvent(str, Enum):
    """Situational events eligible for deterministic cue selection."""

    SPEECH_STOPPED_ACK = "speech_stopped_ack"
    DIRECT_ADDRESS_ACK = "direct_address_ack"


class EmbodimentPolicy:
    """Owns deterministic policy for state-driven embodiment cues."""

    _GLOBAL_COOLDOWN_EXEMPT_GESTURES = {
        "gesture_speaking_posture",
        "gesture_speaking_settle",
    }

    def decide_situational_cue(
        self,
        *,
        event: str | SituationalCueEvent,
        interaction_state: InteractionState,
        response_in_flight: bool,
        motion_busy: bool,
        turn_contract_blocks_gestures: bool,
        followthrough_active: bool,
    ) -> SituatedPostureDecision:
        """Map turn-boundary situations to deterministic embodied cue selections."""
        if response_in_flight or interaction_state == InteractionState.SPEAKING:
            return SituatedPostureDecision(
                posture="suppressed",
                cue_name=None,
                allow_nonessential_gesture=False,
                suppress_reason="situational_cue_suppressed_response_in_flight",
                reason_codes=("situational_cue_suppressed_response_in_flight",),
            )
        if motion_busy:
            return SituatedPostureDecision(
                posture="suppressed",
                cue_name=None,
                allow_nonessential_gesture=False,
                suppress_reason="situational_cue_suppressed_motion_busy",
                reason_codes=("situational_cue_suppressed_motion_busy",),
            )
        if turn_contract_blocks_gestures:
            return SituatedPostureDecision(
                posture="suppressed",
                cue_name=None,
                allow_nonessential_gesture=False,
                suppress_reason="situational_cue_suppressed_turn_contract",
                reason_codes=("situational_cue_suppressed_turn_contract",),
            )
        if followthrough_active:
            return SituatedPostureDecision(
                posture="suppressed",
                cue_name=None,
                allow_nonessential_gesture=False,
                suppress_reason="situational_cue_suppressed_followthrough",
                reason_codes=("situational_cue_suppressed_followthrough",),
            )
        event_value = event.value if isinstance(event, SituationalCueEvent) else str(event).strip().lower()
        if event_value == SituationalCueEvent.SPEECH_STOPPED_ACK.value:
            return SituatedPostureDecision(
                posture="situational",
                cue_name="gesture_nod",
                allow_nonessential_gesture=True,
                suppress_reason=None,
                reason_codes=("speech_stopped_ack",),
            )
        if event_value == SituationalCueEvent.DIRECT_ADDRESS_ACK.value:
            return SituatedPostureDecision(
                posture="situational",
                cue_name="gesture_attention_snap",
                allow_nonessential_gesture=True,
                suppress_reason=None,
                reason_codes=("direct_address_ack",),
            )
        return SituatedPostureDecision(
            posture="idle",
            cue_name=None,
            allow_nonessential_gesture=False,
            suppress_reason="situational_event_no_cue",
            reason_codes=("situational_event_no_cue",),
        )

    def decide_lifecycle_posture(
        self,
        *,
        event: str | LifecyclePostureEvent,
    ) -> SituatedPostureDecision:
        """Map lifecycle situations to named, deterministic embodied cue selections."""

        event_value = event.value if isinstance(event, LifecyclePostureEvent) else str(event).strip().lower()
        if event_value == LifecyclePostureEvent.STARTUP.value:
            return SituatedPostureDecision(
                posture="startup",
                cue_name="gesture_startup_presence",
                allow_nonessential_gesture=False,
                suppress_reason=None,
                reason_codes=("lifecycle_startup_presence",),
            )
        if event_value == LifecyclePostureEvent.SHUTDOWN.value:
            return SituatedPostureDecision(
                posture="shutdown",
                cue_name="gesture_shutdown_rest",
                allow_nonessential_gesture=False,
                suppress_reason=None,
                reason_codes=("lifecycle_shutdown_settle",),
            )
        return SituatedPostureDecision(
            posture="idle",
            cue_name=None,
            allow_nonessential_gesture=False,
            suppress_reason="lifecycle_event_no_cue",
            reason_codes=("lifecycle_event_no_cue",),
        )

    def decide_posture(
        self,
        *,
        state: InteractionState,
        turn_contract_blocks_gestures: bool,
        attention: AttentionSnapshot,
        random_delay_ms: Callable[[int, int], int],
    ) -> SituatedPostureDecision:
        """Returns a deterministic posture/gesture-family decision from runtime facts."""
        if turn_contract_blocks_gestures:
            return SituatedPostureDecision(
                posture="followthrough",
                cue_name=None,
                allow_nonessential_gesture=False,
                suppress_reason="turn_contract_no_gesture",
                reason_codes=("followthrough_nonessential_suppressed",),
            )

        if state == InteractionState.THINKING and attention.active:
            return SituatedPostureDecision(
                posture="thinking",
                cue_name=None,
                allow_nonessential_gesture=False,
                suppress_reason="attention_continuity_hold",
                reason_codes=("attention_continuity_hold",),
            )

        if state == InteractionState.LISTENING:
            return SituatedPostureDecision(
                posture="listening",
                cue_name="gesture_attention_hold",
                allow_nonessential_gesture=True,
                suppress_reason=None,
                reason_codes=("state_listening_attention_hold",),
            )
        if state == InteractionState.THINKING:
            return SituatedPostureDecision(
                posture="thinking",
                cue_name="gesture_curious_tilt",
                allow_nonessential_gesture=True,
                suppress_reason=None,
                reason_codes=("state_thinking_curious_tilt",),
                delay_ms=random_delay_ms(150, 300),
            )
        if state == InteractionState.SPEAKING:
            return SituatedPostureDecision(
                posture="speaking",
                cue_name="gesture_speaking_posture",
                allow_nonessential_gesture=True,
                suppress_reason=None,
                reason_codes=("state_speaking_posture",),
            )
        return SituatedPostureDecision(
            posture="idle",
            cue_name=None,
            allow_nonessential_gesture=False,
            suppress_reason="state_no_cue",
            reason_codes=("no_state_posture",),
        )

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
        posture_decision = self.decide_posture(
            state=state,
            turn_contract_blocks_gestures=turn_contract_blocks_gestures,
            attention=attention,
            random_delay_ms=random_delay_ms,
        )
        if posture_decision.suppress_reason == "turn_contract_no_gesture":
            return EmbodimentDecision(
                action=EmbodimentActionType.SUPPRESS,
                reason="turn_contract_no_gesture",
                reason_codes=posture_decision.reason_codes,
                posture=posture_decision.posture,
                allow_nonessential_gesture=posture_decision.allow_nonessential_gesture,
            )
        if posture_decision.suppress_reason is not None:
            return EmbodimentDecision(
                action=EmbodimentActionType.NONE,
                reason=posture_decision.suppress_reason,
                reason_codes=posture_decision.reason_codes,
                posture=posture_decision.posture,
                allow_nonessential_gesture=posture_decision.allow_nonessential_gesture,
            )
        gesture_name = posture_decision.cue_name
        delay_ms = posture_decision.delay_ms

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
                reason_codes=("cooldown_suppressed",),
                posture=posture_decision.posture,
                allow_nonessential_gesture=False,
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
                reason_codes=("cooldown_suppressed",),
                posture=posture_decision.posture,
                allow_nonessential_gesture=False,
            )

        return EmbodimentDecision(
            action=EmbodimentActionType.EMIT_CUE,
            reason="state_cue_emission",
            cue_name=gesture_name,
            delay_ms=delay_ms,
            reason_codes=posture_decision.reason_codes,
            posture=posture_decision.posture,
            allow_nonessential_gesture=posture_decision.allow_nonessential_gesture,
        )
