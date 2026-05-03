"""Unit tests for embodiment state-cue policy decisions."""

from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.attention_continuity import AttentionSnapshot
from ai.embodiment_policy import (
    EmbodimentActionType,
    EmbodimentDecision,
    EmbodimentPolicy,
    LifecyclePostureEvent,
    SituationalCueEvent,
    embodiment_decision_to_governance,
)
from interaction import InteractionState


def _attention(active: bool = False) -> AttentionSnapshot:
    return AttentionSnapshot(
        active=active,
        user_speaking=False,
        acquired_at_s=10.0 if active else None,
        hold_until_s=11.0 if active else None,
        release_reason="hold:transcript_churn" if active else "never_acquired",
    )


def test_decide_state_cue_suppresses_when_turn_contract_blocks() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.LISTENING,
        previous_state=InteractionState.IDLE,
        turn_contract_blocks_gestures=True,
        now_monotonic_s=100.0,
        last_gesture_time_s=90.0,
        gesture_global_cooldown_s=1.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={},
        random_delay_ms=lambda _low, _high: 200,
        attention=_attention(),
    )

    assert decision.action == EmbodimentActionType.SUPPRESS
    assert decision.reason == "turn_contract_no_gesture"


def test_decide_state_cue_emits_expected_gesture_for_listening() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.LISTENING,
        previous_state=InteractionState.IDLE,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=100.0,
        last_gesture_time_s=0.0,
        gesture_global_cooldown_s=10.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={"gesture_attention_hold": 10.0},
        random_delay_ms=lambda _low, _high: 200,
        attention=_attention(),
    )

    assert decision.action == EmbodimentActionType.EMIT_CUE
    assert decision.cue_name == "gesture_attention_hold"
    assert decision.delay_ms == 0
    assert "state_listening_attention_hold" in decision.reason_codes


def test_decide_state_cue_suppresses_when_global_cooldown_active() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.LISTENING,
        previous_state=InteractionState.IDLE,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=105.0,
        last_gesture_time_s=100.0,
        gesture_global_cooldown_s=10.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={"gesture_attention_hold": 10.0},
        random_delay_ms=lambda _low, _high: 200,
        attention=_attention(),
    )

    assert decision.action == EmbodimentActionType.SUPPRESS
    assert decision.reason == "global_cooldown_active"
    assert decision.cue_name == "gesture_attention_hold"


def test_decide_state_cue_uses_delay_for_thinking_cue() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.THINKING,
        previous_state=InteractionState.LISTENING,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=100.0,
        last_gesture_time_s=0.0,
        gesture_global_cooldown_s=10.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={"gesture_curious_tilt": 6.0},
        random_delay_ms=lambda _low, _high: 222,
        attention=_attention(),
    )

    assert decision.action == EmbodimentActionType.EMIT_CUE
    assert decision.cue_name == "gesture_curious_tilt"
    assert decision.delay_ms == 222
    assert "state_thinking_curious_tilt" in decision.reason_codes


def test_decide_state_cue_suppresses_thinking_gesture_when_attention_is_held() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.THINKING,
        previous_state=InteractionState.LISTENING,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=100.0,
        last_gesture_time_s=0.0,
        gesture_global_cooldown_s=10.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={"gesture_curious_tilt": 6.0},
        random_delay_ms=lambda _low, _high: 222,
        attention=_attention(active=True),
    )

    assert decision.action == EmbodimentActionType.NONE
    assert decision.reason == "attention_continuity_hold"
    assert decision.reason_codes == ("attention_continuity_hold",)


def test_decide_state_cue_emits_speaking_posture() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.SPEAKING,
        previous_state=InteractionState.THINKING,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=100.0,
        last_gesture_time_s=0.0,
        gesture_global_cooldown_s=10.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={"gesture_speaking_posture": 3.0},
        random_delay_ms=lambda _low, _high: 222,
        attention=_attention(),
    )

    assert decision.action == EmbodimentActionType.EMIT_CUE
    assert decision.cue_name == "gesture_speaking_posture"
    assert decision.delay_ms == 0
    assert "state_speaking_posture" in decision.reason_codes


def test_decide_posture_followthrough_suppresses_nonessential_gestures() -> None:
    policy = EmbodimentPolicy()
    posture = policy.decide_posture(
        state=InteractionState.THINKING,
        turn_contract_blocks_gestures=True,
        attention=_attention(),
        random_delay_ms=lambda _low, _high: 200,
    )
    assert posture.allow_nonessential_gesture is False
    assert posture.suppress_reason == "turn_contract_no_gesture"
    assert "followthrough_nonessential_suppressed" in posture.reason_codes


def test_decide_state_cue_speaking_posture_ignores_global_cooldown() -> None:
    policy = EmbodimentPolicy()

    decision = policy.decide_state_cue(
        state=InteractionState.SPEAKING,
        previous_state=InteractionState.THINKING,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=105.0,
        last_gesture_time_s=104.5,
        gesture_global_cooldown_s=2.0,
        gesture_name_last_fired_s={"gesture_curious_tilt": 104.5},
        gesture_cooldowns_s={"gesture_speaking_posture": 3.0},
        random_delay_ms=lambda _low, _high: 222,
        attention=_attention(),
    )

    assert decision.action == EmbodimentActionType.EMIT_CUE
    assert decision.reason == "state_cue_emission"
    assert decision.cue_name == "gesture_speaking_posture"


def test_embodiment_decision_adapter_returns_governance_envelope() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_state_cue(
        state=InteractionState.LISTENING,
        previous_state=InteractionState.IDLE,
        turn_contract_blocks_gestures=False,
        now_monotonic_s=100.0,
        last_gesture_time_s=0.0,
        gesture_global_cooldown_s=10.0,
        gesture_name_last_fired_s={},
        gesture_cooldowns_s={"gesture_attention_hold": 10.0},
        random_delay_ms=lambda _low, _high: 200,
        attention=_attention(),
    )

    envelope = embodiment_decision_to_governance(decision)

    assert envelope.decision == "allow"
    assert envelope.reason_code == "state_cue_emission"
    assert envelope.subsystem == "embodiment"
    assert envelope.metadata["cue_name"] == "gesture_attention_hold"
    assert envelope.metadata["posture"] == "listening"
    assert envelope.metadata["allow_nonessential_gesture"] is True
    assert envelope.metadata["reason_codes"] == ["state_listening_attention_hold"]


def test_embodiment_adapter_maps_none_to_noop_semantics() -> None:
    envelope = embodiment_decision_to_governance(
        EmbodimentDecision(action=EmbodimentActionType.NONE, reason="Attention Continuity Hold")
    )

    assert envelope.decision == "expire"
    assert envelope.reason_code == "attention_continuity_hold"
    assert envelope.metadata["result_class"] == "noop"


def test_embodiment_adapter_keeps_emit_and_suppress_mappings_stable() -> None:
    emitted = embodiment_decision_to_governance(
        EmbodimentDecision(
            action=EmbodimentActionType.EMIT_CUE,
            reason="state_cue_emission",
            cue_name="gesture_attention_hold",
        )
    )
    suppressed = embodiment_decision_to_governance(
        EmbodimentDecision(
            action=EmbodimentActionType.SUPPRESS,
            reason="global_cooldown_active",
            cue_name="gesture_attention_hold",
        )
    )

    assert emitted.decision == "allow"
    assert emitted.metadata["result_class"] == "allow"
    assert suppressed.decision == "suppress"
    assert suppressed.metadata["result_class"] == "suppress"


def test_embodiment_adapter_raises_for_unsupported_action() -> None:
    class _UnsupportedAction:
        value = "unsupported"

    try:
        embodiment_decision_to_governance(
            EmbodimentDecision(action=_UnsupportedAction(), reason="unexpected_action")  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "Unsupported embodiment action" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported embodiment action")


def test_decide_lifecycle_posture_maps_startup_event() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_lifecycle_posture(event=LifecyclePostureEvent.STARTUP)
    assert decision.posture == "startup"
    assert decision.cue_name == "gesture_startup_presence"
    assert decision.reason_codes == ("lifecycle_startup_presence",)


def test_decide_lifecycle_posture_maps_shutdown_event() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_lifecycle_posture(event="shutdown")
    assert decision.posture == "shutdown"
    assert decision.cue_name == "gesture_shutdown_rest"
    assert decision.reason_codes == ("lifecycle_shutdown_settle",)


def test_decide_lifecycle_posture_unmapped_event_returns_noop() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_lifecycle_posture(event="reboot")
    assert decision.cue_name is None
    assert decision.suppress_reason == "lifecycle_event_no_cue"
    assert decision.reason_codes == ("lifecycle_event_no_cue",)


def test_decide_situational_cue_maps_speech_stopped_ack() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_situational_cue(
        event=SituationalCueEvent.SPEECH_STOPPED_ACK,
        interaction_state=InteractionState.LISTENING,
        response_in_flight=False,
        motion_busy=False,
        turn_contract_blocks_gestures=False,
        followthrough_active=False,
    )
    assert decision.cue_name == "gesture_nod"
    assert decision.reason_codes == ("speech_stopped_ack",)


def test_decide_situational_cue_maps_direct_address_ack() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_situational_cue(
        event=SituationalCueEvent.DIRECT_ADDRESS_ACK,
        interaction_state=InteractionState.LISTENING,
        response_in_flight=False,
        motion_busy=False,
        turn_contract_blocks_gestures=False,
        followthrough_active=False,
    )
    assert decision.cue_name == "gesture_attention_snap"
    assert decision.reason_codes == ("direct_address_ack",)


def test_decide_situational_cue_suppresses_on_response_in_flight() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_situational_cue(
        event=SituationalCueEvent.SPEECH_STOPPED_ACK,
        interaction_state=InteractionState.LISTENING,
        response_in_flight=True,
        motion_busy=False,
        turn_contract_blocks_gestures=False,
        followthrough_active=False,
    )
    assert decision.cue_name is None
    assert decision.suppress_reason == "situational_cue_suppressed_response_in_flight"


def test_decide_situational_cue_suppresses_on_turn_contract_and_motion_busy() -> None:
    policy = EmbodimentPolicy()
    blocked = policy.decide_situational_cue(
        event=SituationalCueEvent.SPEECH_STOPPED_ACK,
        interaction_state=InteractionState.LISTENING,
        response_in_flight=False,
        motion_busy=False,
        turn_contract_blocks_gestures=True,
        followthrough_active=False,
    )
    assert blocked.suppress_reason == "situational_cue_suppressed_turn_contract"

    motion_busy = policy.decide_situational_cue(
        event=SituationalCueEvent.SPEECH_STOPPED_ACK,
        interaction_state=InteractionState.LISTENING,
        response_in_flight=False,
        motion_busy=True,
        turn_contract_blocks_gestures=False,
        followthrough_active=False,
    )
    assert motion_busy.suppress_reason == "situational_cue_suppressed_motion_busy"


def test_decide_situational_cue_suppresses_on_followthrough_active() -> None:
    policy = EmbodimentPolicy()
    decision = policy.decide_situational_cue(
        event=SituationalCueEvent.DIRECT_ADDRESS_ACK,
        interaction_state=InteractionState.LISTENING,
        response_in_flight=False,
        motion_busy=False,
        turn_contract_blocks_gestures=False,
        followthrough_active=True,
    )
    assert decision.cue_name is None
    assert decision.suppress_reason == "situational_cue_suppressed_followthrough"
