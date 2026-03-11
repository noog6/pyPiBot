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
