"""Unit tests for embodiment state-cue policy decisions."""

from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.attention_continuity import AttentionSnapshot
from ai.embodiment_policy import EmbodimentActionType, EmbodimentPolicy
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
