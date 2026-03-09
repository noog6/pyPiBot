"""Regression tests for listening attention hold posture lifecycle."""

from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.attention_continuity import AttentionContinuity
from ai.embodiment_policy import EmbodimentActionType, EmbodimentDecision, EmbodimentPolicy
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._attention_continuity = AttentionContinuity(hold_window_s=1.25)
    api._last_interaction_state = InteractionState.IDLE
    api._gesture_last_fired = {}
    api._last_gesture_time = 0.0
    api._gesture_global_cooldown_s = 0.0
    api._gesture_cooldowns_s = {}
    api._listening_attention_hold_active = False
    api._embodiment_policy = EmbodimentPolicy()
    api._turn_contract_blocks_gesture_cues = lambda: False
    return api


def test_listening_hold_starts_once_and_duplicate_is_suppressed(monkeypatch) -> None:
    api = _make_api()

    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.EMIT_CUE,
        reason="state_cue_emission",
        cue_name="gesture_attention_hold",
    )
    emitted: list[tuple[InteractionState, str]] = []
    api._enqueue_gesture_cue = lambda **kwargs: emitted.append((kwargs["state"], kwargs["gesture_name"])) or True

    api._handle_state_gesture(InteractionState.LISTENING)
    api._handle_state_gesture(InteractionState.LISTENING)

    assert emitted == [(InteractionState.LISTENING, "gesture_attention_hold")]
    assert api._listening_attention_hold_active is True
    assert any("attention_hold_started" in msg for msg in messages)


def test_speech_stop_releases_active_hold(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._listening_attention_hold_active = True
    api._enqueue_gesture_cue = lambda **kwargs: True

    api._attention_on_speech_stopped(reason="speech_stopped")

    assert api._listening_attention_hold_active is False
    assert any("attention_hold_released reason=speech_stopped" in msg for msg in messages)


def test_release_not_emitted_when_hold_inactive() -> None:
    api = _make_api()
    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True

    api._emit_attention_hold_release(reason="speech_stopped")

    assert calls == []


def test_speech_stop_logs_deferred_release_when_enqueue_fails(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._listening_attention_hold_active = True
    api._enqueue_gesture_cue = lambda **kwargs: False

    api._attention_on_speech_stopped(reason="speech_stopped")

    assert api._listening_attention_hold_active is True
    assert any("attention_hold_released status=deferred reason=speech_stopped" in msg for msg in messages)
