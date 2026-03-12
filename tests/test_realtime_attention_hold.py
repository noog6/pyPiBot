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
    api._speaking_posture_episode_active = False
    api._speaking_settle_deferred = False
    api._speaking_settle_deferred_at = None
    api._embodiment_policy = EmbodimentPolicy()
    api._pending_image_flush_after_playback = False
    api._pending_image_stimulus = None
    api._turn_contract_blocks_gesture_cues = lambda: False
    return api




def test_embodiment_governance_logs_noop_result_class_for_none(monkeypatch) -> None:
    api = _make_api()
    debug_messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.debug", lambda msg, *args: debug_messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="attention_continuity_hold",
    )

    api._handle_state_gesture(InteractionState.THINKING)

    assert any(
        "embodiment_governance" in msg
        and "turn_id=turn-unknown" in msg
        and "subsystem=embodiment" in msg
        and "decision=expire" in msg
        and "reason_code=attention_continuity_hold" in msg
        and "priority=40" in msg
        and "result_class=noop" in msg
        and "cue_name=none" in msg
        for msg in debug_messages
    )
    assert any(
        "Gesture cue ignored: result_class=noop state=thinking reason=attention_continuity_hold" in msg
        for msg in debug_messages
    )

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


def test_deferred_release_retried_on_non_listening_state_transition(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    calls: list[tuple[InteractionState, str]] = []

    def _enqueue(**kwargs):
        calls.append((kwargs["state"], kwargs["gesture_name"]))
        return len(calls) > 1

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    api._enqueue_gesture_cue = _enqueue

    api._listening_attention_hold_active = True
    api._attention_on_speech_stopped(reason="speech_stopped")
    assert api._listening_attention_hold_active is True

    api._handle_state_gesture(InteractionState.THINKING)

    assert calls == [
        (InteractionState.THINKING, "gesture_attention_release"),
        (InteractionState.THINKING, "gesture_attention_release"),
    ]
    assert api._listening_attention_hold_active is False
    assert any("attention_hold_released status=deferred reason=speech_stopped" in msg for msg in messages)
    assert any("attention_hold_released reason=state_thinking" in msg for msg in messages)


def test_speaking_state_still_releases_continuity_and_hold(monkeypatch) -> None:
    api = _make_api()
    continuity_states: list[InteractionState] = []
    api._attention_on_terminal_state = lambda state: continuity_states.append(state)
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )

    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True
    api._listening_attention_hold_active = True

    api._handle_state_gesture(InteractionState.SPEAKING)

    assert continuity_states == [InteractionState.SPEAKING]
    assert calls == ["gesture_attention_release"]
    assert api._listening_attention_hold_active is False


def test_terminal_state_continuity_runs_without_active_hold() -> None:
    api = _make_api()
    continuity_states: list[InteractionState] = []
    api._attention_on_terminal_state = lambda state: continuity_states.append(state)
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )

    release_calls: list[str] = []
    api._emit_attention_hold_release = lambda **kwargs: release_calls.append(kwargs["reason"])

    api._listening_attention_hold_active = False
    api._handle_state_gesture(InteractionState.IDLE)
    api._handle_state_gesture(InteractionState.SPEAKING)

    assert continuity_states == [InteractionState.IDLE, InteractionState.SPEAKING]
    assert release_calls == []


def test_deferred_release_retried_on_all_non_listening_terminal_transitions() -> None:
    api = _make_api()
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )

    calls: list[InteractionState] = []

    def _emit(*, reason: str) -> None:
        calls.append(InteractionState(reason.removeprefix("state_")))

    api._emit_attention_hold_release = _emit
    api._listening_attention_hold_active = True

    api._handle_state_gesture(InteractionState.THINKING)
    api._handle_state_gesture(InteractionState.SPEAKING)
    api._handle_state_gesture(InteractionState.IDLE)

    assert calls == [InteractionState.THINKING, InteractionState.SPEAKING, InteractionState.IDLE]


def test_deferred_release_attempts_once_per_transition_when_motion_busy(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    api._listening_attention_hold_active = True
    api._enqueue_gesture_cue = lambda **kwargs: False

    api._handle_state_gesture(InteractionState.THINKING)
    api._handle_state_gesture(InteractionState.SPEAKING)
    api._handle_state_gesture(InteractionState.IDLE)

    deferred = [m for m in messages if "attention_hold_released status=deferred" in m]
    assert len(deferred) == 3
    assert any("reason=state_thinking" in m for m in deferred)
    assert any("reason=state_speaking" in m for m in deferred)
    assert any("reason=state_idle" in m for m in deferred)
    assert api._listening_attention_hold_active is True


def test_speaking_posture_emits_once_per_episode(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.EMIT_CUE,
        reason="state_cue_emission",
        cue_name="gesture_speaking_posture",
    )
    emitted: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: emitted.append(kwargs["gesture_name"]) or True

    api._handle_state_gesture(InteractionState.SPEAKING)
    api._handle_state_gesture(InteractionState.SPEAKING)

    assert emitted == ["gesture_speaking_posture"]
    assert api._speaking_posture_episode_active is True
    assert any("speaking_posture_started" in msg for msg in messages)
    assert any("speaking_posture_start_skipped reason=already_started" in msg for msg in messages)


def test_thinking_to_speaking_emits_posture_even_when_global_cooldown_active(monkeypatch) -> None:
    api = _make_api()
    api._gesture_global_cooldown_s = 10.0
    api._gesture_cooldowns_s = {
        "gesture_curious_tilt": 6.0,
        "gesture_speaking_posture": 3.0,
    }

    now_values = iter([100.0, 100.2, 100.2])
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: next(now_values))
    monkeypatch.setattr("ai.realtime_api.random.randint", lambda _low, _high: 200)

    emitted: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: emitted.append(kwargs["gesture_name"]) or True

    api._handle_state_gesture(InteractionState.THINKING)
    api._handle_state_gesture(InteractionState.SPEAKING)

    assert emitted == ["gesture_curious_tilt", "gesture_speaking_posture"]
    assert api._speaking_posture_episode_active is True


def test_speaking_to_idle_emits_single_settle_when_episode_active(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    emitted: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: emitted.append(kwargs["gesture_name"]) or True

    api._speaking_posture_episode_active = True
    api._last_interaction_state = InteractionState.SPEAKING

    api._handle_state_gesture(InteractionState.IDLE)
    api._handle_state_gesture(InteractionState.IDLE)

    assert emitted == ["gesture_speaking_settle"]
    assert any("speaking_settle_emitted" in msg for msg in messages)


def test_speaking_to_idle_skips_settle_without_active_episode(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    emitted: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: emitted.append(kwargs["gesture_name"]) or True

    api._last_interaction_state = InteractionState.SPEAKING
    api._handle_state_gesture(InteractionState.IDLE)

    assert emitted == []
    assert any("speaking_settle_skipped reason=no_active_speaking_episode" in msg for msg in messages)


def test_speaking_settle_deferred_when_motion_busy_on_speaking_to_idle(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.EMIT_CUE,
        reason="state_cue_emission",
        cue_name="gesture_speaking_posture",
    )
    api._enqueue_gesture_cue = lambda **kwargs: False

    api._handle_state_gesture(InteractionState.SPEAKING)
    api._speaking_posture_episode_active = True
    api._last_interaction_state = InteractionState.SPEAKING
    api._handle_state_gesture(InteractionState.IDLE)

    assert any("speaking_posture_start_skipped reason=motion_busy_or_unavailable" in msg for msg in messages)
    assert any("speaking_settle_deferred reason=motion_busy_or_unavailable" in msg for msg in messages)
    assert api._speaking_settle_deferred is True


def test_speaking_settle_retry_emits_when_motion_becomes_available(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )

    calls: list[tuple[InteractionState, str]] = []

    def _enqueue(**kwargs):
        calls.append((kwargs["state"], kwargs["gesture_name"]))
        return len(calls) > 1

    api._enqueue_gesture_cue = _enqueue
    api._speaking_posture_episode_active = True
    api._last_interaction_state = InteractionState.SPEAKING

    api._handle_state_gesture(InteractionState.IDLE)
    api._handle_state_gesture(InteractionState.THINKING)

    assert calls == [
        (InteractionState.IDLE, "gesture_speaking_settle"),
        (InteractionState.THINKING, "gesture_speaking_settle"),
    ]
    assert api._speaking_settle_deferred is False
    assert any("speaking_settle_deferred reason=motion_busy_or_unavailable" in msg for msg in messages)
    assert any("speaking_settle_retry" in msg for msg in messages)
    assert any("speaking_settle_emitted" in msg for msg in messages)


def test_speaking_settle_emits_at_most_once_per_episode_with_retry(monkeypatch) -> None:
    api = _make_api()
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True

    api._speaking_settle_deferred = True
    api._speaking_settle_deferred_at = 10.0
    api._last_interaction_state = InteractionState.IDLE
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: 11.0)

    api._handle_state_gesture(InteractionState.THINKING)
    api._handle_state_gesture(InteractionState.IDLE)

    assert calls == ["gesture_speaking_settle"]
    assert api._speaking_settle_deferred is False


def test_speaking_settle_retry_not_attempted_without_deferred_episode(monkeypatch) -> None:
    api = _make_api()
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True

    api._speaking_settle_deferred = False
    api._last_interaction_state = InteractionState.IDLE

    api._handle_state_gesture(InteractionState.THINKING)

    assert calls == []


def test_speaking_settle_retry_helper_emits_from_safe_seam(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))

    emitted: list[tuple[InteractionState, str]] = []
    api._enqueue_gesture_cue = lambda **kwargs: emitted.append((kwargs["state"], kwargs["policy_reason"])) or True
    api._speaking_settle_deferred = True
    api._speaking_settle_deferred_at = 10.0
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: 11.0)

    retried = api._retry_deferred_speaking_settle(
        state=InteractionState.IDLE,
        source="playback_complete",
    )

    assert retried is True
    assert emitted == [(InteractionState.IDLE, "speaking_settle_retry_playback_complete")]
    assert api._speaking_settle_deferred is False
    assert any("speaking_settle_retry source=playback_complete state=idle" in msg for msg in messages)
    assert any("speaking_settle_emitted source=playback_complete state=idle" in msg for msg in messages)


def test_speaking_settle_retry_helper_noop_without_deferred_episode() -> None:
    api = _make_api()
    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True
    api._speaking_settle_deferred = False

    retried = api._retry_deferred_speaking_settle(
        state=InteractionState.IDLE,
        source="playback_complete",
    )

    assert retried is False
    assert calls == []


def test_speaking_settle_retry_helper_logs_when_retry_still_deferred(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))
    api._enqueue_gesture_cue = lambda **kwargs: False
    api._speaking_settle_deferred = True
    api._speaking_settle_deferred_at = 10.0
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: 11.0)

    retried = api._retry_deferred_speaking_settle(
        state=InteractionState.IDLE,
        source="playback_complete",
    )

    assert retried is False
    assert api._speaking_settle_deferred is True
    assert any("speaking_settle_retry source=playback_complete state=idle" in msg for msg in messages)
    assert any(
        "speaking_settle_retry_deferred source=playback_complete state=idle reason=motion_busy_or_unavailable" in msg
        for msg in messages
    )


def test_speaking_settle_retry_helper_drops_stale_deferred_settle(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: 20.0)

    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True
    api._speaking_settle_deferred = True
    api._speaking_settle_deferred_at = 17.0

    retried = api._retry_deferred_speaking_settle(
        state=InteractionState.IDLE,
        source="playback_complete",
    )

    assert retried is False
    assert calls == []
    assert api._speaking_settle_deferred is False
    assert api._speaking_settle_deferred_at is None
    assert any("speaking_settle_dropped reason=stale_deferred_settle source=playback_complete" in msg for msg in messages)


def test_playback_complete_retry_respects_speaking_settle_freshness_gate(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: 20.0)

    api.exit_event = types.SimpleNamespace(is_set=lambda: False)
    api.state_manager = types.SimpleNamespace(state=InteractionState.IDLE)
    api.mic = types.SimpleNamespace(stop_receiving=lambda: None, start_recording=lambda: None)
    api.websocket = None
    api._response_create_queue = []
    api._runtime_task_registry = lambda: types.SimpleNamespace(spawn=lambda *args, **kwargs: None)

    calls: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: calls.append(kwargs["gesture_name"]) or True
    api._speaking_settle_deferred = True
    api._speaking_settle_deferred_at = 17.0

    api._on_playback_complete()

    assert calls == []
    assert api._speaking_settle_deferred is False
    assert api._speaking_settle_deferred_at is None
    assert any("speaking_settle_dropped reason=stale_deferred_settle source=playback_complete" in msg for msg in messages)


def test_new_speaking_episode_clears_older_deferred_settle_timestamp(monkeypatch) -> None:
    api = _make_api()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda msg, *args: messages.append(msg % args if args else msg))
    api._embodiment_policy.decide_state_cue = lambda **_: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="no_state_cue",
    )
    api._enqueue_gesture_cue = lambda **kwargs: True
    api._last_interaction_state = InteractionState.IDLE
    api._speaking_settle_deferred = True
    api._speaking_settle_deferred_at = 5.0

    api._handle_state_gesture(InteractionState.SPEAKING)

    assert api._speaking_settle_deferred is False
    assert api._speaking_settle_deferred_at is None
    assert any("speaking_settle_dropped reason=new_speaking_episode_started" in msg for msg in messages)
