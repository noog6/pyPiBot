from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

import asyncio

from ai.embodiment_policy import SituationalCueEvent
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._embodiment_policy = __import__("ai.embodiment_policy", fromlist=["EmbodimentPolicy"]).EmbodimentPolicy()
    api._response_in_flight = False
    api._turn_diagnostic_timestamps = {}
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._active_input_event_key_for_turn = lambda _turn: "input-1"
    api._mark_turn_latency_marker = lambda **_kwargs: None
    api._mark_tool_followup_timing = lambda **_kwargs: None
    api._turn_contract_blocks_gesture_cues = lambda: False
    api._is_motion_busy = lambda: False
    api._enqueue_gesture_cue = lambda **_kwargs: True
    api.state_manager = types.SimpleNamespace(state=InteractionState.LISTENING)
    api.mic = types.SimpleNamespace(stop_recording=lambda: None)
    return api


def test_handle_speech_stopped_requests_situational_ack() -> None:
    api = _make_api()
    recorded: list[SituationalCueEvent] = []
    api._emit_situational_cue = lambda **kwargs: recorded.append(kwargs["event"]) or True
    asyncio.run(api.handle_speech_stopped(websocket=None))
    assert recorded == [SituationalCueEvent.SPEECH_STOPPED_ACK]


def test_emit_situational_cue_direct_address_enqueues_expected_gesture() -> None:
    api = _make_api()
    seen: list[str] = []
    api._enqueue_gesture_cue = lambda **kwargs: seen.append(kwargs["gesture_name"]) or True
    emitted = api._emit_situational_cue(event=SituationalCueEvent.DIRECT_ADDRESS_ACK)
    assert emitted is True
    assert seen == ["gesture_attention_snap"]
