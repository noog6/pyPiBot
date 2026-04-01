from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.input_audio_events import InputAudioEventHandlers
from interaction import InteractionState


class _Ws:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


class _Transport:
    async def send_json(self, websocket: _Ws, event: dict[str, str]) -> None:
        await websocket.send(json.dumps(event))


class _Manager:
    def on_user_speech_started(self) -> None:
        return None

    def mark_talk_over_incident(self) -> None:
        return None

    def cancel_all(self, *, reason: str) -> None:
        _ = reason


def _api_stub(*, defer_interrupt: bool, hold_turn: bool = False):
    api = SimpleNamespace()
    api._reset_utterance_info_summary = lambda: None
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api.state_manager = SimpleNamespace(state=InteractionState.SPEAKING, update_state=lambda *_args, **_kwargs: None)
    api._audio_playback_busy = True
    api._micro_ack_manager = _Manager()
    api._mark_active_tool_output_interrupted_before_first_evidence = lambda **_kwargs: defer_interrupt
    api._clear_all_pending_response_creates = lambda **_kwargs: None
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._current_input_event_key = ""
    api._clear_pending_response_contenders = lambda **_kwargs: None
    api._response_in_flight = True
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: _Transport()
    api._utterance_counter = 0
    api._next_response_turn_id = lambda: "turn-2"
    api._should_hold_turn_for_non_substantive_talk_over = lambda: hold_turn

    class _Scope:
        def __init__(self, *, turn_id: str):
            self._turn_id = turn_id

        def __enter__(self):
            api._current_response_turn_id = self._turn_id
            return None

        def __exit__(self, *_args):
            return False

    api._utterance_context_scope = lambda **kwargs: _Scope(turn_id=kwargs.get("turn_id", "turn-unknown"))
    api._active_utterance = None
    api._log_utterance_envelope = lambda *_args, **_kwargs: None
    api._has_active_confirmation_token = lambda: False
    api.orchestration_state = SimpleNamespace(transition=lambda *_args, **_kwargs: None)
    api._attention_on_listening_started = lambda: None
    api._current_run_id = lambda: "run-test"
    api._active_response_id = "resp-1"
    return api


def test_talk_over_cancel_sent_when_not_deferred() -> None:
    api = _api_stub(defer_interrupt=False)
    handler = InputAudioEventHandlers(api)
    ws = _Ws()

    asyncio.run(handler.handle_input_audio_buffer_speech_started({}, ws))

    assert ws.sent == ['{"type": "response.cancel"}']


def test_talk_over_cancel_not_sent_for_deferred_tool_output_candidate() -> None:
    api = _api_stub(defer_interrupt=True)
    handler = InputAudioEventHandlers(api)
    ws = _Ws()

    asyncio.run(handler.handle_input_audio_buffer_speech_started({}, ws))

    assert ws.sent == []


def test_talk_over_hold_keeps_existing_authoritative_turn() -> None:
    api = _api_stub(defer_interrupt=True, hold_turn=True)
    handler = InputAudioEventHandlers(api)
    ws = _Ws()

    asyncio.run(handler.handle_input_audio_buffer_speech_started({}, ws))

    assert api._current_response_turn_id == "turn-1"


def test_talk_over_without_hold_still_advances_turn() -> None:
    api = _api_stub(defer_interrupt=True, hold_turn=False)
    handler = InputAudioEventHandlers(api)
    ws = _Ws()

    asyncio.run(handler.handle_input_audio_buffer_speech_started({}, ws))

    assert api._current_response_turn_id == "turn-2"
