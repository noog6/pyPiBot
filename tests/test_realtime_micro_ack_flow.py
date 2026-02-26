from __future__ import annotations

import asyncio
import base64
from collections import deque

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


class _Manager:
    def __init__(self) -> None:
        self.scheduled: list[tuple[str, str, int | None]] = []
        self.cancelled: list[tuple[str, str]] = []
        self.speech_started = 0
        self.speech_ended = 0

    def maybe_schedule(self, *, turn_id: str, reason: str, loop, expected_delay_ms=None) -> None:
        self.scheduled.append((turn_id, reason, expected_delay_ms))

    def cancel(self, *, turn_id: str, reason: str) -> None:
        self.cancelled.append((turn_id, reason))

    def cancel_all(self, *, reason: str) -> None:
        self.cancelled.append(("*", reason))

    def on_user_speech_started(self) -> None:
        self.speech_started += 1

    def on_user_speech_ended(self) -> None:
        self.speech_ended += 1

    def mark_talk_over_incident(self) -> None:
        return None

    def suppression_baseline_reason(self):
        return None


class _Ws:
    async def send(self, _payload: str) -> None:
        return None


class _Mic:
    is_receiving = False

    def start_receiving(self) -> None:
        self.is_receiving = True


def _api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.loop = asyncio.new_event_loop()
    api.websocket = _Ws()
    api._micro_ack_manager = _Manager()
    api._confirmation_speech_active = False
    api._audio_playback_busy = False
    api._active_utterance = None
    api._assistant_reply_accum = ""
    api.assistant_reply = ""
    api._audio_accum = bytearray()
    api._audio_accum_bytes_target = 9600
    api._mic_receive_on_first_audio = False
    api._speaking_started = False
    api._active_response_confirmation_guarded = False
    api._active_response_preference_guarded = False
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._is_active_response_guarded = lambda: False
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._mark_confirmation_activity = lambda **_kwargs: None
    api.handle_speech_stopped = lambda _ws: asyncio.sleep(0)
    api._response_in_flight = False
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._active_server_auto_input_event_key = None
    api._active_response_origin = "unknown"
    api._pending_server_auto_input_event_keys = deque()
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._extract_response_create_metadata = lambda *_args, **_kwargs: {}
    api._extract_response_create_trigger = lambda *_args, **_kwargs: ""
    api._can_release_queued_response_create = lambda *_args, **_kwargs: True
    api._response_create_queue = deque()
    api._pending_response_create = None
    api._response_done_serial = 0
    api._drain_response_create_queue = lambda: asyncio.sleep(0)
    api._flush_pending_image_stimulus = lambda *_args, **_kwargs: asyncio.sleep(0)
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._current_run_id = lambda: "run-test"
    api._consume_response_origin = lambda: "server_auto"
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._is_user_approved_interrupt_response = lambda *_args, **_kwargs: False
    api.mic = _Mic()
    api.state_manager = type(
        "State",
        (),
        {"state": InteractionState.IDLE, "update_state": lambda *_args, **_kwargs: None},
    )()
    api.orchestration_state = type(
        "State",
        (),
        {"phase": OrchestrationPhase.IDLE, "transition": lambda *_args, **_kwargs: None},
    )()
    api._maybe_handle_confirmation_decision_timeout = lambda *_args, **_kwargs: asyncio.sleep(0)
    return api


def test_speech_stopped_schedules_micro_ack() -> None:
    api = _api_stub()
    asyncio.run(api.handle_event({"type": "input_audio_buffer.speech_stopped"}, api.websocket))
    assert ("turn-1", "speech_stopped", 700) in api._micro_ack_manager.scheduled
    api.loop.close()


def test_response_audio_delta_cancels_pending_micro_ack() -> None:
    api = _api_stub()
    event = {"type": "response.output_audio.delta", "delta": base64.b64encode(b"abc").decode("ascii")}
    asyncio.run(api.handle_event(event, api.websocket))
    assert ("turn-1", "response_started") in api._micro_ack_manager.cancelled
    api.loop.close()
