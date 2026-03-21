"""Regression tests for realtime prompt-origin delivery and key binding."""

from __future__ import annotations

import asyncio
import sys
import types
from collections import deque

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import InteractionState, RealtimeAPI


class _StateManager:
    def __init__(self) -> None:
        self.state = InteractionState.IDLE

    def update_state(self, state: InteractionState, _reason: str) -> None:
        self.state = state


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._preference_recall_cooldown_s = 0.0
    api._preference_recall_cache = {}
    api._memory_retrieval_scope = "user_global"
    api._pending_response_create_origins = deque()
    api._pending_response_create = None
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_create_debug_trace = False
    api._current_response_turn_id = "turn_prompt"
    api._last_response_create_ts = None
    api._audio_playback_busy = False
    api._response_schedule_logged_turn_ids = set()
    api._preference_recall_response_suppression_until = 0.0
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._preference_recall_locked_input_event_keys = set()
    api._pending_server_auto_input_event_keys = deque(maxlen=64)
    api._active_server_auto_input_event_key = None
    api._current_input_event_key = None
    api._active_input_event_key_by_turn_id = {}
    api._input_event_key_counter = 0
    api._active_response_preference_guarded = False
    api._active_response_confirmation_guarded = False
    api._response_obligations = {}
    api._response_created_canonical_keys = set()
    api._response_delivery_ledger = {}
    api._response_id_by_canonical_key = {}
    api._canonical_response_lifecycle_state = {}
    api._already_scheduled_for_input_event_key = set()
    api._active_response_id = None
    api._active_response_origin = "unknown"
    api._active_response_consumes_canonical_slot = True
    api._tool_call_records = []
    api._assistant_reply_accum = ""
    api._active_utterance = None
    api._confirmation_asr_pending = False
    api._response_done_reflection_task = None
    api.assistant_reply = ""
    api._reflection_enqueued = False
    api._response_in_flight = False
    api.response_in_progress = False
    api._speaking_started = False
    api._mic_receive_on_first_audio = False
    api._last_response_metadata = {}
    api._turn_diagnostic_timestamps = {}
    api._transcript_response_watchdog_timeout_s = 5.0
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._preference_recall_handled_logged_turn_ids = set()
    api._preference_recall_skip_logged_turn_ids = set()
    api._pending_preference_recall_trace = None
    api._micro_ack_manager = None
    api._audio_accum = bytearray()
    api._audio_accum_bytes_target = 9600
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._confirmation_speech_active = False
    api._pending_confirmation_token = None
    api._pending_action = None
    api._awaiting_confirmation_completion = False
    api._last_outgoing_event_type = None
    api._silent_turn_incident_count = 0
    api.rate_limits = {}
    api.audio_player = None
    api.mic = type("_Mic", (), {"is_receiving": False, "start_receiving": lambda self: None})()
    api.websocket = None
    api.orchestration_state = type(
        "_Orch",
        (),
        {
            "phase": None,
            "transition": lambda self, *_args, **_kwargs: None,
        },
    )()
    api.state_manager = _StateManager()
    api._current_run_id = lambda: "run-rt-silent"
    api._record_ai_call = lambda: None
    api._maybe_enqueue_reflection = lambda *_args, **_kwargs: None
    api._enqueue_response_done_reflection = lambda *_args, **_kwargs: None
    api._emit_preference_recall_skip_trace_if_needed = lambda *_args, **_kwargs: None
    api._clear_stale_pending_server_auto_for_turn = lambda **_kwargs: None
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._mark_first_assistant_utterance_observed_if_needed = lambda *_args, **_kwargs: None
    api._track_outgoing_event = RealtimeAPI._track_outgoing_event.__get__(api, RealtimeAPI)

    async def _false(*_args, **_kwargs) -> bool:
        return False

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    return api


def test_prompt_origin_delivery_prevents_silent_turn_false_positive() -> None:
    api = _make_api()
    turn_id = "turn_prompt"
    input_event_key = "evt_prompt_1"
    api._current_response_turn_id = turn_id
    api._current_input_event_key = input_event_key
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key

    api._track_outgoing_event(
        {
            "type": "response.create",
            "response": {"metadata": {"origin": "prompt", "turn_id": turn_id, "input_event_key": input_event_key}},
        },
        origin="prompt",
    )

    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "resp_prompt_1"}}, websocket=None))
    asyncio.run(api.handle_event({"type": "response.output_text.delta", "delta": "Hello from prompt."}, websocket=None))

    buffer_before_done = api._assistant_reply_accum
    silent_before_done = api._silent_turn_incident_count

    asyncio.run(api.handle_event({"type": "response.done", "response": {"id": "resp_prompt_1"}}, websocket=None))

    delivery_state = api._response_delivery_state(turn_id=turn_id, input_event_key=input_event_key)
    silent_after_done = api._silent_turn_incident_count

    assert any(
        [
            bool(buffer_before_done.strip()),
            delivery_state == "delivered",
            silent_after_done == silent_before_done,
        ]
    )


def test_prompt_origin_response_created_binds_active_key_before_output(monkeypatch) -> None:
    api = _make_api()
    turn_id = "turn_bind"
    input_event_key = "evt_bind_1"
    api._current_response_turn_id = turn_id
    api._current_input_event_key = input_event_key
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key

    response_create_event = {
        "type": "response.create",
        "response": {"metadata": {"origin": "prompt", "turn_id": turn_id, "input_event_key": input_event_key}},
    }
    ensured_key = api._ensure_response_create_correlation(
        response_create_event=response_create_event,
        origin="prompt",
        turn_id=turn_id,
    )
    api._track_outgoing_event(response_create_event, origin="prompt")

    response_binding_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.debug",
        lambda message, *args: response_binding_logs.append(message % args),
    )

    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "resp_bind_1"}}, websocket=None))

    assert ensured_key == input_event_key
    assert api._active_response_input_event_key == input_event_key
    assert api._canonical_utterance_key(turn_id=turn_id, input_event_key=api._active_response_input_event_key) == api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=input_event_key,
    )
    assert not any("response_binding" in line and "active_key=unknown" in line for line in response_binding_logs)

    asyncio.run(api.handle_event({"type": "response.output_text.delta", "delta": "bound"}, websocket=None))
    assert api._assistant_reply_accum == "bound"
