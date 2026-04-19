from __future__ import annotations

import asyncio
import json
import sys
import types
from collections import deque

import pytest

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.governance import ActionPacket
from ai.micro_ack_manager import MicroAckCategory, MicroAckContext
from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime.types import PendingResponseCreate
from ai.realtime.response_terminal_handlers import ResponseTerminalHandlers
from ai.realtime_api import InteractionState, RealtimeAPI
from ai.interaction_lifecycle_controller import InteractionLifecycleState
from core.logging import logger


class _RecordingWs:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class _Transport:
    async def send_json(self, websocket: _RecordingWs, payload: dict[str, object]) -> None:
        await websocket.send(json.dumps(payload))


def _wire_runtime(api: RealtimeAPI) -> None:
    api._response_create_runtime = ResponseCreateRuntime(api)
    transport = _Transport()
    api._get_or_create_transport = lambda: transport
    api._spoken_research_response_ids = {}
    api._research_spoken_response_dedupe_ttl_s = 60.0


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._preference_recall_cooldown_s = 0.0
    api._assistant_name = "Theo"
    api._preference_recall_cache = {}
    api._memory_retrieval_scope = "user_global"
    api._pending_response_create_origins = deque()
    api._pending_response_create = None
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_create_enqueue_seq = 0
    api._response_create_queued_creates_total = 0
    api._response_create_drains_total = 0
    api._response_create_max_qdepth = 0
    api._response_create_debug_trace = False
    api._current_response_turn_id = "turn_1"
    api._last_response_create_ts = None
    api._audio_playback_busy = False
    api._response_schedule_logged_turn_ids = set()
    api._preference_recall_response_suppression_until = 0.0
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._preference_recall_locked_input_event_keys = set()
    api._pending_server_auto_input_event_keys = deque(maxlen=64)
    api._active_server_auto_input_event_key = None
    api._server_auto_audio_deferral_timeout_ms = 225
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
    api._tool_followup_state_by_canonical_key = {}
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
    api._conversation_efficiency_logged_turns = set()
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
    api.rate_limits = {}
    api.audio_player = None
    api.mic = type(
        "_Mic",
        (),
        {
            "is_receiving": False,
            "is_recording": False,
            "start_recording_calls": 0,
            "stop_recording_calls": 0,
            "start_receiving": lambda self: setattr(self, "is_receiving", True),
            "stop_receiving": lambda self: setattr(self, "is_receiving", False),
            "start_recording": lambda self: (
                setattr(self, "is_recording", True),
                setattr(self, "start_recording_calls", getattr(self, "start_recording_calls", 0) + 1),
            ),
            "stop_recording": lambda self: (
                setattr(self, "is_recording", False),
                setattr(self, "stop_recording_calls", getattr(self, "stop_recording_calls", 0) + 1),
            ),
        },
    )()
    api.websocket = None
    api.orchestration_state = type(
        "_Orch",
        (),
        {
            "phase": None,
            "transition": lambda self, *_args, **_kwargs: None,
        },
    )()
    api.state_manager = type(
        "_State",
        (),
        {
            "state": InteractionState.IDLE,
            "update_state": lambda self, *_args, **_kwargs: None,
        },
    )()
    api._current_run_id = lambda: "run-405-repro"
    api._log_user_transcript_redact_enabled = False
    api._record_ai_call = lambda: None
    api._maybe_enqueue_reflection = lambda *_args, **_kwargs: None
    api._enqueue_response_done_reflection = lambda *_args, **_kwargs: None
    api._emit_preference_recall_skip_trace_if_needed = lambda *_args, **_kwargs: None
    api._clear_stale_pending_server_auto_for_turn = lambda **_kwargs: None
    api._build_confirmation_transition_decision = lambda **_kwargs: type(
        "_Transition",
        (),
        {
            "allow_response_transition": True,
            "emit_reminder": False,
            "recover_mic": False,
            "close_reason": "",
        },
    )()
    api._confirmation_hold_components = lambda: (False, False, False, False)
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = RealtimeAPI._track_outgoing_event.__get__(api, RealtimeAPI)
    api._active_input_event_key_for_turn = lambda turn_id: api._active_input_event_key_by_turn_id.get(turn_id, "")
    api._input_audio_events = type(
        "_InputAudioEvents",
        (),
        {
            "handle_input_audio_buffer_speech_started": lambda self, *_args, **_kwargs: None,
            "handle_input_audio_buffer_speech_stopped": lambda self, *_args, **_kwargs: None,
            "handle_input_audio_buffer_committed": lambda self, *_args, **_kwargs: None,
            "handle_input_audio_transcription_partial": lambda self, *_args, **_kwargs: None,
        },
    )()
    return api


def _prime_response_done_api(
    api: RealtimeAPI,
    *,
    turn_id: str,
    input_event_key: str,
    response_id: str,
    origin: str = "server_auto",
    audio_started: bool = False,
) -> str:
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._current_response_turn_id = turn_id
    api._current_turn_id_or_unknown = lambda: turn_id
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._active_response_id = response_id
    api._active_response_origin = origin
    api._active_response_input_event_key = input_event_key
    api._active_response_canonical_key = canonical_key
    api._active_response_consumes_canonical_slot = True
    api._response_trace_context_by_id = {
        response_id: {
            "turn_id": turn_id,
            "input_event_key": input_event_key,
            "canonical_key": canonical_key,
            "origin": origin,
        }
    }
    api._stale_response_context = lambda _response_id: {}
    api._response_delivery_state = lambda **_kwargs: "created"
    api._response_obligation_key = lambda **_kwargs: f"{turn_id}:{input_event_key}"
    api._lifecycle_controller = lambda: type(
        "_Lifecycle",
        (),
        {
            "on_response_done": lambda *_args, **_kwargs: None,
            "audio_started": lambda *_args, **_kwargs: False,
        },
    )()
    api._log_lifecycle_event = lambda **_kwargs: None
    api._debug_dump_canonical_key_timeline = lambda **_kwargs: None
    api._set_response_delivery_state = lambda **_kwargs: None
    api._tool_followup_state = lambda **_kwargs: "idle"
    api._set_tool_followup_state = lambda **_kwargs: None
    api._reconcile_terminal_substantive_response = lambda **_kwargs: None
    api._release_blocked_tool_followups_for_response_done = lambda **_kwargs: None
    api._log_cancelled_deliverable_once = lambda *_args, **_kwargs: None
    api._record_response_trace_context = lambda *_args, **_kwargs: None
    api._emit_response_lifecycle_trace = lambda **_kwargs: None
    api._log_lifecycle_coherence = lambda **_kwargs: None
    api._clear_terminal_response_text = lambda **_kwargs: None
    api._emit_startup_prompt_terminal = lambda **_kwargs: None
    api._emit_utterance_info_summary = lambda **_kwargs: None
    api._prune_curiosity_surface_candidates = lambda **_kwargs: None
    api._is_empty_response_done = lambda **_kwargs: False
    api._record_silent_turn_incident = lambda **_kwargs: None
    api._maybe_schedule_empty_response_retry = lambda **_kwargs: asyncio.sleep(0)
    api._emit_preference_recall_skip_trace_if_needed = lambda **_kwargs: None
    api._log_turn_conversation_efficiency = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **_kwargs: None
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._response_trace_by_id = lambda: api._response_trace_context_by_id
    api._build_confirmation_transition_decision = lambda **_kwargs: type(
        "_Transition",
        (),
        {
            "allow_response_transition": True,
            "emit_reminder": False,
            "recover_mic": False,
            "close_reason": "",
        },
    )()
    api._confirmation_hold_components = lambda: (False, False, None, False)
    api._mark_confirmation_activity = lambda **_kwargs: None
    api._should_send_response_done_fallback_reminder = lambda: False
    api._is_guarded_server_auto_reminder_allowed = lambda **_kwargs: False
    api._maybe_emit_confirmation_reminder = lambda *_args, **_kwargs: asyncio.sleep(0)
    api._recover_confirmation_guard_microphone = lambda *_args, **_kwargs: None
    api._clear_cancelled_response_tracking = lambda *_args, **_kwargs: None
    api._response_terminal_handlers = ResponseTerminalHandlers(api)
    api.exit_event = type("_ExitEvent", (), {"is_set": lambda self: False})()
    state = api.state_manager
    state.transitions = []
    state.update_state = lambda new_state, reason: (
        state.transitions.append((new_state, reason)),
        setattr(state, "state", new_state),
    )
    orch = api.orchestration_state
    orch.transitions = []
    orch.transition = lambda phase, reason: orch.transitions.append((phase, reason))
    api._canonical_response_state_mutate(
        canonical_key=canonical_key,
        turn_id=turn_id,
        input_event_key=input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", origin),
            setattr(record, "response_id", response_id),
            setattr(record, "created", True),
            setattr(record, "audio_started", audio_started),
        ),
    )
    return canonical_key


def _capture_terminal_audit_logs(monkeypatch) -> list[str]:
    captured: list[str] = []

    def _capture(message, *args, **kwargs) -> None:
        rendered = message % args if args else str(message)
        if "terminal_answer_context_audit" in rendered:
            captured.append(rendered)

    monkeypatch.setattr("ai.realtime.response_terminal_handlers.logger.info", _capture)
    return captured


def test_terminal_answer_context_audit_treats_preference_backed_turn_brief_as_pref_injected(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    logs = _capture_terminal_audit_logs(monkeypatch)
    turn_id = "turn_pref_audit"
    input_event_key = "item_pref_audit"
    response_id = "resp_pref_audit"
    _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=input_event_key,
        response_id=response_id,
        origin="upgraded_response",
    )
    turn_key = api._memory_usage_audit_turn_key(turn_id=turn_id, input_event_key=input_event_key)
    api._memory_usage_audit_store()[turn_key] = {
        "run_id": "run-405-repro",
        "turn_id": turn_id,
        "input_event_key": input_event_key,
        "telemetry_scope": "turn",
        "injection_types": ["turn_brief", "preference_context"],
    }

    asyncio.run(api.handle_response_done({"type": "response.done", "response": {"id": response_id}}))

    assert any(
        "pref_context_injected=true" in entry and "injection_types=turn_brief,preference_context" in entry
        for entry in logs
    )


def test_terminal_answer_context_audit_keeps_generic_turn_brief_pref_false(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    logs = _capture_terminal_audit_logs(monkeypatch)
    turn_id = "turn_turn_brief_only"
    input_event_key = "item_turn_brief_only"
    response_id = "resp_turn_brief_only"
    _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=input_event_key,
        response_id=response_id,
        origin="assistant_message",
    )
    turn_key = api._memory_usage_audit_turn_key(turn_id=turn_id, input_event_key=input_event_key)
    api._memory_usage_audit_store()[turn_key] = {
        "run_id": "run-405-repro",
        "turn_id": turn_id,
        "input_event_key": input_event_key,
        "telemetry_scope": "turn",
        "injection_types": ["turn_brief"],
    }

    asyncio.run(api.handle_response_done({"type": "response.done", "response": {"id": response_id}}))

    assert any("pref_context_injected=false" in entry and "injection_types=turn_brief" in entry for entry in logs)


def test_terminal_answer_context_audit_keeps_direct_preference_context_pref_true(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    logs = _capture_terminal_audit_logs(monkeypatch)
    turn_id = "turn_direct_pref"
    input_event_key = "item_direct_pref"
    response_id = "resp_direct_pref"
    _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=input_event_key,
        response_id=response_id,
        origin="assistant_message",
    )
    turn_key = api._memory_usage_audit_turn_key(turn_id=turn_id, input_event_key=input_event_key)
    api._memory_usage_audit_store()[turn_key] = {
        "run_id": "run-405-repro",
        "turn_id": turn_id,
        "input_event_key": input_event_key,
        "telemetry_scope": "turn",
        "injection_types": ["preference_context"],
    }

    asyncio.run(api.handle_response_done({"type": "response.done", "response": {"id": response_id}}))

    assert any("pref_context_injected=true" in entry and "injection_types=preference_context" in entry for entry in logs)


def test_response_created_syncs_helper_owned_fields_for_assistant_message() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()

    api._pending_response_create_origins.append(
        {
            "origin": "assistant_message",
            "turn_id": "turn_7",
            "input_event_key": "item_assistant_7",
            "consumes_canonical_slot": "true",
        }
    )

    asyncio.run(
        api._handle_response_created_event(
            {"type": "response.created", "response": {"id": "resp-assistant-7", "metadata": {}}},
            ws,
        )
    )

    expected_canonical_key = api._canonical_utterance_key(turn_id="turn_7", input_event_key="item_assistant_7")

    assert api._active_response_lifecycle.response_id == "resp-assistant-7"
    assert api._active_response_lifecycle.origin == "assistant_message"
    assert api._active_response_lifecycle.input_event_key == "item_assistant_7"
    assert api._active_response_lifecycle.canonical_key == expected_canonical_key
    assert api._active_response_lifecycle.consumes_canonical_slot is True
    assert api._active_response_id == "resp-assistant-7"
    assert api._active_response_origin == "assistant_message"
    assert api._active_response_input_event_key == "item_assistant_7"
    assert api._active_response_canonical_key == expected_canonical_key
    assert api._active_server_auto_input_event_key is None


def test_response_created_sparse_provider_metadata_backfills_required_deliverable_trace_markers() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    response_create_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_req",
                "input_event_key": "item_req",
                "followthrough_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
            }
        },
    }
    api._track_outgoing_event(response_create_event, origin="tool_output")

    asyncio.run(
        api._handle_response_created_event(
            {"type": "response.created", "response": {"id": "resp-req", "metadata": {}}},
            ws,
        )
    )

    trace_context = api._response_trace_context_by_id.get("resp-req", {})
    assert trace_context.get("followthrough_step_output_policy") == "required_deliverable"
    assert trace_context.get("tool_followup_step_output_policy") == "required_deliverable"
    assert trace_context.get("followthrough_post_completion_reason") == "required_deliverable_owed"
    assert trace_context.get("tool_followup_post_completion_reason") == "required_deliverable_owed"


def test_response_created_provider_metadata_not_overwritten_by_origin_fallback() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    response_create_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_provider",
                "input_event_key": "item_provider",
                "followthrough_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
            }
        },
    }
    api._track_outgoing_event(response_create_event, origin="tool_output")

    asyncio.run(
        api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-provider",
                    "metadata": {
                        "followthrough_step_output_policy": "final_required_deliverable",
                        "followthrough_post_completion_reason": "final_response_complete",
                    },
                },
            },
            ws,
        )
    )

    trace_context = api._response_trace_context_by_id.get("resp-provider", {})
    assert trace_context.get("followthrough_step_output_policy") == "final_required_deliverable"
    assert trace_context.get("tool_followup_step_output_policy") == "final_required_deliverable"
    assert trace_context.get("followthrough_post_completion_reason") == "final_response_complete"
    assert trace_context.get("tool_followup_post_completion_reason") == "final_response_complete"



def test_response_created_syncs_helper_owned_fields_for_server_auto_guarded_path() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api._current_input_event_key = "synthetic_server_auto_4"
    api._build_confirmation_transition_decision = lambda **_kwargs: type(
        "_Transition",
        (),
        {
            "allow_response_transition": True,
            "emit_reminder": False,
            "recover_mic": False,
            "close_reason": "",
        },
    )()
    api._has_active_confirmation_token = lambda: True
    api._is_awaiting_confirmation_phase = lambda: True
    api._should_guard_confirmation_response = lambda _origin, _response: True
    api._mark_confirmation_activity = lambda **_kwargs: None
    api._schedule_server_auto_audio_deferral = lambda **_kwargs: None
    api._set_server_auto_pre_audio_hold = lambda **_kwargs: None
    api._upgrade_likely_for_server_auto_turn = lambda **_kwargs: (False, "none")
    api._should_delay_server_auto_until_transcript_final = lambda **_kwargs: False
    api._start_audio_response_if_needed = lambda **_kwargs: None

    api._pending_response_create_origins.append(
        {
            "origin": "server_auto",
            "turn_id": "turn_4",
            "input_event_key": "synthetic_server_auto_4",
            "consumes_canonical_slot": "false",
        }
    )

    asyncio.run(
        api._handle_response_created_event(
            {"type": "response.created", "response": {"id": "resp-server-auto-4", "metadata": {}}},
            ws,
        )
    )

    expected_canonical_key = api._canonical_utterance_key(turn_id="turn_4", input_event_key="synthetic_server_auto_4")
    expected_lifecycle_key = f"{expected_canonical_key}:non_consuming:resp-server-auto-4"

    assert api._active_response_lifecycle.response_id == "resp-server-auto-4"
    assert api._active_response_lifecycle.origin == "server_auto"
    assert api._active_response_lifecycle.input_event_key == "synthetic_server_auto_4"
    assert api._active_response_lifecycle.canonical_key == expected_lifecycle_key
    assert api._active_response_lifecycle.consumes_canonical_slot is False
    assert api._active_response_lifecycle.confirmation_guarded is True
    assert api._active_response_id == "resp-server-auto-4"
    assert api._active_response_origin == "server_auto"
    assert api._active_response_input_event_key == "synthetic_server_auto_4"
    assert api._active_response_canonical_key == expected_lifecycle_key
    assert api._active_response_consumes_canonical_slot is False
    assert api._active_response_confirmation_guarded is True
    assert api._active_server_auto_input_event_key == "synthetic_server_auto_4"


def test_response_done_required_deliverable_trace_context_still_activates_guard_when_created_metadata_was_sparse(
    monkeypatch,
) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    turn_id = "turn_guard"
    input_event_key = "item_guard"
    response_id = "resp_guard"
    _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=input_event_key,
        response_id=response_id,
        origin="tool_output",
    )
    api._response_trace_context_by_id[response_id].update(
        {
            "followthrough_step_output_policy": "required_deliverable",
            "followthrough_post_completion_reason": "required_deliverable_owed",
            "tool_followup_step_output_policy": "required_deliverable",
            "tool_followup_post_completion_reason": "required_deliverable_owed",
        }
    )
    api._response_done_deliverable_arbitration = lambda **_kwargs: types.SimpleNamespace(
        selected=True,
        reason_code="normal",
        selected_candidate_id="terminal_selected",
    )
    api._selected_response_has_terminal_text_evidence = lambda **_kwargs: False
    api._required_deliverable_tool_execution_missing = lambda **_kwargs: False
    continuity_payload: dict[str, str] = {}
    api._apply_continuity_event = lambda _event, **kwargs: continuity_payload.update(kwargs)
    log_messages: list[str] = []

    def _capture(message, *args, **kwargs) -> None:
        rendered = message % args if args else str(message)
        if "continuity_response_done_handoff" in rendered:
            log_messages.append(rendered)

    monkeypatch.setattr("ai.realtime.response_terminal_handlers.logger.info", _capture)

    asyncio.run(api.handle_response_done({"type": "response.done", "response": {"id": response_id}}))

    assert any("selection_reason=required_deliverable_missing_substantive_content" in entry for entry in log_messages)
    assert continuity_payload.get("keep_ongoing") == "true"
    assert continuity_payload.get("close_commitment", "") == ""
    assert continuity_payload.get("complete_required_deliverable", "") == ""


def test_response_created_bridge_empty_retry_does_not_claim_pending_server_auto_owner() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api._current_input_event_key = "tool:call_bridge__empty_retry"
    api._pending_response_create_origins.append(
        {
            "origin": "server_auto",
            "turn_id": "turn_bridge",
            "input_event_key": "tool:call_bridge__empty_retry",
            "consumes_canonical_slot": "true",
        }
    )

    asyncio.run(
        api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-bridge-empty-retry",
                    "metadata": {
                        "empty_retry_materialization": "report_followup",
                        "empty_retry_origin": "tool_output_followthrough_bridge",
                    },
                },
            },
            ws,
        )
    )

    assert api._active_response_origin == "server_auto"
    assert api._active_response_input_event_key == "tool:call_bridge__empty_retry"
    assert api._active_server_auto_input_event_key is None
    assert api._pending_server_auto_response_for_turn(turn_id="turn_bridge") is None


def test_duplicate_assistant_message_create_single_flight_guard(monkeypatch) -> None:
    """Deterministic run-405 repro now guarded to single assistant_message response.create."""

    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_mix_5"

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _fake_recall(**_kwargs):
        return {
            "memories": [{"content": "Your pants are gray."}],
            "memory_cards": [{"confidence": "High"}],
            "memory_cards_text": "Relevant memory:\n- \"Your pants are gray.\"",
        }

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False

    async def _run() -> None:
        # 1) server_auto response.created before transcript.final
        await api.handle_event({"type": "response.created", "response": {"id": "resp-server-auto-1"}}, ws)

        # 2/3/4) transcript final => memory intent + preference recall suppression + queued assistant_message create
        await api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_pants",
                "transcript": "Hey Theo, do you remember what color my pants are?",
            },
            ws,
        )

        # Force a second queued assistant_message update while a response is active (replacement scenario).
        await api.send_assistant_message(
            "Yes, your pants are gray.",
            ws,
            response_metadata={
                "turn_id": "turn_1",
                "input_event_key": "item_pants",
                "trigger": "preference_recall",
            },
        )

        # 5) First response.done drains queue and emits first assistant_message response.create
        await api.handle_event({"type": "response.done", "response": {"id": "resp-server-auto-1"}}, ws)

        # Queue another assistant_message while the first assistant_message response is in flight.
        await api.send_assistant_message(
            "I do recall your pants are gray.",
            ws,
            response_metadata={
                "turn_id": "turn_1",
                "input_event_key": "item_pants",
                "trigger": "preference_recall",
            },
        )

        # 6) Later response.done drains pending queue and emits a second assistant_message response.create
        await api.handle_event({"type": "response.done", "response": {"id": "resp-assistant-1"}}, ws)

    asyncio.run(_run())

    response_create_events = [
        event
        for event in ws.sent
        if event.get("type") == "response.create"
        and ((event.get("response") or {}).get("metadata") or {}).get("origin") == "assistant_message"
    ]

    keyed = [
        {
            "order": idx,
            "origin": ((event.get("response") or {}).get("metadata") or {}).get("origin"),
            "input_event_key": ((event.get("response") or {}).get("metadata") or {}).get("input_event_key"),
        }
        for idx, event in enumerate(response_create_events, start=1)
    ]

    same_key = [entry for entry in keyed if entry["input_event_key"] == "item_pants"]

    # Guard expectation: no more than one assistant_message response.create for the same canonical utterance key.
    assert len(same_key) <= 1
    if same_key:
        assert same_key[0]["origin"] == "assistant_message"


def test_active_response_create_gate_queues_tool_followup_and_dedupes() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()

    assistant_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_1", "input_event_key": "item_assistant", "origin": "assistant_message"}},
    }
    tool_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_1",
                "input_event_key": "item_tool_followup",
                "origin": "tool_output",
                "tool_followup": "true",
                "tool_call_id": "call-1",
            }
        },
    }
    duplicate_assistant_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_1", "input_event_key": "item_tool_followup", "origin": "assistant_message"}},
    }

    async def _run() -> None:
        sent_assistant = await api._send_response_create(ws, assistant_event, origin="assistant_message")
        assert sent_assistant is True
        queued_tool = await api._send_response_create(ws, tool_event, origin="tool_output")
        assert queued_tool is False
        duplicate_queued = api._schedule_pending_response_create(
            websocket=ws,
            response_create_event=duplicate_assistant_event,
            origin="assistant_message",
            reason="active_response",
            record_ai_call=False,
            debug_context=None,
            memory_brief_note=None,
        )
        assert duplicate_queued is False

        api._active_response_id = None
        api._response_in_flight = False
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    sent_types = [event.get("type") for event in ws.sent]
    assert sent_types.count("response.create") == 2
    assert api._pending_response_create is None


def test_editor_path_single_assistant_create_with_guarded_server_auto(monkeypatch) -> None:
    """Control case: editor recall produces one assistant_message create, stale server_auto stays guarded."""

    api = _make_api_stub()
    ws = _RecordingWs()
    api.websocket = ws

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _fake_recall(**kwargs):
        query = str(kwargs.get("query") or "")
        if query.strip().lower() == "editor":
            return {
                "memories": [{"content": "Your favorite editor is Vim."}],
                "memory_cards": [{"confidence": "High"}],
                "memory_cards_text": "Relevant memory:\n- \"Your favorite editor is Vim.\"",
            }
        return {"memories": [], "memory_cards": [], "memory_cards_text": ""}

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False

    async def _run() -> None:
        # speculative server_auto before transcript
        await api.handle_event({"type": "response.created", "response": {"id": "resp-server-auto-editor-1"}}, ws)

        # transcript drives preference recall path
        await api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_editor",
                "transcript": "Do you remember what my favorite editor is?",
            },
            ws,
        )

        # response done drains queued assistant_message create
        await api.handle_event({"type": "response.done", "response": {"id": "resp-server-auto-editor-1"}}, ws)

        # consume assistant_message create acknowledgement first
        await api.handle_event({"type": "response.created", "response": {"id": "resp-assistant-editor-1"}}, ws)
        await api.handle_event({"type": "response.done", "response": {"id": "resp-assistant-editor-1"}}, ws)

        # stale/late server_auto for same key should be guarded and cancelled
        api._active_input_event_key_by_turn_id["turn_1"] = "item_editor"
        api._pending_server_auto_input_event_keys.append("item_editor")
        api._preference_recall_suppressed_input_event_keys.add("item_editor")
        await api.handle_event({"type": "response.created", "response": {"id": "resp-server-auto-editor-stale"}}, ws)

    asyncio.run(_run())

    assistant_creates = [
        event
        for event in ws.sent
        if event.get("type") == "response.create"
        and ((event.get("response") or {}).get("metadata") or {}).get("origin") == "assistant_message"
        and ((event.get("response") or {}).get("metadata") or {}).get("input_event_key") == "item_editor"
    ]
    cancels = [event for event in ws.sent if event.get("type") == "response.cancel"]

    assert len(assistant_creates) <= 1
    assert len(cancels) >= 1



def test_server_auto_synthetic_key_rebound_blocks_stale_assistant_message_release(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    api.websocket = ws

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _fake_recall(**_kwargs):
        return {
            "memories": [{"content": "Your pants are gray."}],
            "memory_cards": [{"confidence": "High"}],
            "memory_cards_text": "Relevant memory:\n- \"Your pants are gray.\"",
        }

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False

    async def _run() -> None:
        await api.handle_event({"type": "response.created", "response": {"id": "resp-server-auto-1"}}, ws)
        api._response_in_flight = True

        synthetic_key = str(api._active_server_auto_input_event_key or "")
        assert synthetic_key.startswith("synthetic_server_auto_")

        await api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_pants",
                "transcript": "Hey Theo, do you remember what color my pants are?",
            },
            ws,
        )

        assert api._active_server_auto_input_event_key == "item_pants"

        api._audio_playback_busy = True
        await api.handle_event({"type": "response.done", "response": {"id": "resp-server-auto-1"}}, ws)
        api._audio_playback_busy = False
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    assistant_creates = [
        event
        for event in ws.sent
        if event.get("type") == "response.create"
        and ((event.get("response") or {}).get("metadata") or {}).get("origin") == "assistant_message"
        and ((event.get("response") or {}).get("metadata") or {}).get("input_event_key") == "item_pants"
    ]

    assert assistant_creates == []



def test_empty_transcript_cancels_pending_micro_ack_without_emitting() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    class _MicroAckManager:
        def __init__(self) -> None:
            self.cancelled: list[tuple[str, str]] = []

        def cancel(self, *, turn_id: str, reason: str) -> None:
            self.cancelled.append((turn_id, reason))

    manager = _MicroAckManager()
    api._micro_ack_manager = manager
    api._pending_micro_ack_by_turn_channel = {
        ("turn_1", "voice"): type("_Marker", (), {"category": "start_of_work", "priority": 1, "reason": "speech_stopped"})()
    }

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
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    prompts: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        prompts.append(str(message))

    api.send_assistant_message = _assistant

    async def _run() -> None:
        await api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_empty",
                "transcript": "...",
            },
            ws,
        )

    asyncio.run(_run())

    assert manager.cancelled == [("turn_1", "transcript_completed_empty")]
    assert api._pending_micro_ack_by_turn_channel == {}
    assert prompts == []

def test_realtime_empty_transcript_blocks_speech(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

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
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    prompts: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        prompts.append(str(message))

    api.send_assistant_message = _assistant

    async def _run() -> None:
        api._active_response_origin = "server_auto"
        api._active_response_id = "resp-server-auto-empty-1"
        api._active_server_auto_input_event_key = "item_empty"
        api._record_pending_server_auto_response(
            turn_id="turn_1",
            response_id="resp-server-auto-empty-1",
            canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_empty"),
        )
        api._set_response_obligation(
            turn_id="turn_1",
            input_event_key="item_empty",
            source="input_audio_transcription",
        )
        await api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_empty",
                "transcript": "...",
            },
            ws,
        )

    asyncio.run(_run())

    cancels = [event for event in ws.sent if event.get("type") == "response.cancel"]
    assert cancels
    assert prompts == []

    obligation_key = api._response_obligation_key(turn_id="turn_1", input_event_key="item_empty")
    assert obligation_key not in api._response_obligations

    pending = api._pending_server_auto_response_for_turn(turn_id="turn_1")
    assert pending is not None
    assert pending.active is False
    assert pending.cancelled_for_upgrade is True

    assert api._response_delivery_state(turn_id="turn_1", input_event_key="item_empty") == "blocked_empty_transcript"

    watchdog_tasks = getattr(api, "_transcript_response_watchdog_tasks", {})
    task = watchdog_tasks.get("item_empty")
    assert task is None or task.done()


def test_empty_transcript_rearms_recording_when_not_playing_audio() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    class _Mic:
        def __init__(self) -> None:
            self.is_recording = False
            self.start_calls = 0

        def start_recording(self) -> None:
            self.start_calls += 1
            self.is_recording = True

    class _State:
        def __init__(self) -> None:
            self.state = InteractionState.THINKING
            self.transitions: list[tuple[InteractionState, str]] = []

        def update_state(self, state: InteractionState, reason: str) -> None:
            self.state = state
            self.transitions.append((state, reason))

    api.mic = _Mic()
    api.state_manager = _State()
    api._audio_playback_busy = False

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
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    async def _run() -> None:
        api._active_response_origin = "server_auto"
        api._active_response_id = "resp-server-auto-empty-2"
        api._active_server_auto_input_event_key = "item_empty_2"
        api._record_pending_server_auto_response(
            turn_id="turn_2",
            response_id="resp-server-auto-empty-2",
            canonical_key=api._canonical_utterance_key(turn_id="turn_2", input_event_key="item_empty_2"),
        )
        await api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_empty_2",
                "transcript": "...",
            },
            ws,
        )

    asyncio.run(_run())

    assert api.mic.start_calls == 1
    assert api.mic.is_recording is True
    assert api.state_manager.state == InteractionState.LISTENING
    assert api.state_manager.transitions[-1] == (
        InteractionState.LISTENING,
        "empty transcript blocked",
    )


def test_attention_gate_closed_rearms_recording_and_cancels_pending_server_auto() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    class _Mic:
        def __init__(self) -> None:
            self.is_recording = False
            self.start_calls = 0

        def start_recording(self) -> None:
            self.start_calls += 1
            self.is_recording = True

    class _State:
        def __init__(self) -> None:
            self.state = InteractionState.THINKING
            self.transitions: list[tuple[InteractionState, str]] = []

        def update_state(self, state: InteractionState, reason: str) -> None:
            self.state = state
            self.transitions.append((state, reason))

    api.mic = _Mic()
    api.state_manager = _State()
    api._audio_playback_busy = False

    async def _false(*_args, **_kwargs) -> bool:
        return False

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_verify_on_risk_clarify = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_admission = lambda **_kwargs: (False, "attention_gate_closed")
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}
    signal_calls: list[str] = []
    api._signal_server_auto_transcript_final = lambda *, turn_id: signal_calls.append(turn_id)

    async def _run() -> None:
        api._active_response_origin = "server_auto"
        api._active_response_id = "resp-server-auto-attn-closed"
        api._active_server_auto_input_event_key = "item_attn_closed"
        api._record_pending_server_auto_response(
            turn_id="turn_1",
            response_id="resp-server-auto-attn-closed",
            canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_attn_closed"),
        )
        await api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_attn_closed",
                "transcript": "this should be ignored",
            },
            ws,
        )

    asyncio.run(_run())

    cancels = [event for event in ws.sent if event.get("type") == "response.cancel"]
    assert len(cancels) == 1
    assert cancels[0]["response_id"] == "resp-server-auto-attn-closed"
    pending = api._pending_server_auto_response_for_turn(turn_id="turn_1")
    assert pending is not None
    assert pending.active is False
    assert pending.cancelled_for_upgrade is True
    assert api.mic.start_calls == 1
    assert api.mic.is_recording is True
    assert api.state_manager.state == InteractionState.LISTENING
    assert api.state_manager.transitions[-1] == (
        InteractionState.LISTENING,
        "attention gate closed",
    )
    assert api._attention_gate_blocked_listening_suppress_until_ts > 0.0
    assert api._attention_gate_blocked_listening_cue_suppressed(
        now=api._attention_gate_blocked_listening_suppress_until_ts - 0.01
    )
    assert signal_calls == []


def test_non_addressed_visual_question_skips_verify_and_blocks_attention_gate() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    verify_calls: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _verify(*_args, **kwargs) -> bool:
        verify_calls.append(str(kwargs.get("input_event_key") or ""))
        return False

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_verify_on_risk_clarify = _verify
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_admission = lambda **_kwargs: (False, "attention_gate_closed")
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    asyncio.run(
        api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_blocked_visual",
                "transcript": "what color is that on the screen",
            },
            ws,
        )
    )

    assert verify_calls == []
    verdict = api._get_response_gating_verdict(turn_id="turn_1", input_event_key="item_blocked_visual")
    assert verdict is not None
    assert verdict.action == "BLOCK"
    assert verdict.reason == "attention_gate_closed"


def test_direct_address_visual_question_runs_verify_path_after_attention_admission() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    verify_calls: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _verify(*_args, **kwargs) -> bool:
        verify_calls.append(str(kwargs.get("input_event_key") or ""))
        return False

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_verify_on_risk_clarify = _verify
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_admission = lambda **_kwargs: (True, "direct_address")
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}
    signal_calls: list[str] = []
    api._signal_server_auto_transcript_final = lambda *, turn_id: signal_calls.append(turn_id)

    asyncio.run(
        api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_direct_visual",
                "transcript": "Theo, what color is that?",
            },
            ws,
        )
    )

    assert verify_calls == ["item_direct_visual"]
    assert signal_calls == ["turn_1"]


def test_attention_exception_bypass_runs_verify_without_admission_check() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    verify_calls: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _verify(*_args, **kwargs) -> bool:
        verify_calls.append(str(kwargs.get("input_event_key") or ""))
        return False

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_verify_on_risk_clarify = _verify
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_exception = lambda **_kwargs: (True, "stop_abort_safety_interrupt")
    api._evaluate_attention_gate_admission = lambda **_kwargs: pytest.fail("admission should be skipped for bypass")
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    asyncio.run(
        api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_bypass_visual",
                "transcript": "stop what is this object",
            },
            ws,
        )
    )

    assert verify_calls == ["item_bypass_visual"]


def test_non_admitted_visual_question_preserves_blocked_turn_recovery() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    class _Mic:
        def __init__(self) -> None:
            self.is_recording = False
            self.start_calls = 0

        def start_recording(self) -> None:
            self.start_calls += 1
            self.is_recording = True

    class _State:
        def __init__(self) -> None:
            self.state = InteractionState.THINKING
            self.transitions: list[tuple[InteractionState, str]] = []

        def update_state(self, state: InteractionState, reason: str) -> None:
            self.state = state
            self.transitions.append((state, reason))

    verify_calls: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _verify(*_args, **kwargs) -> bool:
        verify_calls.append(str(kwargs.get("input_event_key") or ""))
        return False

    api.mic = _Mic()
    api.state_manager = _State()
    api._audio_playback_busy = False
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_verify_on_risk_clarify = _verify
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_admission = lambda **_kwargs: (False, "attention_gate_closed")
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    async def _run() -> None:
        api._active_response_origin = "server_auto"
        api._active_response_id = "resp-server-auto-visual-attn-closed"
        api._active_server_auto_input_event_key = "item_visual_blocked"
        api._record_pending_server_auto_response(
            turn_id="turn_1",
            response_id="resp-server-auto-visual-attn-closed",
            canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_visual_blocked"),
        )
        await api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_visual_blocked",
                "transcript": "what color is this",
            },
            ws,
        )

    asyncio.run(_run())

    assert verify_calls == []
    cancels = [event for event in ws.sent if event.get("type") == "response.cancel"]
    assert len(cancels) == 1
    assert cancels[0]["response_id"] == "resp-server-auto-visual-attn-closed"
    assert api._stale_response_ids() == {"resp-server-auto-visual-attn-closed"}
    assert api.mic.start_calls == 1
    assert api.mic.is_recording is True
    assert api.state_manager.state == InteractionState.LISTENING
    assert api.state_manager.transitions[-1] == (
        InteractionState.LISTENING,
        "attention gate closed",
    )


def test_direct_address_question_transcript_suppresses_latency_mask_micro_ack() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    recorded_reasons: list[str] = []

    async def _true(*_args, **_kwargs) -> bool:
        return True

    async def _false(*_args, **_kwargs) -> bool:
        return False

    api._maybe_schedule_micro_ack = lambda **kwargs: recorded_reasons.append(str(kwargs.get("reason") or ""))
    api._rebind_pending_micro_ack_after_transcript_final = lambda **_kwargs: 0
    api._cancel_micro_ack = lambda **_kwargs: None
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_verify_on_risk_clarify = _false
    api._maybe_handle_approval_response = _true
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_admission = lambda **_kwargs: (True, "direct_address")
    api._attention_on_transcript_finalized = lambda: None
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}
    api._utterance_trust_snapshot_by_input_event_key = {}

    async def _run() -> None:
        await api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_direct_question",
                "transcript": "Theo, what time is it",
            },
            ws,
        )

    asyncio.run(_run())

    assert "transcript_finalized" not in recorded_reasons


def test_preference_recall_preserves_parent_input_event_key_during_active_response() -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api._preference_recall_suppressed_input_event_keys.add("item_pants")

    captured: list[dict[str, object]] = []

    async def _capture_send_response_create(websocket, response_create_event, **_kwargs):
        captured.append(response_create_event)
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api.send_assistant_message(
            "Yes, your pants are gray.",
            ws,
            response_metadata={
                "turn_id": "turn_1",
                "input_event_key": "item_pants",
                "trigger": "preference_recall",
            },
        )
    )

    assert len(captured) == 1
    metadata = ((captured[0].get("response") or {}).get("metadata") or {})
    assert metadata.get("input_event_key") == "item_pants"

def test_response_done_preserves_cancelled_canonical_slot() -> None:
    api = _make_api_stub()
    turn_id = "turn_1"
    input_event_key = "item_editor"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._active_response_input_event_key = input_event_key
    api._set_response_delivery_state(turn_id=turn_id, input_event_key=input_event_key, state="cancelled")
    api._drain_response_create_queue = lambda: asyncio.sleep(0)

    asyncio.run(api.handle_response_done())

    assert api._response_delivery_state(turn_id=turn_id, input_event_key=input_event_key) == "cancelled"


def test_response_completed_preserves_cancelled_canonical_slot() -> None:
    api = _make_api_stub()
    turn_id = "turn_1"
    input_event_key = "item_editor"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._active_response_input_event_key = input_event_key
    api._set_response_delivery_state(turn_id=turn_id, input_event_key=input_event_key, state="cancelled")
    api._drain_response_create_queue = lambda: asyncio.sleep(0)

    asyncio.run(api.handle_response_completed())

    assert api._response_delivery_state(turn_id=turn_id, input_event_key=input_event_key) == "cancelled"


def test_canonical_audio_started_blocks_duplicate_response_create(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    api.websocket = ws
    log_messages: list[str] = []

    original_info = __import__("ai.realtime_api", fromlist=["logger"]).logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = message % args if args else message
        log_messages.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(__import__("ai.realtime_api", fromlist=["logger"]).logger, "info", _capture_info)

    async def _run() -> tuple[bool, bool]:
        first_event = {
            "type": "response.create",
            "response": {
                "metadata": {
                    "origin": "assistant_message",
                    "turn_id": "turn_1",
                    "input_event_key": "item_dup",
                }
            },
        }
        first_sent = await api._send_response_create(ws, first_event, origin="assistant_message")
        assert first_sent

        canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_dup")
        api._active_response_canonical_key = canonical_key
        await api.handle_event({"type": "response.output_audio.delta", "delta": "AAA="}, ws)
        api._canonical_lifecycle_state(canonical_key)["first_audio_started"] = True
        api._response_in_flight = False
        api.response_in_progress = False
        api._response_delivery_ledger.clear()
        api._response_created_canonical_keys.clear()

        duplicate_event = {
            "type": "response.create",
            "response": {
                "metadata": {
                    "origin": "assistant_message",
                    "turn_id": "turn_1",
                    "input_event_key": "item_dup",
                }
            },
        }
        second_sent = await api._send_response_create(ws, duplicate_event, origin="assistant_message")
        return first_sent, second_sent

    first_sent, second_sent = asyncio.run(_run())

    assert first_sent is True
    assert second_sent is False
    assistant_creates = [
        event
        for event in ws.sent
        if event.get("type") == "response.create"
        and ((event.get("response") or {}).get("metadata") or {}).get("input_event_key") == "item_dup"
    ]
    assert len(assistant_creates) == 1
    assert api._canonical_first_audio_started(api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_dup"))
    assert any("reason=canonical_audio_already_started" in message for message in log_messages)


def test_canonical_audio_started_blocks_all_canonical_origins() -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    api.websocket = ws
    api._canonical_first_audio_started = lambda _canonical_key: True

    async def _run() -> list[tuple[str, bool]]:
        results: list[tuple[str, bool]] = []
        for origin in ("assistant_message", "tool_output", "server_auto"):
            turn_id = f"turn_{origin}"
            input_event_key = f"item_{origin}"
            api._response_in_flight = False
            api.response_in_progress = False
            sent = await api._send_response_create(
                ws,
                {
                    "type": "response.create",
                    "response": {
                        "metadata": {
                            "origin": origin,
                            "turn_id": turn_id,
                            "input_event_key": input_event_key,
                        }
                    },
                },
                origin=origin,
            )
            results.append((origin, sent))
        return results

    results = asyncio.run(_run())
    assert results == [
        ("assistant_message", False),
        ("tool_output", False),
        ("server_auto", False),
    ]


def test_canonical_audio_started_allows_explicit_multipart_and_micro_ack() -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    api.websocket = ws

    async def _run() -> tuple[bool, bool]:
        canonical_key = api._canonical_utterance_key(turn_id="turn_multi", input_event_key="item_multi")
        api._canonical_lifecycle_state(canonical_key)["first_audio_started"] = True
        multipart_sent = await api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "origin": "assistant_message",
                        "turn_id": "turn_multi",
                        "input_event_key": "item_multi",
                        "explicit_multipart": "true",
                    }
                },
            },
            origin="assistant_message",
        )
        api._response_in_flight = False
        api.response_in_progress = False
        micro_ack_sent = await api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "origin": "assistant_message",
                        "turn_id": "turn_micro",
                        "input_event_key": "item_micro",
                        "consumes_canonical_slot": "false",
                    }
                },
            },
            origin="assistant_message",
        )
        return multipart_sent, micro_ack_sent

    multipart_sent, micro_ack_sent = asyncio.run(_run())
    assert multipart_sent is True
    assert micro_ack_sent is True


def test_tool_followup_response_scheduled_exactly_once_per_tool_call_id(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_1"
    api._current_input_event_key = "item_user_1"
    api._mark_utterance_info_summary = lambda **_kwargs: None

    async def _fake_add_no_tools(*_args, **_kwargs) -> None:
        return None

    async def _fake_research(**_kwargs):
        return {"research_id": "research_1", "summary": "done"}

    api._add_no_tools_follow_up_instruction = _fake_add_no_tools
    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "perform_research", _fake_research)

    asyncio.run(api.execute_function_call("perform_research", "call_research_1", {"query": "uptime"}, ws))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    metadata = ((response_create_events[0].get("response") or {}).get("metadata") or {})
    assert metadata.get("tool_followup") == "true"
    assert metadata.get("tool_call_id") == "call_research_1"
    assert metadata.get("input_event_key") == "tool:call_research_1"


def test_duplicate_tool_result_is_deduped_by_tool_followup_canonical_key(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_2"
    api._current_input_event_key = "item_user_2"
    api._mark_utterance_info_summary = lambda **_kwargs: None

    async def _fake_add_no_tools(*_args, **_kwargs) -> None:
        return None

    async def _fake_research(**_kwargs):
        return {"summary": "done"}

    api._add_no_tools_follow_up_instruction = _fake_add_no_tools
    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "perform_research", _fake_research)
    captured_logs: list[str] = []

    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    asyncio.run(api.execute_function_call("perform_research", "call_research_2", {"query": "logs"}, ws))
    api._response_in_flight = False
    asyncio.run(api.execute_function_call("perform_research", "call_research_2", {"query": "logs"}, ws))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    canonical_key = api._canonical_utterance_key(turn_id="turn_tool_2", input_event_key="tool:call_research_2")
    assert any(
        "tool_followup_arbitration outcome=deny reason=already_" in entry
        and f"canonical_key={canonical_key}" in entry
        for entry in captured_logs
    )


def test_non_tool_response_create_path_still_single_flight_guarded() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_user_1"
    api._current_input_event_key = "item_user_3"

    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create", "response": {"metadata": {"turn_id": "turn_user_1", "input_event_key": "item_user_3"}}},
            origin="assistant_message",
        )
    )
    api._response_in_flight = False
    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create", "response": {"metadata": {"turn_id": "turn_user_1", "input_event_key": "item_user_3"}}},
            origin="assistant_message",
        )
    )

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_reject_tool_call_with_assistant_message_avoids_duplicate_response_create() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_reject_1"
    api._current_input_event_key = "item_user_reject"
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._presented_actions = set()

    action = ActionPacket(
        id="call_reject_1",
        tool_name="perform_research",
        tool_args={"query": "status"},
        tier=1,
        what="Research service status",
        why="Need status for user response",
        impact="No side effects",
        rollback="Not applicable",
        alternatives=["Ask user to retry later"],
        confidence=0.9,
        cost="low",
        risk_flags=[],
        requires_confirmation=False,
    )

    asyncio.run(api._reject_tool_call(action, "invalid_arguments", ws, status="invalid_arguments"))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    metadata = ((response_create_events[0].get("response") or {}).get("metadata") or {})
    assert metadata.get("origin") == "assistant_message"


def test_tool_followup_scheduled_while_active_drains_once_without_active_error(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active-tool"
    api._current_response_turn_id = "turn_tool_active"
    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_active_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 0
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"
        api._response_in_flight = False
        api.response_in_progress = False
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) in {"creating", "created"}
    assert not any("Conversation already has an active response in progress" in entry for entry in captured_logs)
    assert any(
        "tool_followup_state" in entry
        and "state=blocked_active_response" in entry
        and "response_id=resp-active-tool" in entry
        for entry in captured_logs
    )
    assert any("tool_followup_state" in entry and "state=released_on_response_done" in entry for entry in captured_logs)
    assert any("tool_followup_state" in entry and "state=creating" in entry for entry in captured_logs)


def test_duplicate_tool_followup_delivery_events_only_create_once_after_drain(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active-tool"
    api._current_response_turn_id = "turn_tool_dup"
    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_dup_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        api._response_in_flight = False
        api.response_in_progress = False
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) in {"creating", "created"}
    assert any(
        "tool_followup_create_suppressed" in entry and f"canonical_key={canonical_key}" in entry
        for entry in captured_logs
    )




def test_tool_followup_blocked_by_active_response_released_on_response_done(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active-release"
    api._active_response_origin = "assistant_message"
    api._current_response_turn_id = "turn_tool_release"
    api._active_response_input_event_key = "item_active_release"
    api._active_response_canonical_key = api._canonical_utterance_key(
        turn_id="turn_tool_release",
        input_event_key="item_active_release",
    )
    api._active_input_event_key_by_turn_id["turn_tool_release"] = "item_active_release"
    api._record_pending_server_auto_response(
        turn_id="turn_tool_release",
        response_id="resp-active-release",
        canonical_key=api._active_response_canonical_key,
    )
    api._build_confirmation_transition_decision = lambda **_kwargs: type(
        "_Transition",
        (),
        {
            "allow_response_transition": True,
            "emit_reminder": False,
            "recover_mic": False,
            "close_reason": "",
        },
    )()
    api._confirmation_hold_components = lambda: (False, False, False, False)

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_release_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info
    original_warning = logger.warning

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    def _capture_warning(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_warning(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)
    monkeypatch.setattr(logger, "warning", _capture_warning)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 0
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"

        await api.handle_response_done({"type": "response.done", "response_id": "resp-active-release"})
        pending = api._pending_server_auto_response_for_turn(turn_id="turn_tool_release")
        assert pending is not None
        assert pending.active is False

        await api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-tool-release",
                    "metadata": {
                        "turn_id": "turn_tool_release",
                        "input_event_key": "tool:call_release_1",
                        "tool_followup": "true",
                        "tool_followup_release": "true",
                        "tool_call_id": "call_release_1",
                        "parent_turn_id": "turn_tool_release",
                        "parent_input_event_key": "item_active_release",
                    },
                },
            },
            ws,
        )
        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-tool-release"}})

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) == "done"
    selection_parent = api._terminal_deliverable_selection_store().get("resp-active-release")
    selection_tool = api._terminal_deliverable_selection_store().get("resp-tool-release")
    assert selection_parent == {
        "canonical_key": api._canonical_utterance_key(turn_id="turn_tool_release", input_event_key="item_active_release"),
        "selected": False,
        "reason": "tool_followup_precedence",
    }
    assert selection_tool == {
        "canonical_key": canonical_key,
        "semantic_owner_canonical_key": api._canonical_utterance_key(
            turn_id="turn_tool_release",
            input_event_key="item_active_release",
        ),
        "selected": True,
        "reason": "normal",
    }
    assert api._active_input_event_key_for_turn("turn_tool_release") == "tool:call_release_1"
    assert not getattr(api, "_response_create_queue", deque())
    assert any(
        "tool_followup_state" in entry
        and f"canonical_key={canonical_key}" in entry
        and "state=scheduled_release" in entry
        and "reason=response_done response_id=resp-active-release" in entry
        for entry in captured_logs
    )
    assert any("tool_followup_release_handoff" in entry and f"canonical_key={canonical_key}" in entry for entry in captured_logs)
    assert not any(
        "lifecycle_coherence_violation" in entry and "turn_active_key_canonical_mismatch" in entry
        and "response_id=resp-tool-release" in entry
        for entry in captured_logs
    )


def test_tool_output_semantic_answer_ownership_reconciles_to_parent_canonical_key(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_semantic_owner"
    api._active_input_event_key_by_turn_id["turn_semantic_owner"] = "item_parent_semantic"
    api._active_server_auto_input_event_key = "item_parent_semantic"
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp-parent-semantic"
    api._active_response_input_event_key = "item_parent_semantic"
    api._active_response_canonical_key = api._canonical_utterance_key(
        turn_id="turn_semantic_owner",
        input_event_key="item_parent_semantic",
    )
    api._response_in_flight = True
    api.response_in_progress = True
    parent_canonical_key = api._canonical_utterance_key(
        turn_id="turn_semantic_owner",
        input_event_key="item_parent_semantic",
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id="turn_semantic_owner",
        input_event_key="item_parent_semantic",
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", False),
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", "resp-parent-semantic"),
        ),
    )
    api._record_response_trace_context(
        "resp-parent-semantic",
        turn_id="turn_semantic_owner",
        input_event_key="item_parent_semantic",
        canonical_key=parent_canonical_key,
        origin="server_auto",
    )
    response_create_event, child_canonical_key = api._build_tool_followup_response_create_event(call_id="call_semantic_owner")
    api._record_pending_server_auto_response(
        turn_id="turn_semantic_owner",
        response_id="resp-parent-semantic",
        canonical_key=parent_canonical_key,
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=child_canonical_key) == "blocked_active_response"

        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-parent-semantic"}})

        await api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-child-semantic",
                    "metadata": {
                        "turn_id": "turn_semantic_owner",
                        "input_event_key": "tool:call_semantic_owner",
                        "tool_followup": "true",
                        "tool_followup_release": "true",
                        "tool_call_id": "call_semantic_owner",
                        "parent_turn_id": "turn_semantic_owner",
                        "parent_input_event_key": "item_parent_semantic",
                    },
                },
            },
            ws,
        )
        api._record_terminal_response_text(
            response_id="resp-child-semantic",
            text="All set back to center. You're holding a white mug.",
        )
        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-child-semantic"}})

    asyncio.run(_run())

    efficiency = api._conversation_efficiency_state(turn_id="turn_semantic_owner")
    assert efficiency.substantive_count == 1
    assert efficiency.substantive_count_by_canonical == {parent_canonical_key: 1}
    parent_state = api._canonical_response_state(parent_canonical_key)
    child_state = api._canonical_response_state(child_canonical_key)
    assert parent_state is not None
    assert child_state is not None
    assert parent_state.deliverable_observed is True
    assert parent_state.deliverable_class == "final"
    assert parent_state.turn_id == "turn_semantic_owner"
    assert parent_state.input_event_key == "item_parent_semantic"
    assert child_state.done is True
    assert child_state.turn_id == "turn_semantic_owner"
    assert child_state.input_event_key == "tool:call_semantic_owner"
    assert child_state.response_id == "resp-child-semantic"
    assert api._tool_followup_state(canonical_key=child_canonical_key) == "done"
    selection_tool = api._terminal_deliverable_selection_store().get("resp-child-semantic")
    assert selection_tool == {
        "canonical_key": child_canonical_key,
        "semantic_owner_canonical_key": parent_canonical_key,
        "selected": True,
        "reason": "normal",
    }
    assert any(
        "semantic_substantive_owner_reconciled" in entry
        and f"execution_canonical_key={child_canonical_key}" in entry
        and f"semantic_owner_canonical_key={parent_canonical_key}" in entry
        and "response_id=resp-child-semantic" in entry
        for entry in captured_logs
    )
    assert any(
        "conversation_efficiency" in entry
        and f"canonical_key={parent_canonical_key}" in entry
        and "turn_id=turn_semantic_owner" in entry
        for entry in captured_logs
    )


def test_response_done_coherence_allows_execution_or_queued_canonical_after_tool_output_cleanup(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_tool_done_coherence"
    parent_input_event_key = "item_parent_tool_done_coherence"
    parent_response_id = "resp-parent-tool-done-coherence"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )

    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._active_server_auto_input_event_key = parent_input_event_key
    api._active_response_origin = "server_auto"
    api._active_response_id = parent_response_id
    api._active_response_input_event_key = parent_input_event_key
    api._active_response_canonical_key = parent_canonical_key
    api._response_in_flight = True
    api.response_in_progress = True
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", False),
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
        ),
    )
    api._record_response_trace_context(
        parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        canonical_key=parent_canonical_key,
        origin="server_auto",
    )
    api._record_pending_server_auto_response(
        turn_id=turn_id,
        response_id=parent_response_id,
        canonical_key=parent_canonical_key,
    )

    first_event, first_canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_tool_done_coherence_first",
        response_create_event={"type": "response.create"},
    )
    second_event, second_canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_tool_done_coherence_second",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info
    original_warning = logger.warning
    original_debug = logger.debug

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    def _capture_warning(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_warning(message, *args, **kwargs)

    def _capture_debug(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_debug(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "warning", _capture_warning)
    monkeypatch.setattr(logger, "debug", _capture_debug)

    async def _run() -> None:
        await api._send_response_create(ws, first_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=first_canonical_key) == "blocked_active_response"

        await api._send_response_create(ws, second_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=second_canonical_key) == "blocked_active_response"

        await api.handle_response_done({"type": "response.done", "response": {"id": parent_response_id}})

        await api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-child-tool-done-coherence-1",
                    "metadata": {
                        "turn_id": turn_id,
                        "input_event_key": "tool:call_tool_done_coherence_first",
                        "tool_followup": "true",
                        "tool_followup_release": "true",
                        "tool_call_id": "call_tool_done_coherence_first",
                        "parent_turn_id": turn_id,
                        "parent_input_event_key": parent_input_event_key,
                    },
                },
            },
            ws,
        )
        api._record_terminal_response_text(
            response_id="resp-child-tool-done-coherence-1",
            text="The first tool-output turn completed with a substantive answer.",
        )

        api._active_input_event_key_by_turn_id[turn_id] = "tool:call_tool_done_coherence_second"
        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-child-tool-done-coherence-1"}})

    asyncio.run(_run())

    assert api._tool_followup_state(canonical_key=first_canonical_key) == "done"
    assert api._tool_followup_state(canonical_key=second_canonical_key) in {
        "scheduled_release",
        "creating",
        "created",
        "released_on_response_done",
        "dropped",
    }
    assert any(
        "semantic_answer_owner_resolved" in entry
        and "response_id=resp-child-tool-done-coherence-1" in entry
        and f"semantic_owner_canonical_key={parent_canonical_key}" in entry
        and "reason_code=parent_promoted_from_tool_output" in entry
        for entry in captured_logs
    )
    assert not any(
        "lifecycle_coherence_violation" in entry
        and "turn_active_key_canonical_mismatch" in entry
        and "response_id=resp-child-tool-done-coherence-1" in entry
        for entry in captured_logs
    )


def test_parent_without_terminal_text_still_releases_child_followup_after_response_done() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_parent_release_fix"
    parent_input_event_key = "item_parent_release_fix"
    parent_response_id = "resp_parent_release_fix"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._active_response_origin = "server_auto"
    api._active_response_id = parent_response_id
    api._active_response_input_event_key = parent_input_event_key
    api._active_response_canonical_key = parent_canonical_key
    api._response_in_flight = True
    api.response_in_progress = True
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
        ),
    )
    api._record_response_trace_context(
        parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        canonical_key=parent_canonical_key,
        origin="server_auto",
    )
    response_create_event, child_canonical_key = api._build_tool_followup_response_create_event(call_id="call_parent_release_fix")

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=child_canonical_key) == "blocked_active_response"
        await api.handle_response_done({"type": "response.done", "response": {"id": parent_response_id}})

    asyncio.run(_run())

    assert api._tool_followup_state(canonical_key=child_canonical_key) in {"creating", "created", "released_on_response_done"}
    assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 1


def test_same_turn_suppression_still_holds_after_semantic_reconcile() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    turn_id = "turn_owner_after_semantic"
    parent_input_event_key = "item_owner_after_semantic"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    child_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_owner_after_semantic")
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._response_in_flight = True
    api.response_in_progress = True
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: setattr(record, "created", True),
    )
    api._record_terminal_response_text(
        response_id="resp-owner-after-semantic",
        text="The answer is ready.",
    )

    api._apply_terminal_deliverable_selection(
        canonical_key=child_canonical_key,
        semantic_owner_canonical_key=parent_canonical_key,
        response_id="resp-owner-after-semantic",
        turn_id=turn_id,
        input_event_key="tool:call_owner_after_semantic",
        selected=True,
        selection_reason="normal",
    )

    owner_reason = api._assistant_message_same_turn_owner_reason(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        canonical_key=parent_canonical_key,
    )

    assert owner_reason == "terminal_deliverable_owned"


def test_semantic_owner_terminal_text_promotion_bridges_parent_coverage_after_selected_tool_output() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_bridge"
    parent_input_event_key = "item_parent_semantic_bridge"
    child_input_event_key = "tool:call_semantic_bridge"
    response_id = "resp-semantic-bridge"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    child_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=child_input_event_key)

    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: setattr(record, "origin", "assistant_message"),
    )
    api._canonical_response_state_mutate(
        canonical_key=child_canonical_key,
        turn_id=turn_id,
        input_event_key=child_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", response_id),
            setattr(record, "done", True),
        ),
    )

    api._apply_terminal_deliverable_selection(
        canonical_key=child_canonical_key,
        semantic_owner_canonical_key=parent_canonical_key,
        response_id=response_id,
        turn_id=turn_id,
        input_event_key=child_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    parent_state = api._canonical_response_state(parent_canonical_key)
    assert parent_state is not None
    covered, coverage_source, _observed, deliverable_class, terminal_selected, terminal_reason = api._parent_response_coverage_state(
        parent_state=parent_state,
    )
    assert covered is False
    assert coverage_source == "none"
    assert deliverable_class == "unknown"
    assert terminal_selected is False
    assert terminal_reason == "unknown"

    api._record_terminal_response_text(
        response_id=response_id,
        text="The mug is white and your hand is centered again.",
    )

    parent_state = api._canonical_response_state(parent_canonical_key)
    assert parent_state is not None
    covered, coverage_source, observed, deliverable_class, terminal_selected, terminal_reason = api._parent_response_coverage_state(
        parent_state=parent_state,
    )
    assert covered is True
    assert coverage_source == "canonical"
    assert observed is True
    assert deliverable_class == "final"
    assert terminal_selected is True
    assert terminal_reason == "normal"
    assert parent_state.response_id == response_id
    efficiency = api._conversation_efficiency_state(turn_id=turn_id)
    assert efficiency.substantive_count == 1
    assert efficiency.substantive_count_by_canonical == {parent_canonical_key: 1}


def test_semantic_substantive_owner_reconcile_is_idempotent_for_repeated_child_done() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_idempotent"
    execution_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_semantic_idempotent")
    owner_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_semantic_idempotent")
    api._record_substantive_response(turn_id=turn_id, canonical_key=execution_key)
    api._record_terminal_response_text(response_id="resp-semantic-idempotent", text="Here's the answer.")

    api._reconcile_semantic_substantive_owner(
        turn_id=turn_id,
        execution_canonical_key=execution_key,
        semantic_owner_canonical_key=owner_key,
        response_id="resp-semantic-idempotent",
    )
    api._reconcile_terminal_substantive_response(
        turn_id=turn_id,
        canonical_key=owner_key,
        response_id="resp-semantic-idempotent",
        selected=True,
        selection_reason="normal",
    )

    api._reconcile_semantic_substantive_owner(
        turn_id=turn_id,
        execution_canonical_key=execution_key,
        semantic_owner_canonical_key=owner_key,
        response_id="resp-semantic-idempotent",
    )
    api._reconcile_terminal_substantive_response(
        turn_id=turn_id,
        canonical_key=owner_key,
        response_id="resp-semantic-idempotent",
        selected=True,
        selection_reason="normal",
    )

    efficiency = api._conversation_efficiency_state(turn_id=turn_id)
    assert efficiency.substantive_count == 1
    assert efficiency.substantive_count_by_canonical == {owner_key: 1}


def test_missing_parent_lineage_falls_back_to_execution_canonical_key() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    execution_key = api._canonical_utterance_key(turn_id="turn_semantic_fallback", input_event_key="tool:call_fallback")
    api._record_response_trace_context(
        "resp-semantic-fallback",
        turn_id="turn_semantic_fallback",
        input_event_key="tool:call_fallback",
        canonical_key=execution_key,
        origin="tool_output",
    )

    semantic_key = api._resolve_semantic_answer_owner_for_response(
        turn_id="turn_semantic_fallback",
        input_event_key="tool:call_fallback",
        origin="tool_output",
        response_id="resp-semantic-fallback",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )

    assert semantic_key == execution_key


def test_runtime_semantic_owner_decision_wrapper_matches_legacy_key_resolution() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_wrapper"
    execution_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_wrapper")
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_wrapper_parent")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key="item_wrapper_parent",
        mutator=lambda record: setattr(record, "created", True),
    )
    api._record_response_trace_context(
        "resp-semantic-wrapper",
        turn_id=turn_id,
        input_event_key="tool:call_wrapper",
        canonical_key=execution_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key="item_wrapper_parent",
    )

    decision = api._semantic_owner_decision_for_response(
        turn_id=turn_id,
        input_event_key="tool:call_wrapper",
        origin="tool_output",
        response_id="resp-semantic-wrapper",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )
    semantic_key = api._resolve_semantic_answer_owner_for_response(
        turn_id=turn_id,
        input_event_key="tool:call_wrapper",
        origin="tool_output",
        response_id="resp-semantic-wrapper",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )

    assert decision.semantic_owner_canonical_key == parent_key
    assert decision.selected_candidate_id == "semantic_owner_parent"
    assert semantic_key == decision.semantic_owner_canonical_key


def test_runtime_semantic_owner_decision_wrapper_falls_back_when_parent_state_missing() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_missing_parent"
    execution_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_missing_parent")
    api._record_response_trace_context(
        "resp-semantic-missing-parent",
        turn_id=turn_id,
        input_event_key="tool:call_missing_parent",
        canonical_key=execution_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key="item_missing_parent",
    )

    decision = api._semantic_owner_decision_for_response(
        turn_id=turn_id,
        input_event_key="tool:call_missing_parent",
        origin="tool_output",
        response_id="resp-semantic-missing-parent",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )
    semantic_key = api._resolve_semantic_answer_owner_for_response(
        turn_id=turn_id,
        input_event_key="tool:call_missing_parent",
        origin="tool_output",
        response_id="resp-semantic-missing-parent",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )

    assert decision.semantic_owner_canonical_key == execution_key
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "parent_canonical_missing"
    assert semantic_key == decision.semantic_owner_canonical_key


def test_tool_derived_parent_lineage_does_not_create_bogus_semantic_owner() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    execution_key = api._canonical_utterance_key(turn_id="turn_semantic_nested", input_event_key="tool:call_nested")
    api._record_response_trace_context(
        "resp-semantic-nested",
        turn_id="turn_semantic_nested",
        input_event_key="tool:call_nested",
        canonical_key=execution_key,
        origin="tool_output",
        parent_turn_id="turn_semantic_nested",
        parent_input_event_key="tool:call_parent_nested",
    )

    semantic_key = api._resolve_semantic_answer_owner_for_response(
        turn_id="turn_semantic_nested",
        input_event_key="tool:call_nested",
        origin="tool_output",
        response_id="resp-semantic-nested",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )

    assert semantic_key == execution_key


def test_runtime_semantic_owner_decision_wrapper_rejects_bogus_nested_tool_lineage() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_nested_wrapper"
    execution_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_nested_wrapper")
    api._record_response_trace_context(
        "resp-semantic-nested-wrapper",
        turn_id=turn_id,
        input_event_key="tool:call_nested_wrapper",
        canonical_key=execution_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key="tool:call_parent_nested_wrapper",
    )

    decision = api._semantic_owner_decision_for_response(
        turn_id=turn_id,
        input_event_key="tool:call_nested_wrapper",
        origin="tool_output",
        response_id="resp-semantic-nested-wrapper",
        done_canonical_key=execution_key,
        selected=True,
        selection_reason="normal",
    )

    assert decision.semantic_owner_canonical_key == execution_key
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "parent_input_tool_prefixed"


def test_chained_tool_output_semantic_owner_reuses_selected_parent_lineage() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_chain"
    parent_input_event_key = "item_semantic_chain_parent"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    child_input_event_key = "tool:call_chain_child"
    child_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=child_input_event_key)
    grandchild_input_event_key = "tool:call_chain_grandchild"
    grandchild_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=grandchild_input_event_key)

    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "response_id", "resp-semantic-chain-parent"),
            setattr(record, "origin", "server_auto"),
        ),
    )
    api._record_terminal_response_text(response_id="resp-semantic-chain-child", text="You are holding a mug.")
    api._apply_terminal_deliverable_selection(
        canonical_key=child_key,
        semantic_owner_canonical_key=parent_key,
        response_id="resp-semantic-chain-child",
        turn_id=turn_id,
        input_event_key=child_input_event_key,
        selected=True,
        selection_reason="normal",
    )
    api._record_response_trace_context(
        "resp-semantic-chain-child",
        turn_id=turn_id,
        input_event_key=child_input_event_key,
        canonical_key=child_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key=child_input_event_key,
    )

    decision = api._semantic_owner_decision_for_response(
        turn_id=turn_id,
        input_event_key=grandchild_input_event_key,
        origin="tool_output",
        response_id="resp-semantic-chain-child",
        done_canonical_key=grandchild_key,
        selected=True,
        selection_reason="normal",
    )

    assert decision.semantic_owner_canonical_key == parent_key
    assert decision.selected_candidate_id == "semantic_owner_parent"
    assert decision.reason_code == "parent_promoted_from_tool_output"


def test_chained_tool_followup_parent_key_prefers_non_tool_trace_parent_after_bridged_empty_step() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_semantic_chain_bridge"
    parent_input_event_key = "item_semantic_chain_bridge_parent"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._active_server_auto_input_event_key = parent_input_event_key
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp-semantic-chain-bridge-parent"
    api._active_response_input_event_key = parent_input_event_key
    api._active_response_canonical_key = parent_canonical_key
    api._response_in_flight = True
    api.response_in_progress = True
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", "resp-semantic-chain-bridge-parent"),
        ),
    )
    api._record_response_trace_context(
        "resp-semantic-chain-bridge-parent",
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        canonical_key=parent_canonical_key,
        origin="server_auto",
    )
    api._record_pending_server_auto_response(
        turn_id=turn_id,
        response_id="resp-semantic-chain-bridge-parent",
        canonical_key=parent_canonical_key,
    )

    async def _run() -> None:
        first_event, first_canonical_key = api._build_tool_followup_response_create_event(call_id="call_bridge_first")
        await api._send_response_create(ws, first_event, origin="tool_output")
        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-semantic-chain-bridge-parent"}})
        await api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-semantic-chain-bridge-first",
                    "metadata": {
                        "turn_id": turn_id,
                        "input_event_key": "tool:call_bridge_first",
                        "tool_followup": "true",
                        "tool_followup_release": "true",
                        "tool_call_id": "call_bridge_first",
                        "parent_turn_id": turn_id,
                        "parent_input_event_key": parent_input_event_key,
                    },
                },
            },
            ws,
        )
        await api.handle_response_done(
            {"type": "response.done", "response": {"id": "resp-semantic-chain-bridge-first"}}
        )

        second_event, _ = api._build_tool_followup_response_create_event(call_id="call_bridge_second")
        second_metadata = second_event["response"]["metadata"]
        assert second_metadata.get("parent_input_event_key") == parent_input_event_key
        await api._send_response_create(ws, second_event, origin="tool_output")
        await api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-semantic-chain-bridge-second",
                    "metadata": {
                        "turn_id": turn_id,
                        "input_event_key": "tool:call_bridge_second",
                        "tool_followup": "true",
                        "tool_call_id": "call_bridge_second",
                        "parent_turn_id": turn_id,
                        "parent_input_event_key": parent_input_event_key,
                    },
                },
            },
            ws,
        )
        api._record_terminal_response_text(
            response_id="resp-semantic-chain-bridge-second",
            text="Final report with substantive deliverable.",
        )
        await api.handle_response_done(
            {"type": "response.done", "response": {"id": "resp-semantic-chain-bridge-second"}}
        )

        selection = api._terminal_deliverable_selection_store().get("resp-semantic-chain-bridge-second")
        assert isinstance(selection, dict)
        assert selection.get("selected") is True
        assert selection.get("reason") == "normal"
        assert selection.get("semantic_owner_canonical_key") == parent_canonical_key
        assert first_canonical_key != parent_canonical_key

    asyncio.run(_run())


def test_parent_semantic_promotion_requires_terminal_text_evidence() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_promotion_gate"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_semantic_parent")
    child_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_semantic_parent")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key="item_semantic_parent",
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", True),
            setattr(record, "origin", "server_auto"),
        ),
    )

    api._apply_terminal_deliverable_selection(
        canonical_key=child_key,
        semantic_owner_canonical_key=parent_key,
        response_id="resp-semantic-parent",
        turn_id=turn_id,
        input_event_key="tool:call_semantic_parent",
        selected=True,
        selection_reason="normal",
    )

    parent_state = api._canonical_response_state(parent_key)
    assert parent_state is not None
    assert parent_state.deliverable_class != "final"
    assert parent_state.deliverable_observed is False
    assert parent_state.input_event_key == "item_semantic_parent"


def test_parent_semantic_promotion_preserves_parent_input_lineage() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_lineage"
    parent_input_event_key = "item_semantic_lineage_parent"
    child_input_event_key = "tool:call_semantic_lineage_child"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    child_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=child_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "response_id", "resp-parent-lineage"),
            setattr(record, "origin", "server_auto"),
        ),
    )
    api._record_terminal_response_text(
        response_id="resp-child-lineage",
        text="You are holding a mug.",
    )

    api._apply_terminal_deliverable_selection(
        canonical_key=child_key,
        semantic_owner_canonical_key=parent_key,
        response_id="resp-child-lineage",
        turn_id=turn_id,
        input_event_key=child_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    parent_state = api._canonical_response_state(parent_key)
    child_state = api._canonical_response_state(child_key)

    assert parent_state is not None
    assert child_state is not None
    assert parent_state.input_event_key == parent_input_event_key
    assert parent_state.turn_id == turn_id
    assert parent_state.response_id == "resp-parent-lineage"
    assert parent_state.deliverable_observed is True
    assert parent_state.deliverable_class == "final"
    assert child_state.input_event_key == child_input_event_key
    assert child_state.response_id == "resp-child-lineage"


def test_chained_tool_followup_parent_coverage_uses_promoted_semantic_owner_state() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_followup_chain"
    parent_input_event_key = "item_semantic_followup_parent"
    first_child_input_event_key = "tool:call_gesture_look_center"
    second_child_input_event_key = "tool:call_recall_memories"
    parent_response_id = "resp-semantic-followup-parent"
    first_child_response_id = "resp-semantic-followup-child"

    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    first_child_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=first_child_input_event_key)
    second_child_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=second_child_input_event_key)

    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", True),
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
        ),
    )
    api._canonical_response_state_mutate(
        canonical_key=first_child_key,
        turn_id=turn_id,
        input_event_key=first_child_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", True),
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", first_child_response_id),
        ),
    )
    api._record_terminal_response_text(
        response_id=first_child_response_id,
        text="I centered my gaze and found the mug in your hand.",
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=first_child_key,
        semantic_owner_canonical_key=parent_key,
        response_id=first_child_response_id,
        turn_id=turn_id,
        input_event_key=first_child_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    parent_state = api._canonical_response_state(parent_key)
    assert parent_state is not None
    covered, coverage_source, observed, deliverable_class, terminal_selected, terminal_reason = api._parent_response_coverage_state(
        parent_state=parent_state,
        parent_canonical_key=parent_key,
    )
    assert covered is True
    assert coverage_source == "canonical"
    assert observed is True
    assert deliverable_class == "final"
    assert terminal_selected is True
    assert terminal_reason == "normal"
    assert parent_state.response_id == parent_response_id

    parent_entry = api._resolve_parent_state_for_tool_followup(
        response_metadata={
            "turn_id": turn_id,
            "parent_turn_id": turn_id,
            "parent_input_event_key": parent_input_event_key,
            "input_event_key": second_child_input_event_key,
            "tool_call_id": "call_recall_memories",
        },
        blocked_by_response_id=first_child_response_id,
    )

    assert parent_entry is not None
    assert parent_entry[0] == parent_key
    assert parent_entry[1].response_id == parent_response_id

    covered, coverage_source, observed, deliverable_class, terminal_selected, terminal_reason = api._parent_response_coverage_state(
        parent_state=parent_entry[1],
        parent_canonical_key=parent_entry[0],
    )
    assert covered is True
    assert coverage_source == "canonical"
    assert observed is True
    assert deliverable_class == "final"
    assert terminal_selected is True
    assert terminal_reason == "normal"
    assert parent_entry[1].response_id == parent_response_id
    assert second_child_key not in (api._conversation_efficiency_state(turn_id=turn_id).substantive_count_by_canonical or {})


def test_semantic_substantive_owner_reconcile_moves_create_time_count_to_parent_once() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_move_once"
    execution_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_semantic_move_once")
    owner_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_semantic_move_once")
    api._record_substantive_response(turn_id=turn_id, canonical_key=execution_key)
    api._record_terminal_response_text(response_id="resp-semantic-move-once", text="Here is the result.")

    api._reconcile_semantic_substantive_owner(
        turn_id=turn_id,
        execution_canonical_key=execution_key,
        semantic_owner_canonical_key=owner_key,
        response_id="resp-semantic-move-once",
    )
    api._reconcile_terminal_substantive_response(
        turn_id=turn_id,
        canonical_key=owner_key,
        response_id="resp-semantic-move-once",
        selected=True,
        selection_reason="normal",
    )

    efficiency = api._conversation_efficiency_state(turn_id=turn_id)
    assert efficiency.substantive_count == 1
    assert efficiency.substantive_count_by_canonical == {owner_key: 1}
    assert execution_key not in efficiency.duplicate_alerted_canonical_keys


def test_semantic_substantive_owner_reconcile_dedupes_when_owner_already_counted() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_semantic_owner_already_counted"
    execution_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_semantic_owner_existing")
    owner_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_semantic_owner_existing")
    api._record_substantive_response(turn_id=turn_id, canonical_key=owner_key)
    api._record_substantive_response(turn_id=turn_id, canonical_key=execution_key)

    api._reconcile_semantic_substantive_owner(
        turn_id=turn_id,
        execution_canonical_key=execution_key,
        semantic_owner_canonical_key=owner_key,
        response_id="resp-semantic-owner-existing",
    )

    efficiency = api._conversation_efficiency_state(turn_id=turn_id)
    assert efficiency.substantive_count == 1
    assert efficiency.substantive_count_by_canonical == {owner_key: 1}


def test_tool_followup_second_arbitration_denied_for_same_canonical_key(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_arb"
    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_arb_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        await api._send_response_create(ws, response_create_event, origin="tool_output")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert any(
        "tool_followup_arbitration outcome=deny reason=already_creating" in entry
        and f"canonical_key={canonical_key}" in entry
        for entry in captured_logs
    )


def test_response_create_queue_priority_and_fifo_under_contention() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active"
    api._active_response_origin = "assistant_message"
    api._current_response_turn_id = "turn-priority"
    api._active_response_input_event_key = "item-active"

    tool_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn-priority", "input_event_key": "item_tool", "trigger": "tool"}},
    }
    clarify_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn-priority", "input_event_key": "item_clarify", "trigger": "clarify"}},
    }

    async def _run() -> None:
        await api._send_response_create(ws, tool_event, origin="tool_output")
        await api._send_response_create(ws, clarify_event, origin="assistant_message")
        assert len([e for e in ws.sent if e.get("type") == "response.create"]) == 0

        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-active"}})
        api._active_response_id = "resp-tool"
        api._response_in_flight = True
        api.response_in_progress = True
        await api.handle_response_done({"type": "response.done", "response": {"id": "resp-tool"}})

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 2
    ordered_keys = [((event.get("response") or {}).get("metadata") or {}).get("input_event_key") for event in response_create_events]
    assert ordered_keys == ["item_tool", "item_clarify"]


def test_response_create_queue_dedupes_same_canonical_key_when_blocked() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active"

    duplicate_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn-dedupe", "input_event_key": "item_dup", "trigger": "clarify"}},
    }

    async def _run() -> None:
        await api._send_response_create(ws, duplicate_event, origin="assistant_message")
        await api._send_response_create(ws, duplicate_event, origin="assistant_message")
        api._response_in_flight = False
        api.response_in_progress = False
        api._active_response_id = None
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_response_create_queue_drains_on_active_cleared_fallback() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-stuck"

    queued_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn-fallback", "input_event_key": "item_fallback", "trigger": "clarify"}},
    }

    async def _run() -> None:
        await api._send_response_create(ws, queued_event, origin="assistant_message")
        api._response_in_flight = False
        api.response_in_progress = False
        api._active_response_id = None
        await api._drain_response_create_queue(source_trigger="active_cleared")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_response_create_queue_stats_increment_and_drain_under_active_block() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active"
    api._active_response_origin = "assistant_message"

    first_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn-stats", "input_event_key": "item_first", "trigger": "clarify"}},
    }
    second_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn-stats", "input_event_key": "item_second", "trigger": "clarify"}},
    }

    async def _run() -> None:
        await api._send_response_create(ws, first_event, origin="assistant_message")
        await api._send_response_create(ws, second_event, origin="assistant_message")
        assert len([e for e in ws.sent if e.get("type") == "response.create"]) == 0

        api._response_in_flight = False
        api.response_in_progress = False
        api._active_response_id = None
        await api._drain_response_create_queue(source_trigger="response_done")
        api._response_in_flight = False
        api.response_in_progress = False
        api._active_response_id = None
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 2
    assert ((response_create_events[-1].get("response") or {}).get("metadata") or {}).get("input_event_key") == "item_second"
    assert api._response_create_queued_creates_total >= 2
    assert api._response_create_drains_total >= 2


def test_server_auto_short_utterance_defers_audio_until_transcript_final(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn-delay"
    api._active_input_event_key_by_turn_id["turn-delay"] = "item_delay"
    api._active_utterance = {"duration_ms": 900, "transcript": ""}

    deferred_calls: list[tuple[str, str, str]] = []
    started_audio: list[str] = []

    api._upgrade_likely_for_server_auto_turn = lambda **_kwargs: (False, "none")

    def _record_defer(*, turn_id: str, input_event_key: str, response_id: str) -> None:
        deferred_calls.append((turn_id, input_event_key, response_id))

    api._schedule_server_auto_audio_deferral = _record_defer
    api._start_audio_response_if_needed = lambda *, response_id: started_audio.append(str(response_id))
    api._mark_transcript_response_outcome = lambda **_kwargs: None

    cancelled_audio_races: list[str] = []
    api._record_cancelled_audio_race_transition = lambda **kwargs: cancelled_audio_races.append(str(kwargs.get("response_id") or ""))

    async def _run() -> None:
        await api._handle_response_created_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-server-auto-delay",
                    "metadata": {
                        "turn_id": "turn-delay",
                        "input_event_key": "item_delay",
                    },
                },
            },
            ws,
        )

    asyncio.run(_run())

    assert deferred_calls == [("turn-delay", "item_delay", "resp-server-auto-delay")]
    assert started_audio == []
    assert cancelled_audio_races == []


def test_server_auto_response_create_queued_until_transcript_final_then_drained() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    awaiting_transcript_final = {"value": True}
    api._transcript_final_missing_for_turn = lambda **_kwargs: awaiting_transcript_final["value"]

    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn-await",
                "input_event_key": "item-await",
                "origin": "server_auto",
            }
        },
    }

    async def _run() -> None:
        await api._send_response_create(ws, event, origin="server_auto")
        assert len(api._response_create_queue) == 1
        assert api._pending_response_create is not None
        assert api._pending_response_create.reason == "awaiting_transcript_final"

        awaiting_transcript_final["value"] = False
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    response_creates = [payload for payload in ws.sent if payload.get("type") == "response.create"]
    response_cancels = [payload for payload in ws.sent if payload.get("type") == "response.cancel"]

    assert len(response_creates) == 1
    assert response_cancels == []
    assert len(api._response_create_queue) == 0


def test_micro_ack_pending_create_does_not_block_single_flight() -> None:
    api = _make_api_stub()
    api._response_in_flight = True
    api._active_response_id = None
    api._pending_response_create_origins.append({"origin": "micro_ack", "micro_ack": "true"})

    assert api._is_active_response_blocking() is False


def test_tool_followup_suppressed_after_parent_deliverable(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_parent"
    api._active_input_event_key_by_turn_id["turn_tool_parent"] = "item_parent"
    api._set_response_delivery_state(turn_id="turn_tool_parent", input_event_key="item_parent", state="delivered")
    parent_key = api._canonical_utterance_key(turn_id="turn_tool_parent", input_event_key="item_parent")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_tool_parent",
        input_event_key="item_parent",
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "deliverable_class", "progress"),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_parent_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")

    asyncio.run(_run())

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert any(
        "tool_followup_create_suppressed" in entry
        and f"canonical_key={canonical_key}" in entry
        and "reason=final_deliverable_already_sent" in entry
        for entry in captured_logs
    )


def test_tool_followup_not_suppressed_when_parent_response_was_empty(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_parent_empty"
    api._active_input_event_key_by_turn_id["turn_tool_parent_empty"] = "item_parent_empty"
    api._set_response_delivery_state(turn_id="turn_tool_parent_empty", input_event_key="item_parent_empty", state="done")
    parent_key = api._canonical_utterance_key(turn_id="turn_tool_parent_empty", input_event_key="item_parent_empty")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_tool_parent_empty",
        input_event_key="item_parent_empty",
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "deliverable_class", "progress"),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_parent_empty_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) in {"creating", "created"}
    assert not any(
        "tool_followup_create_suppressed" in entry
        and f"canonical_key={canonical_key}" in entry
        and "reason=final_deliverable_already_sent" in entry
        for entry in captured_logs
    )


def test_tool_followup_released_after_playback_complete_when_parent_deliverable_is_progress(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._audio_playback_busy = True
    api._active_response_id = "resp-progress-parent"
    api._active_response_origin = "upgraded_response"
    api._current_response_turn_id = "turn_3"
    api._active_response_input_event_key = "item_parent_progress"
    api._active_input_event_key_by_turn_id["turn_3"] = "item_parent_progress"

    parent_key = api._canonical_utterance_key(turn_id="turn_3", input_event_key="item_parent_progress")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_3",
        input_event_key="item_parent_progress",
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_s2T1Yr4QhXnkEeLs",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"

        api._response_in_flight = False
        api.response_in_progress = False
        api._audio_playback_busy = False
        api._active_response_id = None
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) in {"creating", "created"}
    assert any(
        "tool_followup_state" in entry
        and f"canonical_key={canonical_key}" in entry
        and "state=blocked_active_response" in entry
        for entry in captured_logs
    )


def test_tool_followup_suppressed_when_parent_deliverable_is_final(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_final"
    api._active_input_event_key_by_turn_id["turn_tool_final"] = "item_parent_final"

    parent_key = api._canonical_utterance_key(turn_id="turn_tool_final", input_event_key="item_parent_final")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_tool_final",
        input_event_key="item_parent_final",
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_final_1",
        response_create_event={"type": "response.create"},
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")

    asyncio.run(_run())

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert any(
        "tool_followup_create_suppressed" in entry
        and f"canonical_key={canonical_key}" in entry
        and "reason=final_deliverable_already_sent" in entry
        for entry in captured_logs
    )




def test_transcript_final_rebind_syncs_active_ownership_and_legacy_mirrors() -> None:
    api = _make_api_stub()
    api._active_response_origin = "server_auto"
    api._response_in_flight = True
    api._active_response_id = "resp-server-auto-live"
    api._active_server_auto_input_event_key = "synthetic_server_auto_turn_1_2"
    api._set_active_response_state(
        response_id="resp-server-auto-live",
        origin="server_auto",
        input_event_key="synthetic_server_auto_turn_1_2",
        canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_server_auto_turn_1_2"),
    )

    api._rebind_active_response_correlation_key(
        turn_id="turn_1",
        replacement_input_event_key="item_2",
        cause="transcript_final_rebind",
    )

    expected_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_2")
    assert api._active_response_lifecycle.input_event_key == "item_2"
    assert api._active_response_lifecycle.canonical_key == expected_canonical_key
    assert api._active_response_input_event_key == "item_2"
    assert api._active_response_canonical_key == expected_canonical_key
    assert api._active_server_auto_input_event_key == "item_2"





def test_transcript_final_rebind_coherence_snapshot_stays_consistent() -> None:
    api = _make_api_stub()
    old_key = "synthetic_server_auto_turn_1_2"
    new_key = "item_2"
    old_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=old_key)
    new_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=new_key)
    api._active_response_origin = "server_auto"
    api._response_in_flight = True
    api._active_response_id = "resp-server-auto-live"
    api._active_server_auto_input_event_key = old_key
    api._active_input_event_key_by_turn_id["turn_1"] = new_key
    api._canonical_response_state_mutate(
        canonical_key=old_canonical_key,
        turn_id="turn_1",
        input_event_key=old_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", "resp-server-auto-live"),
        ),
    )
    api._record_pending_server_auto_response(
        turn_id="turn_1",
        response_id="resp-server-auto-live",
        canonical_key=old_canonical_key,
    )
    api._set_active_response_state(
        response_id="resp-server-auto-live",
        origin="server_auto",
        input_event_key=old_key,
        canonical_key=old_canonical_key,
    )

    api._rebind_active_response_correlation_key(
        turn_id="turn_1",
        replacement_input_event_key=new_key,
        cause="transcript_final_rebind",
    )

    snapshot = api._response_runtime_coherence_snapshot(
        turn_id="turn_1",
        canonical_key=new_canonical_key,
        response_id="resp-server-auto-live",
    )

    assert snapshot["violations"] == []
    assert snapshot["active_response"]["input_event_key"] == new_key
    assert snapshot["active_response"]["canonical_key"] == new_canonical_key
    assert snapshot["canonical_state"]["response_id"] == "resp-server-auto-live"
    assert snapshot["canonical_state"]["input_event_key"] == new_key


def test_response_done_snapshot_suppresses_cleared_terminal_turn_active_mismatch() -> None:
    api = _make_api_stub()
    turn_id = "turn_response_done_false_positive_guard"
    parent_input_event_key = "item_parent_guard"
    tool_input_event_key = "tool:call_guard"
    response_id = "resp-tool-guard"
    tool_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._canonical_response_state_mutate(
        canonical_key=tool_canonical_key,
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", True),
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", response_id),
        ),
    )
    api._clear_active_response_state()

    snapshot = api._response_runtime_coherence_snapshot(
        stage="response_done",
        turn_id=turn_id,
        canonical_key=tool_canonical_key,
        response_id=response_id,
    )

    assert "turn_active_key_canonical_mismatch" not in snapshot["violations"]
    assert snapshot["canonical_state"]["done"] is True
    assert snapshot["active_response"]["response_id"] is None
    assert snapshot["active_response"]["canonical_key"] is None


def test_response_done_snapshot_still_flags_turn_active_mismatch_outside_cleared_terminal_case() -> None:
    api = _make_api_stub()
    turn_id = "turn_response_done_negative_guard"
    parent_input_event_key = "item_parent_negative"
    tool_input_event_key = "tool:call_negative"
    response_id = "resp-tool-negative"
    tool_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._canonical_response_state_mutate(
        canonical_key=tool_canonical_key,
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
        mutator=lambda record: (
            setattr(record, "created", True),
            setattr(record, "done", True),
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", response_id),
        ),
    )
    api._clear_active_response_state()

    snapshot = api._response_runtime_coherence_snapshot(
        stage="response_done",
        turn_id=turn_id,
        canonical_key=tool_canonical_key,
        response_id="resp-other-negative",
    )

    assert "turn_active_key_canonical_mismatch" in snapshot["violations"]
    assert snapshot["active_response"]["response_id"] is None
    assert snapshot["active_response"]["canonical_key"] is None


def test_websocket_close_clears_active_ownership_surface_and_legacy_mirrors(monkeypatch) -> None:
    api = _make_api_stub()
    api._transport = type(
        "_TransportClosed",
        (),
        {"recv_json": staticmethod(lambda _websocket: (_ for _ in ()).throw(_ConnectionClosed()))},
    )()
    api._response_create_queue_drain_source = None
    api._drain_response_create_queue = lambda **_kwargs: asyncio.sleep(0)
    api._note_disconnect = lambda _reason: None
    api._active_server_auto_input_event_key = "synthetic_server_auto_turn_3_1"
    api._response_in_flight = True
    api.response_in_progress = True
    api._set_active_response_state(
        response_id="resp-ws-live",
        origin="server_auto",
        input_event_key="synthetic_server_auto_turn_3_1",
        canonical_key=api._canonical_utterance_key(turn_id="turn_3", input_event_key="synthetic_server_auto_turn_3_1"),
        consumes_canonical_slot=False,
        confirmation_guarded=True,
        preference_guarded=True,
    )

    class _ConnectionClosed(Exception):
        pass

    monkeypatch.setattr("ai.realtime_api._require_websockets", lambda: object())
    monkeypatch.setattr(
        "ai.realtime_api._resolve_websocket_exceptions",
        lambda _websockets: (_ConnectionClosed, _ConnectionClosed),
    )

    asyncio.run(api.process_ws_messages(object()))

    snapshot = api._response_runtime_coherence_snapshot(
        turn_id="turn_3",
        canonical_key=api._canonical_utterance_key(turn_id="turn_3", input_event_key="synthetic_server_auto_turn_3_1"),
        response_id="resp-ws-live",
    )

    assert snapshot["violations"] == []
    assert snapshot["active_response"]["response_id"] is None
    assert snapshot["suppression"]["active_input_event_key"] is None
    assert api._response_in_flight is False
    assert api.response_in_progress is False
    assert api._response_create_queue_drain_source == "websocket_close"
    assert api._active_response_lifecycle.response_id is None
    assert api._active_response_lifecycle.origin == "unknown"
    assert api._active_response_lifecycle.input_event_key is None
    assert api._active_response_lifecycle.canonical_key is None
    assert api._active_response_lifecycle.consumes_canonical_slot is True
    assert api._active_response_lifecycle.confirmation_guarded is False
    assert api._active_response_lifecycle.preference_guarded is False
    assert api._active_response_id is None
    assert api._active_response_origin == "unknown"
    assert api._active_response_input_event_key is None
    assert api._active_response_canonical_key is None
    assert api._active_response_consumes_canonical_slot is True
    assert api._active_response_confirmation_guarded is False
    assert api._active_response_preference_guarded is False
    assert api._active_server_auto_input_event_key is None


def test_cancelled_response_unblock_clears_active_ownership_and_legacy_mirrors() -> None:
    api = _make_api_stub()
    api._active_server_auto_input_event_key = "synthetic_server_auto_turn_2_1"
    api._response_in_flight = True
    api.response_in_progress = True
    api._set_active_response_state(
        response_id="resp-cancelled-live",
        origin="server_auto",
        input_event_key="synthetic_server_auto_turn_2_1",
        canonical_key=api._canonical_utterance_key(turn_id="turn_2", input_event_key="synthetic_server_auto_turn_2_1"),
        consumes_canonical_slot=False,
        confirmation_guarded=True,
        preference_guarded=True,
    )

    api._clear_cancelled_response_blocking_state(
        response_id="resp-cancelled-live",
        reason="transcript_final_upgrade",
    )

    assert api._response_in_flight is False
    assert api.response_in_progress is False
    assert api._active_response_lifecycle.response_id is None
    assert api._active_response_lifecycle.origin == "unknown"
    assert api._active_response_lifecycle.input_event_key is None
    assert api._active_response_lifecycle.canonical_key is None
    assert api._active_response_lifecycle.consumes_canonical_slot is True
    assert api._active_response_lifecycle.confirmation_guarded is False
    assert api._active_response_lifecycle.preference_guarded is False
    assert api._active_response_id is None
    assert api._active_response_origin == "unknown"
    assert api._active_response_input_event_key is None
    assert api._active_response_canonical_key is None
    assert api._active_response_consumes_canonical_slot is True
    assert api._active_response_confirmation_guarded is False
    assert api._active_response_preference_guarded is False
    assert api._active_server_auto_input_event_key is None


def test_rebind_active_response_logs_transition_key_rebound_with_cause() -> None:
    api = _make_api_stub()
    api._active_response_origin = "server_auto"
    api._response_in_flight = True
    api._active_server_auto_input_event_key = "synthetic_server_auto_turn_1_1"
    api._active_response_id = "resp-server-auto-1"

    old_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_server_auto_turn_1_1")
    api._lifecycle_controller().on_response_created(old_key, origin="server_auto")

    recorded: list[str] = []

    def _capture(**kwargs):
        recorded.append(str(kwargs.get("decision") or ""))

    api._log_lifecycle_event = _capture

    api._rebind_active_response_correlation_key(
        turn_id="turn_1",
        replacement_input_event_key="item_1",
        cause="transcript_final_rebind",
    )

    assert api._active_server_auto_input_event_key == "item_1"
    assert recorded == ["transition_key_rebound:cause=transcript_final_rebind"]


def test_rebind_active_response_skips_when_new_key_already_active() -> None:
    api = _make_api_stub()
    api._active_response_origin = "server_auto"
    api._response_in_flight = True
    api._active_server_auto_input_event_key = "synthetic_server_auto_turn_1_1"
    api._active_response_input_event_key = "synthetic_server_auto_turn_1_1"
    api._active_response_id = "resp-server-auto-2"

    old_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_server_auto_turn_1_1")
    new_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_1")
    lifecycle = api._lifecycle_controller()
    lifecycle.on_response_created(old_key, origin="server_auto")
    lifecycle.on_response_created(new_key, origin="server_auto")
    lifecycle.on_audio_delta(new_key)

    api._active_response_canonical_key = old_key

    recorded: list[str] = []

    def _capture(**kwargs):
        recorded.append(str(kwargs.get("decision") or ""))

    api._log_lifecycle_event = _capture

    api._rebind_active_response_correlation_key(
        turn_id="turn_1",
        replacement_input_event_key="item_1",
        cause="transcript_final_rebind",
    )

    assert api._active_response_input_event_key == "item_1"
    assert api._active_response_canonical_key == new_key
    assert api._active_server_auto_input_event_key == "item_1"
    assert lifecycle.state_for(old_key) == InteractionLifecycleState.NEW
    assert lifecycle.state_for(new_key) == InteractionLifecycleState.AUDIO_STARTED
    assert recorded == [
        "transition_rebind_skipped:new_key_already_active:new_state=audio_started:cause=transcript_final_rebind"
    ]


def test_assistant_message_not_scheduled_when_same_turn_tool_followup_owner_exists(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_origin = "server_auto"
    api._current_response_turn_id = "turn_owner_tool"
    api._active_input_event_key_by_turn_id["turn_owner_tool"] = "item_owner_tool"
    api._tool_followup_state_by_canonical_key[
        api._canonical_utterance_key(turn_id="turn_owner_tool", input_event_key="tool:call_owner")
    ] = "blocked_active_response"

    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api.send_assistant_message(
            "Working on it.",
            ws,
            response_metadata={
                "turn_id": "turn_owner_tool",
                "input_event_key": "item_owner_tool",
                "trigger": "preference_recall",
            },
        )

    asyncio.run(_run())

    assert api._pending_response_create is None
    assert not list(api._response_create_queue)
    suppression_logs = [
        entry
        for entry in captured_logs
        if "response_not_scheduled" in entry and "reason=same_turn_already_owned" in entry
    ]
    assert len(suppression_logs) == 1
    assert "owner=tool_followup_owned" in suppression_logs[0]


def test_assistant_message_schedules_normally_without_same_turn_owner() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    async def _run() -> None:
        await api.send_assistant_message(
            "All set.",
            ws,
            response_metadata={
                "turn_id": "turn_free",
                "input_event_key": "item_free",
                "trigger": "assistant_message",
            },
        )

    asyncio.run(_run())

    assistant_response_creates = [
        event
        for event in ws.sent
        if event.get("type") == "response.create"
        and ((event.get("response") or {}).get("metadata") or {}).get("origin") == "assistant_message"
    ]

    assert len(assistant_response_creates) == 1
    metadata = (assistant_response_creates[0].get("response") or {}).get("metadata") or {}
    assert metadata.get("turn_id") == "turn_free"
    assert metadata.get("input_event_key") == "item_free"


def test_assistant_message_not_scheduled_when_same_turn_final_deliverable_exists(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_origin = "server_auto"
    api._current_response_turn_id = "turn_owner_final"
    api._active_input_event_key_by_turn_id["turn_owner_final"] = "item_owner_final"

    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)

    final_key = api._canonical_utterance_key(turn_id="turn_owner_final", input_event_key="item_owner_final")
    api._canonical_response_state_mutate(
        canonical_key=final_key,
        turn_id="turn_owner_final",
        input_event_key="item_owner_final",
        mutator=lambda state: (
            setattr(state, "origin", "tool_output"),
            setattr(state, "deliverable_observed", True),
            setattr(state, "deliverable_class", "final"),
        ),
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_info)

    async def _run() -> None:
        await api.send_assistant_message(
            "Already answered.",
            ws,
            response_metadata={
                "turn_id": "turn_owner_final",
                "input_event_key": "item_owner_final",
                "trigger": "preference_recall",
            },
        )

    asyncio.run(_run())

    assert api._pending_response_create is None
    assert not list(api._response_create_queue)
    assert any("owner=terminal_deliverable_owned" in entry for entry in captured_logs)


def test_assistant_message_still_schedules_without_stronger_same_turn_owner() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_origin = "server_auto"
    api._current_response_turn_id = "turn_other"
    api._active_input_event_key_by_turn_id["turn_owner_none"] = "item_owner_none"

    async def _run() -> None:
        await api.send_assistant_message(
            "Queued while active response completes.",
            ws,
            response_metadata={
                "turn_id": "turn_owner_none",
                "input_event_key": "item_owner_none",
                "trigger": "preference_recall",
            },
        )

    asyncio.run(_run())

    assert api._pending_response_create is not None
    assert api._pending_response_create.origin == "assistant_message"


def test_tool_followup_cleanup_stale_assistant_creates_remains_available() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_cleanup", "input_event_key": "tool:call_cleanup"}},
    }
    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event=event,
        origin="assistant_message",
        turn_id="turn_cleanup",
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._response_create_queue.append(
        {
            "websocket": ws,
            "event": event,
            "origin": "assistant_message",
            "turn_id": "turn_cleanup",
            "record_ai_call": False,
            "debug_context": None,
            "memory_brief_note": None,
            "queued_reminder_key": None,
            "enqueued_done_serial": 0,
            "enqueue_seq": 2,
        }
    )

    assert api._pending_response_create is not None
    canonical_key = api._canonical_utterance_key(turn_id="turn_cleanup", input_event_key="tool:call_cleanup")
    api._clear_stale_assistant_message_creates_for_tool_followup(canonical_key=canonical_key, state="created")
    assert api._pending_response_create is None
    assert not list(api._response_create_queue)


def test_tool_followup_release_suppressed_when_upgraded_parent_already_covered_action(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._audio_playback_busy = True
    api._active_response_id = "resp-upgraded-1"
    api._active_response_origin = "upgraded_response"
    api._current_response_turn_id = "turn_mix_1"
    api._active_response_input_event_key = "item_mix_parent"
    api._active_input_event_key_by_turn_id["turn_mix_1"] = "item_mix_parent"

    parent_key = api._canonical_utterance_key(turn_id="turn_mix_1", input_event_key="item_mix_parent")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_mix_1",
        input_event_key="item_mix_parent",
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", "resp-upgraded-1"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_1",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"
        api._release_blocked_tool_followups_for_response_done(response_id="resp-upgraded-1")
        api._response_in_flight = False
        api.response_in_progress = False
        api._audio_playback_busy = False
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert any("tool_followup_release_suppressed" in entry for entry in captured_logs)
    trace = api._turn_arbitration_trace_by_key[("run-405-repro", "turn_mix_1")]
    latest_tool_followup = trace.tool_followup_observations[-1].decision
    assert latest_tool_followup.followup_outcome_posture in {"suppressed", "pruned"}
    assert latest_tool_followup.parent_coverage_state == "covered_canonical"
    assert any(
        observation.decision.followup_distinctness == "redundant"
        for observation in trace.tool_followup_observations
    )
    assert latest_tool_followup.followup_distinctness == "stale"
    assert latest_tool_followup.native_reason_code.startswith("parent_covered_tool_result")


def test_tool_followup_release_preserved_for_distinct_tool_result_information(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._audio_playback_busy = True
    api._active_response_id = "resp-upgraded-2"
    api._active_response_origin = "upgraded_response"
    api._current_response_turn_id = "turn_mix_2"
    api._active_response_input_event_key = "item_mix_parent_2"
    api._active_input_event_key_by_turn_id["turn_mix_2"] = "item_mix_parent_2"

    parent_key = api._canonical_utterance_key(turn_id="turn_mix_2", input_event_key="item_mix_parent_2")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_mix_2",
        input_event_key="item_mix_parent_2",
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", "resp-upgraded-2"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_2",
        response_create_event={"type": "response.create"},
        tool_name="perform_research",
        tool_result_has_distinct_info=True,
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"
        api._release_blocked_tool_followups_for_response_done(response_id="resp-upgraded-2")
        api._response_in_flight = False
        api.response_in_progress = False
        api._audio_playback_busy = False
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) in {"creating", "created"}
    assert not any("tool_followup_release_suppressed" in entry for entry in captured_logs)
    trace = api._turn_arbitration_trace_by_key[("run-405-repro", "turn_mix_2")]
    latest_tool_followup = trace.tool_followup_observations[-1].decision
    assert latest_tool_followup.followup_outcome_posture == "released"
    assert latest_tool_followup.parent_coverage_state == "not_applicable"
    assert latest_tool_followup.followup_distinctness == "distinct"
    assert latest_tool_followup.native_reason_code == "not_suppressible"

def test_tool_followup_release_suppressed_for_assistant_parent_covering_gesture_action() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._audio_playback_busy = True
    api._active_response_id = "resp-assistant-1"
    api._active_response_origin = "assistant_message"
    api._current_response_turn_id = "turn_mix_3"
    api._active_response_input_event_key = "item_mix_parent_3"
    api._active_input_event_key_by_turn_id["turn_mix_3"] = "item_mix_parent_3"

    parent_key = api._canonical_utterance_key(turn_id="turn_mix_3", input_event_key="item_mix_parent_3")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_mix_3",
        input_event_key="item_mix_parent_3",
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-assistant-1"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_3",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"
        api._release_blocked_tool_followups_for_response_done(response_id="resp-assistant-1")
        api._response_in_flight = False
        api.response_in_progress = False
        api._audio_playback_busy = False
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 0
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_tool_followup_release_suppressed_for_server_auto_parent_covering_gesture_action() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._audio_playback_busy = True
    api._active_response_id = "resp-server-auto-1"
    api._active_response_origin = "server_auto"
    api._current_response_turn_id = "turn_mix_server_auto"
    api._active_response_input_event_key = "item_mix_server_auto_parent"
    api._active_input_event_key_by_turn_id["turn_mix_server_auto"] = "item_mix_server_auto_parent"

    parent_key = api._canonical_utterance_key(turn_id="turn_mix_server_auto", input_event_key="item_mix_server_auto_parent")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_mix_server_auto",
        input_event_key="item_mix_server_auto_parent",
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", "resp-server-auto-1"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_server_auto_1",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"
        api._release_blocked_tool_followups_for_response_done(response_id="resp-server-auto-1")
        api._response_in_flight = False
        api.response_in_progress = False
        api._audio_playback_busy = False
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_tool_followup_create_seam_drop_removes_single_matching_queue_entry_without_resuppressing(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    suppressed_logs: list[str] = []

    def _capture_info(msg, *args, **kwargs) -> None:
        rendered = msg % args if args else msg
        if "tool_followup_create_suppressed" in rendered:
            suppressed_logs.append(rendered)

    monkeypatch.setattr(logger, "info", _capture_info)

    turn_id = "turn_drop_once"
    parent_input_event_key = "item_parent_drop_once"
    parent_response_id = "resp-parent-drop-once"
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._active_response_id = parent_response_id
    api._active_response_origin = "upgraded_response"
    api._response_in_flight = True
    api.response_in_progress = True

    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_drop_once",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )

    async def _run() -> None:
        queued = await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert queued is False
        assert len(api._response_create_queue) == 1

        api._response_create_queue.append(
            {
                "websocket": ws,
                "event": {
                    "type": "response.create",
                    "response": {
                        "metadata": {
                            "turn_id": turn_id,
                            "input_event_key": "item_unrelated_queue_entry",
                            "origin": "assistant_message",
                        }
                    },
                },
                "origin": "assistant_message",
                "turn_id": turn_id,
                "record_ai_call": False,
                "debug_context": None,
                "memory_brief_note": None,
                "enqueued_done_serial": 0,
                "enqueue_seq": 99,
            }
        )

        queue_depth_before = len(api._response_create_queue)

        api._response_in_flight = False
        api.response_in_progress = False
        api._active_response_id = None
        await api._drain_response_create_queue(source_trigger="playback_complete")

        assert len(api._response_create_queue) == queue_depth_before - 1
        remaining_input_keys = {
            str((((entry.get("event") or {}).get("response") or {}).get("metadata") or {}).get("input_event_key") or "")
            for entry in api._response_create_queue
        }
        assert "item_unrelated_queue_entry" in remaining_input_keys

        api._audio_playback_busy = True
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    suppressed_for_key = [
        entry
        for entry in suppressed_logs
        if f"canonical_key={canonical_key}" in entry and "reason=parent_covered_tool_result" in entry
    ]
    assert len(suppressed_for_key) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_tool_followup_suppressed_release_does_not_regress_queue_drain_idempotency() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._audio_playback_busy = True
    api._active_response_id = "resp-upgraded-3"
    api._active_response_origin = "upgraded_response"
    api._current_response_turn_id = "turn_mix_4"
    api._active_response_input_event_key = "item_mix_parent_4"
    api._active_input_event_key_by_turn_id["turn_mix_4"] = "item_mix_parent_4"

    parent_key = api._canonical_utterance_key(turn_id="turn_mix_4", input_event_key="item_mix_parent_4")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_mix_4",
        input_event_key="item_mix_parent_4",
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", "resp-upgraded-3"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_4",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )

    async def _run() -> None:
        await api._send_response_create(ws, response_create_event, origin="tool_output")
        api._release_blocked_tool_followups_for_response_done(response_id="resp-upgraded-3")
        api._response_in_flight = False
        api.response_in_progress = False
        api._audio_playback_busy = False
        await api._drain_response_create_queue(source_trigger="playback_complete")
        await api._drain_response_create_queue(source_trigger="playback_complete")

    asyncio.run(_run())

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert api._pending_response_create is None
    assert len(api._response_create_queue) == 0


def test_tool_followup_create_seam_suppresses_parent_covered_even_without_blocked_by_response_id() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_mix_5"

    parent_key = api._canonical_utterance_key(turn_id="turn_mix_5", input_event_key="item_mix_parent_5")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_mix_5",
        input_event_key="item_mix_parent_5",
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", "resp-upgraded-5"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_5",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata.pop("blocked_by_response_id", None)
    metadata["parent_turn_id"] = "turn_mix_5"
    metadata["parent_input_event_key"] = "item_mix_parent_5"

    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event=response_create_event,
        origin="tool_output",
        turn_id="turn_mix_5",
        created_at=0.0,
        reason="legacy_queue_hydration",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )

    asyncio.run(api._drain_response_create_queue(source_trigger="playback_complete"))

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"





def test_tool_followup_release_suppressed_when_parent_has_progress_deliverable_class() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_progress_parent"
    parent_input_event_key = "item_parent_progress"
    parent_response_id = "resp-parent-progress"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_progress_parent",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"

def test_tool_followup_release_not_suppressed_when_parent_only_has_unclassified_deliverable_observed() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_unclassified_parent"
    parent_input_event_key = "item_parent_unclassified"
    parent_response_id = "resp-parent-unclassified"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_unclassified_parent",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "parent_not_coverage_qualified"

def test_tool_followup_create_seam_requires_terminal_text_before_terminal_selection_counts_as_parent_coverage() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_mix_terminal_store"

    parent_turn_id = "turn_mix_terminal_store"
    parent_input_event_key = "item_mix_parent_terminal_store"
    parent_response_id = "resp-parent-terminal-store"
    parent_key = api._canonical_utterance_key(turn_id=parent_turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=parent_turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "audio_started", False),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=parent_turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_mix_terminal_store",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = parent_turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event=response_create_event,
        origin="tool_output",
        turn_id=parent_turn_id,
        created_at=0.0,
        reason="legacy_queue_hydration",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )

    asyncio.run(api._drain_response_create_queue(source_trigger="playback_complete"))

    assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) == "creating"

    api._record_terminal_response_text(response_id=parent_response_id, text="It looks like a phone.")

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"


def test_tool_followup_pruned_when_parent_has_substantive_audio_output() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_substantive"
    parent_input_event_key = "item_parent_substantive"
    parent_response_id = "resp-parent-substantive"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "audio_started", True),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_parent_substantive",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"


def test_response_done_prunes_followup_when_parent_terminal_selection_already_covers_result() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_parent_prune"
    parent_input_event_key = "item_parent_terminal_prune"
    parent_response_id = "resp-parent-terminal-prune"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "audio_started", False),
        ),
    )
    api._record_terminal_response_text(response_id=parent_response_id, text="It looks like a white mug.")
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_terminal_parent_prune",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"


def test_response_done_release_keeps_distinct_tool_followup_when_parent_terminal_selected() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_parent_release"
    parent_input_event_key = "item_parent_terminal_release"
    parent_response_id = "resp-parent-terminal-release"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
        ),
    )
    api._record_terminal_response_text(response_id=parent_response_id, text="I found your keys.")
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    event, _canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_terminal_parent_release",
        response_create_event={"type": "response.create"},
        tool_name="perform_research",
        tool_result_has_distinct_info=True,
    )
    metadata = ((event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "not_suppressible"


def test_tool_followup_blocked_by_active_parent_is_released_and_drained_after_parent_done() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_parent_release"
    parent_input_event_key = "item_parent_release"
    parent_response_id = "resp-parent-release"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_parent_release",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["turn_id"] = turn_id
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=str(metadata.get("input_event_key") or ""))

    api._response_create_queue.append(
        {
            "event": response_create_event,
            "origin": "tool_output",
            "turn_id": turn_id,
            "websocket": ws,
        }
    )
    api._set_tool_followup_state(canonical_key=canonical_key, state="blocked_active_response", reason="test_seed")

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert api._tool_followup_state(canonical_key=canonical_key) == "scheduled_release"

    asyncio.run(api._drain_response_create_queue(source_trigger="response_done"))
    assert len([event for event in ws.sent if event.get("type") == "response.create"]) == 1


def test_terminal_prune_path_is_idempotent_for_terminal_selected_parent() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_idempotent"
    parent_input_event_key = "item_parent_terminal_idempotent"
    parent_response_id = "resp-parent-terminal-idempotent"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_terminal_idempotent",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id

    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._sync_pending_response_create_queue()

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )
    first_pending = api._pending_response_create
    first_queue_size = len(api._response_create_queue)
    first_state = api._tool_followup_state(canonical_key=canonical_key)

    api._drop_dead_tool_followup_creates_after_terminal_selection(
        turn_id=turn_id,
        selected_response_id=parent_response_id,
    )

    assert first_pending is None
    assert first_queue_size == 0
    assert first_state == "dropped"
    assert api._pending_response_create is None
    assert len(api._response_create_queue) == 0
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_gesture_followup_dropped_on_response_done_and_playback_complete_when_parent_already_spoke_action() -> None:
    for trigger in ("response_done", "playback_complete"):
        api = _make_api_stub()
        _wire_runtime(api)
        ws = _RecordingWs()
        api.websocket = ws
        api._current_response_turn_id = f"turn_mix_{trigger}"

        parent_key = api._canonical_utterance_key(turn_id=f"turn_mix_{trigger}", input_event_key="item_parent")
        api._canonical_response_state_mutate(
            canonical_key=parent_key,
            turn_id=f"turn_mix_{trigger}",
            input_event_key="item_parent",
            mutator=lambda record: (
                setattr(record, "origin", "assistant_message"),
                setattr(record, "response_id", f"resp-parent-ack-{trigger}"),
                setattr(record, "deliverable_observed", True),
                setattr(record, "deliverable_class", "progress"),
                setattr(record, "done", True),
            ),
        )

        response_create_event, canonical_key = api._build_tool_followup_response_create_event(
            call_id=f"call_mix_{trigger}",
            response_create_event={"type": "response.create"},
            tool_name="gesture_look_around",
        )
        metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
        metadata["blocked_by_response_id"] = f"resp-parent-ack-{trigger}"
        metadata["parent_turn_id"] = f"turn_mix_{trigger}"
        metadata["parent_input_event_key"] = "item_parent"

        api._pending_response_create = PendingResponseCreate(
            websocket=ws,
            event=response_create_event,
            origin="tool_output",
            turn_id=f"turn_mix_{trigger}",
            created_at=0.0,
            reason="legacy_queue_hydration",
            record_ai_call=False,
            debug_context=None,
            memory_brief_note=None,
            queued_reminder_key=None,
            enqueued_done_serial=0,
            enqueue_seq=1,
        )

        asyncio.run(api._drain_response_create_queue(source_trigger=trigger))

        assert [event for event in ws.sent if event.get("type") == "response.create"] == []
        assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_terminal_deliverable_selection_eagerly_drops_queued_tool_followup() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_drop"
    parent_input_event_key = "item_parent_terminal_drop"
    parent_response_id = "resp-parent-terminal-drop"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_terminal_drop",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=str(metadata.get("input_event_key") or ""))

    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._sync_pending_response_create_queue()

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    assert api._pending_response_create is None
    assert len(api._response_create_queue) == 0
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_terminal_deliverable_selection_prune_invalidates_stale_tool_followup_lineage() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_invalidate"
    parent_input_event_key = "item_parent_terminal_invalidate"
    parent_response_id = "resp-parent-terminal-invalidate"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._current_response_turn_id = turn_id

    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_terminal_invalidate",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    tool_input_event_key = str(metadata.get("input_event_key") or "")
    api._current_input_event_key = tool_input_event_key
    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._pending_response_create_origins.append(
        {
            "origin": "micro_ack",
            "micro_ack": "true",
            "consumes_canonical_slot": "false",
            "turn_id": turn_id,
            "input_event_key": tool_input_event_key,
        }
    )
    api._pending_response_create_origins.append(
        {
            "origin": "assistant_message",
            "micro_ack": "false",
            "consumes_canonical_slot": "true",
            "turn_id": turn_id,
            "input_event_key": parent_input_event_key,
        }
    )
    api._pending_micro_ack_by_turn_channel = {(turn_id, "voice"): object()}

    cancelled: list[tuple[str, str]] = []

    class _AckManager:
        def cancel_matching(self, *, turn_id: str, reason: str, matcher):
            cancelled.append((turn_id, reason))

    api._micro_ack_manager = _AckManager()

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    assert api._pending_response_create is None
    assert len(api._response_create_queue) == 0
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert api._current_input_event_key == parent_input_event_key
    assert list(api._pending_response_create_origins) == [
        {
            "origin": "assistant_message",
            "micro_ack": "false",
            "consumes_canonical_slot": "true",
            "turn_id": turn_id,
            "input_event_key": parent_input_event_key,
        }
    ]
    assert api._pending_micro_ack_by_turn_channel == {}
    assert cancelled == [(turn_id, "tool_followup_pruned:parent_covered_tool_result terminal_deliverable_selected")]


def test_terminal_deliverable_selection_does_not_drop_legitimate_distinct_info_tool_followup() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_keep"
    parent_input_event_key = "item_parent_terminal_keep"
    parent_response_id = "resp-parent-terminal-keep"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_terminal_keep",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["tool_result_has_distinct_info"] = "true"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=str(metadata.get("input_event_key") or ""))

    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._sync_pending_response_create_queue()

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    assert api._pending_response_create is not None
    assert any(
        str(((queued.get("event") or {}).get("response") or {}).get("metadata", {}).get("input_event_key") or "").strip()
        == str(metadata.get("input_event_key") or "").strip()
        for queued in api._response_create_queue
    )
    assert api._tool_followup_state(canonical_key=canonical_key) == "new"


def test_low_risk_gesture_followup_payload_is_status_only() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_payload"
    api._active_input_event_key_by_turn_id["turn_gesture_payload"] = "item_parent_payload"

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_payload",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("tool_followup") == "true"
    assert metadata.get("tool_followup_suppress_if_parent_covered") == "true"
    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_followup_silent_audio") == "true"
    assert metadata.get("tool_followup_silent_user_facing_output") == "true"
    assert "Gesture follow-up only" in instructions
    assert "avoid final completion claims" in instructions


def test_gesture_followup_payload_queued_motion_uses_in_progress_wording() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_payload_queued"
    api._active_input_event_key_by_turn_id["turn_gesture_payload_queued"] = "item_parent_payload_queued"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "queued"}

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_payload_queued",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("gesture_motion_status") == "queued"
    assert "queued and may begin shortly" in instructions
    assert "avoid final completion claims" in instructions


def test_gesture_followup_payload_started_motion_uses_in_progress_wording() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_payload_started"
    api._active_input_event_key_by_turn_id["turn_gesture_payload_started"] = "item_parent_payload_started"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_payload_started",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("gesture_motion_status") == "started"
    assert "started but is not physically complete yet" in instructions
    assert "avoid final completion claims" in instructions


def test_gesture_followup_payload_completed_motion_allows_completion_wording() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_payload_completed"
    api._active_input_event_key_by_turn_id["turn_gesture_payload_completed"] = "item_parent_payload_completed"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_payload_completed",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("gesture_motion_status") == "completed"
    assert "definitive completion wording is allowed" in instructions
    assert "not required" in instructions


def test_gesture_followup_truth_snapshot_logs_continuity_and_motion_status(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_truth_snapshot"
    api._active_input_event_key_by_turn_id["turn_gesture_truth_snapshot"] = "item_parent_truth_snapshot"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="active"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=1,
            recent_completed_step_id="step_1",
            next_pending_step_id="step_3",
            final_followup_pending=True,
        )
    )
    recorded_logs: list[str] = []

    def _fake_info(message: str, *args: object, **_kwargs: object) -> None:
        if args:
            recorded_logs.append(message % args)
        else:
            recorded_logs.append(message)

    monkeypatch.setattr("ai.realtime_api.logger.info", _fake_info)
    api._build_tool_followup_response_create_event(
        call_id="call_gesture_truth_snapshot",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    combined_output = "\n".join(recorded_logs)

    assert "gesture_followup_truth_snapshot" in combined_output
    assert "motion_status=started" in combined_output
    assert "continuity_active_step_kind=gesture" in combined_output
    assert "continuity_active_step_status=active" in combined_output
    assert "continuity_final_followup_pending=true" in combined_output
    assert "continuity_open_non_report_steps=true" in combined_output


def test_gesture_followup_payload_unknown_motion_avoids_completion_claims() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_payload_unknown"
    api._active_input_event_key_by_turn_id["turn_gesture_payload_unknown"] = "item_parent_payload_unknown"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "bogus_future_state"}

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_payload_unknown",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("gesture_motion_status") is None
    assert "Physical completion is unknown" in instructions
    assert "avoid final completion claims" in instructions




def test_tool_followup_post_completion_bucket_classifies_required_deliverable_independently_of_turn_tail() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    bucket, reason = api._classify_tool_followup_post_completion_bucket(
        keep_status_only=False,
        followthrough_remaining=True,
        non_report_followthrough_remaining=True,
        tool_result_has_distinct_info=False,
    )

    assert bucket == "required_deliverable"
    assert reason == "required_deliverable_owed"

def test_low_risk_gesture_followup_drops_status_only_when_only_report_followup_remains() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_report_only"
    api._active_input_event_key_by_turn_id["turn_gesture_report_only"] = "item_parent_report_only"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=None,
            recent_completed_step_id="step_2",
            next_pending_step_id="step_3",
            final_followup_pending=True,
        )
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_report_only",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("tool_followup_status_only") is None
    assert metadata.get("tool_followup_step_output_policy") == "required_deliverable"
    assert metadata.get("followthrough_step_output_policy") == "required_deliverable"
    assert metadata.get("tool_followup_post_completion_bucket") == "final_required_deliverable"
    assert metadata.get("tool_followup_post_completion_reason") == "required_deliverable_owed"
    assert metadata.get("followthrough_post_completion_reason") == "required_deliverable_owed"
    assert metadata.get("tool_followup_suppress_if_parent_covered") is None
    assert metadata.get("tool_followup_silent_audio") is None
    assert metadata.get("tool_followup_silent_user_facing_output") is None
    assert metadata.get("tool_followup_create_suppressed") is None
    assert "Required user deliverable is still owed for the parent turn." in instructions


def test_low_risk_gesture_followup_started_motion_stays_status_only_when_only_report_remains() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_started_report_only"
    api._active_input_event_key_by_turn_id["turn_gesture_started_report_only"] = "item_parent_started_report_only"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=None,
            recent_completed_step_id="step_2",
            next_pending_step_id="step_3",
            final_followup_pending=True,
        )
    )
    api._gesture_followthrough_chain_remaining = lambda *, turn_id: False
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: include_report_followup

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_started_report_only",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_followup_silent_audio") == "true"
    assert metadata.get("tool_followup_silent_user_facing_output") == "true"
    assert metadata.get("tool_followup_step_output_policy") == "silent_intermediate"
    assert metadata.get("tool_followup_post_completion_reason") == "status_only_intermediate_response"
    assert "Required user deliverable is still owed for the parent turn." not in instructions


def test_low_risk_gesture_followup_queued_motion_only_report_remaining_stays_status_only() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_started_report_only_no_override"
    api._active_input_event_key_by_turn_id["turn_gesture_started_report_only_no_override"] = (
        "item_parent_started_report_only_no_override"
    )
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "queued"}
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=None,
            recent_completed_step_id="step_2",
            next_pending_step_id="step_3",
            final_followup_pending=True,
        )
    )
    api._gesture_followthrough_chain_remaining = lambda *, turn_id: True
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: include_report_followup

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_started_report_only_no_override",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_followup_suppress_if_parent_covered") == "true"
    assert metadata.get("tool_followup_silent_audio") == "true"
    assert metadata.get("tool_followup_silent_user_facing_output") == "true"
    assert metadata.get("tool_followup_tool_choice") is None
    assert metadata.get("tool_followup_tool_choice_reason") is None
    assert metadata.get("tool_followup_step_output_policy") == "silent_intermediate"
    assert metadata.get("tool_followup_post_completion_reason") == "status_only_intermediate_response"
    assert "Required user deliverable is still owed for the parent turn." not in instructions


def test_low_risk_gesture_followup_started_motion_stays_status_only_for_true_intermediate_step() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_started_intermediate"
    api._active_input_event_key_by_turn_id["turn_gesture_started_intermediate"] = "item_parent_started_intermediate"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="active"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=1,
            recent_completed_step_id="step_1",
            next_pending_step_id="step_2",
            final_followup_pending=True,
        )
    )
    api._gesture_followthrough_chain_remaining = lambda *, turn_id: True
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: True

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_started_intermediate",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}

    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_followup_step_output_policy") == "model_context_injection_only"
    assert metadata.get("tool_followup_post_completion_bucket") == "model_context_injection_only"
    assert metadata.get("tool_followup_post_completion_reason") == "gesture_intermediate_inject_only"
    assert metadata.get("tool_followup_create_suppressed") == "true"
    assert metadata.get("tool_followup_create_suppression_reason") == "gesture_intermediate_inject_only"
    assert metadata.get("tool_followup_silent_audio") == "true"
    assert metadata.get("tool_followup_silent_user_facing_output") == "true"


def test_low_risk_gesture_followup_marks_inject_only_no_create_for_intermediate_inflight_step() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._response_in_flight = True
    api._current_response_turn_id = "turn_gesture_intermediate_inflight"
    api._active_input_event_key_by_turn_id["turn_gesture_intermediate_inflight"] = "item_parent_intermediate_inflight"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: True
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="active"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=1,
            recent_completed_step_id="step_1",
            next_pending_step_id="step_2",
            final_followup_pending=True,
        )
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_intermediate_inflight",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}

    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_followup_step_output_policy") == "model_context_injection_only"
    assert metadata.get("tool_followup_post_completion_bucket") == "model_context_injection_only"
    assert metadata.get("tool_followup_post_completion_reason") == "gesture_intermediate_inject_only"
    assert metadata.get("tool_followup_create_suppressed") == "true"
    assert metadata.get("tool_followup_create_suppression_reason") == "gesture_intermediate_inject_only"


def test_low_risk_gesture_followup_keeps_create_for_non_gesture_intermediate_without_runtime_descriptor() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_to_diagnostics"
    api._active_input_event_key_by_turn_id["turn_gesture_to_diagnostics"] = "item_parent_gesture_to_diagnostics"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: True
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="diagnostics", status="active"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=1,
            recent_completed_step_id="step_1",
            next_pending_step_id="step_2",
            final_followup_pending=True,
        )
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_to_diagnostics",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}

    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_followup_step_output_policy") == "model_context_injection_only"
    assert metadata.get("tool_followup_post_completion_reason") == "gesture_intermediate_inject_only"
    assert metadata.get("tool_followup_create_suppressed") is None
    assert metadata.get("tool_followup_create_unsuppressed_reason") == (
        "non_gesture_followthrough_requires_model_dispatch"
    )
    assert payload.get("tool_choice") == "required"


def test_low_risk_gesture_followup_marks_no_create_when_followthrough_complete() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_complete"
    api._active_input_event_key_by_turn_id["turn_gesture_complete"] = "item_parent_complete"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: False
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_3", kind="gesture", status="completed"),
            ),
            active_step_index=None,
            recent_completed_step_id="step_3",
            next_pending_step_id=None,
            final_followup_pending=False,
        )
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_complete",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}

    assert metadata.get("tool_followup_step_output_policy") == "execution_state_update"
    assert metadata.get("tool_followup_post_completion_bucket") == "execution_state_only"
    assert metadata.get("tool_followup_post_completion_reason") == "followthrough_complete_non_report"
    assert metadata.get("tool_followup_create_suppressed") == "true"
    assert metadata.get("tool_followup_create_suppression_reason") == "followthrough_complete_non_report"


def test_low_risk_gesture_followup_does_not_mark_no_create_while_motion_started() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_in_motion"
    api._active_input_event_key_by_turn_id["turn_gesture_in_motion"] = "item_parent_in_motion"
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: False
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=None,
            recent_completed_step_id="step_2",
            next_pending_step_id="step_3",
            final_followup_pending=True,
        )
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_in_motion",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = response_create_event.get("response") or {}
    metadata = payload.get("metadata") or {}

    assert metadata.get("tool_followup_create_suppressed") is None
    assert metadata.get("tool_followup_create_suppression_reason") is None
    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("gesture_motion_status") == "started"


def test_execute_function_call_skips_response_create_for_intermediate_inflight_gesture(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api._current_response_turn_id = "turn_gesture_intermediate_exec"
    api._current_input_event_key = "item_parent_intermediate_exec"
    api._active_input_event_key_by_turn_id["turn_gesture_intermediate_exec"] = "item_parent_intermediate_exec"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: True
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="active"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=1,
            recent_completed_step_id="step_1",
            next_pending_step_id="step_2",
            final_followup_pending=True,
        )
    )

    async def _fake_gesture(**_kwargs):
        return {"ok": True, "motion_request_key": "mrk-gesture-intermediate"}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_center",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_center", "call_gesture_intermediate_exec", {"target": "center"}, ws))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert response_create_events == []
    function_output_events = [event for event in ws.sent if event.get("type") == "conversation.item.create"]
    assert len(function_output_events) == 1
    canonical_key = api._canonical_utterance_key(
        turn_id="turn_gesture_intermediate_exec",
        input_event_key="tool:call_gesture_intermediate_exec",
    )
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_execute_function_call_skips_response_create_for_intermediate_gesture_without_inflight_timing(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_gesture_intermediate_exec_no_inflight"
    api._current_input_event_key = "item_parent_intermediate_exec_no_inflight"
    api._active_input_event_key_by_turn_id["turn_gesture_intermediate_exec_no_inflight"] = (
        "item_parent_intermediate_exec_no_inflight"
    )
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "started"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: True
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="active"),
                types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
            ),
            active_step_index=1,
            recent_completed_step_id="step_1",
            next_pending_step_id="step_2",
            final_followup_pending=True,
        )
    )

    async def _fake_gesture(**_kwargs):
        return {"ok": True, "motion_request_key": "mrk-gesture-intermediate-no-inflight"}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_center",
        _fake_gesture,
    )

    asyncio.run(
        api.execute_function_call("gesture_look_center", "call_gesture_intermediate_exec_no_inflight", {"target": "center"}, ws)
    )

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert response_create_events == []
    function_output_events = [event for event in ws.sent if event.get("type") == "conversation.item.create"]
    assert len(function_output_events) == 1
    canonical_key = api._canonical_utterance_key(
        turn_id="turn_gesture_intermediate_exec_no_inflight",
        input_event_key="tool:call_gesture_intermediate_exec_no_inflight",
    )
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"


def test_execute_function_call_intermediate_inject_only_dispatches_next_deterministic_step(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_chain_dispatch"
    api._active_input_event_key_by_turn_id["turn_chain_dispatch"] = "item_parent_chain_dispatch"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._last_user_input_text = "look right then left and report"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_chain_dispatch", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: {
        "request_id": "req_chain_dispatch",
        "step_id": "step_2",
        "tool_name": "gesture_look_left",
        "tool_args": {},
    }

    def _fake_followup_event(**_kwargs):
        return (
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "tool_followup_create_suppressed": "true",
                        "tool_followup_create_suppression_reason": "gesture_intermediate_inject_only",
                    }
                },
            },
            api._canonical_utterance_key(
                turn_id="turn_chain_dispatch",
                input_event_key="tool:call_chain_step_1",
            ),
        )

    api._build_tool_followup_response_create_event = _fake_followup_event
    dispatched: list[dict[str, object]] = []

    async def _fake_dispatch(**kwargs):
        dispatched.append(kwargs)
        return {"outcome": "allow", "executed": True}

    api._submit_companion_gesture_tool_request = _fake_dispatch

    async def _fake_gesture(**_kwargs):
        return {"ok": True, "motion_request_key": "mrk-chain-step-1"}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_right",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_right", "call_chain_step_1", {"target": "right"}, ws))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert response_create_events == []
    assert len(dispatched) == 1
    dispatch_call = dispatched[0]
    assert dispatch_call["tool_name"] == "gesture_look_left"
    assert dispatch_call["source"] == "deterministic_followthrough_inject_only"
    assert dispatch_call["allow_followthrough_execution"] is True


def test_execute_function_call_local_companion_call_skips_provider_function_output_and_followup_state(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_local_compgest"
    api._active_input_event_key_by_turn_id["turn_local_compgest"] = "item_parent_local_compgest"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_local_compgest", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: {
        "request_id": "req_local_compgest",
        "step_id": "step_2",
        "tool_name": "gesture_look_left",
        "tool_args": {},
    }

    def _fake_followup_event(**_kwargs):
        return (
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "tool_followup": "true",
                        "tool_call_id": "compgest_chain_step_1",
                        "tool_followup_create_suppressed": "true",
                        "tool_followup_create_suppression_reason": "gesture_intermediate_inject_only",
                    }
                },
            },
            api._canonical_utterance_key(
                turn_id="turn_local_compgest",
                input_event_key="tool:compgest_chain_step_1",
            ),
        )

    api._build_tool_followup_response_create_event = _fake_followup_event
    dispatched: list[dict[str, object]] = []

    async def _fake_dispatch(**kwargs):
        dispatched.append(kwargs)
        return {"outcome": "allow", "executed": True}

    api._submit_companion_gesture_tool_request = _fake_dispatch

    async def _fake_gesture(**_kwargs):
        return {"ok": True, "motion_request_key": "mrk-local-compgest-step-1"}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_right",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_right", "compgest_chain_step_1", {"target": "right"}, ws))

    conversation_item_events = [event for event in ws.sent if event.get("type") == "conversation.item.create"]
    assert len(conversation_item_events) == 1
    runtime_status_event = conversation_item_events[0]
    assert runtime_status_event["item"]["role"] == "system"
    runtime_status_payload = json.loads(runtime_status_event["item"]["content"][0]["text"])
    assert runtime_status_payload["type"] == "followthrough_runtime_status"
    assert runtime_status_payload["turn_id"] == "turn_local_compgest"
    assert runtime_status_payload["step_id"] == "step_2"
    assert runtime_status_payload["status"] == "completed"
    assert isinstance(runtime_status_payload["required_deliverable_owed"], bool)
    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    canonical_key = api._canonical_utterance_key(
        turn_id="turn_local_compgest",
        input_event_key="tool:compgest_chain_step_1",
    )
    assert api._tool_followup_state(canonical_key=canonical_key) == "new"
    assert len(dispatched) == 1


def test_execute_function_call_local_companion_call_schedules_plain_response_create_for_final_report(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_local_compgest_final"
    api._active_input_event_key_by_turn_id["turn_local_compgest_final"] = "item_parent_local_compgest_final"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_local_compgest_final", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: {
        "request_id": "req_local_compgest_final",
        "step_id": "report_step",
        "tool_name": "gesture_look_center",
        "tool_args": {},
    }

    def _fake_followup_event(**_kwargs):
        return (
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "tool_followup": "true",
                        "tool_call_id": "compgest_chain_final",
                        "tool_followup_step_output_policy": "required_deliverable",
                        "tool_followup_post_completion_bucket": "final_required_deliverable",
                        "tool_followup_post_completion_reason": "required_deliverable_owed",
                    },
                    "instructions": "Final follow-up report is still owed.",
                },
            },
            api._canonical_utterance_key(
                turn_id="turn_local_compgest_final",
                input_event_key="tool:compgest_chain_final",
            ),
        )

    api._build_tool_followup_response_create_event = _fake_followup_event

    async def _fake_gesture(**_kwargs):
        return {"ok": True, "motion_request_key": "mrk-local-compgest-final"}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_center",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_center", "compgest_chain_final", {"target": "center"}, ws))

    conversation_item_events = [event for event in ws.sent if event.get("type") == "conversation.item.create"]
    assert len(conversation_item_events) == 1
    runtime_status_event = conversation_item_events[0]
    assert runtime_status_event["item"]["role"] == "system"
    runtime_status_payload = json.loads(runtime_status_event["item"]["content"][0]["text"])
    assert runtime_status_payload["type"] == "followthrough_runtime_status"
    assert runtime_status_payload["turn_id"] == "turn_local_compgest_final"
    assert runtime_status_payload["step_id"] == "report_step"
    assert isinstance(runtime_status_payload["required_deliverable_owed"], bool)
    response_creates = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_creates) == 1
    metadata = response_creates[0]["response"]["metadata"]
    assert metadata.get("tool_followup") is None
    assert metadata.get("tool_call_id") is None
    assert metadata.get("local_runtime_followthrough") == "true"
    assert metadata.get("followthrough_step_output_policy") == "required_deliverable"
    assert metadata.get("followthrough_post_completion_reason") == "required_deliverable_owed"
    assert metadata.get("input_event_key") == "item_parent_local_compgest_final"


def test_send_followthrough_runtime_status_update_deduplicates_per_step_call() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: turn_id == "turn_dedupe"

    asyncio.run(
        api._send_followthrough_runtime_status_update(
            ws,
            turn_id="turn_dedupe",
            step_id="step_5",
            step_kind="gesture",
            tool_name="gesture_look_left",
            call_id="compgest_step_5",
        )
    )
    asyncio.run(
        api._send_followthrough_runtime_status_update(
            ws,
            turn_id="turn_dedupe",
            step_id="step_5",
            step_kind="gesture",
            tool_name="gesture_look_left",
            call_id="compgest_step_5",
        )
    )

    conversation_item_events = [event for event in ws.sent if event.get("type") == "conversation.item.create"]
    assert len(conversation_item_events) == 1
    payload = json.loads(conversation_item_events[0]["item"]["content"][0]["text"])
    assert payload["type"] == "followthrough_runtime_status"
    assert payload["required_deliverable_owed"] is True


def test_deterministic_inject_only_dispatch_dedupe_only_latches_after_executed() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_chain_dispatch_retry"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_chain_dispatch_retry", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: {
        "request_id": "req_chain_dispatch_retry",
        "step_id": "step_2",
        "tool_name": "gesture_look_left",
        "tool_args": {},
    }

    attempts: list[dict[str, object]] = []

    async def _fake_dispatch(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return {"outcome": "defer", "executed": False}
        return {"outcome": "allow", "executed": True}

    api._submit_companion_gesture_tool_request = _fake_dispatch

    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_right",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )
    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_right",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )
    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_right",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )

    assert len(attempts) == 2


def test_deterministic_inject_only_no_descriptor_dispatches_required_deliverable_when_motion_closed() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_report_dispatch"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_report_dispatch", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: True
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True, **_kwargs: False
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api._parent_input_event_key_for_tool_followup = lambda *, turn_id: "item_parent_report_dispatch"
    sent: list[dict[str, object]] = []

    async def _fake_send_response_create(websocket, event, origin, **_kwargs):
        sent.append({"event": event, "origin": origin})
        return True

    api._send_response_create = _fake_send_response_create

    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_dispatch",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )

    assert len(sent) == 1
    metadata = sent[0]["event"]["response"]["metadata"]
    assert sent[0]["origin"] == "tool_output"
    assert metadata["followthrough_step_output_policy"] == "required_deliverable"
    assert metadata["followthrough_post_completion_reason"] == "required_deliverable_owed"
    assert sent[0]["event"]["response"]["tool_choice"] == "none"
    assert "Do not call tools." in sent[0]["event"]["response"]["instructions"]


def test_deterministic_inject_only_required_deliverable_can_require_tool_execution() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_report_tool_dispatch"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_report_tool_dispatch", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: True
    api._required_deliverable_step_requires_tool_execution = lambda *, turn_id: True
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True, **_kwargs: False
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api._parent_input_event_key_for_tool_followup = lambda *, turn_id: "item_parent_report_tool_dispatch"
    sent: list[dict[str, object]] = []

    async def _fake_send_response_create(websocket, event, origin, **_kwargs):
        sent.append({"event": event, "origin": origin})
        return True

    api._send_response_create = _fake_send_response_create

    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_tool_dispatch",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )

    assert len(sent) == 1
    response = sent[0]["event"]["response"]
    assert response["tool_choice"] == "auto"
    assert "Execute the required tool call(s) first." in response["instructions"]


def test_deterministic_inject_only_no_descriptor_does_not_dispatch_when_motion_queued() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_report_queued"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_report_queued", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: True
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True, **_kwargs: (
        False if include_report_followup is False else True
    )
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "queued"}
    api._parent_input_event_key_for_tool_followup = lambda *, turn_id: "item_parent_report_queued"
    sent: list[dict[str, object]] = []

    async def _fake_send_response_create(websocket, event, origin, **_kwargs):
        sent.append({"event": event, "origin": origin})
        return True

    api._send_response_create = _fake_send_response_create

    async def _run() -> None:
        await api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_queued",
            suppression_reason="gesture_intermediate_inject_only",
        )
        await asyncio.sleep(0.12)

    asyncio.run(_run())
    assert sent == []


def test_deterministic_inject_only_no_descriptor_waits_for_motion_completion_before_dispatch() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_report_wait"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_report_wait", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: True
    state = {"status": "started"}

    def _chain_remaining(*, turn_id, include_report_followup=True, **_kwargs):
        if include_report_followup:
            return True
        return state["status"] != "completed"

    api._turn_followthrough_chain_remaining = _chain_remaining
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": state["status"]}
    api._parent_input_event_key_for_tool_followup = lambda *, turn_id: "item_parent_report_wait"
    sent: list[dict[str, object]] = []

    async def _fake_send_response_create(websocket, event, origin, **_kwargs):
        sent.append({"event": event, "origin": origin})
        return True

    api._send_response_create = _fake_send_response_create

    async def _run() -> None:
        await api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_wait",
            suppression_reason="gesture_intermediate_inject_only",
        )
        await asyncio.sleep(0.12)
        assert sent == []
        state["status"] = "completed"
        await asyncio.sleep(0.12)

    asyncio.run(_run())
    assert len(sent) == 1


def test_deterministic_inject_only_no_descriptor_required_deliverable_dispatch_dedupes() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_report_dedupe"
    api._resolve_continuity_tool_event_owner_turn = lambda *, fallback_turn_id: ("turn_report_dedupe", False, "")
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: True
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True, **_kwargs: False
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api._parent_input_event_key_for_tool_followup = lambda *, turn_id: "item_parent_report_dedupe"
    sent: list[dict[str, object]] = []

    async def _fake_send_response_create(websocket, event, origin, **_kwargs):
        sent.append({"event": event, "origin": origin})
        return True

    api._send_response_create = _fake_send_response_create

    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_dedupe",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )
    asyncio.run(
        api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_dedupe",
            suppression_reason="gesture_intermediate_inject_only",
        )
    )

    assert len(sent) == 1


def test_deterministic_inject_only_required_deliverable_watcher_cleans_up_when_done() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_report_watcher_cleanup"
    api._resolve_continuity_tool_event_owner_turn = (
        lambda *, fallback_turn_id: ("turn_report_watcher_cleanup", False, "")
    )
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: None
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: True
    state = {"status": "queued"}

    def _chain_remaining(*, turn_id, include_report_followup=True, **_kwargs):
        if include_report_followup:
            return True
        return state["status"] != "completed"

    api._turn_followthrough_chain_remaining = _chain_remaining
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": state["status"]}
    api._parent_input_event_key_for_tool_followup = lambda *, turn_id: "item_parent_report_watcher_cleanup"

    async def _fake_send_response_create(websocket, event, origin, **_kwargs):
        return True

    api._send_response_create = _fake_send_response_create

    async def _run() -> None:
        await api._maybe_continue_deterministic_followthrough_after_inject_only(
            websocket=object(),
            triggering_tool_name="gesture_look_center",
            triggering_call_id="call_report_watcher_cleanup",
            suppression_reason="gesture_intermediate_inject_only",
        )
        watchers = getattr(api, "_deterministic_followthrough_required_report_watchers", {})
        assert "turn_report_watcher_cleanup" in watchers
        state["status"] = "completed"
        await asyncio.sleep(0.15)
        watchers = getattr(api, "_deterministic_followthrough_required_report_watchers", {})
        assert "turn_report_watcher_cleanup" not in watchers

    asyncio.run(_run())


def test_prune_deterministic_followthrough_dispatch_registry_keeps_only_owner_and_current_turn() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_current"
    api._continuity_ledger = types.SimpleNamespace(compound_owner_turn_id=lambda: "turn_owner")
    api._deterministic_followthrough_dispatched_steps = {
        ("turn_owner", "step_owner"),
        ("turn_current", "step_current"),
        ("turn_old", "step_old"),
    }

    api._prune_deterministic_followthrough_dispatch_registry()

    assert ("turn_owner", "step_owner") in api._deterministic_followthrough_dispatched_steps
    assert ("turn_current", "step_current") in api._deterministic_followthrough_dispatched_steps
    assert ("turn_old", "step_old") not in api._deterministic_followthrough_dispatched_steps


def test_execute_function_call_skips_response_create_for_completed_non_report_gesture(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_gesture_complete_exec"
    api._current_input_event_key = "item_parent_complete_exec"
    api._active_input_event_key_by_turn_id["turn_gesture_complete_exec"] = "item_parent_complete_exec"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: False
    api.get_continuity_brief = lambda **_kwargs: types.SimpleNamespace(
        compound_request=types.SimpleNamespace(
            steps=(
                types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                types.SimpleNamespace(step_id="step_3", kind="gesture", status="completed"),
            ),
            active_step_index=None,
            recent_completed_step_id="step_3",
            next_pending_step_id=None,
            final_followup_pending=False,
        )
    )

    async def _fake_gesture(**_kwargs):
        return {"ok": True, "motion_request_key": "mrk-gesture-complete"}

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_center",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_center", "call_gesture_complete_exec", {"target": "center"}, ws))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert response_create_events == []
    canonical_key = api._canonical_utterance_key(
        turn_id="turn_gesture_complete_exec",
        input_event_key="tool:call_gesture_complete_exec",
    )
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert any(
        "tool_followup_state" in entry
        and f"canonical_key={canonical_key}" in entry
        and "state=dropped" in entry
        and "reason=followthrough_complete_non_report" in entry
        for entry in captured_logs
    )


def test_execute_function_call_blocks_non_owner_followthrough_gesture(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_turn_id_or_unknown = lambda: "turn_3"
    api._current_response_turn_id = "turn_3"
    api._active_input_event_key_by_turn_id["turn_3"] = "item_turn_3"
    api._active_input_event_key_by_turn_id["turn_2"] = "item_turn_2"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._apply_continuity_event(
        "transcript_final",
        text="look right, then left, then return to center and report done",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    api._apply_continuity_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-owning",
        turn_id="turn_2",
    )
    executed = {"count": 0}

    async def _fake_gesture(**_kwargs):
        executed["count"] += 1
        return {"ok": True}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_left",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_left", "call_owner_block", {"target": "left"}, ws))

    assert executed["count"] == 0
    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert response_create_events == []
    canonical_key = api._canonical_utterance_key(
        turn_id="turn_3",
        input_event_key="tool:call_owner_block",
    )
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    output_items = [event for event in ws.sent if event.get("type") == "conversation.item.create"]
    assert output_items
    output_payload = json.loads(str(output_items[0]["item"]["output"]))
    assert output_payload["result"]["reason"] == "followthrough_owner_turn_mismatch"


def test_execute_function_call_non_owning_empty_and_clarify_turns_do_not_inherit_followthrough_owner(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_2"
    api._active_input_event_key_by_turn_id["turn_2"] = "item_turn_2"
    api._active_input_event_key_by_turn_id["turn_3"] = "item_turn_3_empty"
    api._active_input_event_key_by_turn_id["turn_4"] = "item_turn_4_clarify"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._apply_continuity_event(
        "transcript_final",
        text="look right, then left, then return to center and report done",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    api._apply_continuity_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-owning",
        turn_id="turn_2",
    )
    executed = {"count": 0}

    async def _fake_gesture(**_kwargs):
        executed["count"] += 1
        return {"ok": True}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_center",
        _fake_gesture,
    )

    api._current_turn_id_or_unknown = lambda: "turn_3"
    asyncio.run(api.execute_function_call("gesture_look_center", "call_turn_3", {"target": "center"}, ws))
    api._current_turn_id_or_unknown = lambda: "turn_4"
    asyncio.run(api.execute_function_call("gesture_look_center", "call_turn_4", {"target": "center"}, ws))

    assert executed["count"] == 0
    canonical_turn_3 = api._canonical_utterance_key(turn_id="turn_3", input_event_key="tool:call_turn_3")
    canonical_turn_4 = api._canonical_utterance_key(turn_id="turn_4", input_event_key="tool:call_turn_4")
    assert api._tool_followup_state(canonical_key=canonical_turn_3) == "dropped"
    assert api._tool_followup_state(canonical_key=canonical_turn_4) == "dropped"


def test_execute_function_call_same_owner_followthrough_gesture_still_executes(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_turn_id_or_unknown = lambda: "turn_2"
    api._current_response_turn_id = "turn_2"
    api._active_input_event_key_by_turn_id["turn_2"] = "item_turn_2"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._apply_continuity_event(
        "transcript_final",
        text="look right, then left, then return to center and report done",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    api._apply_continuity_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-owning",
        turn_id="turn_2",
    )
    executed = {"count": 0}

    async def _fake_gesture(**_kwargs):
        executed["count"] += 1
        return {"ok": True}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_left",
        _fake_gesture,
    )

    asyncio.run(api.execute_function_call("gesture_look_left", "call_owner_ok", {"target": "left"}, ws))

    assert executed["count"] == 1


def test_execute_function_call_explicit_cross_turn_rebind_allows_non_owner_followthrough(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_turn_id_or_unknown = lambda: "turn_3"
    api._current_response_turn_id = "turn_3"
    api._active_input_event_key_by_turn_id["turn_2"] = "item_turn_2"
    api._active_input_event_key_by_turn_id["turn_3"] = "item_turn_3"
    api._active_response_id = "resp_tool_followup"
    api._response_trace_context_by_id = {
        "resp_tool_followup": {
            "origin": "tool_output",
            "turn_id": "turn_3",
            "parent_turn_id": "turn_2",
            "semantic_owner_turn_id": "turn_2",
            "allow_cross_turn_rebind": "true",
            "cross_turn_rebind_reason": "semantic_owner_parent_promoted",
        }
    }
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._intent_ledger = {}
    api._apply_continuity_event(
        "transcript_final",
        text="look right, then left, then return to center and report done",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    api._apply_continuity_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-owning",
        turn_id="turn_2",
    )
    executed = {"count": 0}
    dispatched: list[dict[str, object]] = []

    async def _fake_gesture(**_kwargs):
        executed["count"] += 1
        return {"ok": True}

    monkeypatch.setitem(
        __import__("ai.tools", fromlist=["function_map"]).function_map,
        "gesture_look_left",
        _fake_gesture,
    )
    api._deterministic_followthrough_runtime_descriptor = lambda *, turn_id: {
        "request_id": "req_rebind",
        "step_id": "step_4",
        "tool_name": "gesture_look_center",
        "tool_args": {"target": "center"},
    } if turn_id == "turn_2" else None

    async def _fake_dispatch(**kwargs):
        dispatched.append(kwargs)
        return {"outcome": "allow", "executed": True}

    api._submit_companion_gesture_tool_request = _fake_dispatch

    asyncio.run(api.execute_function_call("gesture_look_left", "call_rebind_ok", {"target": "left"}, ws))

    assert executed["count"] == 1
    assert len(dispatched) == 1
    assert dispatched[0]["turn_id"] == "turn_2"
    assert dispatched[0]["allow_followthrough_execution"] is True


def test_low_risk_gesture_followup_payload_preserves_distinct_motion_state() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_gesture_distinct"
    api._active_input_event_key_by_turn_id["turn_gesture_distinct"] = "item_parent_distinct"

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_gesture_distinct",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
        tool_result_has_distinct_info=api._tool_result_has_distinct_followup_info(
            tool_name="gesture_look_center",
            result={"motion_request_state": "already_satisfied", "tool_result_has_distinct_info": True},
        ),
    )

    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})

    assert metadata.get("tool_followup_status_only") == "true"
    assert metadata.get("tool_result_has_distinct_info") == "true"
    assert metadata.get("tool_followup_create_suppressed") is None


def test_non_gesture_tool_followup_never_sets_no_create_marker() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_response_turn_id = "turn_non_gesture"
    api._active_input_event_key_by_turn_id["turn_non_gesture"] = "item_parent_non_gesture"
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: False

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_non_gesture",
        response_create_event={"type": "response.create"},
        tool_name="perform_research",
        tool_result_has_distinct_info=False,
    )

    metadata = ((event.get("response") or {}).get("metadata") or {})

    assert metadata.get("tool_followup_create_suppressed") is None
    assert metadata.get("tool_followup_create_suppression_reason") is None


def test_tool_followup_queue_does_not_rebind_turn_active_key_to_tool_key() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    api._active_input_event_key_by_turn_id["turn_1"] = "item_parent"
    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_owner_queue",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    api._queue_response_origin("tool_output", response_create_event)

    assert api._active_input_event_key_for_turn("turn_1") == "item_parent"


def test_tool_followup_continuity_owner_turn_prefers_semantic_owner_trace() -> None:
    api = _make_api_stub()
    api._active_response_id = "resp_tool_followup"
    api._response_trace_context_by_id = {
        "resp_tool_followup": {
            "origin": "tool_output",
            "turn_id": "turn_3",
            "parent_turn_id": "turn_2",
            "semantic_owner_turn_id": "turn_2",
        }
    }

    owner_turn_id, allow_rebind, reason = api._resolve_continuity_tool_event_owner_turn(
        fallback_turn_id="turn_3",
    )

    assert owner_turn_id == "turn_2"
    assert allow_rebind is True
    assert reason == "tool_followup_semantic_owner_handoff"


def test_build_tool_followup_event_preserves_parent_owner_on_cross_turn_split() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    execution_turn = "turn_3"
    semantic_owner_turn = "turn_2"
    semantic_parent_input_event_key = "item_parent_2"
    api._current_turn_id_or_unknown = lambda: execution_turn
    api._active_input_event_key_by_turn_id[semantic_owner_turn] = semantic_parent_input_event_key
    api._active_input_event_key_by_turn_id[execution_turn] = "tool:call_prev"
    api._active_response_id = "resp_tool_followup_split"
    api._response_trace_context_by_id = {
        "resp_tool_followup_split": {
            "origin": "tool_output",
            "turn_id": execution_turn,
            "parent_turn_id": semantic_owner_turn,
            "semantic_owner_turn_id": semantic_owner_turn,
            # Deliberately stale/mismatched to prove parent input resolution
            # comes from semantic owner turn binding, not this trace field.
            "parent_input_event_key": "item_stale_trace_parent",
        }
    }

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_split_owner",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    assert metadata["turn_id"] == execution_turn
    assert metadata["parent_turn_id"] == semantic_owner_turn
    assert metadata["parent_input_event_key"] == semantic_parent_input_event_key


def test_build_tool_followup_event_uses_parent_turn_followthrough_truth_for_final_report_bridge() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    execution_turn = "turn_3"
    semantic_owner_turn = "turn_2"
    semantic_parent_input_event_key = "item_parent_2"
    api._current_turn_id_or_unknown = lambda: execution_turn
    api._active_input_event_key_by_turn_id[semantic_owner_turn] = semantic_parent_input_event_key
    api._active_input_event_key_by_turn_id[execution_turn] = "tool:call_prev"
    api._active_response_id = "resp_tool_followup_split_report"
    api._response_trace_context_by_id = {
        "resp_tool_followup_split_report": {
            "origin": "tool_output",
            "turn_id": execution_turn,
            "parent_turn_id": semantic_owner_turn,
            "semantic_owner_turn_id": semantic_owner_turn,
        }
    }
    api.get_gesture_motion_state = lambda *, tool_call_id: {"status": "completed"}

    def continuity_brief(*, turn_id: str, **_kwargs):
        if turn_id == semantic_owner_turn:
            return types.SimpleNamespace(
                compound_request=types.SimpleNamespace(
                    steps=(
                        types.SimpleNamespace(step_id="step_1", kind="gesture", status="completed"),
                        types.SimpleNamespace(step_id="step_2", kind="gesture", status="completed"),
                        types.SimpleNamespace(step_id="step_3", kind="report", status="pending"),
                    ),
                    active_step_index=None,
                    recent_completed_step_id="step_2",
                    next_pending_step_id="step_3",
                    final_followup_pending=True,
                )
            )
        return types.SimpleNamespace(compound_request=None)

    api.get_continuity_brief = continuity_brief

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_split_owner_report",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    response_payload = event.get("response") or {}
    metadata = response_payload.get("metadata") or {}
    instructions = str(response_payload.get("instructions") or "")

    assert metadata.get("tool_followup_create_suppressed") is None
    assert metadata.get("tool_followup_status_only") is None
    assert "Required user deliverable is still owed for the parent turn." in instructions


def test_build_tool_followup_event_for_required_tool_result_forces_report_materialization() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._active_input_event_key_by_turn_id["turn_1"] = "item_parent_1"
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: turn_id == "turn_1"
    api._required_deliverable_required_tool_name = lambda *, turn_id: "get_current_time" if turn_id == "turn_1" else None

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_time_1",
        response_create_event={"type": "response.create"},
        tool_name="get_current_time",
    )

    response_payload = event.get("response") or {}
    metadata = response_payload.get("metadata") or {}
    instructions = str(response_payload.get("instructions") or "")

    assert response_payload.get("tool_choice") == "none"
    assert metadata.get("tool_followup_step_output_policy") == "required_deliverable"
    assert metadata.get("followthrough_step_output_policy") == "required_deliverable"
    assert metadata.get("tool_followup_required_tool_name") == "get_current_time"
    assert "tool result is already available" in instructions
    assert "Do not call tools again." in instructions


def test_build_tool_followup_event_non_required_tool_does_not_force_no_tool_followup() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._active_input_event_key_by_turn_id["turn_1"] = "item_parent_1"
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: turn_id == "turn_1"
    api._required_deliverable_required_tool_name = lambda *, turn_id: "get_current_time" if turn_id == "turn_1" else None

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_weather_1",
        response_create_event={"type": "response.create"},
        tool_name="get_weather",
    )

    response_payload = event.get("response") or {}
    metadata = response_payload.get("metadata") or {}

    assert response_payload.get("tool_choice") != "none"
    assert metadata.get("tool_followup_required_tool_name") is None
    assert metadata.get("tool_followup_step_output_policy") is None


def test_required_tool_lineage_survives_redrive_response_created_clear_and_stale_reconstruction() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._active_input_event_key_by_turn_id["turn_1"] = "item_parent_1"
    api._turn_has_active_required_deliverable_step = lambda *, turn_id: turn_id == "turn_1"
    api._required_deliverable_required_tool_name = (
        lambda *, turn_id: "get_current_time" if turn_id == "turn_1" else None
    )
    api._required_deliverable_step_requires_tool_execution = (
        lambda *, turn_id: turn_id == "turn_1"
    )
    api._tool_call_records = [
        {
            "name": "get_current_time",
            "turn_id": "turn_1",
            "call_id": "call_time_0",
        }
    ]

    dispatched = asyncio.run(
        api._dispatch_required_deliverable_followthrough_response_create(
            websocket=ws,
            turn_id="turn_1",
        )
    )
    assert dispatched is True

    asyncio.run(
        api._handle_response_created_event(
            {"type": "response.created", "response": {"id": "resp-redrive-1", "metadata": {}}},
            ws,
        )
    )

    assert api._tool_call_records == []
    trace_context = api._response_trace_context_by_id.get("resp-redrive-1", {})
    assert trace_context.get("followthrough_required_tool_name") == "get_current_time"
    assert trace_context.get("tool_followup_required_tool_name") == "get_current_time"
    assert trace_context.get("followthrough_required_tool_already_executed") == "true"

    missing = api._required_deliverable_tool_execution_missing(
        turn_id="turn_1",
        required_deliverable_followthrough=True,
        response_id="resp-redrive-1",
        trace_context={},
        stale_context={},
    )
    assert missing is False

    api._quarantine_cancelled_response_id(
        response_id="resp-redrive-1",
        turn_id="turn_1",
        input_event_key="item_parent_1",
        origin="tool_output",
        reason="test_lineage_reconstruction",
    )
    stale_context = api._stale_response_context("resp-redrive-1")
    assert stale_context.get("followthrough_required_tool_name") == "get_current_time"
    assert stale_context.get("tool_followup_required_tool_name") == "get_current_time"
    assert stale_context.get("followthrough_required_tool_already_executed") == "true"


def test_catalog_only_descriptive_tool_followup_adds_uncertainty_guardrail_instruction() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._image_response_mode = "catalog_only"
    api._image_response_enabled = False
    api._active_input_event_key_by_turn_id["turn_1"] = "item_visual_parent"
    api._utterance_trust_snapshot_by_input_event_key = {
        "item_visual_parent": {
            "transcript_text": "Can you tell me what I'm holding in my hand?",
        }
    }

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_visual_guardrail",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )

    payload = event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(((event.get("response") or {}).get("instructions")) or "")

    assert metadata.get("tool_followup_status_only") is None
    assert "Descriptive visual follow-up" in instructions
    assert "complete the parent visual identification request" in instructions
    assert "it looks like" in instructions
    assert "Do not guess purpose, accessories, hidden parts" in instructions


def test_respond_mode_descriptive_tool_followup_uses_grounded_instruction_without_catalog_hedge() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    api._image_response_mode = "respond"
    api._image_response_enabled = True
    api._active_input_event_key_by_turn_id["turn_1"] = "item_visual_parent"
    api._utterance_trust_snapshot_by_input_event_key = {
        "item_visual_parent": {
            "transcript_text": "Can you tell me what I'm holding in my hand?",
        }
    }

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_visual_respond_mode",
        response_create_event={"type": "response.create"},
        tool_name="analyze_image",
    )

    payload = event.get("response") or {}
    metadata = payload.get("metadata") or {}
    instructions = str(payload.get("instructions") or "")

    assert metadata.get("tool_followup_status_only") is None
    assert "Descriptive visual follow-up" in instructions
    assert "Ground the answer in directly visible evidence" in instructions
    assert "it looks like" not in instructions


def test_tool_followup_response_created_does_not_rebind_parent_active_key_to_tool_key() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._active_input_event_key_by_turn_id["turn_2"] = "item_parent_2"
    api._current_response_turn_id = "turn_2"

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_owner_created",
        response_create_event={"type": "response.create", "response": {"metadata": {"turn_id": "turn_2"}}},
        tool_name="gesture_look_center",
    )

    asyncio.run(
        api._handle_response_created_event(
            {"type": "response.created", "response": {"id": "resp-tool-created", "metadata": ((event.get("response") or {}).get("metadata") or {})}},
            ws,
        )
    )

    assert api._active_input_event_key_for_turn("turn_2") == "item_parent_2"


def test_second_same_turn_tool_followup_prefers_canonical_parent_over_prior_tool_output() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_same_turn_tool_parent"
    parent_input_event_key = "item_parent_same_turn_tool_parent"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    first_tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key="tool:call_same_turn_first",
    )

    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-parent-same-turn"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )
    api._canonical_response_state_mutate(
        canonical_key=first_tool_canonical_key,
        turn_id=turn_id,
        input_event_key="tool:call_same_turn_first",
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", "resp-tool-first"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_same_turn_second",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    metadata = ((event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = "resp-tool-first"

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id="resp-tool-first",
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"
    assert parent_entry is not None
    assert parent_entry[0] == parent_canonical_key
    assert parent_entry[1].response_id == "resp-parent-same-turn"


def test_second_same_turn_tool_followup_release_uses_canonical_owner_not_prior_tool_response(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_same_turn_tool_release"
    parent_input_event_key = "item_parent_same_turn_tool_release"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    first_tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key="tool:call_same_turn_release_first",
    )

    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-parent-same-turn-release"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._canonical_response_state_mutate(
        canonical_key=first_tool_canonical_key,
        turn_id=turn_id,
        input_event_key="tool:call_same_turn_release_first",
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", "resp-tool-same-turn-release-first"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_same_turn_release_second",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    metadata = ((event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = "resp-tool-same-turn-release-first"

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id="resp-tool-same-turn-release-first",
    )

    assert should_drop is False
    assert reason == "parent_not_coverage_qualified"
    assert parent_entry is not None
    assert parent_entry[0] == parent_canonical_key
    assert api._tool_followup_state(canonical_key=canonical_key) == "new"
    assert any(
        "tool_followup_parent_resolution" in entry
        and "resolved_parent_response_id=resp-parent-same-turn-release" in entry
        and f"resolved_parent_canonical_key={parent_canonical_key}" in entry
        and "resolved_from=parent_key" in entry
        for entry in captured_logs
    )
    assert not any("reason=parent_origin_excluded" in entry for entry in captured_logs)
    assert not any("parent_deliverable_pending" in entry for entry in captured_logs)


def test_tool_followup_parent_resolution_ignores_tool_parent_input_key() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    parent_key = api._canonical_utterance_key(turn_id="turn_3", input_event_key="item_parent_3")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_3",
        input_event_key="item_parent_3",
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-parent-3"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "progress"),
            setattr(record, "done", True),
        ),
    )

    event, _ = api._build_tool_followup_response_create_event(
        call_id="call_parent_resolve",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = "turn_3"
    metadata["parent_turn_id"] = "turn_3"
    metadata["parent_input_event_key"] = "tool:call_parent_resolve"

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=None,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"
    assert parent_entry is not None
    assert parent_entry[0] == parent_key


def test_create_seam_parent_coverage_emits_single_info_source_of_truth(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    parent_key = api._canonical_utterance_key(turn_id="turn_cov", input_event_key="item_parent_cov")
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id="turn_cov",
        input_event_key="item_parent_cov",
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-parent-cov"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_cov",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = "turn_cov"
    metadata["parent_turn_id"] = "turn_cov"
    metadata["parent_input_event_key"] = "item_parent_cov"
    metadata["blocked_by_response_id"] = "resp-parent-cov"

    captured_info: list[str] = []
    captured_debug: list[str] = []
    original_info = logger.info
    original_debug = logger.debug

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_info.append(rendered)
        return original_info(message, *args, **kwargs)

    def _capture_debug(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_debug.append(rendered)
        return original_debug(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)
    monkeypatch.setattr(logger, "debug", _capture_debug)

    should_suppress, _, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id="resp-parent-cov",
    )

    assert should_suppress is True
    assert reason == "parent_covered_tool_result"
    assert not any("parent_coverage_source_of_truth" in entry for entry in captured_info)

    should_drop = api._should_drop_tool_followup_at_create_seam(
        turn_id="turn_cov",
        response_metadata=metadata,
        canonical_key=canonical_key,
        drain_trigger="test",
    )

    assert should_drop is True

    info_parent_coverage = [
        entry for entry in captured_info if "parent_coverage_source_of_truth" in entry
    ]
    debug_parent_coverage = [
        entry for entry in captured_debug if "parent_coverage_source_of_truth" in entry
    ]

    assert len(info_parent_coverage) == 1
    assert all("canonical_class=final" in entry for entry in info_parent_coverage)
    assert all("terminal_selected=" in entry for entry in info_parent_coverage)
    assert all("terminal_reason=" in entry for entry in info_parent_coverage)
    assert len(debug_parent_coverage) >= 1
    assert any("create_seam_parent_coverage_eval" in entry for entry in captured_info)


def test_tool_followup_parent_coverage_verdict_parity_between_create_and_response_done_release(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_cov_parity"
    parent_input_event_key = "item_parent_cov_parity"
    parent_response_id = "resp-parent-cov-parity"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    coverage_calls = 0
    original_parent_coverage_state = api._parent_response_coverage_state

    def _counted_parent_coverage_state(*args, **kwargs):
        nonlocal coverage_calls
        coverage_calls += 1
        return original_parent_coverage_state(*args, **kwargs)

    observation_reasons: list[tuple[str, str, str]] = []
    original_record_observation = api._record_tool_followup_observation

    def _capture_observation(*args, **kwargs):
        observation_reasons.append(
            (
                str(kwargs.get("authority_seam") or ""),
                str(kwargs.get("native_reason_code") or ""),
                str(kwargs.get("native_outcome_action") or ""),
            )
        )
        return original_record_observation(*args, **kwargs)

    monkeypatch.setattr(api, "_parent_response_coverage_state", _counted_parent_coverage_state)
    monkeypatch.setattr(api, "_record_tool_followup_observation", _capture_observation)

    create_event, create_canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_cov_parity_create",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    create_metadata = ((create_event.get("response") or {}).get("metadata") or {})
    create_metadata["turn_id"] = turn_id
    create_metadata["parent_turn_id"] = turn_id
    create_metadata["parent_input_event_key"] = parent_input_event_key
    create_metadata["blocked_by_response_id"] = parent_response_id

    should_drop = api._should_drop_tool_followup_at_create_seam(
        turn_id=turn_id,
        response_metadata=create_metadata,
        canonical_key=create_canonical_key,
        drain_trigger="test",
    )
    assert should_drop is True

    release_event, _release_canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_cov_parity_release",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    release_metadata = ((release_event.get("response") or {}).get("metadata") or {})
    release_metadata["turn_id"] = turn_id
    release_metadata["parent_turn_id"] = turn_id
    release_metadata["parent_input_event_key"] = parent_input_event_key
    release_metadata["blocked_by_response_id"] = parent_response_id
    release_input_event_key = str(release_metadata.get("input_event_key") or "")
    release_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=release_input_event_key)
    api._response_create_queue.append({"event": release_event, "origin": "tool_output", "turn_id": turn_id})
    api._set_tool_followup_state(
        canonical_key=release_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert release_metadata.get("tool_followup_release") != "true"

    assert coverage_calls >= 2
    assert (
        "ai.realtime_api._should_drop_tool_followup_at_create_seam",
        "parent_covered_tool_result",
        "DROP",
    ) in observation_reasons
    assert (
        "ai.realtime_api._release_blocked_tool_followups_for_response_done",
        "parent_covered_tool_result",
        "DROP",
    ) in observation_reasons


def test_tool_followup_parent_coverage_verdict_parity_hold_between_create_and_response_done_release(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_cov_parity_hold"
    parent_input_event_key = "synthetic_server_auto_hold"
    parent_response_id = "resp-parent-cov-parity-hold"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._is_provisional_response = lambda **kwargs: kwargs.get("response_id") == parent_response_id
    api._active_input_event_key_for_turn = lambda _turn_id: parent_input_event_key

    observations: list[dict[str, str]] = []
    original_record_observation = api._record_tool_followup_observation

    def _capture_observation(*args, **kwargs):
        observations.append(
            {
                "authority_seam": str(kwargs.get("authority_seam") or ""),
                "native_reason_code": str(kwargs.get("native_reason_code") or ""),
                "native_outcome_action": str(kwargs.get("native_outcome_action") or ""),
                "parent_coverage_state": str(kwargs.get("parent_coverage_state") or ""),
                "followup_outcome_posture": str(kwargs.get("followup_outcome_posture") or ""),
            }
        )
        return original_record_observation(*args, **kwargs)

    monkeypatch.setattr(api, "_record_tool_followup_observation", _capture_observation)

    create_event, create_canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_cov_parity_hold_create",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    create_metadata = ((create_event.get("response") or {}).get("metadata") or {})
    create_metadata["turn_id"] = turn_id
    create_metadata["parent_turn_id"] = turn_id
    create_metadata["parent_input_event_key"] = parent_input_event_key
    create_metadata["blocked_by_response_id"] = parent_response_id

    should_drop = api._should_drop_tool_followup_at_create_seam(
        turn_id=turn_id,
        response_metadata=create_metadata,
        canonical_key=create_canonical_key,
        drain_trigger="test",
    )
    assert should_drop is False

    release_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_cov_parity_hold_release",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    release_metadata = ((release_event.get("response") or {}).get("metadata") or {})
    release_metadata["turn_id"] = turn_id
    release_metadata["parent_turn_id"] = turn_id
    release_metadata["parent_input_event_key"] = parent_input_event_key
    release_metadata["blocked_by_response_id"] = parent_response_id
    release_input_event_key = str(release_metadata.get("input_event_key") or "")
    release_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=release_input_event_key)
    api._response_create_queue.append({"event": release_event, "origin": "tool_output", "turn_id": turn_id})
    api._set_tool_followup_state(
        canonical_key=release_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert release_metadata.get("tool_followup_release") != "true"
    assert api._tool_followup_state(canonical_key=release_canonical_key) == "blocked_active_response"

    create_observation = next(
        obs for obs in observations if obs["authority_seam"] == "ai.realtime_api._should_drop_tool_followup_at_create_seam"
    )
    release_observation = next(
        obs for obs in observations if obs["authority_seam"] == "ai.realtime_api._release_blocked_tool_followups_for_response_done"
    )
    assert create_observation["native_outcome_action"] == "HOLD"
    assert create_observation["followup_outcome_posture"] == "pending"
    assert release_observation["native_outcome_action"] == "HOLD"
    assert release_observation["followup_outcome_posture"] == "pending"
    assert create_observation["native_reason_code"] == release_observation["native_reason_code"] == "parent_deliverable_pending"
    assert create_observation["parent_coverage_state"] == release_observation["parent_coverage_state"] == "coverage_pending"


def test_tool_followup_parent_coverage_verdict_parity_release_between_create_and_response_done_release(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_cov_parity_release"
    parent_input_event_key = "item_parent_cov_parity_release"
    parent_response_id = "resp-parent-cov-parity-release"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    observations: list[dict[str, str]] = []
    original_record_observation = api._record_tool_followup_observation

    def _capture_observation(*args, **kwargs):
        observations.append(
            {
                "authority_seam": str(kwargs.get("authority_seam") or ""),
                "native_reason_code": str(kwargs.get("native_reason_code") or ""),
                "native_outcome_action": str(kwargs.get("native_outcome_action") or ""),
                "parent_coverage_state": str(kwargs.get("parent_coverage_state") or ""),
                "followup_outcome_posture": str(kwargs.get("followup_outcome_posture") or ""),
            }
        )
        return original_record_observation(*args, **kwargs)

    monkeypatch.setattr(api, "_record_tool_followup_observation", _capture_observation)

    create_event, create_canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_cov_parity_release_create",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    create_metadata = ((create_event.get("response") or {}).get("metadata") or {})
    create_metadata["turn_id"] = turn_id
    create_metadata["parent_turn_id"] = turn_id
    create_metadata["parent_input_event_key"] = parent_input_event_key
    create_metadata["blocked_by_response_id"] = parent_response_id
    api._set_tool_followup_state(
        canonical_key=create_canonical_key,
        state="scheduled_release",
        reason="test_seed_scheduled",
    )

    should_drop = api._should_drop_tool_followup_at_create_seam(
        turn_id=turn_id,
        response_metadata=create_metadata,
        canonical_key=create_canonical_key,
        drain_trigger="test",
    )
    assert should_drop is False

    release_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_cov_parity_release_release",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    release_metadata = ((release_event.get("response") or {}).get("metadata") or {})
    release_metadata["turn_id"] = turn_id
    release_metadata["parent_turn_id"] = turn_id
    release_metadata["parent_input_event_key"] = parent_input_event_key
    release_metadata["blocked_by_response_id"] = parent_response_id
    release_input_event_key = str(release_metadata.get("input_event_key") or "")
    release_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=release_input_event_key)
    api._response_create_queue.append({"event": release_event, "origin": "tool_output", "turn_id": turn_id})
    api._set_tool_followup_state(
        canonical_key=release_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert release_metadata.get("tool_followup_release") == "true"
    assert api._tool_followup_state(canonical_key=release_canonical_key) == "scheduled_release"

    create_observation = next(
        obs for obs in observations if obs["authority_seam"] == "ai.realtime_api._should_drop_tool_followup_at_create_seam"
    )
    release_observation = next(
        obs for obs in observations if obs["authority_seam"] == "ai.realtime_api._release_blocked_tool_followups_for_response_done"
    )
    assert create_observation["native_outcome_action"] == "RELEASE"
    assert create_observation["followup_outcome_posture"] == "released"
    assert release_observation["native_outcome_action"] == "RELEASE"
    assert release_observation["followup_outcome_posture"] == "released"
    assert create_observation["native_reason_code"] == release_observation["native_reason_code"] == "parent_not_coverage_qualified"
    assert create_observation["parent_coverage_state"] == release_observation["parent_coverage_state"] == "uncovered"



def test_dropped_tool_followup_lineage_blocks_assistant_message_create(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_tool_lineage"
    tool_input_event_key = "tool:call_lineage_1"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key)
    api._set_tool_followup_state(canonical_key=canonical_key, state="dropped", reason="test_seed")

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    response_create_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": turn_id, "input_event_key": tool_input_event_key}},
    }

    sent = asyncio.run(api._send_response_create(ws, response_create_event, origin="assistant_message"))

    assert sent is False
    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert any(
        "suppressed_tool_lineage_block origin=assistant_message" in entry
        and f"canonical_key={canonical_key}" in entry
        and "reason=tool_followup_state_dropped" in entry
        for entry in captured_logs
    )


def test_dropped_tool_followup_lineage_blocks_micro_ack_non_consuming_create(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_tool_lineage_ack"
    tool_input_event_key = "tool:call_lineage_2"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key)
    api._set_tool_followup_state(canonical_key=canonical_key, state="dropped", reason="test_seed")

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    response_create_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": turn_id,
                "input_event_key": tool_input_event_key,
                "micro_ack": "true",
                "consumes_canonical_slot": "false",
            }
        },
    }

    sent = asyncio.run(api._send_response_create(ws, response_create_event, origin="assistant_message"))

    assert sent is False
    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
    assert any("micro_ack_lineage_guard outcome=deny reason=tool_followup_state_dropped" in entry for entry in captured_logs)
    assert any(
        "suppressed_tool_lineage_block origin=micro_ack" in entry
        and f"canonical_key={canonical_key}" in entry
        for entry in captured_logs
    )


def test_pruned_tool_followup_rebinds_followon_micro_ack_to_parent_turn_key() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_pruned_followon_ack"
    parent_input_event_key = "item_parent_pruned_followon_ack"
    parent_response_id = "resp-parent-pruned-followon-ack"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._current_response_turn_id = turn_id

    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_pruned_followon_ack",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    tool_input_event_key = str(metadata.get("input_event_key") or "")
    api._current_input_event_key = tool_input_event_key
    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    followon_micro_ack = {
        "type": "response.create",
        "response": {
            "metadata": {
                "micro_ack": "true",
                "consumes_canonical_slot": "false",
                "micro_ack_turn_id": turn_id,
            }
        },
    }

    sent = asyncio.run(api._send_response_create(ws, followon_micro_ack, origin="assistant_message"))

    assert sent is True
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    response_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_events) == 1
    sent_metadata = ((response_events[0].get("response") or {}).get("metadata") or {})
    assert sent_metadata.get("input_event_key") == parent_input_event_key
    assert sent_metadata.get("input_event_key") != tool_input_event_key



def test_response_done_restarts_mic_for_silent_terminal_completion() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    turn_id = "turn_silent_done"
    input_event_key = "item_silent_done"
    response_id = "resp-silent-done"
    canonical_key = _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=input_event_key,
        response_id=response_id,
        origin="server_auto",
        audio_started=False,
    )

    asyncio.run(
        api.handle_response_done(
            {"type": "response.done", "response": {"id": response_id}}
        )
    )

    assert api._canonical_first_audio_started(canonical_key) is False
    assert api.mic.is_recording is True
    assert api.mic.start_recording_calls == 1



def test_response_done_does_not_double_restart_spoken_turns() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    turn_id = "turn_spoken_done"
    input_event_key = "item_spoken_done"
    response_id = "resp-spoken-done"
    canonical_key = _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=input_event_key,
        response_id=response_id,
        origin="server_auto",
        audio_started=True,
    )
    api._audio_playback_busy = True
    api._speaking_started = True
    api.exit_event = type("_ExitEvent", (), {"is_set": lambda self: False})()

    asyncio.run(
        api.handle_response_done(
            {"type": "response.done", "response": {"id": response_id}}
        )
    )

    assert api._canonical_first_audio_started(canonical_key) is True
    assert api.mic.start_recording_calls == 0
    assert api.mic.is_recording is False

    api._on_playback_complete()

    assert api.mic.start_recording_calls == 1
    assert api.mic.is_recording is True



def test_response_done_restarts_mic_when_tool_followup_is_pruned_without_playback() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_response_done_pruned_recovery"
    parent_input_event_key = "item_parent_response_done_pruned_recovery"
    parent_response_id = "resp-parent-response-done-pruned-recovery"
    parent_key = _prime_response_done_api(
        api,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        response_id=parent_response_id,
        origin="server_auto",
        audio_started=False,
    )

    api._tool_followup_state = RealtimeAPI._tool_followup_state.__get__(api, RealtimeAPI)
    api._set_tool_followup_state = RealtimeAPI._set_tool_followup_state.__get__(api, RealtimeAPI)
    api._release_blocked_tool_followups_for_response_done = (
        RealtimeAPI._release_blocked_tool_followups_for_response_done.__get__(api, RealtimeAPI)
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_response_done_pruned_recovery",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_right",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    api._current_input_event_key = str(metadata.get("input_event_key") or "")
    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )

    api._set_tool_followup_state(
        canonical_key=canonical_key,
        state="blocked_active_response",
        reason="test_seed",
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    asyncio.run(
        api.handle_response_done(
            {"type": "response.done", "response": {"id": parent_response_id}}
        )
    )

    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert api._current_input_event_key == parent_input_event_key
    assert api.mic.start_recording_calls == 1
    assert api.mic.is_recording is True



def test_on_playback_complete_restarts_mic_after_tool_followup_prune_path() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_playback_prune"
    parent_input_event_key = "item_parent_playback_prune"
    parent_response_id = "resp-parent-playback-prune"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._current_response_turn_id = turn_id

    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_playback_prune",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    api._current_input_event_key = str(metadata.get("input_event_key") or "")
    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )
    api._audio_playback_busy = True
    api.exit_event = type("_ExitEvent", (), {"is_set": lambda self: False})()

    api._on_playback_complete()

    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert api._current_input_event_key == parent_input_event_key
    assert api._audio_playback_busy is False
    assert api.mic.is_recording is True


def test_unrelated_micro_ack_for_user_turn_still_allowed() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_normal_ack"
    response_create_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": turn_id,
                "input_event_key": "item_turn_normal_ack",
                "micro_ack": "true",
                "consumes_canonical_slot": "false",
            }
        },
    }

    sent = asyncio.run(api._send_response_create(ws, response_create_event, origin="assistant_message"))

    assert sent is True
    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_distinct_info_followup_remains_unsuppressed_where_intended() -> None:
    api = _make_api_stub()

    should_drop, _parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata={
            "tool_followup_suppress_if_parent_covered": "true",
            "tool_name": "gesture_look_around",
            "tool_result_has_distinct_info": "true",
        },
        blocked_by_response_id=None,
    )

    assert should_drop is False
    assert reason == "distinct_info"


def test_tool_followup_parent_covered_not_suppressed_when_non_report_followthrough_remains() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_followthrough_remaining"
    parent_input_event_key = "item_followthrough_remaining"
    parent_response_id = "resp-followthrough-covered"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True, **_kwargs: (
        turn_id == "turn_followthrough_remaining"
    )

    response_create_event, _canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_followthrough_remaining",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id

    should_drop, _parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "followthrough_chain_remaining"


def test_tool_followup_parent_covered_stays_suppressed_without_gesture_followthrough_guard() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_followthrough_guard_off"
    parent_input_event_key = "item_followthrough_guard_off"
    parent_response_id = "resp-followthrough-guard-off"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )
    api._turn_followthrough_chain_remaining = lambda **_kwargs: False

    response_create_event, _canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_followthrough_guard_off",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id

    should_drop, _parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"


def test_tool_followup_parent_covered_not_suppressed_when_only_report_followthrough_remains() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_report_followthrough_remaining"
    parent_input_event_key = "item_report_followthrough_remaining"
    parent_response_id = "resp-report-followthrough-covered"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True, **_kwargs: (
        turn_id == "turn_report_followthrough_remaining" and include_report_followup
    )

    response_create_event, _canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_report_followthrough_remaining",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id

    should_drop, _parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "followthrough_chain_remaining"


def test_suppressed_tool_followup_lineage_is_idempotent_across_repeated_queue_drains() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-active-drop"
    api._current_response_turn_id = "turn_tool_repeat_drop"

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_repeat_drop",
        response_create_event={"type": "response.create"},
    )

    async def _run() -> None:
        queued = await api._send_response_create(ws, response_create_event, origin="tool_output")
        assert queued is False
        assert api._tool_followup_state(canonical_key=canonical_key) == "blocked_active_response"
        api._set_tool_followup_state(canonical_key=canonical_key, state="dropped", reason="manual_test_drop")
        api._response_in_flight = False
        api.response_in_progress = False
        await api._drain_response_create_queue(source_trigger="response_done")
        await api._drain_response_create_queue(source_trigger="response_done")

    asyncio.run(_run())

    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert [event for event in ws.sent if event.get("type") == "response.create"] == []


def test_assistant_message_non_tool_lineage_path_still_allowed() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_assistant_normal"
    response_create_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": turn_id,
                "input_event_key": "item_assistant_normal",
            }
        },
    }

    sent = asyncio.run(api._send_response_create(ws, response_create_event, origin="assistant_message"))

    assert sent is True
    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_transcript_final_recovery_schedules_upgrade_after_provisional_server_auto_done() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._attention_continuity = type(
        "_Attention",
        (),
        {"refresh_hold": lambda self, **_kwargs: type("_Snapshot", (), {"active": False})()},
    )()

    api._current_response_turn_id = "turn_2"
    api._active_response_origin = "unknown"
    api._active_response_id = None
    api._active_server_auto_input_event_key = "synthetic_server_auto_2"
    api._record_pending_server_auto_response(
        turn_id="turn_2",
        response_id="resp-server-auto-2",
        canonical_key="run-405-repro:turn_2:synthetic_server_auto_2",
    )
    pending = api._pending_server_auto_response_for_turn(turn_id="turn_2")
    assert pending is not None
    pending.pre_audio_hold = True
    api._canonical_first_audio_started = lambda _canonical_key: False

    async def _false(*_args, **_kwargs) -> bool:
        return False

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_handle_preference_recall_intent = _false
    api._maybe_process_research_intent = _false
    api._maybe_verify_on_risk_clarify = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._evaluate_attention_gate_admission = lambda **_kwargs: (True, "direct_address")
    api._log_utterance_trust_snapshot = lambda **_kwargs: {"word_count": 9}
    api._asr_verify_short_utterance_ms = 1200
    api._vad_turn_detection = {}

    captured: dict[str, str] = {}

    async def _capture_replace(*, websocket, turn_id, input_event_key, origin_label, memory_intent_subtype="none"):
        captured.update(
            {
                "turn_id": turn_id,
                "input_event_key": input_event_key,
                "origin_label": origin_label,
                "memory_intent_subtype": memory_intent_subtype,
                "websocket": "set" if websocket is ws else "other",
            }
        )
        return True

    api._cancel_and_replace_pending_server_auto_on_transcript_final = _capture_replace

    asyncio.run(
        api._handle_input_audio_transcription_completed_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_user_2",
                "transcript": "Can you look center and then snap to attention?",
            },
            ws,
        )
    )

    assert captured == {
        "turn_id": "turn_2",
        "input_event_key": "item_user_2",
        "origin_label": "upgraded_response",
        "memory_intent_subtype": "none",
        "websocket": "set",
    }


def test_tool_result_send_normalizes_oversized_call_id(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_norm"
    api._current_input_event_key = "item_user_norm"
    api._mark_utterance_info_summary = lambda **_kwargs: None

    async def _fake_add_no_tools(*_args, **_kwargs) -> None:
        return None

    async def _fake_research(**_kwargs):
        return {"summary": "done"}

    api._add_no_tools_follow_up_instruction = _fake_add_no_tools
    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "perform_research", _fake_research)

    oversized_call_id = "compgest_a535dd5098194e50ac1f3de90318b5cc"
    asyncio.run(api.execute_function_call("perform_research", oversized_call_id, {"query": "logs"}, ws))

    function_outputs = [event for event in ws.sent if event.get("item", {}).get("type") == "function_call_output"]
    assert len(function_outputs) == 1
    normalized_call_id = str(function_outputs[0]["item"]["call_id"])
    assert len(normalized_call_id) <= 32
    assert normalized_call_id != oversized_call_id

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    metadata = ((response_create_events[0].get("response") or {}).get("metadata") or {})
    assert metadata.get("tool_call_id") == normalized_call_id
    assert metadata.get("input_event_key") == f"tool:{normalized_call_id}"
    assert api._tool_call_records[0]["call_id"] == normalized_call_id


def test_tool_result_send_keeps_short_call_id_unchanged(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api._current_response_turn_id = "turn_tool_short"
    api._current_input_event_key = "item_user_short"
    api._mark_utterance_info_summary = lambda **_kwargs: None

    async def _fake_add_no_tools(*_args, **_kwargs) -> None:
        return None

    async def _fake_research(**_kwargs):
        return {"summary": "done"}

    api._add_no_tools_follow_up_instruction = _fake_add_no_tools
    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "perform_research", _fake_research)

    short_call_id = "call_short_123"
    asyncio.run(api.execute_function_call("perform_research", short_call_id, {"query": "logs"}, ws))

    function_outputs = [event for event in ws.sent if event.get("item", {}).get("type") == "function_call_output"]
    assert len(function_outputs) == 1
    assert function_outputs[0]["item"]["call_id"] == short_call_id


def test_tool_followup_release_not_allowed_while_parent_deliverable_classification_pending() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_pending"
    parent_input_event_key = "synthetic_server_auto_7"
    parent_response_id = "resp-parent-provisional"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._is_provisional_response = lambda **kwargs: kwargs.get("response_id") == parent_response_id
    api._active_input_event_key_for_turn = lambda _turn_id: parent_input_event_key

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_parent_pending",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "parent_deliverable_pending"


def test_upgraded_response_still_suppresses_redundant_tool_followup_after_pending_parent() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_upgrade"
    parent_input_event_key = "item_parent_upgrade"
    parent_response_id = "resp-parent-upgrade"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_parent_upgrade",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    should_drop, entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "parent_terminal_selection_not_coverage_qualified"
    assert entry is not None
    parent_covered, coverage_source, _observed, _klass, _selected, _sel_reason = api._parent_response_coverage_state(
        parent_state=entry[1],
    )
    assert parent_covered is False
    assert coverage_source == "none"

    api._record_terminal_response_text(response_id=parent_response_id, text="It looks like a phone.")

    should_drop, entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is True
    assert reason == "parent_covered_tool_result"
    assert entry is not None
    parent_covered, coverage_source, _observed, _klass, _selected, _sel_reason = api._parent_response_coverage_state(
        parent_state=entry[1],
    )
    assert parent_covered is True
    assert coverage_source == "canonical"


def test_parent_coverage_does_not_mark_canonical_final_when_terminal_selection_deferred_to_tool_followup() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_terminal_precedence"
    parent_input_event_key = "item_parent_terminal_precedence"
    parent_response_id = "resp-parent-terminal-precedence"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
        ),
    )
    api._record_terminal_response_text(
        response_id=parent_response_id,
        text="Your preferred editor is Neovim.",
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=False,
        selection_reason="tool_followup_precedence",
    )

    parent_state = api._canonical_response_state(parent_key)
    covered, source, observed, deliverable_class, terminal_selected, terminal_reason = api._parent_response_coverage_state(
        parent_state=parent_state,
        parent_canonical_key=parent_key,
    )

    assert covered is False
    assert source == "none"
    assert observed is True
    assert deliverable_class == "final"
    assert terminal_selected is False
    assert terminal_reason == "tool_followup_precedence"


def test_tool_followup_release_is_not_dropped_when_parent_terminal_selection_deferred() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_terminal_precedence_release"
    parent_input_event_key = "item_parent_terminal_precedence_release"
    parent_response_id = "resp-parent-terminal-precedence-release"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "done", True),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
        ),
    )
    api._record_terminal_response_text(
        response_id=parent_response_id,
        text="Your preferred editor is Neovim.",
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=False,
        selection_reason="tool_followup_precedence",
    )

    response_create_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_parent_terminal_precedence_release",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["blocked_by_response_id"] = parent_response_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key

    should_drop, _entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert should_drop is False
    assert reason == "parent_not_coverage_qualified"


def test_status_only_gesture_followup_observation_marks_released_child_redundant() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_status_only_release"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_status_only")
    api._record_tool_followup_metadata(
        canonical_key=canonical_key,
        metadata={
            "tool_name": "gesture_look_center",
            "tool_followup_status_only": "true",
        },
    )

    api._record_tool_followup_observation(
        turn_id=turn_id,
        input_event_key="tool:call_status_only",
        canonical_key=canonical_key,
        origin="tool_output",
        parent_coverage_state="uncovered",
        followup_outcome_posture="released",
        native_reason_code="parent_not_deliverable",
        native_outcome_action="SEND",
        response_metadata={
            "tool_name": "gesture_look_center",
            "tool_followup_status_only": "true",
        },
        parent_canonical_key=f"{turn_id}::item_parent",
        parent_semantic_owner_key=f"{turn_id}::item_parent",
        authority_seam="tool_followup_release_seam",
    )

    trace = api._turn_arbitration_trace_by_key[(api._current_run_id() or "", turn_id)]
    observation = trace.tool_followup_observations[-1]

    assert observation.decision.followup_outcome_posture == "released"
    assert observation.decision.followup_distinctness == "redundant"
    assert observation.decision.native_reason_code == "parent_not_deliverable"
    assert observation.context.authority_retained_by == "ai.tool_followup_arbitration.decide_tool_followup_arbitration"
    assert observation.decision.authority_retained_by == "ai.tool_followup_arbitration.decide_tool_followup_arbitration"
    assert "provenance_source:tool_followup:tool_followup_release_seam" in observation.normalization_warnings


def test_provisional_server_auto_parent_progression_holds_then_suppresses_followup_after_terminal_selection() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_progression"
    parent_input_event_key = "synthetic_server_auto_parent"
    parent_response_id = "resp-parent-provisional-progression"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._mark_response_provisional(response_id=parent_response_id)

    # 1) provisional server_auto hits response.done before transcript final
    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="server_auto",
        delivery_state_before_done="done",
        active_response_was_provisional=True,
        done_canonical_key=parent_canonical_key,
        transcript_final_seen=False,
    )
    # 2) deliverable classification is non-deliverable pending transcript final
    assert selected is False
    assert reason == "provisional_server_auto_awaiting_transcript_final"

    followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_progression_parent",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    followup_metadata = ((followup_event.get("response") or {}).get("metadata") or {})
    followup_metadata["blocked_by_response_id"] = parent_response_id
    followup_metadata["parent_turn_id"] = turn_id
    followup_metadata["parent_input_event_key"] = parent_input_event_key
    tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=str(followup_metadata.get("input_event_key") or ""),
    )
    api._response_create_queue.append(
        {
            "event": followup_event,
            "origin": "tool_output",
            "turn_id": turn_id,
        }
    )
    api._set_tool_followup_state(
        canonical_key=tool_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    # Parent remains provisional (no transcript-final linkage yet).
    api._active_input_event_key_for_turn = lambda _turn_id: parent_input_event_key

    # 3) tool followup stays held
    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert api._tool_followup_state(canonical_key=tool_canonical_key) == "blocked_active_response"
    assert followup_metadata.get("tool_followup_release") != "true"

    # 4) upgraded / transcript-final path lands
    transcript_final_key = "item_parent_progression_final"
    api._active_input_event_key_for_turn = lambda _turn_id: transcript_final_key
    # 5) parent becomes terminally classified as covered
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=transcript_final_key,
        selected=True,
        selection_reason="normal",
    )

    # 6) terminal selection alone does not suppress followup without substantive parent text
    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert api._tool_followup_state(canonical_key=tool_canonical_key) == "scheduled_release"
    assert followup_metadata.get("tool_followup_release") == "true"


def test_tool_followup_release_does_not_reopen_parent_pending_after_tool_rebind() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_tool_rebind"
    parent_input_event_key = "item_parent_tool_rebind"
    parent_response_id = "resp-parent-tool-rebind"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._mark_response_provisional(response_id=parent_response_id)
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    first_followup_response_id = "resp-tool-followup-1"
    first_followup_input_event_key = "tool:call_parent_tool_rebind_1"
    first_followup_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=first_followup_input_event_key,
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=first_followup_canonical_key,
        semantic_owner_canonical_key=parent_canonical_key,
        response_id=first_followup_response_id,
        turn_id=turn_id,
        input_event_key=first_followup_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    # Reproduce the live-run seam: once the first tool followup is released, the
    # active key for the turn can legitimately move to the tool lineage.
    api._active_input_event_key_by_turn_id[turn_id] = first_followup_input_event_key

    second_followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_parent_tool_rebind_2",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    second_followup_metadata = ((second_followup_event.get("response") or {}).get("metadata") or {})
    second_followup_metadata["blocked_by_response_id"] = first_followup_response_id
    second_followup_metadata["parent_turn_id"] = turn_id
    second_followup_metadata["parent_input_event_key"] = first_followup_input_event_key

    decision, entry = api._decide_tool_followup_arbitration(
        response_metadata=second_followup_metadata,
        blocked_by_response_id=first_followup_response_id,
    )

    assert entry is not None
    assert entry[0] == parent_canonical_key
    assert decision.should_hold is False
    assert decision.should_release is True
    assert decision.reason_code == "parent_terminal_selection_not_coverage_qualified"
    assert decision.parent_coverage_state == "uncovered"


def test_tool_followup_parent_pending_falls_back_to_active_turn_key_when_parent_input_key_missing() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_parent_missing_input_key"
    parent_input_event_key = "item_parent_missing_input_key"
    parent_response_id = "resp-parent-missing-input-key"
    parent_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=None,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
            setattr(record, "input_event_key", ""),
        ),
    )
    api._mark_response_provisional(response_id=parent_response_id)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key

    followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_parent_missing_input_key",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    followup_metadata = ((followup_event.get("response") or {}).get("metadata") or {})
    followup_metadata["blocked_by_response_id"] = parent_response_id
    followup_metadata["parent_turn_id"] = turn_id
    followup_metadata["parent_input_event_key"] = parent_input_event_key

    decision, entry = api._decide_tool_followup_arbitration(
        response_metadata=followup_metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert entry is not None
    assert entry[0] == parent_canonical_key
    assert decision.should_hold is False
    assert decision.should_release is True
    assert decision.reason_code == "parent_not_coverage_qualified"
    assert decision.parent_coverage_state == "uncovered"


def test_tool_followup_parent_resolution_rebinds_third_sibling_to_canonical_non_tool_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_tool_chain_canonical_parent"
    canonical_parent_input_event_key = "item_tool_chain_canonical_parent"
    canonical_parent_response_id = "resp-tool-chain-canonical-parent"
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = canonical_parent_input_event_key
    canonical_parent_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=canonical_parent_key,
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", canonical_parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_a_event, tool_a_key = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_a",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    tool_a_metadata = ((tool_a_event.get("response") or {}).get("metadata") or {})
    tool_a_metadata["turn_id"] = turn_id
    tool_a_metadata["parent_turn_id"] = turn_id
    tool_a_metadata["parent_input_event_key"] = canonical_parent_input_event_key
    api._record_tool_followup_metadata(canonical_key=tool_a_key, metadata=tool_a_metadata)

    tool_b_event, tool_b_key = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_b",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    tool_b_metadata = ((tool_b_event.get("response") or {}).get("metadata") or {})
    tool_b_metadata["turn_id"] = turn_id
    tool_b_metadata["parent_turn_id"] = turn_id
    tool_b_metadata["parent_input_event_key"] = "tool:call_tool_chain_a"
    api._record_tool_followup_metadata(canonical_key=tool_b_key, metadata=tool_b_metadata)

    tool_b_response_id = "resp-tool-chain-b"
    api._canonical_response_state_mutate(
        canonical_key=tool_b_key,
        turn_id=turn_id,
        input_event_key="tool:call_tool_chain_b",
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", tool_b_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_c_event, _tool_c_key = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_c",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    tool_c_metadata = ((tool_c_event.get("response") or {}).get("metadata") or {})
    tool_c_metadata["turn_id"] = turn_id
    tool_c_metadata["parent_turn_id"] = turn_id
    tool_c_metadata["parent_input_event_key"] = "tool:call_tool_chain_b"

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=tool_c_metadata,
        blocked_by_response_id=tool_b_response_id,
    )

    assert should_drop is False
    assert reason == "parent_terminal_selection_not_coverage_qualified"
    assert parent_entry is not None
    assert parent_entry[0] == canonical_parent_key
    assert parent_entry[1].response_id == canonical_parent_response_id
    assert any(
        "tool_followup_parent_resolution" in entry
        and f"resolved_parent_canonical_key={canonical_parent_key}" in entry
        and "resolved_from=tool_parent_metadata" in entry
        for entry in captured_logs
    )


def test_tool_followup_parent_resolution_prefers_semantic_owner_promotion_for_later_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_tool_chain_semantic_owner"
    canonical_parent_input_event_key = "item_tool_chain_semantic_owner"
    canonical_parent_response_id = "resp-tool-chain-semantic-owner-parent"
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = canonical_parent_input_event_key
    canonical_parent_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=canonical_parent_key,
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", canonical_parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_b_input_event_key = "tool:call_tool_chain_semantic_b"
    tool_b_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_b_input_event_key)
    tool_b_response_id = "resp-tool-chain-semantic-b"
    api._canonical_response_state_mutate(
        canonical_key=tool_b_key,
        turn_id=turn_id,
        input_event_key=tool_b_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", tool_b_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._record_tool_followup_metadata(
        canonical_key=tool_b_key,
        metadata={
            "tool_name": "gesture_look_center",
            "tool_followup_status_only": "true",
            "parent_turn_id": turn_id,
            "parent_input_event_key": "tool:call_tool_chain_semantic_a",
        },
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=tool_b_key,
        semantic_owner_canonical_key=canonical_parent_key,
        response_id=tool_b_response_id,
        turn_id=turn_id,
        input_event_key=tool_b_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    tool_c_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_semantic_c",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    tool_c_metadata = ((tool_c_event.get("response") or {}).get("metadata") or {})
    tool_c_metadata["turn_id"] = turn_id
    tool_c_metadata["parent_turn_id"] = turn_id
    tool_c_metadata["parent_input_event_key"] = tool_b_input_event_key

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=tool_c_metadata,
        blocked_by_response_id=tool_b_response_id,
    )

    assert should_drop is False
    assert reason == "parent_terminal_selection_not_coverage_qualified"
    assert parent_entry is not None
    assert parent_entry[0] == canonical_parent_key
    assert any(
        "tool_followup_parent_resolution" in entry
        and f"resolved_parent_canonical_key={canonical_parent_key}" in entry
        and "resolved_from=semantic_owner_response_id" in entry
        for entry in captured_logs
    )


def test_tool_followup_parent_resolution_rebinds_semantic_owner_tool_output_via_tool_parent_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_tool_chain_semantic_owner_tool_parent"
    canonical_parent_input_event_key = "item_tool_chain_semantic_owner_tool_parent"
    canonical_parent_response_id = "resp-tool-chain-semantic-owner-tool-parent"
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = canonical_parent_input_event_key
    canonical_parent_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=canonical_parent_key,
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", canonical_parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_b_input_event_key = "tool:call_tool_chain_semantic_owner_tool_parent_b"
    tool_b_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_b_input_event_key)
    tool_b_response_id = "resp-tool-chain-semantic-owner-tool-parent-b"
    api._canonical_response_state_mutate(
        canonical_key=tool_b_key,
        turn_id=turn_id,
        input_event_key=tool_b_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", tool_b_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._record_tool_followup_metadata(
        canonical_key=tool_b_key,
        metadata={
            "tool_name": "gesture_look_center",
            "tool_followup_status_only": "true",
            "parent_turn_id": turn_id,
            "parent_input_event_key": canonical_parent_input_event_key,
        },
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=tool_b_key,
        semantic_owner_canonical_key=tool_b_key,
        response_id=tool_b_response_id,
        turn_id=turn_id,
        input_event_key=tool_b_input_event_key,
        selected=True,
        selection_reason="tool_followup_precedence",
    )

    tool_c_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_semantic_owner_tool_parent_c",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    tool_c_metadata = ((tool_c_event.get("response") or {}).get("metadata") or {})
    tool_c_metadata["turn_id"] = turn_id
    tool_c_metadata["parent_turn_id"] = turn_id
    tool_c_metadata["parent_input_event_key"] = tool_b_input_event_key

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=tool_c_metadata,
        blocked_by_response_id=tool_b_response_id,
    )

    assert should_drop is False
    assert reason == "parent_not_coverage_qualified"
    assert parent_entry is not None
    assert parent_entry[0] == canonical_parent_key
    assert any(
        "tool_followup_parent_resolution" in entry
        and f"resolved_parent_canonical_key={canonical_parent_key}" in entry
        and "resolved_from=semantic_owner_tool_parent_metadata" in entry
        for entry in captured_logs
    )
    assert not any("reason=parent_origin_excluded" in entry for entry in captured_logs)


def test_tool_followup_uncovered_parent_release_uses_canonical_lineage_instead_of_excluded_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_tool_chain_uncovered_release"
    canonical_parent_input_event_key = "item_tool_chain_uncovered_release"
    canonical_parent_response_id = "resp-tool-chain-uncovered-parent"
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = canonical_parent_input_event_key
    canonical_parent_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=canonical_parent_key,
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", canonical_parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_a_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_tool_chain_release_a")
    api._record_tool_followup_metadata(
        canonical_key=tool_a_key,
        metadata={
            "tool_name": "gesture_look_around",
            "tool_followup_status_only": "true",
            "parent_turn_id": turn_id,
            "parent_input_event_key": canonical_parent_input_event_key,
        },
    )

    tool_b_input_event_key = "tool:call_tool_chain_release_b"
    tool_b_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_b_input_event_key)
    api._record_tool_followup_metadata(
        canonical_key=tool_b_key,
        metadata={
            "tool_name": "gesture_look_center",
            "tool_followup_status_only": "true",
            "parent_turn_id": turn_id,
            "parent_input_event_key": "tool:call_tool_chain_release_a",
        },
    )
    tool_b_response_id = "resp-tool-chain-release-b"
    api._canonical_response_state_mutate(
        canonical_key=tool_b_key,
        turn_id=turn_id,
        input_event_key=tool_b_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", tool_b_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_c_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_release_c",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    tool_c_metadata = ((tool_c_event.get("response") or {}).get("metadata") or {})
    tool_c_metadata["turn_id"] = turn_id
    tool_c_metadata["parent_turn_id"] = turn_id
    tool_c_metadata["parent_input_event_key"] = tool_b_input_event_key

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=tool_c_metadata,
        blocked_by_response_id=tool_b_response_id,
    )

    assert should_drop is False
    assert reason == "parent_not_coverage_qualified"
    assert parent_entry is not None
    assert parent_entry[0] == canonical_parent_key
    assert any(
        "tool_followup_parent_resolution" in entry
        and f"resolved_parent_canonical_key={canonical_parent_key}" in entry
        and "resolved_from=tool_parent_metadata" in entry
        for entry in captured_logs
    )
    assert not any("reason=parent_origin_excluded" in entry for entry in captured_logs)


def test_tool_followup_exact_runtime_chain_resolves_tool_c_via_tool_parent_metadata_without_parent_origin_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_tool_chain_exact_runtime_shape"
    canonical_parent_input_event_key = "item_tool_chain_exact_runtime_shape"
    canonical_parent_response_id = "resp-tool-chain-exact-runtime-parent"
    api._current_response_turn_id = turn_id
    api._active_input_event_key_by_turn_id[turn_id] = canonical_parent_input_event_key
    canonical_parent_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=canonical_parent_key,
        turn_id=turn_id,
        input_event_key=canonical_parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", canonical_parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_a_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="tool:call_tool_chain_exact_a")
    api._record_tool_followup_metadata(
        canonical_key=tool_a_key,
        metadata={
            "tool_name": "gesture_look_around",
            "tool_followup_status_only": "true",
            "parent_turn_id": turn_id,
            "parent_input_event_key": canonical_parent_input_event_key,
        },
    )

    tool_b_input_event_key = "tool:call_tool_chain_exact_b"
    tool_b_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_b_input_event_key)
    api._record_tool_followup_metadata(
        canonical_key=tool_b_key,
        metadata={
            "tool_name": "gesture_look_center",
            "tool_followup_status_only": "true",
            "parent_turn_id": turn_id,
            "parent_input_event_key": "tool:call_tool_chain_exact_a",
        },
    )
    tool_b_response_id = "resp-tool-chain-exact-runtime-b"
    api._canonical_response_state_mutate(
        canonical_key=tool_b_key,
        turn_id=turn_id,
        input_event_key=tool_b_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "tool_output"),
            setattr(record, "response_id", tool_b_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    tool_c_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_tool_chain_exact_c",
        response_create_event={"type": "response.create"},
        tool_name="gesture_idle",
    )
    tool_c_metadata = ((tool_c_event.get("response") or {}).get("metadata") or {})
    tool_c_metadata["turn_id"] = turn_id
    tool_c_metadata["parent_turn_id"] = turn_id
    tool_c_metadata["parent_input_event_key"] = tool_b_input_event_key

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    should_drop, parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata=tool_c_metadata,
        blocked_by_response_id=tool_b_response_id,
    )

    assert should_drop is False
    assert reason == "parent_not_coverage_qualified"
    assert parent_entry is not None
    assert parent_entry[0] == canonical_parent_key
    assert parent_entry[1].response_id == canonical_parent_response_id
    assert any(
        "tool_followup_parent_resolution" in entry
        and f"resolved_parent_canonical_key={canonical_parent_key}" in entry
        and "resolved_from=tool_parent_metadata" in entry
        for entry in captured_logs
    )
    assert not any("reason=parent_origin_excluded" in entry for entry in captured_logs)


def test_response_done_suppression_prunes_blocked_tool_followup_lineage_artifacts() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_response_done_prune"
    parent_input_event_key = "item_parent_response_done_prune"
    parent_response_id = "resp_parent_response_done_prune"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._current_response_turn_id = turn_id

    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_response_done_prune",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    tool_input_event_key = str(metadata.get("input_event_key") or "")
    api._current_input_event_key = tool_input_event_key
    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._response_create_queue.append(
        {
            "event": {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": turn_id,
                        "input_event_key": parent_input_event_key,
                        "origin": "assistant_message",
                    }
                },
            },
            "origin": "assistant_message",
            "turn_id": turn_id,
        }
    )
    api._pending_response_create_origins.append(
        {
            "origin": "micro_ack",
            "micro_ack": "true",
            "consumes_canonical_slot": "false",
            "turn_id": turn_id,
            "input_event_key": tool_input_event_key,
        }
    )
    api._pending_response_create_origins.append(
        {
            "origin": "assistant_message",
            "micro_ack": "false",
            "consumes_canonical_slot": "true",
            "turn_id": turn_id,
            "input_event_key": parent_input_event_key,
        }
    )
    api._pending_micro_ack_by_turn_channel = {(turn_id, "voice"): object()}

    cancelled: list[tuple[str, str]] = []

    class _AckManager:
        def cancel_matching(self, *, turn_id: str, reason: str, matcher):
            cancelled.append((turn_id, reason))

    api._micro_ack_manager = _AckManager()
    api._set_tool_followup_state(
        canonical_key=canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)

    assert api._pending_response_create is None
    assert list(api._response_create_queue) == [
        {
            "event": {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": turn_id,
                        "input_event_key": parent_input_event_key,
                        "origin": "assistant_message",
                    }
                },
            },
            "origin": "assistant_message",
            "turn_id": turn_id,
        }
    ]
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert list(api._pending_response_create_origins) == [
        {
            "origin": "assistant_message",
            "micro_ack": "false",
            "consumes_canonical_slot": "true",
            "turn_id": turn_id,
            "input_event_key": parent_input_event_key,
        }
    ]
    assert api._pending_micro_ack_by_turn_channel == {}
    assert api._current_input_event_key == parent_input_event_key
    assert cancelled == [(turn_id, f"tool_followup_pruned:parent_covered_tool_result response_id={parent_response_id}")]


def test_terminal_selection_prunes_mirrored_tool_followup_once(monkeypatch) -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_prune_mirror"
    parent_input_event_key = "item_parent_terminal_prune_mirror"
    parent_response_id = "resp_parent_terminal_prune_mirror"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._current_response_turn_id = turn_id

    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_terminal_prune_mirror",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    tool_input_event_key = str(metadata.get("input_event_key") or "")

    api._pending_response_create = PendingResponseCreate(
        websocket=_RecordingWs(),
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._sync_pending_response_create_queue()
    api._set_tool_followup_state(
        canonical_key=canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    captured_logs: list[str] = []
    original_info = logger.info

    def _capture_info(message: str, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered % args
        captured_logs.append(rendered)
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(logger, "info", _capture_info)

    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=True,
        selection_reason="normal",
    )

    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert sum("tool_followup_parent_resolution" in entry for entry in captured_logs) == 1
    assert sum("tool_followup_lineage_invalidated" in entry for entry in captured_logs) == 1
    assert any(
        "terminal_deliverable_tool_followup_prune" in entry
        and "dropped_pending=1" in entry
        and "dropped_queue=1" in entry
        for entry in captured_logs
    )
    assert all(tool_input_event_key in entry for entry in captured_logs if "tool_followup_lineage_invalidated" in entry)


def test_response_done_suppression_rebinds_followon_micro_ack_to_parent_turn_key() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_response_done_followon_ack"
    parent_input_event_key = "item_parent_response_done_followon_ack"
    parent_response_id = "resp_parent_response_done_followon_ack"
    parent_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=parent_input_event_key)
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key
    api._current_response_turn_id = turn_id

    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "server_auto"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )

    response_create_event, canonical_key = api._build_tool_followup_response_create_event(
        call_id="call_response_done_followon_ack",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_center",
    )
    metadata = ((response_create_event.get("response") or {}).get("metadata") or {})
    metadata["turn_id"] = turn_id
    metadata["parent_turn_id"] = turn_id
    metadata["parent_input_event_key"] = parent_input_event_key
    metadata["blocked_by_response_id"] = parent_response_id
    tool_input_event_key = str(metadata.get("input_event_key") or "")
    api._current_input_event_key = tool_input_event_key
    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event=response_create_event,
        origin="tool_output",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=0,
        enqueue_seq=1,
    )
    api._set_tool_followup_state(
        canonical_key=canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)

    followon_micro_ack = {
        "type": "response.create",
        "response": {
            "metadata": {
                "micro_ack": "true",
                "consumes_canonical_slot": "false",
                "micro_ack_turn_id": turn_id,
            }
        },
    }

    sent = asyncio.run(api._send_response_create(ws, followon_micro_ack, origin="assistant_message"))

    assert sent is True
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    response_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_events) == 1
    sent_metadata = ((response_events[0].get("response") or {}).get("metadata") or {})
    assert sent_metadata.get("input_event_key") == parent_input_event_key
    assert sent_metadata.get("input_event_key") != tool_input_event_key


def test_emitted_micro_ack_freezes_parent_utterance_context_before_tool_followup_rebind() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws
    api.loop = asyncio.new_event_loop()

    turn_id = "turn_micro_ack_context_freeze"
    parent_input_event_key = "item_parent_micro_ack_context_freeze"
    tool_input_event_key = "tool:call_micro_ack_context_freeze"
    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
    )
    api._current_response_turn_id = turn_id
    api._current_input_event_key = parent_input_event_key
    api._active_input_event_key_by_turn_id[turn_id] = parent_input_event_key

    class _MutatingTransport:
        async def send_json(self, websocket: _RecordingWs, payload: dict[str, object]) -> None:
            await websocket.send(json.dumps(payload))
            if payload.get("type") == "conversation.item.create":
                api._current_input_event_key = tool_input_event_key
                api._active_input_event_key_by_turn_id[turn_id] = tool_input_event_key
                api._set_tool_followup_state(
                    canonical_key=tool_canonical_key,
                    state="dropped",
                    reason="test_rebind_after_emit",
                )

    api._get_or_create_transport = lambda: _MutatingTransport()

    async def _run() -> None:
        api._emit_micro_ack(
            MicroAckContext(
                category=MicroAckCategory.LATENCY_MASK,
                channel="voice",
                run_id=api._current_run_id(),
                session_id=None,
                turn_id=turn_id,
                intent=None,
                action=parent_canonical_key,
                tool_call_id=None,
            ),
            "latency_mask_hmm",
            "Hmm.",
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    try:
        api.loop.run_until_complete(_run())
    finally:
        api.loop.close()

    response_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_events) == 1
    sent_metadata = ((response_events[0].get("response") or {}).get("metadata") or {})
    assert sent_metadata.get("micro_ack") == "true"
    assert sent_metadata.get("input_event_key") == parent_input_event_key
    assert sent_metadata.get("input_event_key") != tool_input_event_key
    assert api._tool_followup_state(canonical_key=tool_canonical_key) == "dropped"


def test_terminal_substantive_reconcile_requires_terminal_text_evidence() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_terminal_text_required"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_terminal_text_required")
    response_id = "resp-terminal-text-required"

    api._reconcile_terminal_substantive_response(
        turn_id=turn_id,
        canonical_key=canonical_key,
        response_id=response_id,
        selected=True,
        selection_reason="normal",
    )

    state = api._conversation_efficiency_state(turn_id=turn_id)
    assert state.substantive_count == 0

    api._record_terminal_response_text(response_id=response_id, text="I can see a small object in your hand.")
    api._reconcile_terminal_substantive_response(
        turn_id=turn_id,
        canonical_key=canonical_key,
        response_id=response_id,
        selected=True,
        selection_reason="normal",
    )

    state = api._conversation_efficiency_state(turn_id=turn_id)
    assert state.substantive_count == 1
    assert state.substantive_count_by_canonical == {canonical_key: 1}


def test_tool_followup_release_after_response_done_drains_for_nonsubstantive_parent() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_2"
    parent_input_event_key = "item_parent_2"
    parent_response_id = "resp_parent_2"

    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_Au1PiLqWAAgpbHwW",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    followup_metadata = ((followup_event.get("response") or {}).get("metadata") or {})
    followup_metadata["blocked_by_response_id"] = parent_response_id
    followup_metadata["parent_turn_id"] = turn_id
    followup_metadata["parent_input_event_key"] = parent_input_event_key
    api._response_create_queue.append(
        {
            "websocket": ws,
            "event": followup_event,
            "origin": "tool_output",
            "turn_id": turn_id,
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=str(followup_metadata.get("input_event_key") or ""),
    )
    api._set_tool_followup_state(
        canonical_key=tool_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert api._tool_followup_state(canonical_key=tool_canonical_key) == "scheduled_release"
    assert followup_metadata.get("tool_followup_release") == "true"

    asyncio.run(api._drain_response_create_queue(source_trigger="response_done"))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_tool_followup_release_after_response_done_holds_when_exact_phrase_repair_is_open() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_exact_phrase_hold"
    parent_input_event_key = "item_parent_exact_phrase_hold"
    parent_response_id = "resp_parent_exact_phrase_hold"

    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._apply_terminal_deliverable_selection(
        canonical_key=parent_canonical_key,
        response_id=parent_response_id,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        selected=False,
        selection_reason="exact_phrase_obligation_open",
    )

    followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_exact_phrase_hold",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    followup_metadata = ((followup_event.get("response") or {}).get("metadata") or {})
    followup_metadata["blocked_by_response_id"] = parent_response_id
    followup_metadata["parent_turn_id"] = turn_id
    followup_metadata["parent_input_event_key"] = parent_input_event_key
    api._response_create_queue.append(
        {
            "websocket": ws,
            "event": followup_event,
            "origin": "tool_output",
            "turn_id": turn_id,
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=str(followup_metadata.get("input_event_key") or ""),
    )
    api._set_tool_followup_state(
        canonical_key=tool_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    decision, entry = api._decide_tool_followup_arbitration(
        response_metadata=followup_metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert entry is not None
    assert entry[0] == parent_canonical_key
    assert decision.should_hold is True
    assert decision.reason_code == "parent_deliverable_pending"

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)

    assert api._tool_followup_state(canonical_key=tool_canonical_key) == "blocked_active_response"
    assert followup_metadata.get("tool_followup_release") != "true"


def test_tool_followup_release_after_response_done_does_not_hold_for_unrelated_open_contract() -> None:
    api = _make_api_stub()
    _wire_runtime(api)

    turn_id = "turn_unrelated_contract"
    parent_input_event_key = "item_parent_unrelated_contract"
    parent_response_id = "resp_parent_unrelated_contract"

    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )
    api._turn_contracts_by_turn_id = {
        turn_id: {"exact_phrase": "Ready.", "exact_phrase_repair_scheduled": False}
    }

    followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_unrelated_contract",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    followup_metadata = ((followup_event.get("response") or {}).get("metadata") or {})
    followup_metadata["blocked_by_response_id"] = parent_response_id
    followup_metadata["parent_turn_id"] = turn_id
    followup_metadata["parent_input_event_key"] = parent_input_event_key

    decision, entry = api._decide_tool_followup_arbitration(
        response_metadata=followup_metadata,
        blocked_by_response_id=parent_response_id,
    )

    assert entry is not None
    assert entry[0] == parent_canonical_key
    assert decision.should_hold is False
    assert decision.should_release is True
    assert decision.reason_code == "parent_not_coverage_qualified"


def test_tool_followup_release_after_response_done_drains_after_listening_clears() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_2"
    parent_input_event_key = "item_parent_2"
    parent_response_id = "resp_parent_2"

    parent_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
    )
    api._canonical_response_state_mutate(
        canonical_key=parent_canonical_key,
        turn_id=turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "upgraded_response"),
            setattr(record, "response_id", parent_response_id),
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
            setattr(record, "done", True),
        ),
    )

    followup_event, _ = api._build_tool_followup_response_create_event(
        call_id="call_Au1PiLqWAAgpbHwW",
        response_create_event={"type": "response.create"},
        tool_name="gesture_look_around",
    )
    followup_metadata = ((followup_event.get("response") or {}).get("metadata") or {})
    followup_metadata["blocked_by_response_id"] = parent_response_id
    followup_metadata["parent_turn_id"] = turn_id
    followup_metadata["parent_input_event_key"] = parent_input_event_key
    api._response_create_queue.append(
        {
            "websocket": ws,
            "event": followup_event,
            "origin": "tool_output",
            "turn_id": turn_id,
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    tool_canonical_key = api._canonical_utterance_key(
        turn_id=turn_id,
        input_event_key=str(followup_metadata.get("input_event_key") or ""),
    )
    api._set_tool_followup_state(
        canonical_key=tool_canonical_key,
        state="blocked_active_response",
        reason="test_seed_blocked",
    )

    api._release_blocked_tool_followups_for_response_done(response_id=parent_response_id)
    assert api._tool_followup_state(canonical_key=tool_canonical_key) == "scheduled_release"

    api.state_manager.state = InteractionState.LISTENING
    asyncio.run(api._drain_response_create_queue(source_trigger="response_done"))
    assert ws.sent == []

    api.state_manager.state = InteractionState.IDLE
    asyncio.run(api._drain_response_create_queue(source_trigger="active_cleared"))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1


def test_stale_queued_upgraded_response_is_dropped_after_final_deliverable() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_interrupt_chain"
    input_event_key = "item_interrupt_followup"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=canonical_key,
        turn_id=turn_id,
        input_event_key=input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-substantive"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )
    api._response_done_serial = 2
    api._response_create_queue.append(
        {
            "websocket": ws,
            "event": {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": turn_id,
                        "input_event_key": input_event_key,
                        "origin": "upgraded_response",
                    }
                },
            },
            "origin": "upgraded_response",
            "turn_id": turn_id,
            "record_ai_call": False,
            "debug_context": None,
            "memory_brief_note": None,
            "enqueued_done_serial": 1,
            "enqueue_seq": 1,
        }
    )

    asyncio.run(api._drain_response_create_queue(source_trigger="playback_complete"))

    assert ws.sent == []
    assert api._pending_response_create is None
    assert len(api._response_create_queue) == 0


def test_stale_pending_upgraded_response_is_dropped_before_send() -> None:
    api = _make_api_stub()
    _wire_runtime(api)
    ws = _RecordingWs()
    api.websocket = ws

    turn_id = "turn_interrupt_chain_pending"
    input_event_key = "item_interrupt_followup_pending"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._canonical_response_state_mutate(
        canonical_key=canonical_key,
        turn_id=turn_id,
        input_event_key=input_event_key,
        mutator=lambda record: (
            setattr(record, "origin", "assistant_message"),
            setattr(record, "response_id", "resp-substantive-pending"),
            setattr(record, "deliverable_observed", True),
            setattr(record, "deliverable_class", "final"),
            setattr(record, "done", True),
        ),
    )
    api._response_done_serial = 3
    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event={
            "type": "response.create",
            "response": {
                "metadata": {
                    "turn_id": turn_id,
                    "input_event_key": input_event_key,
                    "origin": "upgraded_response",
                }
            },
        },
        origin="upgraded_response",
        turn_id=turn_id,
        created_at=0.0,
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
        queued_reminder_key=None,
        enqueued_done_serial=1,
        enqueue_seq=7,
    )

    asyncio.run(api._drain_response_create_queue(source_trigger="playback_complete"))

    assert ws.sent == []
    assert api._pending_response_create is None
