from __future__ import annotations

import asyncio
import json
import sys
import types
from collections import deque

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.governance import ActionPacket
from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime.types import PendingResponseCreate
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
            "start_receiving": lambda self: None,
            "start_recording": lambda self: setattr(self, "is_recording", True),
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
    return api


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
    api._active_input_event_key_by_turn_id["turn_tool_release"] = "item_active_release"
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

        await api.handle_response_done({"type": "response.done", "response_id": "resp-active-release"})

    asyncio.run(_run())

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    assert api._tool_followup_state(canonical_key=canonical_key) in {"creating", "created", "released_on_response_done"}
    assert any(
        "tool_followup_state" in entry
        and f"canonical_key={canonical_key}" in entry
        and "state=scheduled_release" in entry
        and "reason=response_done response_id=resp-active-release" in entry
        for entry in captured_logs
    )
    assert not any(
        "tool_followup_create_suppressed" in entry
        and f"canonical_key={canonical_key}" in entry
        and "reason=final_deliverable_already_sent" in entry
        for entry in captured_logs
    )


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
    assert reason == "parent_not_deliverable"

def test_tool_followup_create_seam_uses_terminal_selection_when_parent_canonical_coverage_lags() -> None:
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
    # Simulate stale canonical coverage fields lagging behind terminal selection state.
    api._canonical_response_state_mutate(
        canonical_key=parent_key,
        turn_id=parent_turn_id,
        input_event_key=parent_input_event_key,
        mutator=lambda record: (
            setattr(record, "deliverable_observed", False),
            setattr(record, "deliverable_class", "unknown"),
        ),
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

    assert [event for event in ws.sent if event.get("type") == "response.create"] == []
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
    assert "Gesture follow-up only" in instructions
    assert "Do not restate or re-answer semantic memory/preferences content" in instructions
    assert "Do not narrate environment/vision context" in instructions


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
        "active_key_transition" in entry
        and "cause=prepare_response_create" in entry
        and "origin=micro_ack" in entry
        for entry in captured_logs
    )
    assert any(
        "derived_response_lineage_eval origin=micro_ack" in entry
        and f"canonical_key={canonical_key}" in entry
        and "allowed=false" in entry
        for entry in captured_logs
    )
    assert any(
        "suppressed_tool_lineage_block origin=micro_ack" in entry
        and f"canonical_key={canonical_key}" in entry
        for entry in captured_logs
    )


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

    oversized_call_id = "mixed_intent_a535dd5098194e50ac1f3de90318b5cc"
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
