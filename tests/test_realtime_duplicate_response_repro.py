from __future__ import annotations

import asyncio
import json
from collections import deque

from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime_api import InteractionState, RealtimeAPI
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
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = RealtimeAPI._track_outgoing_event.__get__(api, RealtimeAPI)
    return api


def test_duplicate_assistant_message_create_single_flight_guard(monkeypatch) -> None:
    """Deterministic run-405 repro now guarded to single assistant_message response.create."""

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

    asyncio.run(api.execute_function_call("perform_research", "call_research_2", {"query": "logs"}, ws))
    api._response_in_flight = False
    asyncio.run(api.execute_function_call("perform_research", "call_research_2", {"query": "logs"}, ws))

    response_create_events = [event for event in ws.sent if event.get("type") == "response.create"]
    assert len(response_create_events) == 1
    canonical_key = api._canonical_utterance_key(turn_id="turn_tool_2", input_event_key="tool:call_research_2")
    assert any(
        "tool_followup_arbitration outcome=deny reason=already_created" in entry
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
