"""Tests deterministic preference-recall tool invocation before replying."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import ai.tools as ai_tools
import ai.realtime_api as realtime_api
from ai.realtime_api import InteractionState, RealtimeAPI
from core.logging import logger
from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


class _Ws:
    async def send(self, _payload: str) -> None:
        return None


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._preference_recall_cooldown_s = 10.0
    api._preference_recall_cache = {}
    api._memory_retrieval_scope = "user_global"
    api._pending_response_create_origins = deque()
    api._preference_recall_response_suppression_until = 0.0
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._pending_server_auto_input_event_keys = deque(maxlen=64)
    api._active_server_auto_input_event_key = None
    api._current_input_event_key = None
    api._input_event_key_counter = 0
    api._active_response_preference_guarded = False
    api._active_response_confirmation_guarded = False
    api._active_response_id = None
    api._active_response_origin = "unknown"
    api._audio_accum = bytearray()
    api._audio_accum_bytes_target = 9600
    api._tool_call_records = []
    api._last_tool_call_results = []
    api._assistant_reply_accum = ""
    api.assistant_reply = ""
    api._reflection_enqueued = False
    api._response_in_flight = False
    api.response_in_progress = False
    api._speaking_started = False
    api._mic_receive_on_first_audio = False
    api._last_response_metadata = {}
    api.audio_player = None
    api._active_utterance = None
    api.orchestration_state = type("_Orch", (), {"transition": lambda self, *_args, **_kwargs: None})()
    api._current_run_id = lambda: "run-test"
    api.state_manager = type(
        "_State",
        (),
        {
            "state": InteractionState.IDLE,
            "update_state": lambda self, *_args, **_kwargs: None,
        },
    )()
    return api




def _make_memory_manager(store: MemoryStore) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = "default"
    manager._active_session_id = None
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._store = store
    manager._embedding_worker = None
    manager._semantic_config = SimpleNamespace(enabled=False, rerank_enabled=False)
    manager._last_turn_retrieval_at = {}
    manager._auto_pin_min_importance = 5
    manager._auto_pin_requires_review = True
    manager._auto_reflection_semantic_dedupe_enabled = False
    manager._auto_reflection_dedupe_recent_limit = 24
    manager._auto_reflection_dedupe_high_risk_cosine = 0.9
    manager._auto_reflection_dedupe_policy = "skip_write"
    manager._auto_reflection_dedupe_importance = 2
    manager._auto_reflection_dedupe_clear_pin = True
    manager._auto_reflection_dedupe_needs_review = True
    manager._auto_reflection_dedupe_apply_to_manual_tool = False
    manager._recall_trace_enabled = True
    manager._recall_trace_level = "info"
    return manager

def test_preference_question_calls_recall_before_response(monkeypatch) -> None:
    api = _make_api_stub()
    call_order: list[str] = []
    sent_messages: list[str] = []

    async def _fake_recall(**_kwargs):
        call_order.append("recall")
        return {
            "memories": [
                {
                    "content": "User's favorite editor is Vim.",
                }
            ],
            "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"\nWhy it's relevant:\n- \"Matches editor preference.\"\nConfidence: High",
        }

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        call_order.append("respond")
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert call_order == ["recall", "respond"]
    assert "Vim" in sent_messages[0]
    assert "Relevant memory:" in sent_messages[0]
    assert "Confidence: High" in sent_messages[0]


def test_preference_recall_response_sanitizes_memory_id_phrasing(monkeypatch) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []

    async def _fake_recall(**_kwargs):
        return {
            "memories": [{"content": "User's favorite editor is Vim."}],
            "memory_cards_text": (
                "Relevant memory:\n"
                "- \"Prefers jasmine tea\"\n"
                "Why it's relevant:\n"
                "- \"memory #14 matched memory ID: 14\"\n"
                "Confidence: Medium"
            ),
            "memory_cards": [{"confidence": "Medium"}],
        }

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert sent_messages
    assert "Relevant memory:" in sent_messages[0]
    assert "memory #" not in sent_messages[0].lower()
    assert "memory id" not in sent_messages[0].lower()
    assert "id:" not in sent_messages[0].lower()


def test_preference_recall_truthfulness_guard_for_checking_phrase(monkeypatch) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []
    recall_calls = 0

    async def _fake_recall(**_kwargs):
        nonlocal recall_calls
        recall_calls += 1
        return {
            "memories": [{"content": "User's favorite editor is Vim.", "tags": ["preference"]}],
            "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"\nWhy it's relevant:\n- \"Matches editor preference.\"\nConfidence: High",
            "memory_cards": [{"confidence": "High"}],
        }

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))
    asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))

    assert recall_calls == 1
    assert sent_messages[0].startswith("I’m checking what I remember.")
    assert "I’m checking what I remember." not in sent_messages[1]


def test_preference_question_empty_recall_returns_saved_yet_message(monkeypatch) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []

    async def _fake_recall(**_kwargs):
        return {"memories": []}

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert "don’t have that saved yet" in sent_messages[0]


def test_preference_question_uses_editor_fallback_query(monkeypatch) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []
    queries: list[str] = []

    async def _fake_recall(**kwargs):
        queries.append(kwargs.get("query", ""))
        if kwargs.get("query") == "editor":
            return {
                "memories": [{"content": "User's favorite editor is Vim."}],
                "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
            }
        return {"memories": []}

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert queries[0] != "editor"
    assert "editor" in queries
    assert "Vim" in sent_messages[0]


def test_preference_recall_query_builder_includes_editor_variants() -> None:
    api = _make_api_stub()

    matched, keywords = api._is_preference_recall_intent("Which editor do I prefer?")
    assert matched is True

    query = api._build_preference_recall_query("which editor do i prefer?", keywords=keywords)

    assert "editor" in query
    assert "user preferred editor" in query




def test_preference_question_accepts_items_payload(monkeypatch) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []

    async def _fake_recall(**_kwargs):
        return {
            "items": [{"content": "User prefers Vim for editing."}],
            "memory_cards_text": "Relevant memory:\n- \"User prefers Vim for editing.\"",
        }

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert sent_messages
    assert "Vim" in sent_messages[0]
    assert "don’t have that saved yet" not in sent_messages[0]

def test_preference_recall_uses_cache_within_cooldown(monkeypatch) -> None:
    api = _make_api_stub()
    recall_calls = 0

    async def _fake_recall(**_kwargs):
        nonlocal recall_calls
        recall_calls += 1
        return {
            "memories": [{"content": "User's favorite editor is Vim."}],
            "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"\nWhy it's relevant:\n- \"Matches editor preference.\"\nConfidence: High",
        }

    async def _fake_send(_message: str, _ws, **_kwargs) -> None:
        return None

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))
    asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))

    assert recall_calls == 1


def test_preference_question_without_recall_tool_emits_skip_trace(monkeypatch) -> None:
    api = _make_api_stub()
    api._tool_call_records = []
    api._preference_recall_skip_logged_turn_ids = set()
    api._pending_preference_recall_trace = None
    api._current_run_id = lambda: "run-123"
    logged: list[str] = []

    def _fake_info(message: str, *args) -> None:
        logged.append(message % args if args else message)

    monkeypatch.setattr(logger, "info", _fake_info)
    api._mark_preference_recall_candidate("Which editor do I prefer?", source="text_message")
    api._emit_preference_recall_skip_trace_if_needed(turn_id="turn-9")

    assert logged
    assert "preference_recall_decision_trace" in logged[0]
    assert "intent=preference_recall" in logged[0]
    assert "decision=skipped_tool" in logged[0]
    assert "reason=model_did_not_request_tool" in logged[0]
    assert "run_id=run-123" in logged[0]
    assert "turn_id=turn-9" in logged[0]


def test_preference_question_with_recall_tool_does_not_emit_skip_trace(monkeypatch) -> None:
    api = _make_api_stub()
    api._tool_call_records = [{"name": "recall_memories", "call_id": "c1", "args": {}, "result": {}}]
    api._preference_recall_skip_logged_turn_ids = set()
    api._pending_preference_recall_trace = None
    api._current_run_id = lambda: "run-123"
    logged: list[str] = []

    def _fake_info(message: str, *args) -> None:
        logged.append(message % args if args else message)

    monkeypatch.setattr(logger, "info", _fake_info)
    api._mark_preference_recall_candidate("Which editor do I prefer?", source="text_message")
    api._emit_preference_recall_skip_trace_if_needed(turn_id="turn-10")

    assert not logged


def test_preference_recall_records_tool_and_updates_trace(monkeypatch) -> None:
    api = _make_api_stub()
    api._tool_call_records = []
    api._preference_recall_skip_logged_turn_ids = set()
    api._current_run_id = lambda: "run-456"
    api._current_response_turn_id = "turn-42"
    logged: list[str] = []

    async def _fake_recall(**_kwargs):
        assert api._pending_preference_recall_trace["decision"] == "invoked_tool"
        assert api._pending_preference_recall_trace["reason"] == "preference_intent_matched"
        return {
            "memories": [{"content": "User's favorite editor is Vim."}],
            "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
        }

    async def _fake_send(_message: str, _ws, **_kwargs) -> None:
        return None

    def _fake_info(message: str, *args) -> None:
        logged.append(message % args if args else message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)
    monkeypatch.setattr(logger, "info", _fake_info)

    handled = asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))
    api._emit_preference_recall_skip_trace_if_needed(turn_id="turn-42")

    assert handled is True
    assert api._tool_call_records[-1]["name"] == "recall_memories"
    assert api._tool_call_records[-1]["source"] == "preference_recall"
    assert api._tool_call_records[-1]["turn_id"] == "turn-42"
    assert "editor" in api._tool_call_records[-1]["query"]
    assert not any("preference_recall_decision_trace" in entry for entry in logged)


class _RecordingWs:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_preference_recall_short_circuits_server_auto_response(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    sent_messages: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _handled(*_args, **_kwargs) -> bool:
        sent_messages.append("handled")
        api._preference_recall_response_suppression_until = 9999999999.0
        api._preference_recall_suppressed_turns.add(api._current_turn_id_or_unknown())
        return True

    api.websocket = ws
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_handle_preference_recall_intent = _handled
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = lambda *_args, **_kwargs: None

    asyncio.run(
        api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Which editor do I prefer?",
            },
            ws,
        )
    )

    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "r-1"}}, ws))
    asyncio.run(api.handle_event({"type": "response.text.delta", "delta": "fallback"}, ws))

    assert sent_messages == ["handled"]
    assert api._active_response_preference_guarded is True
    assert api.assistant_reply == ""
    assert ws.sent == ['{"type": "response.cancel"}']




class _FailingCancelWs:
    def __init__(self, error_message: str) -> None:
        self.error_message = error_message
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)
        if payload == '{"type": "response.cancel"}':
            raise RuntimeError(self.error_message)


def test_preference_recall_cancel_no_active_response_is_noop(monkeypatch) -> None:
    api = _make_api_stub()
    api._active_response_origin = "server_auto"
    api._response_in_flight = True
    sent_messages: list[str] = []

    async def _fake_recall(**_kwargs):
        return {"memories": [{"content": "User prefers Vim."}]}

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    ws = _FailingCancelWs("Cancellation failed: no active response found")
    info_logs: list[str] = []
    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            ws,
            source="text_message",
        )
    )

    assert handled is True
    assert sent_messages
    assert "Vim" in sent_messages[0]
    assert any("response_cancel_noop" in entry and "reason=no_active_response" in entry for entry in info_logs)


def test_handle_error_treats_no_active_response_cancel_as_noop(monkeypatch) -> None:
    api = _make_api_stub()
    info_logs: list[str] = []
    error_logs: list[str] = []
    emitted_error_events: list[str] = []

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))
    monkeypatch.setattr(logger, "error", lambda msg, *args, **kwargs: error_logs.append(msg % args if args else msg))
    monkeypatch.setattr(realtime_api, "log_error", lambda message: emitted_error_events.append(message))

    asyncio.run(
        api.handle_error(
            {"error": {"message": "Cancellation failed: no active response found"}},
            _Ws(),
        )
    )

    assert any("response_cancel_noop" in entry and "reason=no_active_response" in entry for entry in info_logs)
    assert not any("Unhandled error" in entry for entry in error_logs)
    assert not emitted_error_events

def test_preference_recall_transcript_path_emits_memory_answer_without_model_followup(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    response_create_calls: list[dict[str, str]] = []

    async def _fake_recall(**_kwargs):
        return {
            "memories": [{"content": "User's favorite editor is Vim."}],
            "memory_cards_text": (
                "Relevant memory:\n"
                '- "User\'s favorite editor is Vim."\n'
                "Why it's relevant:\n"
                '- "Matches editor preference."\n'
                "Confidence: High"
            ),
        }

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _capture_response_create(_ws, event, *, origin="unknown", **_kwargs) -> None:
        response_create_calls.append({"type": event.get("type", ""), "origin": origin})

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "_send_response_create", _capture_response_create)

    api.websocket = ws
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = lambda *_args, **_kwargs: None

    asyncio.run(
        api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Which editor do I prefer?",
            },
            ws,
        )
    )

    payloads = [json.loads(item) for item in ws.sent]
    assistant_payloads = [
        payload for payload in payloads if payload.get("type") == "conversation.item.create"
    ]

    assert len(assistant_payloads) == 1
    message = assistant_payloads[0]["item"]["content"][0]["text"]
    assert "Vim" in message
    assert "Relevant memory:" in message
    assert response_create_calls == []


def test_preference_recall_drops_queued_server_auto_response_create(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()

    api.websocket = ws
    api._current_response_turn_id = "turn_42"
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._pending_response_create = None
    api._response_done_serial = 0
    api._response_create_debug_trace = False
    api._response_schedule_logged_turn_ids = set()
    api._turn_diagnostic_timestamps = {}
    api._audio_playback_busy = False
    api._last_response_create_ts = None

    async def _fake_recall(**_kwargs):
        return {
            "memories": [{"content": "User's favorite editor is Vim."}],
            "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
        }

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)

    api._response_in_flight = True
    scheduled = asyncio.run(api._send_response_create(ws, {"type": "response.create"}, origin="server_auto"))

    assert scheduled is False
    assert api._pending_response_create is not None

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            ws,
            source="input_audio_transcription",
        )
    )

    assert handled is True
    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []

    api._response_in_flight = False
    asyncio.run(api._drain_response_create_queue())

    payloads = [json.loads(item) for item in ws.sent]
    assistant_payloads = [
        payload
        for payload in payloads
        if payload.get("type") == "conversation.item.create"
        and payload.get("item", {}).get("role") == "assistant"
    ]
    response_create_payloads = [payload for payload in payloads if payload.get("type") == "response.create"]

    assert len(assistant_payloads) == 1
    assert "Vim" in assistant_payloads[0]["item"]["content"][0]["text"]
    assert response_create_payloads == []


def test_preference_question_falls_back_to_preference_tag_query(monkeypatch) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []
    queries: list[str] = []

    async def _fake_recall(**kwargs):
        queries.append(kwargs.get("query", ""))
        if kwargs.get("query") == "preference":
            return {
                "memories": [{"content": "User's favorite editor is Vim.", "tags": ["preference"]}],
                "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
            }
        return {"memories": []}

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert queries[:2]
    assert queries[1] == "editor"
    assert "preference" in queries
    assert "Vim" in sent_messages[0]


def test_preference_recall_regression_returns_seeded_vim_memory(monkeypatch, tmp_path) -> None:
    api = _make_api_stub()
    sent_messages: list[str] = []

    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager.remember_memory(
        content="User's favorite editor is Vim.",
        tags=["preference", "editor"],
        importance=4,
        scope=MemoryScope.USER_GLOBAL,
    )

    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(
        api._maybe_handle_preference_recall_intent(
            "Which editor do I prefer?",
            _Ws(),
            source="text_message",
        )
    )

    assert handled is True
    assert sent_messages
    assert "Vim" in sent_messages[0]

def test_preference_recall_cancels_only_matching_server_auto_response(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    sent_messages: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _handled(*_args, **_kwargs) -> bool:
        sent_messages.append("handled")
        await api._suppress_preference_recall_server_auto_response(ws)
        return True

    api.websocket = ws
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_handle_preference_recall_intent = _handled
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = lambda *_args, **_kwargs: None

    # Simulate run-390 ordering where server_auto response.created arrives before recall handling.
    api._current_response_turn_id = "turn_1"
    api._pending_server_auto_input_event_keys.append("item-1")
    api._current_input_event_key = "item-1"
    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "r-1"}}, ws))

    asyncio.run(
        api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "transcript": "Which editor do I prefer?",
            },
            ws,
        )
    )

    # Next utterance keeps same turn_id but must not be suppressed by stale turn state.
    api._current_response_turn_id = "turn_1"
    api._response_in_flight = True
    api._pending_server_auto_input_event_keys.append("item-2")
    api._current_input_event_key = "item-2"
    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "r-2"}}, ws))

    assert sent_messages == ["handled"]
    assert ws.sent == ['{"type": "response.cancel"}']
    assert api._active_server_auto_input_event_key == "item-2"
    assert "item-1" in api._preference_recall_suppressed_input_event_keys
    assert "item-2" not in api._preference_recall_suppressed_input_event_keys


def test_transcript_watchdog_logs_when_response_not_scheduled(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    info_logs: list[str] = []

    async def _false(*_args, **_kwargs) -> bool:
        return False

    api._transcript_response_watchdog_timeout_s = 0.5
    api._response_in_flight = True
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp-stuck"
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._maybe_handle_preference_recall_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = lambda *_args, **_kwargs: None

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    async def _run() -> None:
        await api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-stalled",
                "transcript": "Theo, do you have any memories related to Vim?",
            },
            ws,
        )
        await asyncio.sleep(0.65)

    asyncio.run(_run())

    assert any(
        "response_not_scheduled" in entry
        and "reason=active_response_in_flight" in entry
        and "input_event_key=item-stalled" in entry
        for entry in info_logs
    )


def test_memory_intent_server_auto_race_upgrades_after_transcript(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _RecordingWs()
    sent_messages: list[str] = []
    queries: list[str] = []

    async def _fake_recall(**kwargs):
        queries.append(str(kwargs.get("query") or ""))
        return {
            "memories": [{"content": "Your eyes are blue."}],
            "memory_cards_text": "Relevant memory:\n- \"Your eyes are blue.\"",
        }

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _fake_send(message: str, _ws, **_kwargs) -> None:
        sent_messages.append(message)

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    api.websocket = ws
    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._track_outgoing_event = lambda *_args, **_kwargs: None

    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "r-1"}}, ws))

    asyncio.run(
        api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-remember",
                "transcript": "Do you remember I have blue eyes?",
            },
            ws,
        )
    )

    assert queries
    assert any("blue" in query for query in queries)
    assert sent_messages
    assert "blue" in sent_messages[0].lower()


def test_non_memory_smalltalk_does_not_trigger_recall(monkeypatch) -> None:
    api = _make_api_stub()
    recall_calls = 0

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _fake_recall(**_kwargs):
        nonlocal recall_calls
        recall_calls += 1
        return {"memories": []}

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
    api._is_user_approved_interrupt_response = lambda _response: False
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._record_user_input = lambda *_args, **_kwargs: None

    asyncio.run(
        api.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-smalltalk",
                "transcript": "Nice weather today.",
            },
            _Ws(),
        )
    )

    assert recall_calls == 0


def test_preference_recall_suppression_clears_all_pending_contenders() -> None:
    api = _make_api_stub()
    api._sync_pending_response_create_queue = lambda: None
    api._response_create_queue = deque(
        [
            {
                "turn_id": "turn_1",
                "origin": "assistant_message",
                "event": {"type": "response.create", "response": {"metadata": {"input_event_key": "item-1"}}},
            },
            {
                "turn_id": "turn_1",
                "origin": "tool_output",
                "event": {"type": "response.create", "response": {"metadata": {"input_event_key": "item-1"}}},
            },
        ]
    )
    api._pending_response_create = None

    api._clear_pending_response_contenders(
        turn_id="turn_1",
        input_event_key="item-1",
        reason="test",
    )

    assert list(api._response_create_queue) == []
