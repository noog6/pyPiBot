"""Tests deterministic preference-recall tool invocation before replying."""

from __future__ import annotations

import asyncio
from ai.realtime_api import RealtimeAPI
from core.logging import logger


class _Ws:
    async def send(self, _payload: str) -> None:
        return None


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._preference_recall_cooldown_s = 10.0
    api._preference_recall_cache = {}
    api._memory_retrieval_scope = "user_global"
    return api


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

    async def _fake_recall(**_kwargs):
        assert api._pending_preference_recall_trace["decision"] == "invoked_tool"
        assert api._pending_preference_recall_trace["reason"] == "preference_intent_matched"
        return {
            "memories": [{"content": "User's favorite editor is Vim."}],
            "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
        }

    async def _fake_send(_message: str, _ws, **_kwargs) -> None:
        return None

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    handled = asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))

    assert handled is True
    assert api._tool_call_records[-1]["name"] == "recall_memories"
    assert api._tool_call_records[-1]["source"] == "preference_recall"
    assert "editor" in api._tool_call_records[-1]["query"]
