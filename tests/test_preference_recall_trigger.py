"""Tests deterministic preference-recall tool invocation before replying."""

from __future__ import annotations

import asyncio

from ai.realtime_api import RealtimeAPI


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
            ]
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
        return {"memories": [{"content": "User's favorite editor is Vim."}]}

    async def _fake_send(_message: str, _ws, **_kwargs) -> None:
        return None

    monkeypatch.setitem(__import__("ai.tools", fromlist=["function_map"]).function_map, "recall_memories", _fake_recall)
    monkeypatch.setattr(api, "send_assistant_message", _fake_send)

    asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))
    asyncio.run(api._maybe_handle_preference_recall_intent("Which editor do I prefer?", _Ws(), source="text_message"))

    assert recall_calls == 1
