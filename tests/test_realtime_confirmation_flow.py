"""Tests for confirmation gating in realtime tool orchestration."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import time

from ai.governance import ActionPacket
from ai.orchestration import OrchestrationPhase
from ai.realtime_api import PendingAction, RealtimeAPI


class _FakeOrchestrationState:
    def __init__(self) -> None:
        self.phase = OrchestrationPhase.IDLE
        self.transitions: list[tuple[OrchestrationPhase, str | None]] = []

    def transition(self, phase: OrchestrationPhase, reason: str | None = None) -> None:
        self.phase = phase
        self.transitions.append((phase, reason))


class _FakeWebsocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


class _FailGovernance:
    def review(self, _action):  # pragma: no cover - should never run in this test
        raise AssertionError("governance.review should not run while awaiting confirmation")


class _FailBuildGovernance:
    def build_action_packet(self, *args, **kwargs):  # pragma: no cover - should never run in this test
        raise AssertionError("governance.build_action_packet should not run for duplicate call")



def _action_packet() -> ActionPacket:
    return ActionPacket(
        id="call-1",
        tool_name="perform_research",
        tool_args={"query": "datasheet vin"},
        tier=2,
        what="Run research",
        why="asked by user",
        impact="read only",
        rollback="none",
        alternatives=[],
        confidence=0.3,
        cost="med",
        risk_flags=[],
        requires_confirmation=True,
        expiry_ts=time.monotonic() + 60,
    )


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stop_words = []
    api._stop_word_cooldown_s = 0.0
    api._tool_execution_disabled_until = 0.0
    api._presented_actions = set()
    api.orchestration_state = _FakeOrchestrationState()
    api._response_in_flight = False
    api._response_create_queue = deque()
    api._tool_call_dedupe_ttl_s = 30.0
    api._last_executed_tool_call = None
    api._awaiting_confirmation_completion = False
    api._tool_call_records = []
    api._last_tool_call_results = []
    api.function_call = None
    api.function_call_args = ""
    api._spoken_research_response_ids = {}
    api._research_spoken_response_dedupe_ttl_s = 60.0
    api._response_create_debug_trace = False
    api._last_response_create_ts = None
    api._active_response_id = None
    api._track_outgoing_event = lambda *args, **kwargs: None
    return api


def test_confirmation_parser_handles_yes_no_unclear() -> None:
    api = _make_api_stub()

    assert api._parse_confirmation_decision("yes") == "yes"
    assert api._parse_confirmation_decision("Yes!") == "yes"
    assert api._parse_confirmation_decision("okay") == "yes"
    assert api._parse_confirmation_decision("no.") == "no"
    assert api._parse_confirmation_decision("nope") == "no"
    assert api._parse_confirmation_decision("maybe later") == "unclear"


def test_yes_executes_pending_action_once_and_clears_state() -> None:
    api = _make_api_stub()
    action = _action_packet()
    api._pending_action = PendingAction(
        action=action,
        staging={"valid": True},
        original_intent="please research waveshare vin",
        created_at=time.monotonic(),
    )

    executed: list[str] = []

    async def _execute_action(action, staging, websocket, **kwargs):
        executed.append(action.id)

    async def _reject_tool_call(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("reject path should not run")

    async def _send_assistant_message(message: str, websocket) -> None:
        return None

    api._execute_action = _execute_action
    api._reject_tool_call = _reject_tool_call
    api.send_assistant_message = _send_assistant_message

    handled = asyncio.run(api._maybe_handle_approval_response("yes!", _FakeWebsocket()))

    assert handled is True
    assert executed == ["call-1"]
    assert api._pending_action is None


def test_handle_function_call_suppresses_duplicate_while_awaiting_confirmation() -> None:
    api = _make_api_stub()
    action = _action_packet()
    api._pending_action = PendingAction(
        action=action,
        staging={"valid": True},
        original_intent="research request",
        created_at=time.monotonic(),
    )
    api._governance = _FailGovernance()
    api.function_call = {"name": "perform_research", "call_id": "call-duplicate"}
    api.function_call_args = json.dumps({"query": "datasheet vin"})

    suppressed: list[str] = []

    async def _send_awaiting_confirmation_output(call_id: str, websocket) -> None:
        suppressed.append(call_id)

    api._send_awaiting_confirmation_output = _send_awaiting_confirmation_output

    asyncio.run(api.handle_function_call({}, _FakeWebsocket()))

    assert suppressed == ["call-duplicate"]
    assert api.function_call is None
    assert api.function_call_args == ""


def test_post_confirmation_duplicate_tool_call_is_deduped_without_reentering_confirmation() -> None:
    api = _make_api_stub()
    action = _action_packet()
    api._pending_action = PendingAction(
        action=action,
        staging={"valid": True},
        original_intent="research request",
        created_at=time.monotonic(),
    )
    api._governance = _FailBuildGovernance()

    async def _send_assistant_message(message: str, websocket) -> None:
        return None

    async def _execute_action(action, staging, websocket, **kwargs):
        api._record_executed_tool_call(action.tool_name, action.tool_args)

    api.send_assistant_message = _send_assistant_message
    api._execute_action = _execute_action

    websocket = _FakeWebsocket()
    handled = asyncio.run(api._maybe_handle_approval_response("yes", websocket))
    assert handled is True
    assert api._pending_action is None

    api.orchestration_state.phase = OrchestrationPhase.IDLE
    api.function_call = {"name": "perform_research", "call_id": "call-2"}
    api.function_call_args = json.dumps({"query": "datasheet vin"})

    asyncio.run(api.handle_function_call({}, websocket))

    assert api.orchestration_state.phase != OrchestrationPhase.AWAITING_CONFIRMATION
    assert api._pending_action is None

    function_outputs = [
        msg
        for msg in websocket.messages
        if msg.get("type") == "conversation.item.create"
        and msg.get("item", {}).get("type") == "function_call_output"
    ]
    assert any(
        json.loads(msg["item"]["output"]).get("status") == "redundant"
        for msg in function_outputs
    )

    response_creates = [msg for msg in websocket.messages if msg.get("type") == "response.create"]
    assert not response_creates


def test_execute_function_call_suppresses_duplicate_research_spoken_response() -> None:
    api = _make_api_stub()
    websocket = _FakeWebsocket()

    call_count = 0

    async def _fake_research(**kwargs):
        return {"research_id": "research-123", "answer_summary": "done"}

    async def _fake_send_response_create(websocket, event, *, origin, record_ai_call=False, debug_context=None):
        nonlocal call_count
        call_count += 1
        websocket.messages.append(event)
        return True

    from ai import realtime_api as realtime_module

    original = realtime_module.function_map.get("perform_research")
    realtime_module.function_map["perform_research"] = _fake_research
    api._send_response_create = _fake_send_response_create
    try:
        asyncio.run(api.execute_function_call("perform_research", "call-1", {"query": "q"}, websocket))
        asyncio.run(api.execute_function_call("perform_research", "call-2", {"query": "q"}, websocket))
    finally:
        if original is not None:
            realtime_module.function_map["perform_research"] = original

    assert call_count == 1
