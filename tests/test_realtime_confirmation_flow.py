"""Tests for confirmation gating in realtime tool orchestration."""

from __future__ import annotations

import asyncio
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

    async def _execute_action(action, staging, websocket):
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
