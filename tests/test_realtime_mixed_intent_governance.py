from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.realtime_api import PendingAction, RealtimeAPI


def _decision(**overrides):
    payload = {
        "approved": False,
        "needs_confirmation": False,
        "confirm_reason": None,
        "confirm_prompt": None,
        "reason": "within_bounds",
        "status": "denied",
        "idempotency_key": "idem-1",
        "max_reminders": None,
        "reminder_schedule_seconds": (),
        "action_summary": "summary",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _setup_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._governance = SimpleNamespace(
        build_action_packet=Mock(
            return_value=SimpleNamespace(
                tool_name="gesture_look_around",
                tool_args={},
                requires_confirmation=False,
                expiry_ts=None,
            )
        ),
        decide_tool_call=Mock(return_value=SimpleNamespace(status="approved", reason="within bounds")),
    )
    api._stage_action = Mock(return_value={"valid": True})
    api._build_tool_runtime_context = Mock(return_value={})
    api._tool_execution_cooldown_remaining = Mock(return_value=0.0)
    api._normalize_confirmation_decision = Mock(return_value=_decision(approved=True, status="approved"))
    api._evaluate_intent_guard = Mock(return_value=(False, None, None))
    api._execute_action = AsyncMock()
    api._record_intent_state = Mock()
    api._create_confirmation_token = Mock(return_value=SimpleNamespace(prompt_sent=False))
    api._sync_confirmation_legacy_fields = Mock()
    api._request_tool_confirmation = AsyncMock()
    api._last_user_input_text = "look around and remind me"
    api._approval_timeout_s = 20.0
    api._pending_action = None
    return api


def test_mixed_intent_tool_request_allow_executes() -> None:
    api = _setup_api()

    result = asyncio.run(
        RealtimeAPI._submit_mixed_intent_tool_request(
            api,
            tool_name="gesture_look_around",
            tool_args={},
            websocket=object(),
            source="preference_recall_mixed_intent",
            turn_id="turn_1",
            query="look around",
        )
    )

    assert result["outcome"] == "allow"
    assert result["executed"] is True
    api._execute_action.assert_awaited_once()
    execute_kwargs = api._execute_action.await_args.kwargs
    assert execute_kwargs["force_no_tools_followup"] is True
    assert execute_kwargs["inject_no_tools_instruction"] is False
    assert "turn=turn_1" in api._governance.build_action_packet.call_args.kwargs["reason"]


def test_mixed_intent_tool_request_defer_does_not_execute() -> None:
    api = _setup_api()
    api._normalize_confirmation_decision = Mock(return_value=_decision(status="deferred", reason="autonomy window closed"))

    result = asyncio.run(
        RealtimeAPI._submit_mixed_intent_tool_request(
            api,
            tool_name="gesture_look_around",
            tool_args={},
            websocket=object(),
            source="preference_recall_mixed_intent",
            turn_id="turn_1",
            query="look around",
        )
    )

    assert result["outcome"] == "defer"
    assert result["executed"] is False
    api._execute_action.assert_not_awaited()
    assert api._record_intent_state.call_args.args[2] == "deferred"


def test_mixed_intent_tool_request_suppress_does_not_execute() -> None:
    api = _setup_api()
    api._normalize_confirmation_decision = Mock(return_value=_decision(status="denied", reason="risk threshold exceeded"))

    result = asyncio.run(
        RealtimeAPI._submit_mixed_intent_tool_request(
            api,
            tool_name="gesture_look_around",
            tool_args={},
            websocket=object(),
            source="preference_recall_mixed_intent",
            turn_id="turn_1",
            query="look around",
        )
    )

    assert result["outcome"] == "suppress"
    assert result["executed"] is False
    api._execute_action.assert_not_awaited()
    assert api._record_intent_state.call_args.args[2] == "denied"


def test_mixed_intent_tool_request_confirmation_uses_governed_confirmation_path() -> None:
    api = _setup_api()
    api._normalize_confirmation_decision = Mock(
        return_value=_decision(
            status="needs_confirmation",
            reason="autonomy_level_requires_confirmation",
            confirm_reason="autonomy_level_requires_confirmation",
            needs_confirmation=True,
        )
    )

    result = asyncio.run(
        RealtimeAPI._submit_mixed_intent_tool_request(
            api,
            tool_name="gesture_look_around",
            tool_args={},
            websocket=object(),
            source="preference_recall_mixed_intent",
            turn_id="turn_1",
            query="look around",
        )
    )

    assert result["outcome"] == "confirm"
    assert result["executed"] is False
    api._request_tool_confirmation.assert_awaited_once()
    assert isinstance(api._pending_action, PendingAction)
    token_metadata = api._create_confirmation_token.call_args.kwargs["metadata"]
    assert token_metadata["mixed_intent"] is True
    assert token_metadata["turn_id"] == "turn_1"
    api._execute_action.assert_not_awaited()


def test_mixed_intent_tool_request_blocked_execution_defer_records_deferred_state() -> None:
    api = _setup_api()
    api._evaluate_intent_guard = Mock(return_value=(True, "deferred", "deferred"))

    result = asyncio.run(
        RealtimeAPI._submit_mixed_intent_tool_request(
            api,
            tool_name="gesture_look_around",
            tool_args={},
            websocket=object(),
            source="preference_recall_mixed_intent",
            turn_id="turn_1",
            query="look around",
        )
    )

    assert result["outcome"] == "defer"
    assert result["executed"] is False
    assert api._record_intent_state.call_args.args[2] == "deferred"
    api._execute_action.assert_not_awaited()
