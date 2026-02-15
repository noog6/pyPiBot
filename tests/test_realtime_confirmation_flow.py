"""Tests for confirmation-flow response suppression."""

from __future__ import annotations

import asyncio
from collections import deque

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


class _Ws:
    async def send(self, payload: str) -> None:  # pragma: no cover - not used directly
        return None


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.websocket = _Ws()
    api._pending_action = object()
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.AWAITING_CONFIRMATION})()
    api._injection_response_triggers = {}
    api._injection_response_cooldown_s = 0.0
    api._max_injection_responses_per_minute = 0
    api._injection_response_trigger_timestamps = {}
    api._injection_response_timestamps = deque()
    api.rate_limits = None
    api.response_in_progress = False
    api.state_manager = type("State", (), {"state": InteractionState.IDLE})()
    api._response_in_flight = False
    api._response_create_queue = deque()
    api._audio_playback_busy = False
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    return api


def test_parse_confirmation_decision_accepts_go_ahead_phrasing() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    assert api._parse_confirmation_decision("Please go ahead.") == "yes"
    assert api._parse_confirmation_decision("go ahead and do it") == "yes"
    assert api._parse_confirmation_decision("proceed") == "yes"


def test_maybe_request_response_blocks_image_trigger_during_confirmation() -> None:
    api = _make_api_stub()
    sent: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def _send_response_create(*args, **kwargs):
        sent.append((args, kwargs))
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api.maybe_request_response("image_message", {"source": "camera"}))

    assert sent == []


def test_maybe_request_response_allows_user_text_trigger_during_confirmation() -> None:
    api = _make_api_stub()
    sent: list[tuple[tuple[object, ...], dict[str, object]]] = []
    api._allow_ai_call = lambda *args, **kwargs: True

    async def _send_response_create(*args, **kwargs):
        sent.append((args, kwargs))
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api.maybe_request_response("text_message", {"source": "user_text"}))

    assert sent
    event = sent[0][0][1]
    assert event["response"]["metadata"]["trigger"] == "text_message"


def test_drain_response_create_queue_defers_injection_while_confirmation_pending() -> None:
    api = _make_api_stub()
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"trigger": "image_message", "origin": "injection"}},
            },
            "origin": "injection",
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    sent: list[str] = []

    async def _send_response_create(*args, **kwargs):
        sent.append("sent")
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api._drain_response_create_queue())

    assert sent == []
    assert len(api._response_create_queue) == 1


def test_drain_response_create_queue_allows_approval_flow_prompt() -> None:
    api = _make_api_stub()
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "origin": "assistant_message",
                        "approval_flow": "true",
                    }
                },
            },
            "origin": "assistant_message",
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    sent: list[str] = []

    async def _send_response_create(*args, **kwargs):
        sent.append("sent")
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api._drain_response_create_queue())

    assert sent == ["sent"]
    assert len(api._response_create_queue) == 0


def test_drain_response_create_queue_skips_blocked_head_and_releases_approval_prompt() -> None:
    api = _make_api_stub()
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"trigger": "image_message", "origin": "injection"}},
            },
            "origin": "injection",
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "origin": "assistant_message",
                        "approval_flow": "true",
                    }
                },
            },
            "origin": "assistant_message",
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    sent: list[str] = []

    async def _send_response_create(*args, **kwargs):
        sent.append(kwargs["origin"])
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api._drain_response_create_queue())

    assert sent == ["assistant_message"]
    assert len(api._response_create_queue) == 1
    remaining = api._response_create_queue[0]
    metadata = remaining["event"]["response"]["metadata"]
    assert metadata["trigger"] == "image_message"


def test_send_response_create_defers_while_audio_playback_busy() -> None:
    api = _make_api_stub()
    api._audio_playback_busy = True
    sent_payloads: list[str] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(payload)

    websocket = _SendWs()
    sent_now = asyncio.run(
        api._send_response_create(
            websocket,
            {"type": "response.create"},
            origin="tool_output",
        )
    )

    assert sent_now is False
    assert sent_payloads == []
    assert len(api._response_create_queue) == 1


def test_drain_response_create_queue_waits_for_audio_playback_complete() -> None:
    api = _make_api_stub()
    api._audio_playback_busy = True
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {"type": "response.create"},
            "origin": "tool_output",
            "record_ai_call": False,
            "debug_context": None,
        }
    )
    sent: list[str] = []

    async def _send_response_create(*args, **kwargs):
        sent.append("sent")
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api._drain_response_create_queue())

    assert sent == []
    assert len(api._response_create_queue) == 1


def test_request_tool_confirmation_sends_single_spoken_prompt() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.IDLE, "transition": lambda *args, **kwargs: None})()
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api.function_call = "perform_research"
    api.function_call_args = '{"query":"x"}'
    api._governance = type("Gov", (), {"describe_tool": lambda *args, **kwargs: {"tier": 2}})()
    api._build_approval_prompt = lambda action: "Need approval"

    calls = {"assistant": 0, "response_create": 0}

    async def _assistant_message(*args, **kwargs):
        calls["assistant"] += 1

    async def _send_response_create(*args, **kwargs):
        calls["response_create"] += 1
        return True

    api.send_assistant_message = _assistant_message
    api._send_response_create = _send_response_create

    sent_payloads: list[str] = []

    class _Ws:
        async def send(self, payload: str) -> None:
            sent_payloads.append(payload)

    from ai.governance import ActionPacket

    action = ActionPacket(
        id="call_123",
        tool_name="perform_research",
        tool_args={"query": "waveshare"},
        tier=2,
        what="research",
        why="user asked",
        impact="none",
        rollback="n/a",
        alternatives=[],
        confidence=0.3,
        cost="expensive",
        risk_flags=[],
        requires_confirmation=True,
    )

    asyncio.run(api._request_tool_confirmation(action, "needs confirmation", _Ws(), {"valid": True}))

    assert calls["assistant"] == 1
    assert calls["response_create"] == 0
    assert sent_payloads
