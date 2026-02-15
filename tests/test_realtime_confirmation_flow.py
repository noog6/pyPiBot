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
    return api


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
