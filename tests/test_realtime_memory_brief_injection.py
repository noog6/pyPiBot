"""Tests for memory brief injection sequencing in realtime responses."""

from __future__ import annotations

import asyncio
import json
from collections import deque

from ai.realtime.transport import RealtimeTransport
from ai.realtime_api import RealtimeAPI


class _Ws:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.events.append(json.loads(payload))


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._transport = RealtimeTransport(
        connect_fn=lambda *args, **kwargs: None,
        validate_outbound_endpoint=lambda _url: None,
    )
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._active_response_id = None
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._response_create_queue = deque()
    api._pending_response_create = None
    api._response_create_turn_counter = 0
    api._current_response_turn_id = None
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._record_ai_call = lambda: None
    api._track_outgoing_event = lambda *args, **kwargs: None
    return api


def test_send_response_create_injects_memory_brief_before_response() -> None:
    api = _make_api_stub()
    ws = _Ws()

    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="injection",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]


def test_send_response_create_preserves_brief_when_deferred() -> None:
    api = _make_api_stub()
    api._response_in_flight = True
    ws = _Ws()

    sent = asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="injection",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert sent is False
    assert len(api._response_create_queue) == 1
    assert api._response_create_queue[0]["memory_brief_note"] == "Turn memory brief: ..."


def test_initialize_session_injects_startup_memory_digest_note() -> None:
    api = _make_api_stub()
    api.profile_manager = type("P", (), {"get_profile_context": lambda self: type("Ctx", (), {"to_instruction_block": lambda self: "profile"})()})()
    api._vad_turn_detection = {
        "profile": "default",
        "threshold": 0.2,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 900,
        "create_response": True,
        "interrupt_response": True,
    }
    api._build_startup_memory_digest_note = lambda: "Startup memory digest: ..."

    ws = _Ws()
    asyncio.run(api.initialize_session(ws))

    assert ws.events[0]["type"] == "session.update"
    assert ws.events[1]["type"] == "conversation.item.create"
