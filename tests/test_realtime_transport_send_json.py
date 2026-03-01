"""Tests realtime outbound events route through RealtimeTransport.send_json."""

from __future__ import annotations

import asyncio
from collections import deque

from ai.realtime.transport import RealtimeTransport
from ai.realtime_api import RealtimeAPI


class _Ws:
    async def send(self, payload: str) -> None:  # pragma: no cover - should not be called
        raise AssertionError("websocket.send should not be called directly")


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
    api.profile_manager = type(
        "P",
        (),
        {
            "get_profile_context": lambda self: type(
                "Ctx", (), {"to_instruction_block": lambda self: "profile"}
            )()
        },
    )()
    api._vad_turn_detection = {
        "profile": "default",
        "threshold": 0.2,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 900,
        "create_response": True,
        "interrupt_response": True,
    }
    api._build_startup_memory_digest_note = lambda: None
    return api


def test_outbound_session_update_and_response_create_use_transport_send_json(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _Ws()
    sent_types: list[str] = []

    async def _capture_send_json(self, websocket, payload):
        assert websocket is ws
        sent_types.append(str(payload.get("type", "")))

    monkeypatch.setattr(RealtimeTransport, "send_json", _capture_send_json)

    asyncio.run(api.initialize_session(ws))
    asyncio.run(api._send_response_create(ws, {"type": "response.create"}, origin="assistant_message"))

    assert "session.update" in sent_types
    assert "response.create" in sent_types
