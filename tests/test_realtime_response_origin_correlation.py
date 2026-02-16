"""Tests for response.created origin correlation."""

from __future__ import annotations

import asyncio
from collections import deque

from ai.realtime_api import RealtimeAPI


class _OrchestrationState:
    def __init__(self) -> None:
        self.transitions: list[tuple[object, str | None]] = []

    def transition(self, *args, **kwargs) -> None:
        phase = args[0] if args else None
        reason = kwargs.get("reason")
        self.transitions.append((phase, reason))


class _StateManager:
    def update_state(self, *args, **kwargs) -> None:
        return None


class _Mic:
    is_receiving = False

    def start_receiving(self) -> None:
        return None


def _build_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._pending_response_create_origins = deque(maxlen=64)
    api._last_outgoing_event_type = None
    api._active_response_id = None
    api.orchestration_state = _OrchestrationState()
    api.audio_player = None
    api._audio_accum = bytearray()
    api._mic_receive_on_first_audio = False
    api.response_in_progress = False
    api._response_in_flight = False
    api._speaking_started = False
    api._assistant_reply_accum = ""
    api._tool_call_records = []
    api._last_tool_call_results = []
    api._last_response_metadata = {}
    api._reflection_enqueued = False
    api.state_manager = _StateManager()
    api.mic = _Mic()
    return api


def test_response_created_origin_correlation_distinguishes_tool_and_server_auto(monkeypatch) -> None:
    api = _build_api()
    logs: list[str] = []
    monkeypatch.setattr("ai.realtime_api.log_info", logs.append)

    api._track_outgoing_event({"type": "response.create"}, origin="tool_output")

    asyncio.run(
        api.handle_event(
            {"type": "response.created", "response": {"id": "resp_tool"}},
            websocket=None,
        )
    )
    asyncio.run(
        api.handle_event(
            {"type": "response.created", "response": {"id": "resp_vad"}},
            websocket=None,
        )
    )

    origin_logs = [entry for entry in logs if entry.startswith("response.created:")]
    assert origin_logs == [
        "response.created: origin=tool_output",
        "response.created: origin=server_auto",
    ]
    assert list(api._pending_response_create_origins) == []


def test_response_created_from_confirmation_prompt_keeps_awaiting_phase(monkeypatch) -> None:
    api = _build_api()
    logs: list[str] = []
    monkeypatch.setattr("ai.realtime_api.log_info", logs.append)
    api._pending_action = object()

    api._track_outgoing_event({"type": "response.create"}, origin="assistant_message")

    asyncio.run(
        api.handle_event(
            {"type": "response.created", "response": {"id": "resp_confirmation_prompt"}},
            websocket=None,
        )
    )

    assert api.orchestration_state.transitions == []
    assert (
        "response.created consumed by confirmation flow; phase remains AWAITING_CONFIRMATION"
        in logs
    )
