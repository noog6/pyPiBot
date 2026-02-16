"""Tests for response.created origin correlation."""

from __future__ import annotations

import asyncio
from collections import deque

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI


class _OrchestrationState:
    def __init__(self) -> None:
        self.phase = OrchestrationPhase.IDLE
        self.transitions: list[tuple[object, str | None]] = []

    def transition(self, *args, **kwargs) -> None:
        phase = args[0] if args else None
        reason = kwargs.get("reason")
        self.transitions.append((phase, reason))


class _StateManager:
    def __init__(self) -> None:
        self.state = None

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


def test_response_created_from_pending_action_with_server_auto_keeps_awaiting_phase(
    monkeypatch,
) -> None:
    api = _build_api()
    logs: list[str] = []
    monkeypatch.setattr("ai.realtime_api.log_info", logs.append)
    api._pending_action = object()
    api.orchestration_state.phase = OrchestrationPhase.AWAITING_CONFIRMATION

    asyncio.run(
        api.handle_event(
            {"type": "response.created", "response": {"id": "resp_confirmation_prompt"}},
            websocket=None,
        )
    )

    assert api.orchestration_state.transitions == []
    assert api.orchestration_state.phase == OrchestrationPhase.AWAITING_CONFIRMATION
    assert "response.created consumed by confirmation flow; origin=server_auto" in logs


def test_rate_limits_updated_tokens_only_steady_payload_logs_info_only(monkeypatch) -> None:
    api = _build_api()
    info_logs: list[str] = []
    warning_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.info",
        lambda message, *args: info_logs.append(message % args),
    )
    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {
                "name": "tokens",
                "remaining": 995,
                "limit": 1000,
                "reset_seconds": None,
            }
        ],
    }

    asyncio.run(api.handle_event(event, websocket=None))
    asyncio.run(api.handle_event(event, websocket=None))

    assert warning_logs == []
    assert info_logs[-1] == "Rate limits: requests n/a/n/a reset=n/a | tokens 995/1000 reset=n/a"


def test_rate_limits_updated_missing_requests_does_not_suffix_na_reset(monkeypatch) -> None:
    api = _build_api()
    info_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.info",
        lambda message, *args: info_logs.append(message % args),
    )

    event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {
                "name": "tokens",
                "remaining": 995,
                "limit": 1000,
                "reset_seconds": None,
            }
        ],
    }

    asyncio.run(api.handle_event(event, websocket=None))

    assert "reset=n/a" in info_logs[-1]
    assert "n/as" not in info_logs[-1]


def test_rate_limits_updated_empty_payload_warns_once_for_stable_omission(monkeypatch) -> None:
    api = _build_api()
    warning_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    event = {"type": "rate_limits.updated", "rate_limits": []}

    asyncio.run(api.handle_event(event, websocket=None))
    asyncio.run(api.handle_event(event, websocket=None))

    assert warning_logs == [
        "Realtime API rate_limits.updated missing expected bucket(s): requests, tokens"
    ]


def test_rate_limits_updated_warns_when_bucket_disappears_mid_session(monkeypatch) -> None:
    api = _build_api()
    warning_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    baseline_event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {"name": "requests", "remaining": 99, "limit": 100, "reset_seconds": 1},
            {"name": "tokens", "remaining": 995, "limit": 1000, "reset_seconds": 1},
        ],
    }
    tokens_only_event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {"name": "tokens", "remaining": 990, "limit": 1000, "reset_seconds": 1}
        ],
    }

    asyncio.run(api.handle_event(baseline_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))

    assert warning_logs == [
        "Realtime API rate_limits.updated missing expected bucket(s): requests"
    ]
