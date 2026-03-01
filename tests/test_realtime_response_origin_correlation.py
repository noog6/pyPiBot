"""Tests for response.created origin correlation."""

from __future__ import annotations

import asyncio
from collections import deque
import sys
import types


if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

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


class _MemoryManager:
    def get_active_session_id(self) -> str:
        return "sess-test"


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
    api._memory_manager = _MemoryManager()
    api._input_audio_events = type(
        "_InputAudioEvents",
        (),
        {
            "handle_input_audio_buffer_speech_started": staticmethod(lambda *_args, **_kwargs: None),
            "handle_input_audio_buffer_speech_stopped": staticmethod(lambda *_args, **_kwargs: None),
            "handle_input_audio_buffer_committed": staticmethod(lambda *_args, **_kwargs: None),
            "handle_input_audio_transcription_partial": staticmethod(lambda *_args, **_kwargs: None),
        },
    )()
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


def test_response_created_origin_correlation_prefers_non_micro_ack_for_server_auto(monkeypatch) -> None:
    api = _build_api()
    logs: list[str] = []
    monkeypatch.setattr("ai.realtime_api.log_info", logs.append)

    api._track_outgoing_event(
        {
            "type": "response.create",
            "response": {"metadata": {"origin": "assistant_message", "micro_ack": "true"}},
        },
        origin="assistant_message",
    )

    asyncio.run(
        api.handle_event(
            {"type": "response.created", "response": {"id": "resp_vad"}},
            websocket=None,
        )
    )

    origin_logs = [entry for entry in logs if entry.startswith("response.created:")]
    assert origin_logs == ["response.created: origin=server_auto"]
    assert list(api._pending_response_create_origins) == [
        {
            "origin": "assistant_message",
            "micro_ack": "true",
            "consumes_canonical_slot": "false",
            "turn_id": "",
            "input_event_key": "",
        }
    ]


def test_response_created_binding_prompt_happy_path_never_logs_unknown_active_key(monkeypatch) -> None:
    api = _build_api()
    binding_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.info",
        lambda message, *args: binding_logs.append(message % args),
    )

    api._track_outgoing_event(
        {
            "type": "response.create",
            "response": {"metadata": {"turn_id": "turn-7", "input_event_key": "evt-7"}},
        },
        origin="prompt",
    )

    asyncio.run(
        api.handle_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-prompt",
                    "metadata": {"turn_id": "turn-7", "input_event_key": "evt-7"},
                },
            },
            websocket=None,
        )
    )

    assert any("response_binding" in line for line in binding_logs)
    assert not any("response_binding" in line and "active_key=unknown" in line for line in binding_logs)


def test_response_created_binding_server_auto_happy_path_never_logs_unknown_active_key(monkeypatch) -> None:
    api = _build_api()
    binding_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.info",
        lambda message, *args: binding_logs.append(message % args),
    )

    api._current_response_turn_id = "turn-9"
    api._current_input_event_key = "evt-9"
    api._active_input_event_key_by_turn_id = {"turn-9": "evt-9"}
    api._pending_server_auto_input_event_keys = deque(["evt-9"])

    asyncio.run(
        api.handle_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-auto",
                    "metadata": {"turn_id": "turn-9", "input_event_key": "evt-9"},
                },
            },
            websocket=None,
        )
    )

    assert any("response_binding" in line for line in binding_logs)
    assert not any("response_binding" in line and "active_key=unknown" in line for line in binding_logs)


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


def test_rate_limits_updated_tokens_only_steady_payload_logs_info_and_throttled_warning(
    monkeypatch,
) -> None:
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
    asyncio.run(api.handle_event(event, websocket=None))

    assert warning_logs[-1] == (
        "Realtime API requests bucket missing for 3 consecutive rate_limits.updated events "
        "(session_id=sess-test)"
    )
    assert (
        info_logs[-1]
        == "Rate limits: requests_bucket=missing | requests n/a/n/a reset=n/a | "
        "tokens 995/1000 reset=n/a"
    )


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

    assert "requests_bucket=missing" in info_logs[-1]
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


def test_resolve_response_create_turn_id_keeps_micro_ack_turn_id() -> None:
    api = _build_api()
    api._response_create_turn_counter = 0
    api._current_response_turn_id = None
    event = {
        "type": "response.create",
        "response": {"metadata": {"origin": "assistant_message", "micro_ack": "true", "micro_ack_turn_id": "turn_1"}},
    }

    turn_id = api._resolve_response_create_turn_id(origin="assistant_message", response_create_event=event)

    assert turn_id == "turn_1"
    assert api._current_response_turn_id == "turn_1"


def test_response_lifecycle_trace_continuity_created_content_done(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._active_response_id = "resp_trace_1"
    api._active_response_origin = "assistant_message"
    api._active_response_input_event_key = "input_evt_trace_1"
    api._active_response_canonical_key = "turn_trace_1::input_evt_trace_1"
    api._response_trace_context_by_id = {}
    api._current_turn_id_or_unknown = lambda: "turn_trace_1"
    api._canonical_utterance_key = lambda turn_id, input_event_key: f"{turn_id}::{input_event_key}"

    async def _legacy_handler(_event, _websocket) -> None:
        return None

    api._handle_event_legacy = _legacy_handler

    info_logs: list[str] = []
    debug_logs: list[str] = []

    def _capture_info(message: str, *args) -> None:
        info_logs.append(message % args)

    def _capture_debug(message: str, *args) -> None:
        debug_logs.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture_info)
    monkeypatch.setattr("ai.realtime_api.logger.debug", _capture_debug)

    api._record_response_trace_context(
        "resp_trace_1",
        turn_id="turn_trace_1",
        input_event_key="input_evt_trace_1",
        canonical_key="turn_trace_1::input_evt_trace_1",
        origin="assistant_message",
    )
    api._emit_response_lifecycle_trace(
        event_type="response.created",
        response_id="resp_trace_1",
        turn_id="turn_trace_1",
        input_event_key="input_evt_trace_1",
        canonical_key="turn_trace_1::input_evt_trace_1",
        origin="assistant_message",
        active_input_event_key="input_evt_trace_1",
        active_canonical_key="turn_trace_1::input_evt_trace_1",
        payload_summary="anchor=created",
    )

    asyncio.run(
        api._handle_response_lifecycle_event(
            {"type": "response.text.delta", "response": {"id": "resp_trace_1"}, "delta": "hello"},
            websocket=None,
        )
    )

    api._emit_response_lifecycle_trace(
        event_type="response.done",
        response_id="resp_trace_1",
        turn_id="turn_trace_1",
        input_event_key="input_evt_trace_1",
        canonical_key="turn_trace_1::input_evt_trace_1",
        origin="assistant_message",
        active_input_event_key="input_evt_trace_1",
        active_canonical_key="turn_trace_1::input_evt_trace_1",
        payload_summary="anchor=done",
    )

    lifecycle_anchors = [
        line
        for line in info_logs
        if line.startswith("response_lifecycle_trace response_id=resp_trace_1")
    ]
    assert any("event_type=response.created" in line for line in lifecycle_anchors)
    assert any("event_type=response.text.delta" in line for line in lifecycle_anchors)
    assert any("event_type=response.done" in line for line in lifecycle_anchors)
    assert all("turn_id=turn_trace_1" in line for line in lifecycle_anchors)
    assert all("input_event_key=input_evt_trace_1" in line for line in lifecycle_anchors)
    assert all("canonical_key=turn_trace_1::input_evt_trace_1" in line for line in lifecycle_anchors)
    assert any(
        "response_lifecycle_trace_detail response_id=resp_trace_1 event_type=response.text.delta" in line
        for line in debug_logs
    )
