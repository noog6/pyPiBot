"""Tests for response.created origin correlation."""

from __future__ import annotations

import asyncio
from collections import deque
import sys
import types


if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI, parse_rate_limits


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

    class _ConfirmationRuntime:
        async def maybe_handle_confirmation_decision_timeout(self, *_args, **_kwargs) -> bool:
            return False

        def mark_confirmation_activity(self, *_args, **_kwargs) -> None:
            return None

    api._confirmation_runtime = _ConfirmationRuntime()
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
    api.rate_limits = None
    api.rate_limits_supports_tokens = False
    api.rate_limits_supports_requests = False
    api.rate_limits_last_present_names = set()
    api.rate_limits_last_event_id = ""
    api._rate_limits_regression_missing_counts = {"tokens": 0, "requests": 0}
    api._rate_limits_strict = False
    api._rate_limits_regression_warning_threshold = 3
    api._rate_limits_debug_samples = False
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




def test_parse_rate_limits_parses_expected_buckets() -> None:
    rl_map, meta = parse_rate_limits(
        {
            "rate_limits": [
                {"name": "requests", "remaining": 9},
                {"name": "tokens", "remaining": 99},
            ]
        }
    )

    assert set(rl_map.keys()) == {"requests", "tokens"}
    assert meta["present_names"] == ["requests", "tokens"]
    assert meta["unknown_names"] == []
    assert meta["entry_count"] == 2
    assert meta["malformed_count"] == 0


def test_parse_rate_limits_normalizes_name_variants() -> None:
    rl_map, meta = parse_rate_limits(
        {"rate_limits": [{"name": " Requests ", "remaining": 9}, {"name": "TOKENS"}]}
    )

    assert set(rl_map.keys()) == {"requests", "tokens"}
    assert meta["present_names"] == ["requests", "tokens"]


def test_parse_rate_limits_handles_non_list_and_missing_names() -> None:
    rl_map, meta = parse_rate_limits({"rate_limits": None})

    assert rl_map == {}
    assert meta["rate_limits_is_list"] is False

    rl_map, meta = parse_rate_limits({"rate_limits": [{"remaining": 1}, "oops", {}]})
    assert rl_map == {}
    assert meta["entry_count"] == 3
    assert meta["malformed_count"] == 3


def test_parse_rate_limits_tracks_unknown_bucket_names() -> None:
    rl_map, meta = parse_rate_limits({"rate_limits": [{"name": "compute"}, {"name": "tokens"}]})

    assert set(rl_map.keys()) == {"tokens"}
    assert meta["unknown_names"] == ["compute"]

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


def test_rate_limits_updated_tokens_only_steady_payload_logs_info_without_early_warning(
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

    assert warning_logs == []
    assert "present_names=['tokens']" in info_logs[-1]
    assert "tokens 995/1000 reset=n/a" in info_logs[-1]
    assert api.rate_limits_supports_tokens is True
    assert api.rate_limits_supports_requests is False


def test_rate_limits_updated_requests_only_logs_info_without_warning(monkeypatch) -> None:
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
        "rate_limits": [{"name": "requests", "remaining": 9, "limit": 10, "reset_seconds": 1}],
    }

    asyncio.run(api.handle_event(event, websocket=None))

    assert warning_logs == []
    assert "present_names=['requests']" in info_logs[-1]
    assert "requests 9/10 reset=1s" in info_logs[-1]
    assert api.rate_limits_supports_requests is True


def test_rate_limits_updated_both_buckets_logs_info_without_warning(monkeypatch) -> None:
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
            {"name": "requests", "remaining": 9, "limit": 10, "reset_seconds": 1},
            {"name": "tokens", "remaining": 995, "limit": 1000, "reset_seconds": 1},
        ],
    }

    asyncio.run(api.handle_event(event, websocket=None))

    assert warning_logs == []
    assert "present_names=['requests', 'tokens']" in info_logs[-1]
    assert "requests 9/10 reset=1s" in info_logs[-1]
    assert "tokens 995/1000 reset=1s" in info_logs[-1]
    assert api.rate_limits_supports_requests is True
    assert api.rate_limits_supports_tokens is True


def test_rate_limits_updated_empty_payload_warns_for_suspicious_signal(monkeypatch) -> None:
    api = _build_api()
    warning_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    event = {"type": "rate_limits.updated", "rate_limits": []}

    asyncio.run(api.handle_event(event, websocket=None))

    assert warning_logs == [
        "Realtime API rate_limits.updated has no buckets: event_id=unknown entry_count=0 "
        "malformed_count=0 unknown_names=[]"
    ]


def test_rate_limits_updated_missing_rate_limits_field_warns(monkeypatch) -> None:
    api = _build_api()
    warning_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    asyncio.run(api.handle_event({"type": "rate_limits.updated"}, websocket=None))

    assert warning_logs == [
        "Realtime API rate_limits.updated has no buckets: event_id=unknown entry_count=0 "
        "malformed_count=0 unknown_names=[]"
    ]


def test_rate_limits_updated_missing_requests_three_updates_strict_false_no_warning(monkeypatch) -> None:
    api = _build_api()
    warning_logs: list[str] = []
    api._rate_limits_regression_warning_threshold = 3

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    baseline_event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {"name": "requests", "remaining": 9, "limit": 10, "reset_seconds": 1},
            {"name": "tokens", "remaining": 995, "limit": 1000, "reset_seconds": 1},
        ],
    }
    tokens_only_event = {
        "type": "rate_limits.updated",
        "rate_limits": [{"name": "tokens", "remaining": 990, "limit": 1000, "reset_seconds": 1}],
    }

    asyncio.run(api.handle_event(baseline_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))

    assert warning_logs == []


def test_rate_limits_updated_missing_requests_three_updates_strict_true_warns(monkeypatch) -> None:
    api = _build_api()
    warning_logs: list[str] = []
    api._rate_limits_strict = True

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warning_logs.append(message % args),
    )

    baseline_event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {"name": "requests", "remaining": 9, "limit": 10, "reset_seconds": 1},
            {"name": "tokens", "remaining": 995, "limit": 1000, "reset_seconds": 1},
        ],
    }
    tokens_only_event = {
        "type": "rate_limits.updated",
        "rate_limits": [{"name": "tokens", "remaining": 990, "limit": 1000, "reset_seconds": 1}],
    }

    asyncio.run(api.handle_event(baseline_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))
    asyncio.run(api.handle_event(tokens_only_event, websocket=None))

    regression_warnings = [line for line in warning_logs if "regression: bucket=requests" in line]
    assert len(regression_warnings) == 1
    assert "missing_count=1" in regression_warnings[0]


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


def test_response_output_audio_transcript_delta_trace_is_debug_with_sampling(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._lifecycle_trace_transcript_delta_sample_n = 3
    api._lifecycle_trace_transcript_delta_inactivity_ms = 750
    api._lifecycle_trace_transcript_delta_state = {}
    api._lifecycle_trace_item_added_unknown_events = deque()
    api._lifecycle_trace_item_added_unknown_threshold = 3
    api._lifecycle_trace_item_added_unknown_window_s = 10.0
    api._lifecycle_trace_item_added_unknown_cooldown_s = 30.0
    api._lifecycle_trace_item_added_unknown_last_escalation_ts = 0.0

    debug_logs: list[str] = []
    info_logs: list[str] = []
    monotonic_values = iter([10.0, 10.1, 10.2, 10.3])

    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("ai.realtime_api.logger.debug", lambda message, *args: debug_logs.append(message % args))
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: info_logs.append(message % args))

    for _ in range(4):
        api._emit_response_lifecycle_trace(
            event_type="response.output_audio_transcript.delta",
            response_id="resp_trace_2",
            turn_id="turn_trace_2",
            input_event_key="input_evt_trace_2",
            canonical_key="turn_trace_2::input_evt_trace_2",
            origin="assistant_message",
            active_input_event_key="input_evt_trace_2",
            active_canonical_key="turn_trace_2::input_evt_trace_2",
            payload_summary="has_delta=true",
        )

    transcript_trace_logs = [
        line for line in debug_logs if "response_lifecycle_trace response_id=resp_trace_2" in line
    ]
    assert len(transcript_trace_logs) == 2
    assert "seq=1 sampled=false first=true last=false" in transcript_trace_logs[0]
    assert "seq=3 sampled=true first=false last=false" in transcript_trace_logs[1]
    assert all("event_type=response.output_audio_transcript.delta" in line for line in transcript_trace_logs)
    assert not any(
        "response_lifecycle_trace response_id=resp_trace_2 event_type=response.output_audio_transcript.delta" in line
        for line in info_logs
    )


def test_conversation_item_added_trace_debug_with_unknown_escalation_cooldown(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._lifecycle_trace_transcript_delta_sample_n = 20
    api._lifecycle_trace_transcript_delta_inactivity_ms = 750
    api._lifecycle_trace_transcript_delta_state = {}
    api._lifecycle_trace_item_added_unknown_events = deque()
    api._lifecycle_trace_item_added_unknown_threshold = 3
    api._lifecycle_trace_item_added_unknown_window_s = 10.0
    api._lifecycle_trace_item_added_unknown_cooldown_s = 30.0
    api._lifecycle_trace_item_added_unknown_last_escalation_ts = 0.0

    debug_logs: list[str] = []
    info_logs: list[str] = []
    monotonic_values = iter([0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0])

    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("ai.realtime_api.logger.debug", lambda message, *args: debug_logs.append(message % args))
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: info_logs.append(message % args))

    for _ in range(7):
        api._emit_response_lifecycle_trace(
            event_type="conversation.item.added",
            response_id="",
            turn_id="turn_trace_3",
            input_event_key="input_evt_trace_3",
            canonical_key="turn_trace_3::input_evt_trace_3",
            origin="assistant_message",
            active_input_event_key="input_evt_trace_3",
            active_canonical_key="turn_trace_3::input_evt_trace_3",
            payload_summary="has_item=true item_type=message",
        )

    item_added_debug_logs = [
        line
        for line in debug_logs
        if "event_type=conversation.item.added" in line and line.startswith("response_lifecycle_trace ")
    ]
    assert len(item_added_debug_logs) == 7
    escalation_logs = [
        line for line in info_logs if "response_lifecycle_trace_unknown_item_added_spike" in line
    ]
    assert len(escalation_logs) == 1
