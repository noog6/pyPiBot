"""Tests for confirmation-flow response suppression."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from unittest.mock import patch

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import ConfirmationState, PendingConfirmationToken, RealtimeAPI
from interaction import InteractionState


class _Ws:
    async def send(self, payload: str) -> None:  # pragma: no cover - not used directly
        return None


class _Mic:
    def __init__(self) -> None:
        self.is_recording = True
        self.start_recording_calls = 0

    def start_recording(self) -> None:
        self.start_recording_calls += 1
        self.is_recording = True

    def stop_recording(self) -> None:
        self.is_recording = False


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
    api.state_manager = type("State", (), {"state": InteractionState.IDLE, "update_state": lambda *args, **kwargs: None})()
    api._response_in_flight = False
    api._response_create_queue = deque()
    api._audio_playback_busy = False
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._response_done_serial = 0
    api._active_response_confirmation_guarded = False
    api._active_response_origin = "unknown"
    api._active_response_id = None
    api._confirmation_timeout_markers = {}
    api._confirmation_timeout_causes = {}
    api._confirmation_timeout_debounce_window_s = 5.0
    api._approval_timeout_s = 30.0
    api._awaiting_confirmation_completion = False
    api._last_user_input_text = None
    api._stop_words = []
    api._tool_execution_disabled_until = 0.0
    api._pending_research_request = None
    api._prior_research_permission_marker = None
    api._prior_research_permission_grace_s = 8.0
    api._presented_actions = set()
    api._pending_confirmation_token = None
    api._confirmation_state = ConfirmationState.IDLE
    api._last_executed_tool_call = None
    api._debug_governance_decisions = False
    api.mic = _Mic()
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


def test_drain_response_create_queue_waits_until_not_listening() -> None:
    api = _make_api_stub()
    api.state_manager.state = InteractionState.LISTENING
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

    # Simulate speech-stopped/idle transition before draining again.
    api.state_manager.state = InteractionState.IDLE
    api._pending_action = None
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.IDLE})()

    asyncio.run(api._drain_response_create_queue())

    assert sent == ["sent"]
    assert len(api._response_create_queue) == 0


def test_drain_response_create_queue_drops_stale_tool_output_entry() -> None:
    api = _make_api_stub()
    api._response_done_serial = 7
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {"type": "response.create"},
            "origin": "tool_output",
            "record_ai_call": False,
            "debug_context": None,
            "enqueued_done_serial": 5,
        }
    )
    sent: list[str] = []

    async def _send_response_create(*args, **kwargs):
        sent.append("sent")
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api._drain_response_create_queue())

    assert sent == []
    assert len(api._response_create_queue) == 0


def test_drain_response_create_queue_drops_stale_assistant_message_after_confirmation() -> None:
    api = _make_api_stub()
    api._response_done_serial = 9
    api._pending_action = None
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.IDLE})()
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"origin": "assistant_message", "approval_flow": "true"}},
            },
            "origin": "assistant_message",
            "record_ai_call": False,
            "debug_context": None,
            "enqueued_done_serial": 7,
        }
    )
    sent: list[str] = []

    async def _send_response_create(*args, **kwargs):
        sent.append("sent")
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api._drain_response_create_queue())

    assert sent == []
    assert len(api._response_create_queue) == 0


def test_drain_response_create_queue_drops_non_approval_assistant_message_after_one_completion() -> None:
    api = _make_api_stub()
    api._response_done_serial = 10
    api._pending_action = None
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.IDLE})()
    api._response_create_queue.append(
        {
            "websocket": api.websocket,
            "event": {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "origin": "assistant_message",
                        "trigger": "research_summary",
                    }
                },
            },
            "origin": "assistant_message",
            "record_ai_call": False,
            "debug_context": None,
            "enqueued_done_serial": 9,
        }
    )

    info_logs = ["research summary window=alpha"]

    async def _send_response_create(*args, **kwargs):
        info_logs.append("origin=assistant_message replay window=alpha")
        return True

    api._send_response_create = _send_response_create

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api._drain_response_create_queue())

    info_logs.extend(call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args)

    assert info_logs.count("research summary window=alpha") == 1
    assert "origin=assistant_message replay window=alpha" not in info_logs
    assert any(
        "Dropping stale queued response.create origin=assistant_message" in log
        for log in info_logs
    )
    assert len(api._response_create_queue) == 0


def test_handle_response_done_keeps_listening_state_without_idle_transition() -> None:
    api = _make_api_stub()
    transitions: list[tuple[InteractionState, str]] = []
    api.state_manager = type(
        "State",
        (),
        {
            "state": InteractionState.LISTENING,
            "update_state": lambda *args: transitions.append((args[1], args[2])),
        },
    )()
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    assert api.state_manager.state == InteractionState.LISTENING
    assert transitions == []


def test_handle_response_done_token_active_with_idle_phase_skips_reflect_transition() -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api._pending_confirmation_token = _build_confirmation_token(
        kind="tool_governance",
        pending_action=_build_pending_action(),
    )
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    assert transitions == []


def test_handle_response_completed_token_active_with_idle_phase_skips_reflect_transition() -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api._pending_confirmation_token = _build_confirmation_token(
        kind="tool_governance",
        pending_action=_build_pending_action(),
    )
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False

    asyncio.run(api.handle_response_completed({"type": "response.completed"}))

    assert transitions == []



def test_maybe_process_research_intent_emits_single_permission_prompt_for_duplicate_request() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._pending_research_request = None
    api._pending_action = None
    api._research_permission_required = True
    api._clip_text = lambda value, limit=80: value[:limit]

    prompted: list[str] = []

    async def _send_assistant_message(message, *args, **kwargs):
        prompted.append(message)

    api.send_assistant_message = _send_assistant_message

    first = asyncio.run(
        api._maybe_process_research_intent(
            "search the web for market updates",
            _Ws(),
            source="input_audio_transcription",
        )
    )
    second = asyncio.run(
        api._maybe_process_research_intent(
            "search the web for market updates",
            _Ws(),
            source="input_audio_transcription",
        )
    )

    assert first is True
    assert second is True
    assert len(prompted) == 1


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


def _build_pending_action(*, expiry_ts=None):
    from ai.governance import ActionPacket
    from ai.realtime_api import PendingAction

    action = ActionPacket(
        id="call_pending",
        tool_name="perform_research",
        tool_args={"query": "status"},
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
        expiry_ts=expiry_ts,
    )
    return PendingAction(
        action=action,
        staging={"valid": True},
        original_intent="test",
        created_at=0.0,
    )


def test_handle_event_speech_started_stays_in_awaiting_confirmation() -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []

    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    state_updates: list[tuple[InteractionState, str]] = []
    api.state_manager = type(
        "State",
        (),
        {
            "state": InteractionState.IDLE,
            "update_state": lambda *args: state_updates.append((args[1], args[2])),
        },
    )()
    api._pending_action = _build_pending_action()
    api._utterance_counter = 0
    api._active_utterance = None
    api._debug_vad = False

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api.handle_event({"type": "input_audio_buffer.speech_started"}, api.websocket))

    assert transitions == []
    assert state_updates == [(InteractionState.LISTENING, "speech started")]
    assert any(
        "confirmation mode remains active" in str(call.args[0])
        for call in mock_info.call_args_list
        if call.args
    )


def test_handle_event_response_created_marks_server_auto_as_confirmation_guarded() -> None:
    api = _make_api_stub()
    api._pending_response_create_origins = deque(["server_auto"])
    api._audio_accum = bytearray()
    api._audio_accum_bytes_target = 9600
    api._mic_receive_on_first_audio = False
    api._speaking_started = False
    api._assistant_reply_accum = ""
    api._tool_call_records = []
    api._last_tool_call_results = []
    api._last_response_metadata = {}
    api._reflection_enqueued = False
    api.audio_player = None
    api._is_user_approved_interrupt_response = lambda *_args, **_kwargs: False
    api.state_manager = type("State", (), {"state": InteractionState.IDLE, "update_state": lambda *args: None})()

    asyncio.run(api.handle_event({"type": "response.created", "response": {"id": "resp_1"}}, api.websocket))

    assert api._active_response_confirmation_guarded is True
    assert api._active_response_origin == "server_auto"


def test_handle_transcribe_response_done_suppresses_confirmation_guarded_response() -> None:
    api = _make_api_stub()
    api.websocket = _Ws()
    api.assistant_reply = "Thanks for confirming."
    api._assistant_reply_accum = "Thanks for confirming."
    api._active_response_confirmation_guarded = True
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp_2"
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._maybe_enqueue_reflection = lambda *_args, **_kwargs: None

    sent_prompts: list[str] = []

    async def _send_assistant_message(message, *_args, **_kwargs):
        sent_prompts.append(message)

    api.send_assistant_message = _send_assistant_message

    asyncio.run(api.handle_transcribe_response_done())

    assert sent_prompts == ["Please reply with: yes or no."]
    assert api.assistant_reply == ""
    assert api._assistant_reply_accum == ""


def test_confirmation_guarded_transcribe_done_restarts_mic_once_when_recording_stopped() -> None:
    api = _make_api_stub()
    api.websocket = _Ws()
    api.assistant_reply = "Thanks for confirming."
    api._assistant_reply_accum = "Thanks for confirming."
    api._active_response_confirmation_guarded = True
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp_2"
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._maybe_enqueue_reflection = lambda *_args, **_kwargs: None
    api._audio_playback_busy = False

    sent_prompts: list[str] = []

    async def _send_assistant_message(message, *_args, **_kwargs):
        sent_prompts.append(message)

    api.send_assistant_message = _send_assistant_message
    api.mic = _Mic()
    api.mic.is_recording = False

    asyncio.run(api.handle_speech_stopped(api.websocket))
    assert api.mic.is_recording is False
    asyncio.run(api.handle_transcribe_response_done())

    assert sent_prompts == ["Please reply with: yes or no."]
    assert api.mic.start_recording_calls == 1
    assert api.orchestration_state.phase == OrchestrationPhase.AWAITING_CONFIRMATION


def test_pending_confirmation_still_accepts_after_speech_started() -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []

    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api.state_manager = type(
        "State",
        (),
        {
            "state": InteractionState.IDLE,
            "update_state": lambda *args, **kwargs: None,
        },
    )()
    api._pending_action = _build_pending_action()
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}
    api._utterance_counter = 0
    api._active_utterance = None
    api._debug_vad = False

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action

    asyncio.run(api.handle_event({"type": "input_audio_buffer.speech_started"}, api.websocket))
    consumed = asyncio.run(api._maybe_handle_approval_response("yes", api.websocket))

    assert consumed is True
    assert executed == ["done"]
    assert api._pending_action is None
    assert transitions == []


def test_maybe_handle_approval_response_accept_transitions_after_execution() -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []

    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api._pending_action = _build_pending_action()
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action

    consumed = asyncio.run(api._maybe_handle_approval_response("yes", api.websocket))

    assert consumed is True
    assert executed == ["done"]
    assert api._pending_action is None
    assert transitions == []


def test_maybe_handle_approval_response_reject_transitions_to_idle() -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []

    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api._pending_action = _build_pending_action()
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0

    rejected: list[str] = []

    async def _reject_tool_call(*args, **kwargs):
        rejected.append("rejected")

    api._reject_tool_call = _reject_tool_call

    consumed = asyncio.run(api._maybe_handle_approval_response("no", api.websocket))

    assert consumed is True
    assert rejected == ["rejected"]
    assert api._pending_action is None
    assert transitions == [(OrchestrationPhase.IDLE, "confirmation rejected")]


def test_maybe_handle_approval_response_timeout_transitions_to_idle(caplog) -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []

    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api._pending_action = _build_pending_action(expiry_ts=-1.0)
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0

    sent_messages: list[str] = []

    async def _send_assistant_message(message, *args, **kwargs):
        sent_messages.append(message)

    api.send_assistant_message = _send_assistant_message

    caplog.set_level(logging.INFO)
    consumed = asyncio.run(api._maybe_handle_approval_response("anything", api.websocket))

    assert consumed is True
    assert sent_messages
    assert api._pending_action is None
    assert transitions == [(OrchestrationPhase.IDLE, "confirmation timeout")]


def test_maybe_handle_approval_response_retry_exhaustion_logs_timeout_cause(caplog) -> None:
    api = _make_api_stub()
    transitions: list[tuple[object, str | None]] = []

    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    pending = _build_pending_action(expiry_ts=None)
    pending.retry_count = pending.max_retries
    api._pending_action = pending
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0

    sent_messages: list[str] = []

    async def _send_assistant_message(message, *args, **kwargs):
        sent_messages.append(message)

    api.send_assistant_message = _send_assistant_message

    caplog.set_level(logging.INFO)
    consumed = asyncio.run(api._maybe_handle_approval_response("maybe", api.websocket))

    assert consumed is True
    assert sent_messages
    assert api._pending_action is None
    assert transitions == [(OrchestrationPhase.IDLE, "confirmation timeout")]


def test_record_confirmation_timeout_tracks_cause_metadata() -> None:
    api = _make_api_stub()
    pending = _build_pending_action()

    api._record_confirmation_timeout(pending.action, cause="retry_exhausted")

    fingerprint = api._build_tool_call_fingerprint(
        pending.action.tool_name,
        pending.action.tool_args,
    )
    assert api._confirmation_timeout_causes[fingerprint] == "retry_exhausted"


def test_handle_function_call_suppresses_during_pending_confirmation_without_act_transition(
    caplog,
) -> None:
    api = _make_api_stub()
    api._pending_action = _build_pending_action()
    api.function_call = {"name": "perform_research", "call_id": "call_suppress"}
    api.function_call_args = '{"query":"status"}'
    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    caplog.set_level(logging.INFO)
    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert transitions == []
    assert len(sent_payloads) == 1
    output_payload = json.loads(sent_payloads[0]["item"]["output"])
    assert output_payload["status"] == "awaiting_confirmation"
    assert api.function_call is None
    assert api.function_call_args == ""


def test_handle_function_call_suppresses_research_while_intent_permission_pending() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._pending_research_request = object()
    api.function_call = {"name": "perform_research", "call_id": "call_intent_permission"}
    api.function_call_args = '{"query":"status"}'
    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()

    governance_calls = {"build": 0, "review": 0}

    class _Gov:
        def build_action_packet(self, *args, **kwargs):
            governance_calls["build"] += 1
            return object()

        def review(self, *args, **kwargs):
            governance_calls["review"] += 1
            return object()

    api._governance = _Gov()

    confirmation_prompts = {"count": 0}

    async def _request_tool_confirmation(*args, **kwargs):
        confirmation_prompts["count"] += 1

    api._request_tool_confirmation = _request_tool_confirmation

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert transitions == []
    assert confirmation_prompts["count"] == 0
    assert governance_calls == {"build": 0, "review": 0}
    assert len(sent_payloads) == 1
    output_payload = json.loads(sent_payloads[0]["item"]["output"])
    assert output_payload["status"] == "awaiting_intent_permission"
    assert api.function_call is None
    assert api.function_call_args == ""


def test_handle_function_call_transitions_to_act_when_execution_approved() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_exec"}
    api.function_call_args = '{"query":"status"}'
    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
                "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api._extract_dry_run_flag = lambda args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}
    api._governance = type(
        "Gov",
        (),
        {
            "build_action_packet": lambda *args, **kwargs: type(
                "Action",
                (),
                {
                    "id": "call_exec",
                    "tool_name": "perform_research",
                    "summary": lambda self: "summary",
                },
            )(),
            "review": lambda *args, **kwargs: type(
                "Decision",
                (),
                {
                    "approved": True,
                    "needs_confirmation": False,
                    "status": "approved",
                    "reason": "ok",
                },
            )(),
        },
    )()

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action

    asyncio.run(api.handle_function_call({}, _Ws()))

    assert executed == ["done"]
    assert transitions == [(OrchestrationPhase.ACT, "function_call perform_research")]


def test_handle_function_call_duplicate_logs_skip_without_execution(caplog) -> None:
    api = _make_api_stub()
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_duplicate"}
    api.function_call_args = '{"query":"status"}'
    api._is_duplicate_tool_call = lambda *args, **kwargs: True
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    caplog.set_level(logging.INFO)
    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert len(sent_payloads) == 1
    output_payload = json.loads(sent_payloads[0]["item"]["output"])
    assert output_payload["status"] == "redundant"
    assert executed == []


def test_handle_function_call_logs_tool_and_call_id_with_parse_status(monkeypatch, caplog) -> None:
    api = _make_api_stub()
    api.function_call = {"name": "perform_research", "call_id": "call_123"}
    api.function_call_args = '{"query":'
    api._pending_action = type("Pending", (), {"action": type("Action", (), {"tool_name": "noop"})()})()

    async def _send_awaiting_confirmation_output(*args, **kwargs):
        return None

    monkeypatch.setattr(api, "_send_awaiting_confirmation_output", _send_awaiting_confirmation_output)

    caplog.set_level(logging.INFO)
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert (
        "Function call payload parsed | tool=perform_research "
        "call_id=call_123 parse=invalid_json"
    ) in caplog.text


def test_handle_function_call_logs_consolidated_governance_review_summary(caplog) -> None:
    api = _make_api_stub()
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_gov_1"}
    api.function_call_args = '{"query":"status"}'
    api._extract_dry_run_flag = lambda args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}
    api._governance = type(
        "Gov",
        (),
        {
            "build_action_packet": lambda *args, **kwargs: type(
                "Action",
                (),
                {
                    "id": "call_gov_1",
                    "tool_name": "perform_research",
                    "tool_args": {"query": "status"},
                    "summary": lambda self: "summary",
                },
            )(),
            "review": lambda *args, **kwargs: type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "expensive_read",
                },
            )(),
        },
    )()

    async def _request_tool_confirmation(*args, **kwargs):
        return None

    api._request_tool_confirmation = _request_tool_confirmation

    caplog.set_level(logging.INFO)
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert (
        "Governance review summary | call_id=call_gov_1 tool=perform_research "
        "initial_status=needs_confirmation initial_reason=expensive_read "
        "prior_permission_override=False final_execution_decision=request_confirmation"
    ) in caplog.text



def test_handle_function_call_suppresses_immediate_recall_after_confirmation_timeout() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_timeout_suppressed"}
    api.function_call_args = '{"query":"status"}'
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._extract_dry_run_flag = lambda args: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: True

    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
                "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()

    governance_calls = {"build": 0}
    request_confirmation_calls = {"count": 0}

    class _Gov:
        def build_action_packet(self, *args, **kwargs):
            governance_calls["build"] += 1
            return object()

    api._governance = _Gov()

    async def _request_tool_confirmation(*args, **kwargs):
        request_confirmation_calls["count"] += 1

    api._request_tool_confirmation = _request_tool_confirmation

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert len(sent_payloads) == 1
    output_payload = json.loads(sent_payloads[0]["item"]["output"])
    assert output_payload["status"] == "suppressed_after_confirmation_timeout"
    assert transitions == []
    assert governance_calls["build"] == 0
    assert request_confirmation_calls["count"] == 0


def test_handle_function_call_allows_recall_after_timeout_debounce_window() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_timeout_allowed"}
    api.function_call_args = '{"query":"status"}'
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._extract_dry_run_flag = lambda args: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False

    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
                "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()

    from ai.governance import ActionPacket

    action = ActionPacket(
        id="call_timeout_allowed",
        tool_name="perform_research",
        tool_args={"query": "status"},
        tier=2,
        what="research",
        why="user asked",
        impact="none",
        rollback="n/a",
        alternatives=[],
        confidence=0.5,
        cost="moderate",
        risk_flags=[],
        requires_confirmation=False,
    )

    class _Gov:
        def build_action_packet(self, *args, **kwargs):
            return action

        def review(self, *args, **kwargs):
            return type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "confirm",
                },
            )()

    api._governance = _Gov()

    request_confirmation_calls = {"count": 0}

    async def _request_tool_confirmation(*args, **kwargs):
        request_confirmation_calls["count"] += 1

    api._request_tool_confirmation = _request_tool_confirmation

    asyncio.run(api.handle_function_call({}, _Ws()))

    assert request_confirmation_calls["count"] == 1
    assert transitions == []


def _build_confirmation_token(*, kind: str, pending_action=None, request=None, token_id: str = "tok_1"):
    return PendingConfirmationToken(
        id=token_id,
        kind=kind,
        tool_name="perform_research",
        request=request,
        pending_action=pending_action,
        created_at=0.0,
        expiry_ts=None,
        metadata={"approval_flow": True},
    )


def test_stale_drop_preserves_active_confirmation_prompt() -> None:
    api = _make_api_stub()
    api._response_done_serial = 7
    api._pending_confirmation_token = _build_confirmation_token(kind="research_permission")
    api._pending_action = None
    queued = {
        "origin": "assistant_message",
        "event": {
            "type": "response.create",
            "response": {
                "metadata": {
                    "origin": "assistant_message",
                    "approval_flow": "true",
                    "confirmation_token": "tok_1",
                }
            },
        },
        "enqueued_done_serial": 6,
    }

    assert api._is_stale_queued_response_create(queued) is False


def test_handle_response_done_fallback_reminder_without_transcript_callback() -> None:
    api = _make_api_stub()
    api._pending_confirmation_token = _build_confirmation_token(kind="tool_governance", pending_action=_build_pending_action())
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._active_response_confirmation_guarded = True
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False

    reminders: list[str] = []

    async def _send_confirmation_reminder(websocket, *, reason: str):
        reminders.append(reason)

    api._send_confirmation_reminder = _send_confirmation_reminder

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    assert reminders == ["response_done_fallback"]


def test_research_permission_parser_accepts_natural_language_yes_no_variants() -> None:
    api_yes = _make_api_stub()
    api_yes._pending_confirmation_token = _build_confirmation_token(
        kind="research_permission",
        request=type("Req", (), {"prompt": "status"})(),
    )
    dispatched: list[str] = []

    async def _dispatch(_request, _ws):
        dispatched.append("yes")

    api_yes._dispatch_research_request = _dispatch

    assert asyncio.run(api_yes._maybe_handle_research_permission_response("yes please", _Ws())) is True
    assert dispatched == ["yes"]
    assert api_yes._pending_confirmation_token is None

    api_no = _make_api_stub()
    api_no._pending_confirmation_token = _build_confirmation_token(
        kind="research_permission",
        request=type("Req", (), {"prompt": "status"})(),
    )
    denied: list[str] = []

    async def _assistant(*args, **kwargs):
        denied.append("no")

    api_no.send_assistant_message = _assistant
    assert asyncio.run(api_no._maybe_handle_research_permission_response("no thanks", _Ws())) is True
    assert denied == ["no"]
    assert api_no._pending_confirmation_token is None


def test_research_permission_yes_executes_without_second_governance_confirmation() -> None:
    api = _make_api_stub()
    api.function_call = {"name": "perform_research", "call_id": "call_research_once"}
    api.function_call_args = '{"query":"status"}'
    api._extract_dry_run_flag = lambda args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}

    created_tokens: list[str] = []

    def _create_confirmation_token(*args, **kwargs):
        created_tokens.append(kwargs["kind"])
        raise AssertionError("unexpected second confirmation token")

    api._create_confirmation_token = _create_confirmation_token

    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.IDLE,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()

    action = type(
        "Action",
        (),
        {
            "id": "call_research_once",
            "tool_name": "perform_research",
            "tool_args": {"query": "status"},
            "summary": lambda self: "summary",
        },
    )()
    api._governance = type(
        "Gov",
        (),
        {
            "build_action_packet": lambda *args, **kwargs: action,
            "review": lambda *args, **kwargs: type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "expensive read",
                },
            )(),
        },
    )()

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action

    api._pending_confirmation_token = _build_confirmation_token(
        kind="research_permission",
        request=type("Req", (), {"prompt": "status"})(),
    )

    async def _dispatch(_request, _ws):
        return None

    api._dispatch_research_request = _dispatch

    assert asyncio.run(api._maybe_handle_research_permission_response("yes", _Ws())) is True

    asyncio.run(api.handle_function_call({}, _Ws()))

    assert executed == ["done"]
    assert created_tokens == []
    assert transitions == [(OrchestrationPhase.ACT, "function_call perform_research")]


def test_tool_confirmation_token_closes_and_state_returns_idle_on_reject() -> None:
    api = _make_api_stub()
    pending = _build_pending_action()
    api._pending_confirmation_token = _build_confirmation_token(kind="tool_governance", pending_action=pending)
    api._pending_action = pending
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._awaiting_confirmation_completion = False
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.AWAITING_CONFIRMATION, "transition": lambda *args, **kwargs: None})()

    rejected: list[str] = []

    async def _reject(*args, **kwargs):
        rejected.append("rejected")

    api._reject_tool_call = _reject

    assert asyncio.run(api._maybe_handle_approval_response("cancel that", _Ws())) is True
    assert rejected == ["rejected"]
    assert api._pending_confirmation_token is None
    assert api._confirmation_state == ConfirmationState.IDLE
