"""Tests for confirmation-flow response suppression."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from unittest.mock import patch

from ai.governance import ActionPacket, build_normalized_idempotency_key
from ai.orchestration import OrchestrationPhase
from ai.realtime_api import ConfirmationState, PendingConfirmationToken, RealtimeAPI
from interaction import InteractionState
from services.research import OpenAIResearchService
from services.research import ResearchRequest


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
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: None,
        },
    )()
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
    api._pending_response_create_origins = deque()
    api._audio_playback_busy = False
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._response_done_serial = 0
    api._active_response_confirmation_guarded = False
    api._active_response_origin = "unknown"
    api._active_response_id = None
    api._confirmation_timeout_markers = {}
    api._confirmation_timeout_causes = {}
    api._recent_confirmation_outcomes = {}
    api._confirmation_timeout_debounce_window_s = 5.0
    api._approval_timeout_s = 30.0
    api._confirmation_awaiting_decision_timeout_s = 20.0
    api._research_permission_awaiting_decision_timeout_s = 60.0
    api._confirmation_late_decision_grace_s = 15.0
    api._confirmation_decision_expiry_grace_s = 1.5
    api._confirmation_unclear_max_reprompts = 1
    api._confirmation_reminder_interval_s = 6.0
    api._confirmation_reminder_max_count = 2
    api._confirmation_reminder_tracker = {}
    api._confirmation_token_created_at = None
    api._confirmation_last_activity_at = None
    api._confirmation_speech_active = False
    api._confirmation_asr_pending = False
    api._confirmation_pause_started_at = None
    api._confirmation_paused_accum_s = 0.0
    api._confirmation_timeout_check_log_interval_s = 1.0
    api._confirmation_timeout_check_last_logged_at = {}
    api._confirmation_timeout_check_last_pause_reason = {}
    api._confirmation_last_closed_token = None
    api._awaiting_confirmation_completion = False
    api._last_user_input_text = None
    api._stop_words = []
    api._tool_execution_disabled_until = 0.0
    api._pending_research_request = None
    api._prior_research_permission_marker = None
    api._prior_research_permission_grace_s = 8.0
    api._research_permission_outcome_ttl_s = 20.0
    api._research_permission_outcomes = {}
    api._research_suppressed_fingerprints = {}
    api._research_pending_call_ids = set()
    api._deferred_research_tool_call = None
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()
    api._presented_actions = set()
    api._pending_confirmation_token = None
    api._confirmation_state = ConfirmationState.IDLE
    api._last_executed_tool_call = None
    api._intent_ledger = {}
    api._intent_state_ttl_s = 300.0
    api._debug_governance_decisions = False
    api._active_utterance = None
    api.mic = _Mic()
    return api


def _configure_budget_storage(monkeypatch, tmp_path):
    import config.controller as config_controller
    from storage.controller import StorageController

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: type("_Cfg", (), {"get_config": lambda self: {"var_dir": str(var_dir), "log_dir": str(log_dir)}})(),
    )
    StorageController._instance = None
    return StorageController


def _wire_real_research_budget(monkeypatch, tmp_path, api: RealtimeAPI, *, daily_budget: int = 3):
    storage_controller_cls = _configure_budget_storage(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service = OpenAIResearchService(
        daily_budget=daily_budget,
        budget_state_file=str(tmp_path / "research_budget.json"),
        cache_dir=str(tmp_path / "research_cache"),
        firecrawl_enabled=False,
    )
    service._search_candidates = lambda *_args, **_kwargs: {
        "best_url": "",
        "candidate_urls": [],
        "sources": [],
        "search_summary": "stub search summary",
        "safety_notes": [],
    }
    api._research_service = service
    return storage_controller_cls


def _usage_count(storage_controller_cls) -> int:
    conn = storage_controller_cls.get_instance().conn
    return int(conn.execute("SELECT COUNT(*) FROM research_budget_usage").fetchone()[0])


def test_parse_confirmation_decision_accepts_go_ahead_phrasing() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    assert api._parse_confirmation_decision("Please go ahead.") == "yes"
    assert api._parse_confirmation_decision("go ahead and do it") == "yes"
    assert api._parse_confirmation_decision("proceed") == "yes"


def test_build_approval_prompt_has_two_sentences_with_reason_and_options() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._governance = type("Gov", (), {"describe_tool": lambda *_args, **_kwargs: {"dry_run_supported": True}})()
    action = ActionPacket(
        id="call_1",
        tool_name="perform_research",
        tool_args={"query": "weather"},
        tier=2,
        what="Run research",
        why="Need latest weather summary\nfor planning",
        impact="Read only",
        rollback="None",
        alternatives=["skip"],
        confidence=0.91,
        cost="low",
        risk_flags=[],
        requires_confirmation=True,
    )

    prompt = api._build_approval_prompt(
        action,
        action_summary="tool=perform_research tier=2 cost=low confidence=0.91 requires_confirmation=True",
        confirm_reason="expensive_read",
    )

    assert prompt == (
        "Action summary: tool=perform_research tier=2 cost=low confidence=0.91 requires_confirmation=True. "
        "Reason: expensive_read; options: Approve / Deny / Dry-run."
    )
    assert len([sentence for sentence in prompt.split(". ") if sentence]) == 2


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



def test_maybe_process_research_intent_emits_single_permission_prompt_for_duplicate_request(monkeypatch, tmp_path) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
    api._pending_research_request = None
    api._pending_action = None
    api._research_mode = "ask"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()
    api._clip_text = lambda value, limit=80: value[:limit]

    prompted: list[str] = []

    async def _send_assistant_message(message, *args, **kwargs):
        prompted.append(message)

    api.send_assistant_message = _send_assistant_message

    try:
        usage_before = _usage_count(storage_controller_cls)

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
        usage_after = _usage_count(storage_controller_cls)

        assert first is True
        assert second is True
        assert len(prompted) == 1
        assert usage_after == usage_before
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None


def test_maybe_process_research_intent_ignores_duplicate_when_confirmation_token_pending_without_usage_change(
    monkeypatch,
    tmp_path,
) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
    api._pending_research_request = None
    api._pending_action = None
    api._research_mode = "ask"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()
    api._clip_text = lambda value, limit=80: value[:limit]
    api._pending_confirmation_token = _build_confirmation_token(
        kind="research_permission",
        request=ResearchRequest(prompt="search the web for market updates", context={"source": "input_audio_transcription"}),
    )

    prompted: list[str] = []

    async def _send_assistant_message(message, *args, **kwargs):
        prompted.append(message)

    api.send_assistant_message = _send_assistant_message

    try:
        usage_before = _usage_count(storage_controller_cls)

        result = asyncio.run(
            api._maybe_process_research_intent(
                "search the web for market updates",
                _Ws(),
                source="input_audio_transcription",
            )
        )

        usage_after = _usage_count(storage_controller_cls)

        assert result is True
        assert prompted == []
        assert usage_after == usage_before
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None


def test_request_tool_confirmation_sends_single_spoken_prompt() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.IDLE, "transition": lambda *args, **kwargs: None})()
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api.function_call = "perform_research"
    api.function_call_args = '{"query":"x"}'
    api._governance = type("Gov", (), {"describe_tool": lambda *args, **kwargs: {"tier": 2}})()
    api._build_approval_prompt = lambda action, **_kwargs: "Need approval"

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


def _build_pending_action(*, expiry_ts=None, idempotency_key: str | None = "idem-pending"):
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
        idempotency_key=idempotency_key,
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
    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("executed")

    api._execute_action = _execute_action

    sent_messages: list[str] = []

    async def _send_assistant_message(message, *args, **kwargs):
        sent_messages.append(message)

    api.send_assistant_message = _send_assistant_message

    caplog.set_level(logging.INFO)
    consumed = asyncio.run(api._maybe_handle_approval_response("anything", api.websocket))

    assert consumed is True
    assert sent_messages
    assert sent_messages[0].startswith("No action taken.")
    assert executed == []
    assert api._pending_action is None
    assert transitions == [(OrchestrationPhase.IDLE, "confirmation timeout")]


def test_maybe_handle_approval_response_pauses_expiry_while_asr_pending() -> None:
    api = _make_api_stub()
    api._pending_action = _build_pending_action(expiry_ts=10.0)
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._confirmation_asr_pending = True
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action

    with patch("ai.realtime_api.time.monotonic", return_value=11.0):
        consumed = asyncio.run(api._maybe_handle_approval_response("yes", api.websocket))

    assert consumed is True
    assert executed == ["done"]


def test_maybe_handle_approval_response_yes_wins_within_expiry_grace_window() -> None:
    api = _make_api_stub()
    api._pending_action = _build_pending_action(expiry_ts=10.0)
    api._awaiting_confirmation_completion = False
    api._presented_actions = set()
    api._confirmation_decision_expiry_grace_s = 1.5
    api._handle_stop_word = lambda *args, **kwargs: asyncio.sleep(0, result=False)
    api._tool_execution_cooldown_remaining = lambda: 0.0

    executed: list[str] = []
    sent_messages: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    async def _send_assistant_message(message, *args, **kwargs):
        sent_messages.append(message)

    api._execute_action = _execute_action
    api.send_assistant_message = _send_assistant_message

    with patch("ai.realtime_api.time.monotonic", return_value=11.0):
        consumed = asyncio.run(api._maybe_handle_approval_response("Yes.", api.websocket))

    assert consumed is True
    assert executed == ["done"]

    with patch("ai.realtime_api.time.monotonic", return_value=11.1):
        follow_up = asyncio.run(api._maybe_handle_approval_response("", api.websocket))

    assert follow_up is False
    assert sent_messages == []


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
    assert str(output_payload["message"]).startswith("No action taken.")
    assert output_payload["no_op"] == {"executed": False, "category": "suppression"}
    assert api.function_call is None
    assert api.function_call_args == ""


def test_handle_function_call_suppresses_research_while_research_permission_pending(monkeypatch, tmp_path) -> None:
    api = _make_api_stub()
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
    api._pending_action = None
    api._pending_research_request = object()
    api.function_call = {"name": "perform_research", "call_id": "call_research_permission"}
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

    with patch("ai.realtime_api.logger.info") as mock_info:
        budget_before = api._research_budget_remaining()
        usage_before = _usage_count(storage_controller_cls)

        asyncio.run(api.handle_function_call({}, _SendWs()))

        budget_after = api._research_budget_remaining()
        usage_after = _usage_count(storage_controller_cls)

    logged = [call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args]

    try:
        assert transitions == []
        assert confirmation_prompts["count"] == 0
        assert governance_calls == {"build": 0, "review": 0}
        assert len(sent_payloads) == 1
        output_payload = json.loads(sent_payloads[0]["item"]["output"])
        assert output_payload["status"] == "waiting_for_permission"
        assert str(output_payload["message"]).startswith("No action taken.")
        assert any("NO_OP_EVENT" in line and "outcome=waiting_for_permission" in line and "reason=permission_pending" in line for line in logged)
        assert budget_after == budget_before
        assert usage_after == usage_before
        assert api.function_call is None
        assert api.function_call_args == ""
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None


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
                        "tool_args": {"query": "status"},
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


def test_handle_function_call_duplicate_logs_skip_without_execution(caplog, monkeypatch, tmp_path) -> None:
    api = _make_api_stub()
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
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

    try:
        budget_before = api._research_budget_remaining()
        usage_before = _usage_count(storage_controller_cls)

        caplog.set_level(logging.INFO)
        asyncio.run(api.handle_function_call({}, _SendWs()))

        budget_after = api._research_budget_remaining()
        usage_after = _usage_count(storage_controller_cls)

        assert len(sent_payloads) == 1
        output_payload = json.loads(sent_payloads[0]["item"]["output"])
        assert output_payload["status"] == "redundant"
        assert budget_after == budget_before
        assert usage_after == usage_before
        assert executed == []
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None


def test_handle_function_call_executed_research_decrements_budget_and_records_usage(monkeypatch, tmp_path) -> None:
    api = _make_api_stub()
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_exec_budget"}
    api.function_call_args = '{"query":"status"}'
    api._extract_dry_run_flag = lambda _args: False
    api._is_duplicate_tool_call = lambda *_args, **_kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *_args, **_kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda _action: {"valid": True}
    api._governance = type(
        "Gov",
        (),
        {
            "build_action_packet": lambda *_args, **_kwargs: type(
                "Action",
                (),
                {"id": "call_exec_budget", "tool_name": "perform_research", "tool_args": {"query": "status"}, "tier": 0},
            )(),
            "review": lambda *_args, **_kwargs: type(
                "Decision",
                (),
                {"approved": True, "needs_confirmation": False, "status": "approved", "reason": "ok"},
            )(),
            "record_execution": lambda *_args, **_kwargs: None,
        },
    )()

    async def _execute_action(action, *_args, **_kwargs):
        request = ResearchRequest(prompt=action.tool_args["query"], context={"source": "text_message"})
        api._research_service.request_research(request)

    api._execute_action = _execute_action

    try:
        budget_before = api._research_budget_remaining()
        usage_before = _usage_count(storage_controller_cls)

        asyncio.run(api.handle_function_call({}, _Ws()))

        budget_after = api._research_budget_remaining()
        usage_after = _usage_count(storage_controller_cls)

        assert budget_after == budget_before - 1
        assert usage_after == usage_before + 1
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None


def test_handle_function_call_logs_tool_and_call_id_with_parse_status(monkeypatch) -> None:
    api = _make_api_stub()
    api.function_call = {"name": "perform_research", "call_id": "call_123"}
    api.function_call_args = '{"query":'
    api._pending_action = type("Pending", (), {"action": type("Action", (), {"tool_name": "noop"})()})()

    async def _send_awaiting_confirmation_output(*args, **kwargs):
        return None

    monkeypatch.setattr(api, "_send_awaiting_confirmation_output", _send_awaiting_confirmation_output)

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api.handle_function_call({}, _Ws()))

    logged = [call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args]
    assert any(
        "Function call received | tool=perform_research call_id=call_123 args_parsed=False" in line
        for line in logged
    )




def test_normalize_confirmation_decision_preserves_confirmation_required_defaults() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    decision = type(
        "Decision",
        (),
        {
            "status": "needs_confirmation",
            "reason": "expensive_read",
            "needs_confirmation": True,
            "confirm_required": False,
            "confirm_reason": None,
            "confirm_prompt": None,
            "idempotency_key": None,
            "cooldown_seconds": 0.0,
            "dry_run_supported": False,
        },
    )()

    normalized = api._normalize_confirmation_decision(
        "perform_research",
        {"query": "status update"},
        decision,
        {"privacy_flag": False},
    )

    assert normalized.status == "needs_confirmation"
    assert normalized.action_summary == "tool=perform_research requires confirmation."
    assert normalized.needs_confirmation is True
    assert normalized.confirm_required is True
    assert normalized.confirm_reason == "expensive_read"
    assert normalized.confirm_prompt is None
    assert normalized.idempotency_key == build_normalized_idempotency_key(
        "perform_research",
        {"query": "status update"},
    )


def test_normalize_confirmation_decision_keeps_explicit_confirmation_fields() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    decision = type(
        "Decision",
        (),
        {
            "status": "needs_confirmation",
            "reason": "within_bounds",
            "needs_confirmation": True,
            "confirm_required": True,
            "confirm_reason": "autonomy level requires confirmation",
            "confirm_prompt": "Confirm execution?",
            "idempotency_key": "fixed-key",
            "cooldown_seconds": 12.0,
            "dry_run_supported": True,
        },
    )()

    normalized = api._normalize_confirmation_decision(
        "perform_research",
        {"query": "status"},
        decision,
        {"privacy_flag": True},
    )

    assert normalized.action_summary == "tool=perform_research requires confirmation."
    assert normalized.confirm_required is True
    assert normalized.confirm_reason == "autonomy_level_requires_confirmation"
    assert normalized.confirm_prompt == "Confirm execution?"
    assert normalized.idempotency_key == "fixed-key"
    assert normalized.cooldown_seconds == 12.0
    assert normalized.dry_run_supported is True

def test_handle_function_call_logs_consolidated_governance_review_summary() -> None:
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

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api.handle_function_call({}, _Ws()))

    logged = [call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args]
    assert any(
        "Governance review summary | call_id=call_gov_1 tool=perform_research "
        "initial_status=needs_confirmation initial_reason=expensive_read "
        "confirm_required=True confirm_reason=expensive_read idempotency_key=perform_research:" in line
        and "prior_permission_override=False final_execution_decision=request_confirmation" in line
        for line in logged
    )



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
    assert str(output_payload["message"]).startswith("No action taken.")
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


def test_tool_confirmation_reask_after_deny_within_cooldown_suppresses_prompt() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._extract_dry_run_flag = lambda _args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda _action: {"valid": True}
    api._intent_denial_cooldown_s = 0.0

    prompt_calls = {"count": 0}

    async def _request_tool_confirmation(*_args, **_kwargs):
        prompt_calls["count"] += 1

    async def _assistant(*_args, **_kwargs):
        return None

    api._request_tool_confirmation = _request_tool_confirmation
    api.send_assistant_message = _assistant

    class _Gov:
        def build_action_packet(self, function_name, call_id, args, **_kwargs):
            return ActionPacket(
                id=call_id,
                tool_name=function_name,
                tool_args=args,
                tier=2,
                what="research",
                why="test",
                impact="none",
                rollback="n/a",
                alternatives=[],
                confidence=0.8,
                cost="low",
                risk_flags=[],
                requires_confirmation=True,
            )

        def decide_tool_call(self, *_args, **_kwargs):
            return type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "confirm",
                    "confirm_required": True,
                    "confirm_prompt": None,
                    "confirm_reason": "confirm",
                    "idempotency_key": "idem-1",
                    "cooldown_seconds": 30.0,
                    "dry_run_supported": True,
                },
            )()

        def describe_tool(self, *_args, **_kwargs):
            return {}

    api._governance = _Gov()

    api.function_call = {"name": "perform_research", "call_id": "call_1"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert prompt_calls["count"] == 1
    assert api._pending_confirmation_token is not None

    asyncio.run(api._maybe_handle_approval_response("no", _Ws()))

    api.function_call = {"name": "perform_research", "call_id": "call_2"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert prompt_calls["count"] == 1


def test_tool_confirmation_reask_after_expired_within_cooldown_suppresses_prompt() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._extract_dry_run_flag = lambda _args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda _action: {"valid": True}
    api._intent_denial_cooldown_s = 0.0

    prompt_calls = {"count": 0}

    async def _request_tool_confirmation(*_args, **_kwargs):
        prompt_calls["count"] += 1

    async def _assistant(*_args, **_kwargs):
        return None

    api._request_tool_confirmation = _request_tool_confirmation
    api.send_assistant_message = _assistant

    class _Gov:
        def build_action_packet(self, function_name, call_id, args, **_kwargs):
            return ActionPacket(
                id=call_id,
                tool_name=function_name,
                tool_args=args,
                tier=2,
                what="research",
                why="test",
                impact="none",
                rollback="n/a",
                alternatives=[],
                confidence=0.8,
                cost="low",
                risk_flags=[],
                requires_confirmation=True,
            )

        def decide_tool_call(self, *_args, **_kwargs):
            return type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "confirm",
                    "confirm_required": True,
                    "confirm_prompt": None,
                    "confirm_reason": "confirm",
                    "idempotency_key": "idem-2",
                    "cooldown_seconds": 30.0,
                    "dry_run_supported": True,
                },
            )()

        def describe_tool(self, *_args, **_kwargs):
            return {}

    api._governance = _Gov()

    api.function_call = {"name": "perform_research", "call_id": "call_1"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert prompt_calls["count"] == 1
    pending = api._pending_confirmation_token.pending_action
    assert pending is not None
    pending.action.expiry_ts = 0.0

    with patch("ai.realtime_api.time.monotonic", return_value=100.0):
        assert asyncio.run(api._maybe_handle_approval_response("yes", _Ws())) is True
    api._recent_confirmation_outcomes["idem-2"]["timestamp"] = time.monotonic()

    api.function_call = {"name": "perform_research", "call_id": "call_2"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert prompt_calls["count"] == 1


def test_tool_confirmation_reask_after_cooldown_allows_prompt() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._extract_dry_run_flag = lambda _args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda _action: {"valid": True}
    api._intent_denial_cooldown_s = 0.0

    prompt_calls = {"count": 0}

    async def _request_tool_confirmation(*_args, **_kwargs):
        prompt_calls["count"] += 1

    async def _assistant(*_args, **_kwargs):
        return None

    api._request_tool_confirmation = _request_tool_confirmation
    api.send_assistant_message = _assistant

    class _Gov:
        def build_action_packet(self, function_name, call_id, args, **_kwargs):
            return ActionPacket(
                id=call_id,
                tool_name=function_name,
                tool_args=args,
                tier=2,
                what="research",
                why="test",
                impact="none",
                rollback="n/a",
                alternatives=[],
                confidence=0.8,
                cost="low",
                risk_flags=[],
                requires_confirmation=True,
            )

        def decide_tool_call(self, *_args, **_kwargs):
            return type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "confirm",
                    "confirm_required": True,
                    "confirm_prompt": None,
                    "confirm_reason": "confirm",
                    "idempotency_key": "idem-3",
                    "cooldown_seconds": 10.0,
                    "dry_run_supported": True,
                },
            )()

        def describe_tool(self, *_args, **_kwargs):
            return {}

    api._governance = _Gov()

    api.function_call = {"name": "perform_research", "call_id": "call_1"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, _Ws()))
    assert prompt_calls["count"] == 1

    asyncio.run(api._maybe_handle_approval_response("no", _Ws()))

    api._recent_confirmation_outcomes["idem-3"]["timestamp"] = time.monotonic() - 20.0
    api.function_call = {"name": "perform_research", "call_id": "call_2"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, _Ws()))

    assert prompt_calls["count"] == 2

def test_intent_guard_prefers_idempotency_key_for_semantically_equivalent_args() -> None:
    api = _make_api_stub()
    api._intent_denial_cooldown_s = 30.0

    api._record_intent_state(
        "perform_research",
        {"query": "market update"},
        "denied",
        idempotency_key="same-key",
    )

    blocked, status, _message = api._evaluate_intent_guard(
        "perform_research",
        {"query": "  market   update  "},
        phase="confirmation_prompt",
        idempotency_key="same-key",
    )

    assert blocked is True
    assert status == "blocked_recent_denial"


def test_intent_guard_equivalent_args_normalization_blocks_retry() -> None:
    api = _make_api_stub()
    api._intent_denial_cooldown_s = 30.0

    api._record_intent_state(
        "perform_research",
        {"query": "status   report"},
        "timeout",
        idempotency_key="idem-timeout",
    )

    blocked, status, _message = api._evaluate_intent_guard(
        "perform_research",
        {"query": "status report"},
        phase="confirmation_prompt",
        idempotency_key="idem-timeout",
    )

    assert blocked is True
    assert status == "blocked_recent_timeout"


def test_intent_guard_changed_args_creates_new_intent() -> None:
    api = _make_api_stub()
    api._intent_denial_cooldown_s = 30.0

    api._record_intent_state(
        "perform_research",
        {"query": "status"},
        "denied",
        idempotency_key="idem-status",
    )

    blocked, status, _message = api._evaluate_intent_guard(
        "perform_research",
        {"query": "different topic"},
        phase="confirmation_prompt",
        idempotency_key="idem-different",
    )

    assert blocked is False
    assert status is None


def test_tool_confirmation_retry_after_denial_returns_noop_without_prompt() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._extract_dry_run_flag = lambda _args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda _action: {"valid": True}
    api._intent_denial_cooldown_s = 30.0

    prompts: list[str] = []

    class _CaptureWs:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, payload: str) -> None:
            self.messages.append(json.loads(payload))

    websocket = _CaptureWs()
    api.websocket = websocket

    async def _request_tool_confirmation(*_args, **_kwargs):
        prompts.append("prompted")

    async def _assistant(message: str, _ws):
        prompts.append(message)

    api._request_tool_confirmation = _request_tool_confirmation
    api.send_assistant_message = _assistant

    class _Gov:
        def build_action_packet(self, function_name, call_id, args, **_kwargs):
            return ActionPacket(
                id=call_id,
                tool_name=function_name,
                tool_args=args,
                tier=2,
                what="research",
                why="test",
                impact="none",
                rollback="n/a",
                alternatives=[],
                confidence=0.8,
                cost="low",
                risk_flags=[],
                requires_confirmation=True,
            )

        def decide_tool_call(self, *_args, **_kwargs):
            return type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "confirm",
                    "confirm_required": True,
                    "confirm_prompt": None,
                    "confirm_reason": "confirm",
                    "idempotency_key": "idem-noop",
                    "cooldown_seconds": 30.0,
                    "dry_run_supported": True,
                },
            )()

        def describe_tool(self, *_args, **_kwargs):
            return {}

    api._governance = _Gov()

    api.function_call = {"name": "perform_research", "call_id": "call_1"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, websocket))
    assert prompts == ["prompted"]

    asyncio.run(api._maybe_handle_approval_response("no", websocket))

    prompts.clear()
    api.function_call = {"name": "perform_research", "call_id": "call_2"}
    api.function_call_args = '{"query":"status"}'
    asyncio.run(api.handle_function_call({}, websocket))

    assert prompts == []
    function_outputs = [
        event
        for event in websocket.messages
        if event.get("type") == "conversation.item.create"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "function_call_output"
    ]
    payload = json.loads(function_outputs[-1]["item"]["output"])
    assert payload["status"] == "suppressed_recent_confirmation_outcome"
    assert payload["no_op"]["executed"] is False



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




def test_handle_function_call_replaces_pending_confirmation_for_new_intent_same_tool() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._pending_confirmation_token = None
    api._extract_dry_run_flag = lambda args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}

    request_confirmation_calls: list[dict[str, object]] = []
    executed_calls: list[str] = []
    assistant_messages: list[str] = []

    async def _request_tool_confirmation(action, *_args, **_kwargs):
        request_confirmation_calls.append(action.tool_args)

    async def _execute_action(*_args, **_kwargs):
        executed_calls.append("executed")

    api._request_tool_confirmation = _request_tool_confirmation
    api._execute_action = _execute_action

    async def _send_assistant_message(message, *_args, **_kwargs):
        assistant_messages.append(message)

    api.send_assistant_message = _send_assistant_message

    class _Gov:
        def build_action_packet(self, function_name, call_id, args, **_kwargs):
            return ActionPacket(
                id=call_id,
                tool_name=function_name,
                tool_args=args,
                tier=2,
                what="update profile",
                why="test",
                impact="profile update",
                rollback="manual",
                alternatives=["skip"],
                confidence=0.9,
                cost="low",
                risk_flags=[],
                requires_confirmation=True,
            )

        def review(self, *_args, **_kwargs):
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

    api.function_call = {"name": "update_user_profile", "call_id": "call_name_a"}
    api.function_call_args = '{"name":"A"}'
    asyncio.run(api.handle_function_call({}, _Ws()))

    first_token_id = api._pending_confirmation_token.id

    api.function_call = {"name": "update_user_profile", "call_id": "call_name_b"}
    api.function_call_args = '{"name":"B"}'

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api.handle_function_call({}, _Ws()))

    logged = [call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args]

    assert api._pending_confirmation_token is not None
    assert api._pending_confirmation_token.id != first_token_id
    assert api._pending_confirmation_token.pending_action is not None
    assert api._pending_confirmation_token.pending_action.action.tool_args["name"] == "B"
    assert request_confirmation_calls == [{"name": "A"}, {"name": "B"}]
    assert executed_calls == []
    assert assistant_messages == ["No action taken."]
    assert any("NO_OP_EVENT" in line and "outcome=cancelled" in line and "reason=replaced_by_new_intent" in line for line in logged)
    assert any("CONFIRMATION_TOKEN_CLOSED" in line and "outcome=replaced" in line for line in logged)



def test_pending_confirmation_suppression_does_not_emit_noop_user_text_while_token_active() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._pending_confirmation_token = PendingConfirmationToken(
        id="tok_active",
        kind="research_permission",
        tool_name="perform_research",
        request=ResearchRequest(prompt="status"),
        pending_action=None,
        created_at=0.0,
        expiry_ts=None,
        metadata={"approval_flow": True},
    )
    api.function_call = {"name": "perform_research", "call_id": "call_waiting"}
    api.function_call_args = '{"query":"status"}'

    assistant_messages: list[str] = []
    executed_calls: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        assistant_messages.append(message)

    async def _execute(*_args, **_kwargs):
        executed_calls.append("executed")

    api.send_assistant_message = _assistant
    api._execute_action = _execute

    class _SendWs:
        async def send(self, _payload: str) -> None:
            return None

    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert executed_calls == []
    assert assistant_messages == []


def test_duplicate_suppression_emits_noop_user_text_when_no_token_active() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._pending_confirmation_token = None
    api._is_duplicate_tool_call = lambda *_args, **_kwargs: True
    api._is_suppressed_after_confirmation_timeout = lambda *_args, **_kwargs: False
    api.function_call = {"name": "perform_research", "call_id": "call_duplicate"}
    api.function_call_args = '{"query":"status"}'

    assistant_messages: list[str] = []
    executed_calls: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        assistant_messages.append(message)

    async def _execute(*_args, **_kwargs):
        executed_calls.append("executed")

    api.send_assistant_message = _assistant
    api._execute_action = _execute

    class _SendWs:
        async def send(self, _payload: str) -> None:
            return None

    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert executed_calls == []
    assert assistant_messages == ["No action taken."]


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
    token = _build_confirmation_token(kind="tool_governance", pending_action=_build_pending_action())
    token.metadata = {
        "approval_flow": True,
        "max_reminders": 1,
        "reminder_schedule_seconds": [0.0],
    }
    api._pending_confirmation_token = token
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


def test_handle_response_done_fallback_suppressed_when_budget_exhausted() -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="tool_governance", pending_action=_build_pending_action(idempotency_key="idem-fallback"))
    token.metadata = {
        "approval_flow": True,
        "max_reminders": 1,
        "reminder_schedule_seconds": [0.0],
    }
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._active_response_confirmation_guarded = True
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._confirmation_reminder_tracker["idem-fallback"] = {"count": 1, "last_sent_at": 0.0}

    reminders: list[str] = []

    async def _send_confirmation_reminder(websocket, *, reason: str):
        reminders.append(reason)

    api._send_confirmation_reminder = _send_confirmation_reminder

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    assert reminders == []


def test_send_confirmation_reminder_respects_interval_and_max_count() -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="tool_governance", pending_action=_build_pending_action())
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_reminder_interval_s = 5.0
    api._confirmation_reminder_max_count = 2

    sent_messages: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        sent_messages.append(message)

    api.send_assistant_message = _assistant

    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="test_repeat_1"))
    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="test_repeat_2"))

    reminder_key = api._confirmation_reminder_key(token)
    assert reminder_key is not None
    api._confirmation_reminder_tracker[reminder_key]["last_sent_at"] = 0.0

    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="test_repeat_3"))
    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="test_repeat_4"))

    assert sent_messages == [
        "Please reply with: yes or no.",
        "Please reply with: yes or no.",
    ]


def test_non_decision_reminders_are_bounded_by_token_schedule() -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="tool_governance", pending_action=_build_pending_action(idempotency_key="idem-non-decision"))
    token.metadata = {
        "approval_flow": True,
        "max_reminders": 2,
        "reminder_schedule_seconds": [0.0, 0.0],
    }
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_reminder_interval_s = 0.0

    sent_messages: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        sent_messages.append(message)

    api.send_assistant_message = _assistant

    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="non_decision_input"))
    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="non_decision_input"))
    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="non_decision_input"))

    assert len(sent_messages) == 2


def test_same_pending_idempotency_key_does_not_duplicate_reminder_burst() -> None:
    api = _make_api_stub()
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_reminder_interval_s = 0.0

    first_token = _build_confirmation_token(
        kind="tool_governance",
        pending_action=_build_pending_action(idempotency_key="idem-shared"),
        token_id="tok_first",
    )
    first_token.metadata = {
        "approval_flow": True,
        "max_reminders": 1,
        "reminder_schedule_seconds": [0.0],
    }
    api._pending_confirmation_token = first_token

    sent_messages: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        sent_messages.append(message)

    api.send_assistant_message = _assistant

    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="non_decision_input"))

    replacement_token = _build_confirmation_token(
        kind="tool_governance",
        pending_action=_build_pending_action(idempotency_key="idem-shared"),
        token_id="tok_second",
    )
    replacement_token.metadata = {
        "approval_flow": True,
        "max_reminders": 1,
        "reminder_schedule_seconds": [0.0],
    }
    api._pending_confirmation_token = replacement_token

    asyncio.run(api._send_confirmation_reminder(api.websocket, reason="non_decision_input"))

    assert len(sent_messages) == 1


def test_confirmation_reminder_logs_sent_and_suppressed() -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="tool_governance", pending_action=_build_pending_action())
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_reminder_interval_s = 10.0
    api._confirmation_reminder_max_count = 1

    async def _assistant(*_args, **_kwargs):
        return None

    api.send_assistant_message = _assistant

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api._send_confirmation_reminder(api.websocket, reason="structured_1"))
        asyncio.run(api._send_confirmation_reminder(api.websocket, reason="structured_2"))

    logged = [call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args]
    assert any("CONFIRMATION_REMINDER_SENT" in line for line in logged)
    assert any("CONFIRMATION_REMINDER_SUPPRESSED" in line for line in logged)


def test_parse_confirmation_decision_supports_explicit_cancel_terms() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    assert api._parse_confirmation_decision("cancel") == "cancel"
    assert api._parse_confirmation_decision("never mind") == "cancel"
    assert api._parse_confirmation_decision("ignore that") == "cancel"


def test_research_permission_unclear_reprompts_once_then_cancels() -> None:
    api = _make_api_stub()
    request = type("Req", (), {"prompt": "status"})()
    token = _build_confirmation_token(kind="research_permission", request=request)
    token.metadata = {"approval_flow": True, "domains": ["example.com"], "domains_known": True}
    api._pending_confirmation_token = token

    messages: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        messages.append(message)

    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()
    api.send_assistant_message = _assistant

    assert asyncio.run(api._maybe_handle_research_permission_response("Thanks.", _Ws())) is True
    assert messages == ["Please reply yes, no, or cancel for example.com."]
    assert api._pending_confirmation_token is not None

    assert asyncio.run(api._maybe_handle_research_permission_response("Thanks.", _Ws())) is True
    assert api._pending_confirmation_token is None
    assert messages[-1].startswith("I couldn't confirm research permission")
    assert transitions == [(OrchestrationPhase.IDLE, "research permission unclear cancel")]


def test_awaiting_confirmation_timeout_clears_token_and_transitions_idle() -> None:
    api = _make_api_stub()
    request = type("Req", (), {"prompt": "status"})()
    token = _build_confirmation_token(kind="research_permission", request=request)
    token.metadata = {"approval_flow": True, "awaiting_decision_since": 0.0}
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_awaiting_decision_timeout_s = 20.0
    api._research_permission_awaiting_decision_timeout_s = 20.0

    transitions: list[tuple[object, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda *args, **kwargs: transitions.append((args[1], kwargs.get("reason"))),
        },
    )()

    messages: list[str] = []
    executed_calls: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        messages.append(message)

    async def _execute(*_args, **_kwargs):
        executed_calls.append("executed")

    api.send_assistant_message = _assistant
    api._execute_action = _execute

    with patch("ai.realtime_api.time.monotonic", return_value=25.0):
        asyncio.run(api.handle_event({"type": "rate_limits.updated", "rate_limits": []}, _Ws()))

    assert api._pending_confirmation_token is None
    assert api._confirmation_state == ConfirmationState.IDLE
    assert transitions == [(OrchestrationPhase.IDLE, "confirmation timeout")]
    assert executed_calls == []
    assert messages == ["No action taken."]


def test_process_ws_messages_recovers_from_event_handler_exception_and_continues(
    monkeypatch,
) -> None:
    api = _make_api_stub()
    transitions: list[tuple[OrchestrationPhase, str | None]] = []
    api.orchestration_state = type(
        "S",
        (),
        {
            "phase": OrchestrationPhase.AWAITING_CONFIRMATION,
            "transition": lambda _self, phase, reason=None: transitions.append((phase, reason)),
        },
    )()
    api._pending_confirmation_token = PendingConfirmationToken(
        id="token_123",
        kind="research_budget",
        tool_name="perform_research",
        request=ResearchRequest(prompt="weather"),
        pending_action=None,
        created_at=0.0,
        expiry_ts=None,
    )
    api._active_response_confirmation_guarded = True
    api._log_user_transcripts_enabled = False

    async def _false(*_args, **_kwargs) -> bool:
        return False

    async def _raise_budget(*_args, **_kwargs) -> bool:
        raise RuntimeError("boom")

    sent_messages: list[str] = []

    async def _send_assistant_message(message: str, _websocket, **_kwargs) -> None:
        sent_messages.append(message)

    api._maybe_handle_confirmation_decision_timeout = _false
    api._maybe_handle_approval_response = _false
    api._handle_stop_word = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _raise_budget
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_process_research_intent = _false
    api.send_assistant_message = _send_assistant_message

    class _ConnectionClosed(Exception):
        pass

    class _LoopWs:
        def __init__(self) -> None:
            self._messages = [
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "yes",
                    }
                ),
                json.dumps(
                    {
                        "type": "rate_limits.updated",
                        "rate_limits": [{"name": "requests", "remaining": 9}],
                    }
                ),
            ]
            self._index = 0

        async def recv(self) -> str:
            if self._index >= len(self._messages):
                raise _ConnectionClosed()
            payload = self._messages[self._index]
            self._index += 1
            return payload

    monkeypatch.setattr("ai.realtime_api._require_websockets", lambda: object())
    monkeypatch.setattr(
        "ai.realtime_api._resolve_websocket_exceptions",
        lambda _websockets: (_ConnectionClosed, _ConnectionClosed),
    )

    with (
        patch("ai.realtime_api.logger.exception") as mock_exception,
        patch("ai.realtime_api.logger.error") as mock_error,
    ):
        asyncio.run(api.process_ws_messages(_LoopWs()))

    assert mock_exception.call_count == 1
    assert mock_exception.call_args.args[0].startswith("Unhandled event handler exception event=%s")
    assert mock_error.call_args_list[0].args == (
        "EVENT_HANDLER_ERROR event=%s",
        "conversation.item.input_audio_transcription.completed",
    )
    assert api.rate_limits == {"requests": {"name": "requests", "remaining": 9}}
    assert sent_messages == [
        "Sorry—I hit an internal error while handling that confirmation/research request, so I cancelled it. Please try again.",
    ]
    assert api._active_response_confirmation_guarded is False
    assert transitions == [(OrchestrationPhase.IDLE, "event handler error recovery")]


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


def test_accepted_research_permission_short_circuits_governance_confirmation_for_same_fingerprint() -> None:
    api = _make_api_stub()
    api.function_call = {"name": "perform_research", "call_id": "call_research_short_circuit"}
    api.function_call_args = '{"query":"Status", "context": {"source": "user_text"}}'
    api._extract_dry_run_flag = lambda args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}

    created_tokens: list[str] = []

    def _create_confirmation_token(*args, **kwargs):
        created_tokens.append(kwargs["kind"])
        raise AssertionError("tool_governance token should not be created for approved intent fingerprint")

    api._create_confirmation_token = _create_confirmation_token

    action = type(
        "Action",
        (),
        {
            "id": "call_research_short_circuit",
            "tool_name": "perform_research",
            "tool_args": {"query": "Status", "context": {"source": "user_text"}},
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
        request=type("Req", (), {"prompt": "Status", "context": {"source": "user_text"}})(),
    )
    api._dispatch_research_request = lambda *_args, **_kwargs: asyncio.sleep(0)

    assert asyncio.run(api._maybe_handle_research_permission_response("yes", _Ws())) is True

    asyncio.run(api.handle_function_call({}, _Ws()))

    assert executed == ["done"]
    assert created_tokens == []


def test_accepted_research_permission_short_circuits_governance_confirmation_when_function_call_omits_source() -> None:
    api = _make_api_stub()
    api.function_call = {"name": "perform_research", "call_id": "call_research_without_source"}
    api.function_call_args = '{"query":"Status"}'
    api._extract_dry_run_flag = lambda args: False
    api._is_duplicate_tool_call = lambda *args, **kwargs: False
    api._is_suppressed_after_confirmation_timeout = lambda *args, **kwargs: False
    api._tool_execution_cooldown_remaining = lambda: 0.0
    api._stage_action = lambda action: {"valid": True}

    created_tokens: list[str] = []

    def _create_confirmation_token(*args, **kwargs):
        created_tokens.append(kwargs["kind"])
        raise AssertionError("tool_governance token should not be created for approved intent fingerprint")

    api._create_confirmation_token = _create_confirmation_token

    governance_calls = {"build": 0, "review": 0}
    api._debug_governance_decisions = True

    action = type(
        "Action",
        (),
        {
            "id": "call_research_without_source",
            "tool_name": "perform_research",
            "tool_args": {"query": "Status"},
            "summary": lambda self: "summary",
        },
    )()
    class _Gov:
        def build_action_packet(self, *args, **kwargs):
            governance_calls["build"] += 1
            return action

        def review(self, *args, **kwargs):
            governance_calls["review"] += 1
            return type(
                "Decision",
                (),
                {
                    "approved": False,
                    "needs_confirmation": True,
                    "status": "needs_confirmation",
                    "reason": "expensive read",
                },
            )()

    api._governance = _Gov()

    executed: list[str] = []

    async def _execute_action(*args, **kwargs):
        executed.append("done")

    api._execute_action = _execute_action
    api._pending_confirmation_token = _build_confirmation_token(
        kind="research_permission",
        request=type(
            "Req",
            (),
            {"prompt": "Status", "context": {"source": "input_audio_transcription"}},
        )(),
    )
    api._dispatch_research_request = lambda *_args, **_kwargs: asyncio.sleep(0)

    assert asyncio.run(api._maybe_handle_research_permission_response("yes", _Ws())) is True

    with patch("ai.realtime_api.logger.info") as mock_info:
        asyncio.run(api.handle_function_call({}, _Ws()))

    logged = [call.args[0] % call.args[1:] for call in mock_info.call_args_list if call.args]

    assert executed == ["done"]
    assert created_tokens == []
    assert governance_calls == {"build": 1, "review": 1}
    assert any(
        "Function call outcome: executing tool | tool=perform_research call_id=call_research_without_source" in line
        for line in logged
    )


def test_handle_function_call_replays_stable_blocked_research_permission_output_for_same_fingerprint(
    monkeypatch,
    tmp_path,
) -> None:
    api = _make_api_stub()
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
    fingerprint = api._build_research_fingerprint(query="status", source="user_text")
    api._record_research_permission_outcome(fingerprint, approved=False)
    api._pending_action = None
    api.function_call = {"name": "perform_research", "call_id": "call_blocked_one"}
    api.function_call_args = '{"query":"status", "context": {"source": "user_text"}}'

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    try:
        budget_before = api._research_budget_remaining()
        usage_before = _usage_count(storage_controller_cls)

        asyncio.run(api.handle_function_call({}, _SendWs()))

        api.function_call = {"name": "perform_research", "call_id": "call_blocked_two"}
        api.function_call_args = '{"query":"status", "context": {"source": "user_text"}}'
        asyncio.run(api.handle_function_call({}, _SendWs()))

        budget_after = api._research_budget_remaining()
        usage_after = _usage_count(storage_controller_cls)

        assert len(sent_payloads) == 2
        first_output = json.loads(sent_payloads[0]["item"]["output"])
        second_output = json.loads(sent_payloads[1]["item"]["output"])
        assert first_output["status"] == "blocked_by_research_permission"
        assert second_output["status"] == "blocked_by_research_permission"
        assert first_output == second_output
        assert budget_after == budget_before
        assert usage_after == usage_before
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None


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


def test_tool_confirmation_deny_does_not_execute_tool_and_emits_noop_payload() -> None:
    api = _make_api_stub()
    pending = _build_pending_action()
    token = _build_confirmation_token(kind="tool_governance", pending_action=pending)
    api._pending_confirmation_token = token
    api._pending_action = pending
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._awaiting_confirmation_completion = False
    api.orchestration_state = type(
        "S",
        (),
        {"phase": OrchestrationPhase.AWAITING_CONFIRMATION, "transition": lambda *args, **kwargs: None},
    )()
    api._governance = type("Gov", (), {"describe_tool": lambda *_args, **_kwargs: {}})()
    api._intent_ledger = {}
    api._intent_state_ttl_s = 300.0

    executed: list[str] = []

    async def _execute(*_args, **_kwargs):
        executed.append("executed")

    api._execute_action = _execute
    api._send_response_create = lambda *_args, **_kwargs: asyncio.sleep(0)
    api.send_assistant_message = lambda *_args, **_kwargs: asyncio.sleep(0)

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    assert asyncio.run(api._maybe_handle_approval_response("deny", _SendWs())) is True

    assert executed == []
    function_outputs = [p for p in sent_payloads if p.get("item", {}).get("type") == "function_call_output"]
    assert len(function_outputs) == 1
    output_payload = json.loads(function_outputs[0]["item"]["output"])
    assert output_payload["status"] == "cancelled"
    assert str(output_payload["message"]).startswith("No action taken.")
    assert output_payload["no_op"] == {"executed": False, "category": "rejection"}


def test_research_permission_yes_after_25s_still_approves_when_timeout_paused() -> None:
    api = _make_api_stub()
    request = type("Req", (), {"prompt": "status"})()
    token = _build_confirmation_token(kind="research_permission", request=request)
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_token_created_at = 0.0
    api._research_permission_awaiting_decision_timeout_s = 20.0

    dispatched: list[str] = []

    async def _dispatch(_request, _ws):
        dispatched.append("yes")

    api._dispatch_research_request = _dispatch

    with patch("ai.realtime_api.time.monotonic", return_value=25.0):
        api._confirmation_speech_active = True
        assert api._expire_confirmation_awaiting_decision_timeout() is None

    api._confirmation_speech_active = False
    api._confirmation_asr_pending = False
    assert asyncio.run(api._maybe_handle_research_permission_response("yes", _Ws())) is True
    assert dispatched == ["yes"]


def test_late_yes_within_grace_applies_to_recently_timed_out_research_permission() -> None:
    api = _make_api_stub()
    request = type("Req", (), {"prompt": "status"})()
    token = _build_confirmation_token(kind="research_permission", request=request, token_id="tok_late")
    api._confirmation_last_closed_token = {
        "token": token,
        "kind": "research_permission",
        "closed_at": 100.0,
        "outcome": "awaiting_decision_timeout",
    }

    dispatched: list[str] = []

    async def _dispatch(_request, _ws):
        dispatched.append("yes")

    api._dispatch_research_request = _dispatch

    with patch("ai.realtime_api.time.monotonic", return_value=108.0):
        assert asyncio.run(api._maybe_apply_late_confirmation_decision("yes", _Ws())) is True

    assert dispatched == ["yes"]


def test_research_permission_no_cancels_pending_request() -> None:
    api = _make_api_stub()
    api._pending_confirmation_token = _build_confirmation_token(
        kind="research_permission",
        request=type("Req", (), {"prompt": "status"})(),
    )
    denied: list[str] = []

    async def _assistant(*args, **kwargs):
        denied.append("no")

    api.send_assistant_message = _assistant

    assert asyncio.run(api._maybe_handle_research_permission_response("nope", _Ws())) is True
    assert denied == ["no"]
    assert api._pending_confirmation_token is None


def test_confirmation_empty_transcript_reprompts_once_and_keeps_token_active() -> None:
    api = _make_api_stub()
    request = type("Req", (), {"prompt": "status"})()
    token = _build_confirmation_token(kind="research_permission", request=request)
    token.metadata = {"approval_flow": True}
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_token_created_at = 1e12
    api._extract_transcript = lambda _event: ""
    api._log_user_transcript = lambda *args, **kwargs: None
    api._log_utterance_envelope = lambda *_args, **_kwargs: None

    messages: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        messages.append(message)

    api.send_assistant_message = _assistant

    asyncio.run(
        api.handle_event(
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""},
            _Ws(),
        )
    )

    assert messages == ["Sorry—was that a yes or no?"]
    assert api._pending_confirmation_token is not None
    assert api._pending_confirmation_token.metadata.get("empty_transcript_reprompted") is True

def test_confirmation_timeout_check_logs_at_most_once_per_interval(monkeypatch) -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="research_permission")
    api._pending_confirmation_token = token
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_timeout_check_log_interval_s = 1.0
    api._confirmation_remaining_seconds = lambda: 42.0
    api._refresh_confirmation_pause = lambda: None
    api._confirmation_pause_reason = lambda: None
    api._get_confirmation_timeout_s = lambda _token: 60.0

    ticks = iter([100.0, 100.4, 101.2])
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: next(ticks))
    logs: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.debug", lambda msg, *a: logs.append(msg % a if a else msg))

    api._expire_confirmation_awaiting_decision_timeout()
    api._expire_confirmation_awaiting_decision_timeout()
    api._expire_confirmation_awaiting_decision_timeout()

    assert len([line for line in logs if "CONFIRMATION_TIMEOUT_CHECK" in line]) == 2


def test_handle_function_call_suppresses_research_for_budget_confirmation() -> None:
    api = _make_api_stub()
    api._pending_action = None
    api._pending_confirmation_token = _build_confirmation_token(kind="research_budget")
    api._pending_research_request = object()
    api.function_call = {"name": "perform_research", "call_id": "call_budget_pending"}
    api.function_call_args = '{"query":"status"}'

    sent_payloads: list[dict[str, object]] = []

    class _SendWs:
        async def send(self, payload: str) -> None:
            sent_payloads.append(json.loads(payload))

    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert len(sent_payloads) == 1
    output_payload = json.loads(sent_payloads[0]["item"]["output"])
    assert output_payload["status"] == "waiting_for_permission"


def test_research_budget_approval_replays_deferred_tool_once() -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="research_budget")
    request = ResearchRequest(prompt="find board dimensions", context={"source": "user"})
    token.request = request
    api._pending_confirmation_token = token
    api._pending_research_request = request
    api.function_call = {"name": "perform_research", "call_id": "call_deferred_1"}
    api.function_call_args = '{"query":"board dimensions"}'
    api._governance = type(
        "Gov",
        (),
        {
            "build_action_packet": lambda *_args, **_kwargs: type(
                "Action",
                (),
                {"id": "call_deferred_1", "tool_name": "perform_research", "tool_args": {"query": "board dimensions"}, "tier": 0},
            )(),
            "record_execution": lambda *_args, **_kwargs: None,
        },
    )()
    api._stage_action = lambda *_args, **_kwargs: {"valid": True}

    dispatched: list[str] = []

    async def _execute_action(action, *_args, **_kwargs):
        dispatched.append(action.id)

    async def _dispatch_research_request(*_args, **_kwargs):
        dispatched.append("fallback_request")

    api._execute_action = _execute_action
    api._dispatch_research_request = _dispatch_research_request

    class _SendWs:
        async def send(self, payload: str) -> None:
            return None

    asyncio.run(api.handle_function_call({}, _SendWs()))

    assert api._deferred_research_tool_call is not None

    asyncio.run(api._maybe_handle_research_budget_response("Yes, go ahead.", api.websocket))
    asyncio.run(api._maybe_handle_research_budget_response("Yes.", api.websocket))

    assert dispatched == ["call_deferred_1"]
    assert api._deferred_research_tool_call is None


def test_research_budget_approval_fallback_dispatch_passes_over_budget_context() -> None:
    api = _make_api_stub()
    token = _build_confirmation_token(kind="research_budget")
    request = ResearchRequest(prompt="find board dimensions", context={"source": "user"})
    token.request = request
    api._pending_confirmation_token = token
    api._pending_research_request = request

    fallback_contexts: list[dict[str, object]] = []

    async def _dispatch_research_request(passed_request, *_args, **_kwargs):
        fallback_contexts.append(dict(passed_request.context))

    api._dispatch_research_request = _dispatch_research_request

    assert asyncio.run(api._maybe_handle_research_budget_response("Yes.", api.websocket)) is True

    assert fallback_contexts == [{"source": "user", "over_budget_approved": True}]
    assert request.context == {"source": "user"}


def test_research_budget_rejection_clears_deferred_tool(monkeypatch, tmp_path) -> None:
    api = _make_api_stub()
    storage_controller_cls = _wire_real_research_budget(monkeypatch, tmp_path, api)
    token = _build_confirmation_token(kind="research_budget")
    token.request = type("ResearchRequest", (), {"prompt": "find board dimensions", "context": {"source": "user"}})()
    api._pending_confirmation_token = token
    api._pending_research_request = token.request
    api._deferred_research_tool_call = {
        "token_id": token.id,
        "tool_name": "perform_research",
        "call_id": "call_cancel",
        "args": {"query": "board dimensions"},
    }

    messages: list[str] = []

    async def _assistant(message, *_args, **_kwargs):
        messages.append(message)

    api.send_assistant_message = _assistant

    try:
        budget_before = api._research_budget_remaining()
        usage_before = _usage_count(storage_controller_cls)

        asyncio.run(api._maybe_handle_research_budget_response("no", api.websocket))

        budget_after = api._research_budget_remaining()
        usage_after = _usage_count(storage_controller_cls)

        assert api._deferred_research_tool_call is None
        assert messages
        assert budget_after == budget_before
        assert usage_after == usage_before
    finally:
        controller = storage_controller_cls._instance
        if controller is not None:
            controller.close()
        storage_controller_cls._instance = None
