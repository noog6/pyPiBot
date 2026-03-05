from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI, ResponseGatingVerdict


def _api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._latest_partial_transcript_by_turn_id = {}
    api._pending_confirmation_token = None
    api._deferred_research_tool_call = None
    api._pending_research_request = None
    api._asr_verify_on_risk_enabled = False
    api._asr_verify_short_utterance_ms = 450
    api._asr_verify_min_confidence = 0.65
    api._current_run_id = lambda: "run-476"
    api.get_vision_state = lambda: {"available": False, "can_capture": False}
    api._server_auto_audio_waiters_by_turn_id = {}
    api._server_auto_audio_defer_tasks_by_turn_id = {}
    api._audio_response_started_ids = set()
    api._server_auto_audio_deferral_timeout_ms = 20
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api._active_response_id = "resp-1"
    api._response_gating_verdict_by_input_event_key = {}
    api._server_auto_pre_audio_hold_by_turn_id = {}
    api._maybe_schedule_micro_ack = lambda **_kwargs: None
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-476:{turn_id}:{input_event_key}"
    return api


def test_upgrade_likely_detects_memory_intent_from_partial() -> None:
    api = _api_stub()
    api._latest_partial_transcript_by_turn_id["turn-1"] = "remember that I prefer vim"
    api._is_memory_intent = lambda text: "remember" in text

    is_likely, reasons = api._upgrade_likely_for_server_auto_turn(turn_id="turn-1", input_event_key="item-1")

    assert is_likely is True
    assert "memory_intent_likely" in reasons


def test_upgrade_likely_detects_pending_tool_followup() -> None:
    api = _api_stub()
    api._is_memory_intent = lambda _text: False
    api._pending_confirmation_token = SimpleNamespace(pending_action=SimpleNamespace(action=SimpleNamespace()))

    is_likely, reasons = api._upgrade_likely_for_server_auto_turn(turn_id="turn-1", input_event_key="item-1")

    assert is_likely is True
    assert "pending_tool_followup" in reasons


def test_deferred_audio_not_started_when_upgrade_verdict_arrives() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        key = api._canonical_utterance_key(turn_id="turn-1", input_event_key="item-1")
        api._response_gating_verdict_by_input_event_key[key] = SimpleNamespace(action="UPGRADE")
        api._signal_server_auto_transcript_final(turn_id="turn-1")
        await asyncio.sleep(0.03)

    asyncio.run(_run())

    assert api._started is False


def test_deferred_audio_starts_after_timeout_when_no_upgrade_signal() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    scheduled_micro_acks: list[dict[str, object]] = []
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    key = api._canonical_utterance_key(turn_id="turn-1", input_event_key="item-1")
    api._response_gating_verdict_by_input_event_key[key] = ResponseGatingVerdict(action="CLARIFY", reason="awaiting_transcript_final", decided_at=0.0)

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        await asyncio.sleep(0.08)
        api._response_gating_verdict_by_input_event_key[key] = ResponseGatingVerdict(action="NOOP", reason="proceed_without_transcript_final", decided_at=0.0)
        await asyncio.sleep(0.04)

    asyncio.run(_run())

    assert api._started is True
    assert scheduled_micro_acks
    assert scheduled_micro_acks[0]["reason"] == "awaiting_transcript_final"


def test_deferred_audio_short_utterance_transcript_final_arrives_quickly_no_micro_ack_spam() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    scheduled_micro_acks: list[dict[str, object]] = []
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    key = api._canonical_utterance_key(turn_id="turn-1", input_event_key="item-1")
    api._response_gating_verdict_by_input_event_key[key] = ResponseGatingVerdict(action="CLARIFY", reason="awaiting_transcript_final", decided_at=0.0)

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        await asyncio.sleep(0.005)
        api._response_gating_verdict_by_input_event_key[key] = ResponseGatingVerdict(action="NOOP", reason="transcript_final_ready", decided_at=0.0)
        api._signal_server_auto_transcript_final(turn_id="turn-1")
        await asyncio.sleep(0.03)

    asyncio.run(_run())

    assert api._started is True
    assert scheduled_micro_acks == []
