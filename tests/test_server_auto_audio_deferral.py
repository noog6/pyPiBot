from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI


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
    api._server_auto_pre_audio_hold_by_turn_id = {}
    api._server_auto_pre_audio_hold_phase_by_key = {}
    api._audio_response_started_ids = set()
    api._server_auto_audio_deferral_timeout_ms = 20
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api._active_response_id = "resp-1"
    api._response_gating_verdict_by_input_event_key = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-476:{turn_id}:{input_event_key}"
    api.loop = asyncio.new_event_loop()
    api._pending_server_auto_response_by_turn_id = {}
    api._pending_micro_ack_by_turn_channel = {}
    api._micro_ack_manager = None
    return api


def test_upgrade_likely_detects_memory_intent_from_partial() -> None:
    api = _api_stub()
    api._is_memory_intent = lambda text: "remember" in text
    api._latest_partial_transcript_by_turn_id["turn-1"] = "remember that I prefer vim"

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


def test_long_utterance_pre_audio_hold_emits_single_micro_ack_and_waits_for_transcript_final() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    micro_acks: list[tuple[str, str]] = []
    api._maybe_schedule_micro_ack = lambda **kwargs: micro_acks.append((kwargs["turn_id"], kwargs["reason"]))
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-1", canonical_key="run-476:turn-1:item-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final")

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        await asyncio.sleep(0.05)
        assert api._started is False
        assert micro_acks == [("turn-1", "transcript_finalized")]
        api._signal_server_auto_transcript_final(turn_id="turn-1")
        await asyncio.sleep(0.03)

    asyncio.run(_run())

    assert api._started is True


def test_short_utterance_transcript_final_arrives_quickly_without_micro_ack_spam() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    micro_acks: list[tuple[str, str]] = []
    api._maybe_schedule_micro_ack = lambda **kwargs: micro_acks.append((kwargs["turn_id"], kwargs["reason"]))
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-1", canonical_key="run-476:turn-1:item-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final")

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        await asyncio.sleep(0.005)
        api._signal_server_auto_transcript_final(turn_id="turn-1")
        await asyncio.sleep(0.03)

    asyncio.run(_run())

    assert api._started is True
    assert micro_acks == []


def test_pre_audio_hold_timeout_schedules_single_transcript_finalized_micro_ack_before_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    api._server_auto_audio_deferral_timeout_ms = 1
    scheduled_micro_acks: list[dict[str, object]] = []
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-1", canonical_key="run-476:turn-1:item-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final")

    original_wait_for = asyncio.wait_for
    wait_for_calls = {"count": 0}

    async def _controlled_wait_for(awaitable: object, timeout: float) -> object:
        wait_for_calls["count"] += 1
        if wait_for_calls["count"] == 1:
            close_awaitable = getattr(awaitable, "close", None)
            if callable(close_awaitable):
                close_awaitable()
            raise asyncio.TimeoutError
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr("ai.realtime_api.asyncio.wait_for", _controlled_wait_for)

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        await asyncio.sleep(0)

        assert len(scheduled_micro_acks) == 1
        assert scheduled_micro_acks[0]["reason"] == "transcript_finalized"
        assert scheduled_micro_acks[0]["expected_delay_ms"] == 700
        assert api._started is False

        api._signal_server_auto_transcript_final(turn_id="turn-1")
        await asyncio.sleep(0.01)

    asyncio.run(_run())

    assert api._started is True
    assert len(scheduled_micro_acks) == 1


def test_pre_audio_hold_rejects_duplicate_and_out_of_order_transitions() -> None:
    api = _api_stub()
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-1", canonical_key="run-476:turn-1:item-1")

    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final", response_id="resp-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="duplicate_hold", response_id="resp-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=False, reason="transcript_final_linked", response_id="resp-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=False, reason="duplicate_release", response_id="resp-1")

    hold_record = api._server_auto_pre_audio_hold_by_turn_id["turn-1"]
    assert hold_record.turn_id == "turn-1"
    assert hold_record.response_id == "resp-1"
    assert hold_record.hold_started_at <= hold_record.hold_released_at
    assert hold_record.last_reason == "transcript_final_linked"
    assert api._server_auto_pre_audio_hold_phase_by_key[("turn-1", "resp-1")] == "released"
    assert api._server_auto_pre_audio_hold_active(turn_id="turn-1", response_id="resp-1") is False


def test_audio_deferral_waiter_terminates_when_different_response_releases_hold() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    api._active_response_id = "resp-2"
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-1", canonical_key="run-476:turn-1:item-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final", response_id="resp-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=False, reason="pending_replaced", response_id="resp-1")
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-2", canonical_key="run-476:turn-1:item-2")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final", response_id="resp-2")

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        await asyncio.sleep(0.01)

    asyncio.run(_run())

    assert api._started is False
    assert "turn-1" not in api._server_auto_audio_waiters_by_turn_id


def test_audio_deferral_waiter_terminates_when_pending_response_changes() -> None:
    api = _api_stub()
    api.audio_player = SimpleNamespace(start_response=lambda: setattr(api, "_started", True))
    api._started = False
    api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-1", canonical_key="run-476:turn-1:item-1")
    api._set_server_auto_pre_audio_hold(turn_id="turn-1", enabled=True, reason="awaiting_transcript_final", response_id="resp-1")

    async def _run() -> None:
        api._schedule_server_auto_audio_deferral(turn_id="turn-1", input_event_key="item-1", response_id="resp-1")
        api._record_pending_server_auto_response(turn_id="turn-1", response_id="resp-2", canonical_key="run-476:turn-1:item-2")
        await asyncio.sleep(0.02)

    asyncio.run(_run())

    assert api._started is False
    assert "turn-1" not in api._server_auto_audio_waiters_by_turn_id
