from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._current_run_id = lambda: "run-timing"
    api._tool_followup_timing_by_turn_id = {}
    return api


def test_tool_followup_timing_summary_emits_for_successful_followup_turn(monkeypatch) -> None:
    api = _make_api_stub()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: messages.append(message % args))

    marker_times = {
        "speech_stopped": 1.000,
        "transcript_final_received": 1.120,
        "first_tool_call_received": 1.300,
        "tool_result_received": 1.480,
        "tool_followup_release": 1.520,
        "followup_response_created": 1.640,
        "first_output_audio_delta_for_followup": 1.900,
        "followup_output_audio_done": 2.550,
        "followup_response_done": 2.700,
    }
    for marker, when in marker_times.items():
        api._mark_tool_followup_timing(
            turn_id="turn-1",
            marker=marker,
            when=when,
            call_id="call-1",
            canonical_key="turn-1::tool:call-1",
            response_id="resp-followup" if marker.startswith("followup_") or "audio" in marker else None,
            is_tool_followup=True,
            released=marker in {"tool_followup_release", "followup_response_created", "first_output_audio_delta_for_followup", "followup_output_audio_done", "followup_response_done"},
        )

    api._maybe_emit_tool_followup_timing_summary(turn_id="turn-1")

    assert len(messages) == 1
    assert "tool_followup_turn_timing_summary run_id=run-timing turn_id=turn-1 tool_call_id=call-1" in messages[0]
    assert "speech_to_transcript_final_ms=120" in messages[0]
    assert "transcript_final_to_tool_call_ms=179" in messages[0]
    assert "tool_call_to_tool_result_ms=179" in messages[0]
    assert "tool_result_to_followup_release_ms=40" in messages[0]
    assert "followup_release_to_response_created_ms=119" in messages[0]
    assert "response_created_to_first_audio_ms=260" in messages[0]
    assert "first_audio_to_audio_done_ms=649" in messages[0]
    assert "total_turn_elapsed_ms=1700" in messages[0]


def test_tool_followup_timing_summary_degrades_gracefully_with_missing_markers(monkeypatch) -> None:
    api = _make_api_stub()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: messages.append(message % args))

    api._mark_tool_followup_timing(
        turn_id="turn-2",
        marker="tool_followup_release",
        when=5.0,
        call_id="call-2",
        canonical_key="turn-2::tool:call-2",
        is_tool_followup=True,
        released=True,
    )
    api._mark_tool_followup_timing(
        turn_id="turn-2",
        marker="followup_response_done",
        when=5.8,
        call_id="call-2",
        canonical_key="turn-2::tool:call-2",
        response_id="resp-followup-2",
        is_tool_followup=True,
        released=True,
    )

    api._maybe_emit_tool_followup_timing_summary(turn_id="turn-2")

    assert len(messages) == 1
    assert "tool_followup_turn_timing_summary" in messages[0]
    assert "speech_to_transcript_final_ms=na" in messages[0]
    assert "response_created_to_first_audio_ms=na" in messages[0]
    assert "total_turn_elapsed_ms=na" in messages[0]


def test_tool_followup_timing_summary_not_emitted_for_non_tool_turn(monkeypatch) -> None:
    api = _make_api_stub()
    messages: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: messages.append(message % args))

    api._mark_tool_followup_timing(
        turn_id="turn-3",
        marker="followup_response_done",
        when=9.0,
        call_id="call-3",
        canonical_key="turn-3::tool:call-3",
        is_tool_followup=False,
        released=False,
    )

    api._maybe_emit_tool_followup_timing_summary(turn_id="turn-3")

    assert messages == []
