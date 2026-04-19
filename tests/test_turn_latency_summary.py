from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._current_run_id = lambda: "run-latency"
    api._turn_diagnostic_timestamps = {}
    api._turn_latency_summary_emitted = set()
    api._canonical_utterance_key = lambda turn_id, input_event_key: f"{turn_id}::{input_event_key}"
    return api


def _capture_turn_latency(monkeypatch):
    messages: list[str] = []

    def _capture(message: str, *args):
        rendered = message % args
        if rendered.startswith("turn_latency_summary"):
            messages.append(rendered)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)
    return messages


def test_normal_admitted_turn_emits_one_latency_summary(monkeypatch) -> None:
    api = _make_api_stub()
    messages = _capture_turn_latency(monkeypatch)

    key = "item_1"
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="speech_stopped", when=1.0)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="transcript_final_received", when=1.1)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="attention_decision_finalized", when=1.15)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="response_create_sent", when=1.2)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="response_created_received", when=1.3)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="first_audio_delta_received", when=1.45)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="playback_start", when=1.5)
    api._mark_turn_latency_marker(turn_id="turn-1", input_event_key=key, marker="response_done", when=2.0)
    api._set_turn_latency_classification(turn_id="turn-1", input_event_key=key, path="normal_admitted_answer", outcome="completed")

    api._emit_turn_latency_summary(turn_id="turn-1", input_event_key=key)

    assert len(messages) == 1
    assert "speech_to_transcript_final_ms=100" in messages[0]
    assert "transcript_final_to_attention_decision_ms=49" in messages[0]
    assert "response_create_to_response_created_ms=100" in messages[0]
    assert "transcript_final_to_first_audible_answer_ms=399" in messages[0]
    assert "path=normal_admitted_answer" in messages[0]


def test_attention_gate_blocked_turn_emits_without_answer_audio_phases(monkeypatch) -> None:
    api = _make_api_stub()
    messages = _capture_turn_latency(monkeypatch)

    key = "item_2"
    api._mark_turn_latency_marker(turn_id="turn-2", input_event_key=key, marker="speech_stopped", when=3.0)
    api._mark_turn_latency_marker(turn_id="turn-2", input_event_key=key, marker="transcript_final_received", when=3.2)
    api._mark_turn_latency_marker(turn_id="turn-2", input_event_key=key, marker="attention_decision_finalized", when=3.23)
    api._set_turn_latency_classification(turn_id="turn-2", input_event_key=key, path="attention_gate_blocked", outcome="blocked")

    api._emit_turn_latency_summary(turn_id="turn-2", input_event_key=key)

    assert len(messages) == 1
    assert "transcript_final_to_response_create_ms=na" in messages[0]
    assert "response_created_to_first_audio_delta_ms=na" in messages[0]
    assert "path=attention_gate_blocked outcome=blocked" in messages[0]


def test_empty_transcript_blocked_turn_emits_explicit_blocked_summary(monkeypatch) -> None:
    api = _make_api_stub()
    messages = _capture_turn_latency(monkeypatch)

    key = "item_empty"
    api._mark_turn_latency_marker(turn_id="turn-3", input_event_key=key, marker="speech_stopped", when=5.0)
    api._mark_turn_latency_marker(turn_id="turn-3", input_event_key=key, marker="transcript_final_received", when=5.05)
    api._mark_turn_latency_marker(turn_id="turn-3", input_event_key=key, marker="attention_decision_finalized", when=5.06)
    api._set_turn_latency_classification(turn_id="turn-3", input_event_key=key, path="empty_transcript_blocked", outcome="blocked")

    api._emit_turn_latency_summary(turn_id="turn-3", input_event_key=key)

    assert len(messages) == 1
    assert "path=empty_transcript_blocked outcome=blocked" in messages[0]


def test_upgraded_response_emits_single_authoritative_summary(monkeypatch) -> None:
    api = _make_api_stub()
    messages = _capture_turn_latency(monkeypatch)

    key = "item_upgrade"
    api._mark_turn_latency_marker(turn_id="turn-4", input_event_key=key, marker="speech_stopped", when=10.0)
    api._mark_turn_latency_marker(turn_id="turn-4", input_event_key=key, marker="transcript_final_received", when=10.2)
    api._mark_turn_latency_marker(turn_id="turn-4", input_event_key=key, marker="response_create_sent", when=10.3)
    api._mark_turn_latency_marker(turn_id="turn-4", input_event_key=key, marker="response_created_received", when=10.4)
    api._mark_turn_latency_marker(turn_id="turn-4", input_event_key=key, marker="response_done", when=10.9)
    api._set_turn_latency_classification(turn_id="turn-4", input_event_key=key, path="upgraded_response", outcome="completed")

    api._emit_turn_latency_summary(turn_id="turn-4", input_event_key=key)
    api._emit_turn_latency_summary(turn_id="turn-4", input_event_key=key)

    assert len(messages) == 1
    assert "path=upgraded_response" in messages[0]


def test_tool_involved_turn_rolls_up_tool_timing(monkeypatch) -> None:
    api = _make_api_stub()
    messages = _capture_turn_latency(monkeypatch)

    key = "item_tool"
    state = api._turn_latency_state(turn_id="turn-5", input_event_key=key, create=True)
    assert state is not None
    state["tool_count"] = 2
    state["tool_total_ms"] = 312.0
    api._mark_turn_latency_marker(turn_id="turn-5", input_event_key=key, marker="speech_stopped", when=1.0)
    api._mark_turn_latency_marker(turn_id="turn-5", input_event_key=key, marker="transcript_final_received", when=1.1)
    api._mark_turn_latency_marker(turn_id="turn-5", input_event_key=key, marker="response_done", when=1.9)

    api._emit_turn_latency_summary(turn_id="turn-5", input_event_key=key)

    assert len(messages) == 1
    assert "tool_total_ms=312 tool_count=2" in messages[0]
    assert "path=tool_involved" in messages[0]


def test_cancelled_provisional_server_auto_can_be_suppressed_without_polluting_final_lineage(monkeypatch) -> None:
    api = _make_api_stub()
    messages = _capture_turn_latency(monkeypatch)

    provisional_key = "synthetic_server_auto_1"
    final_key = "item_final_1"
    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=provisional_key, marker="speech_stopped", when=2.0)
    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=provisional_key, marker="response_created_received", when=2.1)
    api._set_turn_latency_classification(
        turn_id="turn-6",
        input_event_key=provisional_key,
        path="server_auto_cancelled_before_audio",
        outcome="deferred",
    )

    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=final_key, marker="speech_stopped", when=2.0)
    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=final_key, marker="transcript_final_received", when=2.2)
    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=final_key, marker="response_create_sent", when=2.3)
    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=final_key, marker="response_created_received", when=2.4)
    api._mark_turn_latency_marker(turn_id="turn-6", input_event_key=final_key, marker="response_done", when=3.0)
    api._set_turn_latency_classification(turn_id="turn-6", input_event_key=final_key, path="normal_admitted_answer", outcome="completed")
    api._emit_turn_latency_summary(turn_id="turn-6", input_event_key=final_key)

    assert len(messages) == 1
    assert "input_event_key=item_final_1" in messages[0]
    assert "transcript_final_to_response_create_ms=99" in messages[0]
