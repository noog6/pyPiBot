from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI
from ai.interruption_recovery import InterruptedToolOutputResolution


def test_resolve_interrupted_candidates_scoped_to_matching_interruption_turn() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._interrupted_tool_output_candidates_by_response_id = {
        "resp-a": {"resolution": "pending", "interruption_turn_id": "turn-a", "canonical_key": "k-a"},
        "resp-b": {"resolution": "pending", "interruption_turn_id": "turn-b", "canonical_key": "k-b"},
    }
    api._current_run_id = lambda: "run-test"

    updated = RealtimeAPI._resolve_interrupted_tool_output_candidates(
        api,
        interruption_turn_id="turn-a",
        resolution="interruption_merged_into_followup",
    )

    assert updated == 1
    assert api._interrupted_tool_output_candidates_by_response_id["resp-a"]["resolution"] == "interruption_merged_into_followup"
    assert api._interrupted_tool_output_candidates_by_response_id["resp-b"]["resolution"] == "pending"


def test_resume_interrupted_tool_output_after_noise_schedules_followup_create() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    sent_events: list[dict[str, object]] = []
    state_updates: list[tuple[str, str, str]] = []
    api._interrupted_tool_output_candidates_by_response_id = {
        "resp-1": {"resolution": "interruption_resume_after_noise", "resume_scheduled": False}
    }

    api._tool_call_id_from_input_event_key = lambda key: "call-123" if "tool:call-123" in str(key) else ""

    def _build_event(*, call_id: str):
        assert call_id == "call-123"
        return ({"type": "response.create", "response": {"metadata": {"tool_call_id": call_id}}}, "turn-1::tool:call-123")

    api._build_tool_followup_response_create_event = _build_event
    api._set_tool_followup_state = lambda *, canonical_key, state, reason: state_updates.append((canonical_key, state, reason))

    async def _send_response_create(_websocket, event, *, origin, record_ai_call):
        sent_events.append({"event": event, "origin": origin, "record_ai_call": record_ai_call})
        return True

    api._send_response_create = _send_response_create
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn-fallback"

    resumed = asyncio.run(
        RealtimeAPI._resume_interrupted_tool_output_candidates_after_noise(
            api,
            websocket=SimpleNamespace(),
            interruption_turn_id="turn-noise",
            candidates=[
                {
                    "response_id": "resp-1",
                    "turn_id": "turn-1",
                    "input_event_key": "tool:call-123",
                }
            ],
        )
    )

    assert resumed == 1
    assert len(sent_events) == 1
    metadata = sent_events[0]["event"]["response"]["metadata"]
    assert metadata["interruption_recovery"] == "resume_after_noise"
    assert metadata["consumes_canonical_slot"] == "false"
    assert sent_events[0]["origin"] == "tool_output"
    assert state_updates == [("turn-1::tool:call-123", "scheduled_release", "interruption_resume_after_noise")]
    assert api._interrupted_tool_output_candidates_by_response_id["resp-1"]["resume_scheduled"] is True


def test_resume_interrupted_tool_output_is_not_scheduled_twice_for_same_candidate() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    sent_events: list[dict[str, object]] = []
    api._interrupted_tool_output_candidates_by_response_id = {
        "resp-2": {"resolution": "interruption_resume_after_noise", "resume_scheduled": True}
    }
    api._tool_call_id_from_input_event_key = lambda key: "call-123"
    api._build_tool_followup_response_create_event = (
        lambda *, call_id: ({"type": "response.create", "response": {"metadata": {"tool_call_id": call_id}}}, "k")
    )
    api._set_tool_followup_state = lambda **_kwargs: None

    async def _send_response_create(_websocket, event, *, origin, record_ai_call):
        sent_events.append({"event": event, "origin": origin, "record_ai_call": record_ai_call})
        return True

    api._send_response_create = _send_response_create
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn-fallback"

    resumed = asyncio.run(
        RealtimeAPI._resume_interrupted_tool_output_candidates_after_noise(
            api,
            websocket=SimpleNamespace(),
            interruption_turn_id="turn-noise",
            candidates=[{"response_id": "resp-2", "turn_id": "turn-1", "input_event_key": "tool:call-123"}],
        )
    )

    assert resumed == 0
    assert sent_events == []


def test_superseded_candidate_cannot_be_resumed() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    sent_events: list[dict[str, object]] = []
    api._interrupted_tool_output_candidates_by_response_id = {
        "resp-3": {"resolution": "interruption_superseded_by_new_turn", "resume_scheduled": False}
    }
    api._tool_call_id_from_input_event_key = lambda key: "call-123"
    api._build_tool_followup_response_create_event = (
        lambda *, call_id: ({"type": "response.create", "response": {"metadata": {"tool_call_id": call_id}}}, "k")
    )
    api._set_tool_followup_state = lambda **_kwargs: None

    async def _send_response_create(_websocket, event, *, origin, record_ai_call):
        sent_events.append({"event": event, "origin": origin, "record_ai_call": record_ai_call})
        return True

    api._send_response_create = _send_response_create
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn-fallback"

    resumed = asyncio.run(
        RealtimeAPI._resume_interrupted_tool_output_candidates_after_noise(
            api,
            websocket=SimpleNamespace(),
            interruption_turn_id="turn-noise",
            candidates=[{"response_id": "resp-3", "turn_id": "turn-1", "input_event_key": "tool:call-123"}],
        )
    )

    assert resumed == 0
    assert sent_events == []


def test_superseded_interruption_marks_tool_followup_state_dropped() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    resolved: list[tuple[str, str]] = []
    dropped: list[tuple[str, str, str]] = []
    api._resolve_interrupted_tool_output_candidates = (
        lambda *, interruption_turn_id, resolution: resolved.append((interruption_turn_id, resolution))
    )
    api._set_tool_followup_state = lambda *, canonical_key, state, reason: dropped.append((canonical_key, state, reason))

    RealtimeAPI._apply_interrupted_tool_output_transcript_resolution(
        api,
        interruption_turn_id="turn-2",
        resolution=InterruptedToolOutputResolution.SUPERSEDED_BY_NEW_TURN,
        candidates=[{"canonical_key": "turn-2::tool:call-2"}],
    )

    assert resolved == [("turn-2", "interruption_superseded_by_new_turn")]
    assert dropped == [("turn-2::tool:call-2", "dropped", "interruption_superseded_by_new_turn")]


def test_merged_interruption_preserves_followup_chain_state() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    resolved: list[tuple[str, str]] = []
    dropped: list[tuple[str, str, str]] = []
    replay_events: list[object] = []
    api._resolve_interrupted_tool_output_candidates = (
        lambda *, interruption_turn_id, resolution: resolved.append((interruption_turn_id, resolution))
    )
    api._set_tool_followup_state = lambda *, canonical_key, state, reason: dropped.append((canonical_key, state, reason))
    api._send_response_create = lambda *_args, **_kwargs: replay_events.append("unexpected")

    RealtimeAPI._apply_interrupted_tool_output_transcript_resolution(
        api,
        interruption_turn_id="turn-3",
        resolution=InterruptedToolOutputResolution.MERGED_INTO_FOLLOWUP,
        candidates=[{"canonical_key": "turn-3::tool:call-3"}],
    )

    assert resolved == [("turn-3", "interruption_merged_into_followup")]
    assert dropped == []
    assert replay_events == []


def test_resolved_candidate_not_re_resolved_on_unrelated_turn() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._interrupted_tool_output_candidates_by_response_id = {
        "resp-a": {"resolution": "pending", "interruption_turn_id": "turn-a", "canonical_key": "k-a"},
    }
    api._current_run_id = lambda: "run-test"

    first = RealtimeAPI._resolve_interrupted_tool_output_candidates(
        api,
        interruption_turn_id="turn-a",
        resolution="interruption_merged_into_followup",
    )
    second = RealtimeAPI._resolve_interrupted_tool_output_candidates(
        api,
        interruption_turn_id="turn-b",
        resolution="interruption_resume_after_noise",
    )

    assert first == 1
    assert second == 0


def test_prune_interrupted_candidates_cleans_resolved_entries_for_turn_only() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._interrupted_tool_output_candidates_by_response_id = {
        "resp-a": {"resolution": "interruption_merged_into_followup", "interruption_turn_id": "turn-a"},
        "resp-b": {"resolution": "pending", "interruption_turn_id": "turn-a"},
        "resp-c": {"resolution": "interruption_superseded_by_new_turn", "interruption_turn_id": "turn-b"},
    }
    api._current_run_id = lambda: "run-test"

    removed = RealtimeAPI._prune_interrupted_tool_output_candidates(api, interruption_turn_id="turn-a")

    assert removed == 1
    assert "resp-a" not in api._interrupted_tool_output_candidates_by_response_id
    assert "resp-b" in api._interrupted_tool_output_candidates_by_response_id
    assert "resp-c" in api._interrupted_tool_output_candidates_by_response_id
