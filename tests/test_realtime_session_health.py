"""Tests for realtime session health payload composition."""

from __future__ import annotations

import sys
import types
from threading import Event

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.continuity import ContinuityItem, ContinuityLedger
from ai.realtime_api import RealtimeAPI


class _FakeMemoryManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None]] = []

    def get_active_session_id(self) -> str | None:
        return "session-abc"

    def get_retrieval_health_metrics(
        self,
        *,
        scope: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, float | int]:
        self.calls.append((scope, session_id))
        return {
            "retrieval_count": 3,
            "average_retrieval_latency_ms": 8.5,
            "semantic_provider_attempts": 2,
            "semantic_provider_errors": 0,
            "semantic_provider_error_rate_pct": 0.0,
            "embedding_total_memories": 6,
            "embedding_ready_memories": 5,
            "embedding_coverage_pct": 83.33,
            "pending_count": 4,
            "retry_blocked_count": 2,
            "consecutive_failures": 1,
            "oldest_pending_age_ms": 1250,
        }


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._memory_manager = _FakeMemoryManager()
    api._memory_retrieval_scope = "session_local"
    api._session_connected = True
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True
    api.ready_event = Event()
    api.ready_event.set()
    api._session_connection_attempts = 4
    api._session_connections = 2
    api._session_reconnects = 1
    api._session_failures = 0
    api._last_connect_time = 123.0
    api._last_disconnect_reason = None
    api._last_failure_reason = None
    api._silent_turn_incident_count = 2
    api._sensor_event_aggregation_metrics = {}
    api.rate_limits_supports_tokens = False
    api.rate_limits_supports_requests = False
    api.rate_limits_last_present_names = set()
    api.rate_limits_last_event_id = ""
    return api


def _bind_continuity_helpers(api: RealtimeAPI) -> None:
    api._continuity_ledger = ContinuityLedger()
    api._apply_continuity_event = RealtimeAPI._apply_continuity_event.__get__(api, RealtimeAPI)
    api.get_continuity_brief = RealtimeAPI.get_continuity_brief.__get__(api, RealtimeAPI)
    api.get_continuity_turn_settlement = RealtimeAPI.get_continuity_turn_settlement.__get__(api, RealtimeAPI)
    api.get_continuity_diagnostics = RealtimeAPI.get_continuity_diagnostics.__get__(api, RealtimeAPI)
    api._continuity_ledger_instance = RealtimeAPI._continuity_ledger_instance.__get__(api, RealtimeAPI)


def test_get_session_health_passes_scope_and_active_session_to_memory_metrics() -> None:
    api = _make_api_stub()

    health = api.get_session_health()

    assert api._memory_manager.calls == [("session_local", "session-abc")]
    assert health["memory_retrieval"]["embedding_coverage_pct"] == 83.33
    assert health["memory_retrieval"]["retrieval_count"] == 3
    assert health["memory_retrieval"]["pending_count"] == 4
    assert health["memory_retrieval"]["oldest_pending_age_ms"] == 1250


def test_get_session_health_includes_silent_turn_incident_counter() -> None:
    api = _make_api_stub()

    health = api.get_session_health()

    assert health["silent_turn_incidents"] == 2


def test_get_session_health_ready_uses_injection_readiness_semantics() -> None:
    api = _make_api_stub()
    api.is_ready_for_injections = lambda with_reason=False: (False, "not_ready") if with_reason else False

    health = api.get_session_health()

    assert health["ready"] is False
    assert health["injection_ready"] is False
    assert health["session_ready"] is True


def test_get_session_health_includes_compact_continuity_section_when_settled() -> None:
    api = _make_api_stub()

    health = api.get_session_health()

    assert health["continuity"] == {
        "stance": "idle",
        "stance_detail": "",
        "settlement": "settled",
        "settlement_detail": "no_open_continuity_items",
        "counts": {
            "current": 0,
            "ongoing": 0,
            "commitments": 0,
            "unresolved": 0,
            "blockers": 0,
            "constraints": 0,
            "recently_closed": 0,
        },
    }


def test_get_session_health_continuity_section_reports_awaiting_tool_without_mutation() -> None:
    api = _make_api_stub()
    _bind_continuity_helpers(api)
    api._current_run_id = lambda: "run-health"
    api._current_response_turn_id = "turn-health"
    api._response_create_queue = ["unchanged"]

    api._apply_continuity_event(
        "tool_call_started",
        run_id="run-health",
        turn_id="turn-health",
        tool_name="perform_research",
        call_id="call-1",
        commitment_summary="Look up the latest battery status.",
    )

    before_items = dict(api._continuity_ledger._items)
    before_queue = list(api._response_create_queue)
    health = api.get_session_health()
    after_items = dict(api._continuity_ledger._items)
    after_queue = list(api._response_create_queue)

    continuity = health["continuity"]
    assert continuity["stance"] == "awaiting_tool"
    assert continuity["stance_detail"] == "tool=perform_research call_id=call-1"
    assert continuity["settlement"] == "awaiting_tool"
    assert continuity["settlement_detail"] == "tool=perform_research call_id=call-1"
    assert continuity["counts"] == {
        "current": 2,
        "ongoing": 0,
        "commitments": 1,
        "unresolved": 0,
        "blockers": 1,
        "constraints": 0,
        "recently_closed": 0,
    }
    assert before_items == after_items
    assert before_queue == after_queue


def test_get_continuity_diagnostics_reports_followthrough_unresolved_and_active_cases() -> None:
    api = _make_api_stub()
    _bind_continuity_helpers(api)

    api._apply_continuity_event(
        "tool_call_started",
        run_id="run-followthrough",
        turn_id="turn-followthrough",
        tool_name="gesture_wave",
        call_id="call-followthrough",
        commitment_summary="Wave hello to the audience.",
    )
    api._apply_continuity_event(
        "tool_result_received",
        run_id="run-followthrough",
        turn_id="turn-followthrough",
        tool_name="gesture_wave",
        call_id="call-followthrough",
    )
    followthrough = api.get_continuity_diagnostics(
        run_id="run-followthrough",
        turn_id="turn-followthrough",
        reason="test_case",
    )
    assert followthrough["settlement"] == "followthrough_remaining"
    assert followthrough["settlement_detail"] == "origin=tool_followthrough tool=gesture_wave call_id=call-followthrough"
    assert followthrough["counts"] == {
        "current": 1,
        "ongoing": 0,
        "commitments": 1,
        "unresolved": 0,
        "blockers": 0,
        "constraints": 0,
        "recently_closed": 1,
    }

    api._continuity_ledger = ContinuityLedger()
    api._continuity_ledger._set_item(
        ContinuityItem(
            id="commitment:current",
            kind="commitment",
            summary="Check the battery and report whether it is low.",
            status="active",
            priority="medium",
            source="test_case",
            detail="origin=user_request",
            expires_after_turns=4,
        )
    )
    api._continuity_ledger._set_item(
        ContinuityItem(
            id="question:followup",
            kind="unresolved",
            summary="Check the battery and tell me if it is low?",
            status="pending",
            priority="medium",
            source="test_case",
            detail="opened_by=transcript_final",
            expires_after_turns=4,
        )
    )
    api._continuity_ledger._stance = "awaiting_user"
    unresolved = api.get_continuity_diagnostics(
        run_id="run-unresolved",
        turn_id="turn-unresolved",
        reason="test_case",
    )
    assert unresolved["settlement"] == "unresolved_followup"
    assert unresolved["settlement_detail"] == "opened_by=transcript_final"
    assert unresolved["counts"] == {
        "current": 2,
        "ongoing": 0,
        "commitments": 1,
        "unresolved": 1,
        "blockers": 0,
        "constraints": 0,
        "recently_closed": 0,
    }

    api._continuity_ledger = ContinuityLedger()
    api._apply_continuity_event(
        "transcript_final",
        run_id="run-active",
        turn_id="turn-active",
        text="What is your battery voltage right now?",
        source="input_audio_transcription",
    )
    active = api.get_continuity_diagnostics(run_id="run-active", turn_id="turn-active", reason="test_case")
    assert active["stance"] == "assisting_query"
    assert active["stance_detail"] == "read_query_detected"
    assert active["settlement"] == "active_items_only"
    assert active["settlement_detail"] == "origin=user_transcript"
    assert active["counts"] == {
        "current": 1,
        "ongoing": 1,
        "commitments": 0,
        "unresolved": 0,
        "blockers": 0,
        "constraints": 0,
        "recently_closed": 0,
    }
