"""Focused tests for healthy-path realtime logging cleanup."""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI


LOGGER_NAME = "ai.realtime_api"


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.orchestration_state = SimpleNamespace(phase=OrchestrationPhase.AWAITING_CONFIRMATION)
    api._current_run_id = lambda: "run-test"
    api._conversation_efficiency_state = lambda **_kwargs: SimpleNamespace(
        substantive_count_by_canonical={},
        substantive_count=0,
        duplicate_alerted_canonical_keys=set(),
    )
    return api


def test_log_suppressed_stimulus_demotes_expected_camera_gating_to_debug() -> None:
    api = _make_api()

    with patch("ai.realtime_api.logger.log") as log_mock:
        api._log_suppressed_stimulus("camera", "image", "awaiting_confirmation_policy")

    log_mock.assert_called_once()
    level, message, source, kind, phase_name, reason = log_mock.call_args.args
    assert level == logging.DEBUG
    assert message == "suppressed_external_stimulus source=%s kind=%s phase=%s reason=%s"
    assert (source, kind, phase_name, reason) == ("camera", "image", "awaiting_confirmation", "awaiting_confirmation_policy")


def test_log_suppressed_stimulus_keeps_operator_relevant_sources_at_info() -> None:
    api = _make_api()

    with patch("ai.realtime_api.logger.log") as log_mock:
        api._log_suppressed_stimulus("battery", "status", "awaiting_confirmation_policy")

    level, message, source, kind, phase_name, reason = log_mock.call_args.args
    assert level == logging.INFO
    assert message == "suppressed_external_stimulus source=%s kind=%s phase=%s reason=%s"
    assert (source, kind, phase_name, reason) == ("battery", "status", "awaiting_confirmation", "awaiting_confirmation_policy")


def test_intent_guard_bypass_logs_once_per_turn_pair() -> None:
    api = _make_api()

    with patch("ai.realtime_api.logger.info") as info_mock:
        api._log_intent_guard_bypass_once(
            phase="execution",
            tool_name="gesture_attention_release",
            current_turn_id="turn-2",
            prior_turn_id="turn-1",
            explicit_repeat=False,
        )
        api._log_intent_guard_bypass_once(
            phase="execution",
            tool_name="gesture_attention_release",
            current_turn_id="turn-2",
            prior_turn_id="turn-1",
            explicit_repeat=False,
        )

    info_mock.assert_called_once_with(
        "INTENT_GUARD_BYPASS reason=allow_repeated_reversible_gesture phase=%s tool=%s current_turn_id=%s prior_turn_id=%s explicit_repeat=%s",
        "execution",
        "gesture_attention_release",
        "turn-2",
        "turn-1",
        False,
    )


def test_terminal_substantive_reconcile_missing_terminal_text_is_debug_only() -> None:
    api = _make_api()
    api._selected_response_has_substantive_evidence = lambda **_kwargs: False

    with patch("ai.realtime_api.logger.debug") as debug_mock:
        api._reconcile_terminal_substantive_response(
            turn_id="turn-1",
            canonical_key="turn-1::input-1",
            response_id="resp-1",
            selected=True,
            selection_reason="normal",
        )

    debug_mock.assert_called_once_with(
        "terminal_substantive_reconcile_skipped run_id=%s turn_id=%s canonical_key=%s response_id=%s selected=%s reason=%s evidence=missing_terminal_text",
        "run-test",
        "turn-1",
        "turn-1::input-1",
        "resp-1",
        "true",
        "normal",
    )


def test_tool_followup_state_transition_logs_info_with_prior_state() -> None:
    api = _make_api()
    api._tool_followup_state_by_canonical_key = {}
    api._clear_stale_assistant_message_creates_for_tool_followup = lambda **_kwargs: None

    with patch("ai.realtime_api.logger.info") as info_mock, patch("ai.realtime_api.logger.debug") as debug_mock:
        RealtimeAPI._set_tool_followup_state(
            api,
            canonical_key="turn-1::tool-call-1",
            state="scheduled",
            reason="test_transition",
        )

    info_mock.assert_called_once_with(
        "tool_followup_state canonical_key=%s state=%s reason=%s prior_state=%s",
        "turn-1::tool-call-1",
        "scheduled",
        "test_transition",
        "new",
    )
    debug_mock.assert_not_called()


def test_tool_followup_state_same_state_reemission_logs_debug_only() -> None:
    api = _make_api()
    api._tool_followup_state_by_canonical_key = {"turn-1::tool-call-1": "scheduled"}
    api._clear_stale_assistant_message_creates_for_tool_followup = lambda **_kwargs: None

    with patch("ai.realtime_api.logger.info") as info_mock, patch("ai.realtime_api.logger.debug") as debug_mock:
        RealtimeAPI._set_tool_followup_state(
            api,
            canonical_key="turn-1::tool-call-1",
            state="scheduled",
            reason="test_repeat",
        )

    info_mock.assert_not_called()
    debug_mock.assert_called_once_with(
        "tool_followup_state canonical_key=%s state=%s reason=%s",
        "turn-1::tool-call-1",
        "scheduled",
        "test_repeat",
    )


def test_tool_followup_terminal_states_keep_cleanup_behavior() -> None:
    for terminal_state in ("done", "dropped"):
        api = _make_api()
        api._tool_followup_state_by_canonical_key = {"turn-1::tool-call-1": "created"}
        cleared: list[dict[str, str]] = []
        api._clear_stale_assistant_message_creates_for_tool_followup = lambda **kwargs: cleared.append(kwargs)

        with patch("ai.realtime_api.logger.info") as info_mock, patch("ai.realtime_api.logger.debug") as debug_mock:
            RealtimeAPI._set_tool_followup_state(
                api,
                canonical_key="turn-1::tool-call-1",
                state=terminal_state,
                reason="terminal_transition",
            )

        assert api._tool_followup_state_by_canonical_key["turn-1::tool-call-1"] == terminal_state
        assert cleared == [{"canonical_key": "turn-1::tool-call-1", "state": terminal_state}]
        info_mock.assert_called_once_with(
            "tool_followup_state canonical_key=%s state=%s reason=%s prior_state=%s",
            "turn-1::tool-call-1",
            terminal_state,
            "terminal_transition",
            "created",
        )
        debug_mock.assert_not_called()
