"""Contract guard for confirmation reminder flow."""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("audioop", types.SimpleNamespace())

from ai.governance import ActionPacket
from ai.realtime.confirmation_runtime import ConfirmationRuntime
from ai.realtime_api import ConfirmationState, PendingConfirmationToken, RealtimeAPI


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._confirmation_reminder_interval_s = 6.0
    api._confirmation_reminder_max_count = 2
    api._confirmation_awaiting_decision_timeout_s = 20.0
    api._research_permission_awaiting_decision_timeout_s = 60.0
    api._confirmation_timeout_check_log_interval_s = 1.0
    api._confirmation_reminder_tracker = {}
    api._confirmation_state = ConfirmationState.PENDING_PROMPT
    api._confirmation_speech_active = False
    api._confirmation_asr_pending = False
    api._confirmation_pause_started_at = None
    api._confirmation_paused_accum_s = 0.0
    api._confirmation_timeout_check_last_logged_at = {}
    api._confirmation_timeout_check_last_pause_reason = {}
    api._current_run_id = lambda: "run-contract"
    from ai.realtime.confirmation import ConfirmationCoordinator
    from ai.realtime.confirmation_service import ConfirmationService

    api._confirmation_coordinator = ConfirmationCoordinator(
        reminder_interval_s=api._confirmation_reminder_interval_s,
        reminder_max_count=api._confirmation_reminder_max_count,
        awaiting_decision_timeout_s=api._confirmation_awaiting_decision_timeout_s,
        research_permission_timeout_s=api._research_permission_awaiting_decision_timeout_s,
        timeout_check_log_interval_s=api._confirmation_timeout_check_log_interval_s,
    )
    api._pending_action = None
    api._confirmation_runtime = ConfirmationRuntime(api)
    action = ActionPacket(
        id="call_contract",
        tool_name="perform_research",
        tool_args={"query": "contract"},
        tier=2,
        what="Run research",
        why="Contract test",
        impact="Read only",
        rollback="None",
        alternatives=["skip"],
        confidence=0.91,
        cost="low",
        risk_flags=[],
        requires_confirmation=True,
    )
    api._pending_action = type("PendingAction", (), {"action": action, "idempotency_key": "idem-contract"})()
    api._pending_research_request = None
    api._deferred_research_tool_call = None
    api._research_pending_call_ids = set()
    api._presented_actions = set()
    api._pending_confirmation_prompt_latches = set()
    api._confirmation_last_closed_token = None
    api._clear_queued_confirmation_reminder_markers = lambda *_args, **_kwargs: None
    api._confirmation_prompt_latch_key = lambda *_args, **_kwargs: None
    api._confirmation_reminder_key = lambda token: token.id if token is not None else None
    api._record_intent_state = lambda *_args, **_kwargs: None
    api._record_recent_confirmation_outcome = lambda *_args, **_kwargs: None
    api._log_confirmation_transition = lambda *_args, **_kwargs: None
    api.orchestration_state = type("Orch", (), {"transition": lambda *_args, **_kwargs: None})()
    api._governance = type("Gov", (), {"describe_tool": lambda *_args, **_kwargs: {"dry_run_supported": True}})()
    api._confirmation_service = ConfirmationService(
        awaiting_timeout_s=api._confirmation_awaiting_decision_timeout_s,
        late_decision_grace_s=15.0,
    )
    api._pending_confirmation_token = PendingConfirmationToken(
        id="confirm_contract",
        kind="tool_governance",
        tool_name="perform_research",
        request=None,
        pending_action=api._pending_action,
        created_at=0.0,
        expiry_ts=None,
        metadata={},
    )
    api._confirmation_token_created_at = 0.0
    api._confirmation_last_activity_at = 0.0
    api._confirmation_coordinator.on_token_started(api._pending_confirmation_token, now=0.0)
    return api


def test_minimal_event_sequence_does_not_introduce_extra_confirmation_loops(monkeypatch) -> None:
    api = _make_api()

    times = iter([0.0, 1.0, 8.1, 9.0, 30.0])
    monkeypatch.setattr("ai.realtime_api.time.monotonic", lambda: next(times))

    allowed_1 = api._allow_confirmation_reminder(api._pending_confirmation_token, reason="prompt")
    denied_burst = api._allow_confirmation_reminder(api._pending_confirmation_token, reason="response_done")
    allowed_2 = api._allow_confirmation_reminder(api._pending_confirmation_token, reason="response_done")
    denied_interval = api._allow_confirmation_reminder(api._pending_confirmation_token, reason="response_done")
    allowed_3 = api._allow_confirmation_reminder(api._pending_confirmation_token, reason="response_done")

    assert allowed_1[0] is False
    assert allowed_1[4] == "schedule"
    assert denied_burst[0] is False
    assert denied_burst[4] == "schedule"
    assert allowed_2[0] is True
    assert denied_interval[0] is False
    assert denied_interval[4] == "interval"
    assert allowed_3[0] is True
    assert len([item for item in (allowed_1, denied_burst, allowed_2, denied_interval, allowed_3) if item[0]]) == 2


def test_realtime_api_build_prompt_routes_through_confirmation_service() -> None:
    api = _make_api()
    action = api._pending_confirmation_token.pending_action.action

    prompt = api._build_approval_prompt(
        action,
        action_summary="tool=perform_research tier=2 cost=low confidence=0.91 requires_confirmation=True",
        confirm_reason="expensive_read",
    )

    assert prompt == (
        "Action summary: tool=perform_research tier=2 cost=low confidence=0.91 requires_confirmation=True. "
        "Reason: expensive_read.; options: Approve / Deny / Dry-run."
    )


def test_confirmation_runtime_timeout_and_transition_contract(monkeypatch) -> None:
    api = _make_api()
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._awaiting_confirmation_completion = True

    monkeypatch.setattr("ai.realtime.confirmation_runtime.time.monotonic", lambda: 22.0)

    expired = api._expire_confirmation_awaiting_decision_timeout()

    assert expired is not None
    assert expired.id == "confirm_contract"
    assert api._pending_confirmation_token is None
    assert api._confirmation_state == ConfirmationState.IDLE


def test_confirmation_runtime_ttl_pause_contract(monkeypatch) -> None:
    api = _make_api()
    api._confirmation_state = ConfirmationState.AWAITING_DECISION
    api._confirmation_speech_active = True

    monkeypatch.setattr("ai.realtime.confirmation_runtime.time.monotonic", lambda: 40.0)

    assert api._confirmation_pause_reason() == "speech_active"
    assert api._is_confirmation_ttl_paused() is True
    assert api._confirmation_remaining_seconds() == 0.0
    assert api._expire_confirmation_awaiting_decision_timeout() is None


def test_confirmation_runtime_guard_reused() -> None:
    api = _make_api()

    import asyncio

    lock_one = asyncio.run(api._confirmation_transition_guard())
    lock_two = asyncio.run(api._confirmation_transition_guard())

    assert lock_one is lock_two
