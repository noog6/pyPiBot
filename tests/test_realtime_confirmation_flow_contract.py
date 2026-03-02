"""Contract guard for confirmation reminder flow."""

from __future__ import annotations

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
    api._confirmation_service = ConfirmationService(
        awaiting_timeout_s=api._confirmation_awaiting_decision_timeout_s,
        late_decision_grace_s=15.0,
    )
    api._pending_confirmation_token = PendingConfirmationToken(
        id="confirm_contract",
        kind="tool_governance",
        tool_name="perform_research",
        request=None,
        pending_action=None,
        created_at=0.0,
        expiry_ts=None,
        metadata={},
    )
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
