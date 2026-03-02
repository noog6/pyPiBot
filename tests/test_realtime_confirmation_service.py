from __future__ import annotations

from ai.realtime.confirmation import ConfirmationState
from ai.realtime.confirmation_service import ConfirmationService
from ai.realtime_api import PendingAction, PendingConfirmationToken
from ai.governance import ActionPacket


def _token() -> PendingConfirmationToken:
    action = ActionPacket(
        id="c1",
        tool_name="perform_research",
        tool_args={"query": "x"},
        tier=2,
        what="x",
        why="x",
        impact="x",
        rollback="x",
        alternatives=[],
        confidence=0.9,
        cost="low",
        risk_flags=[],
        requires_confirmation=True,
    )
    pending = PendingAction(action=action, staging={}, original_intent="x", created_at=0.0)
    return PendingConfirmationToken(
        id="t1",
        kind="tool_governance",
        tool_name="perform_research",
        request=None,
        pending_action=pending,
        created_at=0.0,
        expiry_ts=None,
    )


def test_decision_matrix_yes_no_unclear_and_retry_exhaustion() -> None:
    service = ConfirmationService(awaiting_timeout_s=20.0, late_decision_grace_s=15.0)
    token = _token()
    service.start_pending(token, token.pending_action, now=0.0)
    service.state = ConfirmationState.AWAITING_DECISION

    yes = service.handle_user_text("yes", now=1.0)
    assert yes.should_execute is True

    service.start_pending(token, token.pending_action, now=0.0)
    service.state = ConfirmationState.AWAITING_DECISION
    no = service.handle_user_text("no", now=1.0)
    assert no.should_reject is True

    service.start_pending(token, token.pending_action, now=0.0)
    service.state = ConfirmationState.AWAITING_DECISION
    token.retry_count = token.max_retries
    unclear = service.handle_user_text("maybe", now=1.0)
    assert unclear.retry_exhausted is True


def test_timeout_tick_and_prompt_contract() -> None:
    service = ConfirmationService(awaiting_timeout_s=5.0, late_decision_grace_s=15.0)
    token = _token()
    service.start_pending(token, token.pending_action, now=0.0)
    service.state = ConfirmationState.AWAITING_DECISION

    not_yet = service.on_timeout_tick(now=3.0)
    assert not_yet.expired is False
    assert not_yet.remaining_s == 2.0

    expired = service.on_timeout_tick(now=6.0)
    assert expired.expired is True

    prompt = service.build_prompt(
        action_summary="tool=perform_research tier=2 cost=low confidence=0.91 requires_confirmation=True",
        confirm_reason="expensive_read",
        dry_run_supported=True,
        confirm_prompt=None,
    )
    assert prompt.endswith("options: Approve / Deny / Dry-run.")
