from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from unittest.mock import patch

sys.modules.setdefault("audioop", types.SimpleNamespace())

from ai.realtime.confirmation import ConfirmationCoordinator, ConfirmationState
from ai.realtime_api import RealtimeAPI


@dataclass
class _Token:
    id: str
    kind: str
    created_at: float
    metadata: dict


def _make_coordinator() -> ConfirmationCoordinator:
    return ConfirmationCoordinator(
        reminder_interval_s=6.0,
        reminder_max_count=2,
        awaiting_decision_timeout_s=20.0,
        research_permission_timeout_s=60.0,
        timeout_check_log_interval_s=1.0,
    )


def test_timeout_pauses_during_speech_and_resumes() -> None:
    coordinator = _make_coordinator()
    token = _Token(id="tok-1", kind="tool_governance", created_at=0.0, metadata={})
    coordinator.on_token_started(token, now=0.0)
    coordinator.state = ConfirmationState.AWAITING_DECISION

    check = coordinator.check_timeout(now=10.0)
    assert check.expired is False
    assert round(check.remaining_s, 2) == 10.0

    coordinator.speech_active = True
    check = coordinator.check_timeout(now=15.0)
    assert check.expired is False
    assert check.pause_reason == "speech_active"

    coordinator.speech_active = False
    check = coordinator.check_timeout(now=25.0)
    assert check.expired is False
    assert round(check.remaining_s, 2) == 5.0

    check = coordinator.check_timeout(now=31.0)
    assert check.expired is True
    assert check.remaining_s == 0.0


def test_reminder_schedule_and_interval_gating() -> None:
    coordinator = _make_coordinator()
    token = _Token(id="tok-2", kind="tool_governance", created_at=0.0, metadata={})
    coordinator.on_token_started(token, now=0.0)

    first = coordinator.evaluate_reminder(key="idempotency:abc", schedule=(0.0, 8.0), now=0.0)
    assert first.allowed is True
    coordinator.mark_reminder_sent(first, reason="prompt")

    blocked_schedule_early = coordinator.evaluate_reminder(key="idempotency:abc", schedule=(0.0, 8.0), now=4.0)
    assert blocked_schedule_early.allowed is False
    assert blocked_schedule_early.suppress_reason == "schedule"

    blocked_schedule = coordinator.evaluate_reminder(key="idempotency:abc", schedule=(0.0, 8.0), now=7.5)
    assert blocked_schedule.allowed is False
    assert blocked_schedule.suppress_reason == "schedule"

    second = coordinator.evaluate_reminder(key="idempotency:abc", schedule=(0.0, 8.0), now=8.1)
    assert second.allowed is True
    coordinator.mark_reminder_sent(second, reason="fallback")

    blocked_interval = coordinator.evaluate_reminder(key="idempotency:abc", schedule=(0.0, 8.0), now=9.0)
    assert blocked_interval.allowed is False
    assert blocked_interval.suppress_reason == "max_count"


def test_transition_callback_accepts_pending_token_positional_shape() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    coordinator = ConfirmationCoordinator(
        reminder_interval_s=6.0,
        reminder_max_count=2,
        awaiting_decision_timeout_s=20.0,
        research_permission_timeout_s=60.0,
        timeout_check_log_interval_s=1.0,
        on_transition=api._log_confirmation_transition,
    )
    token = _Token(id="tok-r", kind="research_budget", created_at=1.0, metadata={"budget_remaining": 0})
    coordinator.pending_token = token

    with patch("ai.realtime_api.logger.info") as log_info:
        coordinator.transition(
            "token_created:research_budget",
            {"state": ConfirmationState.PENDING_PROMPT, "now": 1.0},
        )

    log_info.assert_called_once()
    assert log_info.call_args.args[0].startswith("CONFIRMATION_FSM transition=%s->%s")
    assert log_info.call_args.args[3] == "tok-r"
    assert log_info.call_args.args[4] == "research_budget"
    assert token.metadata["budget_remaining"] == 0


def test_transition_callback_still_logs_non_research_confirmation() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    coordinator = ConfirmationCoordinator(
        reminder_interval_s=6.0,
        reminder_max_count=2,
        awaiting_decision_timeout_s=20.0,
        research_permission_timeout_s=60.0,
        timeout_check_log_interval_s=1.0,
        on_transition=api._log_confirmation_transition,
    )
    token = _Token(id="tok-g", kind="tool_governance", created_at=2.0, metadata={"approval_flow": True})
    coordinator.pending_token = token
    coordinator.state = ConfirmationState.PENDING_PROMPT

    with patch("ai.realtime_api.logger.info") as log_info:
        coordinator.transition(
            "prompt_sent",
            {"state": ConfirmationState.AWAITING_DECISION, "now": 2.0},
        )

    log_info.assert_called_once()
    assert log_info.call_args.args[3] == "tok-g"
    assert log_info.call_args.args[4] == "tool_governance"
    assert isinstance(token.metadata.get("awaiting_decision_since"), float)
