from __future__ import annotations

from dataclasses import dataclass

from ai.realtime.confirmation import ConfirmationCoordinator, ConfirmationState


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
