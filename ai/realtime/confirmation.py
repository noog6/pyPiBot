"""Confirmation coordinator for realtime approval flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Callable


class ConfirmationState(str, Enum):
    IDLE = "idle"
    PENDING_PROMPT = "pending_prompt"
    AWAITING_DECISION = "awaiting_decision"
    RESOLVING = "resolving"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ConfirmationTransitionDecision:
    allow_response_transition: bool
    emit_reminder: bool
    recover_mic: bool
    close_reason: str | None


@dataclass(frozen=True)
class ConfirmationTimeoutDecision:
    expired: bool
    pause_reason: str | None
    remaining_s: float


@dataclass(frozen=True)
class ConfirmationReminderDecision:
    allowed: bool
    key: str | None
    sent_count: int
    sent_at: float | None
    suppress_reason: str | None
    now: float


# Pure decision helpers.
_YES_PATTERN = re.compile(
    r"^(yes|y|yeah|yep|sure|ok|okay|approve|proceed|go ahead|go ahead and do it|do it|please do|please go ahead)( please| thanks| thank you)?$"
)
_NO_PATTERN = re.compile(
    r"^(no|n|nope|nah|deny|do not|dont|don t|don t do that|don t do it|do not do that|do not do it)( please| thanks| thank you)?$"
)
_CANCEL_PATTERN = re.compile(r"^(cancel|cancel that|never mind|nevermind|stop|ignore that|ignore it)( please)?$")


def normalize_confirmation_decision(text: str) -> str:
    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    if not normalized:
        return "unclear"
    if _YES_PATTERN.fullmatch(normalized):
        return "yes"
    if _NO_PATTERN.fullmatch(normalized):
        return "no"
    if _CANCEL_PATTERN.fullmatch(normalized):
        return "cancel"
    return "unclear"


class ConfirmationCoordinator:
    """Owns confirmation-state transitions and timeout/reminder decisions."""

    def __init__(
        self,
        *,
        reminder_interval_s: float,
        reminder_max_count: int,
        awaiting_decision_timeout_s: float,
        research_permission_timeout_s: float,
        timeout_check_log_interval_s: float,
        on_transition: Callable[[ConfirmationState, ConfirmationState, str, Any], None] | None = None,
        on_timeout_check: Callable[[Any, float, str | None], None] | None = None,
    ) -> None:
        self.state = ConfirmationState.IDLE
        self.pending_token: Any | None = None
        self.awaiting_confirmation_completion = False
        self.reminder_interval_s = max(0.0, float(reminder_interval_s))
        self.reminder_max_count = max(0, int(reminder_max_count))
        self.awaiting_decision_timeout_s = max(0.0, float(awaiting_decision_timeout_s))
        self.research_permission_timeout_s = max(0.0, float(research_permission_timeout_s))
        self.timeout_check_log_interval_s = max(0.25, float(timeout_check_log_interval_s))

        self.token_created_at: float | None = None
        self.last_activity_at: float | None = None
        self.speech_active = False
        self.asr_pending = False
        self.pause_started_at: float | None = None
        self.paused_accum_s = 0.0
        self.reminder_tracker: dict[str, dict[str, Any]] = {}
        self.timeout_check_last_logged_at: dict[str, float] = {}
        self.timeout_check_last_pause_reason: dict[str, str] = {}

        self._on_transition = on_transition
        self._on_timeout_check = on_timeout_check

    # Side-effect wrappers via callbacks.
    def transition(self, reason: str, context: dict[str, Any]) -> ConfirmationTransitionDecision:
        next_state = context.get("state")
        if isinstance(next_state, str):
            next_state = ConfirmationState(next_state)
        if not isinstance(next_state, ConfirmationState):
            next_state = self.state

        prev_state = self.state
        if prev_state != next_state and self._on_transition is not None:
            self._on_transition(prev_state, next_state, reason, self.pending_token)
        self.state = next_state
        if self.state == ConfirmationState.AWAITING_DECISION and self.pending_token is not None:
            metadata = self.pending_token.metadata if isinstance(self.pending_token.metadata, dict) else {}
            metadata.setdefault("awaiting_decision_since", context.get("now"))
            self.pending_token.metadata = metadata

        token_active = context.get("token_active", self.pending_token is not None)
        hold_active = context.get("hold_active", bool(token_active or self.awaiting_confirmation_completion))
        phase_is_awaiting = context.get("phase_is_awaiting", False)
        was_confirmation_guarded = bool(context.get("was_confirmation_guarded", False))
        allow_response_transition = bool((not token_active and not phase_is_awaiting) or hold_active)
        emit_reminder = bool(context.get("emit_reminder", False))
        recover_mic = bool(was_confirmation_guarded and hold_active)
        return ConfirmationTransitionDecision(
            allow_response_transition=allow_response_transition,
            emit_reminder=emit_reminder,
            recover_mic=recover_mic,
            close_reason=context.get("close_reason"),
        )

    def on_token_started(self, token: Any, now: float) -> None:
        self.pending_token = token
        self.token_created_at = float(now)
        self.last_activity_at = float(now)
        self.pause_started_at = None
        self.paused_accum_s = 0.0

    def on_user_activity(self, kind: str, now: float) -> None:
        _ = kind
        self.last_activity_at = float(now)
        self._refresh_pause(now=now)

    def check_timeout(self, now: float) -> ConfirmationTimeoutDecision:
        token = self.pending_token
        if token is None or self.state != ConfirmationState.AWAITING_DECISION:
            return ConfirmationTimeoutDecision(expired=False, pause_reason=None, remaining_s=0.0)
        self._refresh_pause(now=now)
        pause_reason = self._pause_reason()
        remaining_s = self.remaining_seconds(now=now)
        if self._on_timeout_check is not None:
            self._on_timeout_check(token, remaining_s, pause_reason)
        if pause_reason is not None:
            return ConfirmationTimeoutDecision(expired=False, pause_reason=pause_reason, remaining_s=remaining_s)
        return ConfirmationTimeoutDecision(expired=remaining_s <= 0.0, pause_reason=None, remaining_s=remaining_s)

    def evaluate_reminder(self, *, key: str | None, schedule: tuple[float, ...], now: float) -> ConfirmationReminderDecision:
        if key is None:
            return ConfirmationReminderDecision(False, None, 0, None, "no_confirmation_key", now)
        if self.reminder_max_count <= 0:
            return ConfirmationReminderDecision(False, key, 0, None, "max_count", now)
        entry = self.reminder_tracker.get(key, {})
        sent_count = int(entry.get("count", 0))
        last_sent_at = entry.get("last_sent_at")
        if isinstance(last_sent_at, (int, float)):
            last_sent_at = float(last_sent_at)
        else:
            last_sent_at = None
        token_created_at = float(self.token_created_at) if isinstance(self.token_created_at, (int, float)) else now
        elapsed_s = max(0.0, now - token_created_at)
        if sent_count >= self.reminder_max_count:
            return ConfirmationReminderDecision(False, key, sent_count, last_sent_at, "max_count", now)
        if sent_count < len(schedule) and elapsed_s < float(schedule[sent_count]):
            return ConfirmationReminderDecision(False, key, sent_count, last_sent_at, "schedule", now)
        if last_sent_at is not None and (now - last_sent_at) < self.reminder_interval_s:
            return ConfirmationReminderDecision(False, key, sent_count, last_sent_at, "interval", now)
        return ConfirmationReminderDecision(True, key, sent_count + 1, now, None, now)

    def mark_reminder_sent(self, decision: ConfirmationReminderDecision, *, reason: str) -> None:
        if not decision.allowed or decision.key is None:
            return
        self.reminder_tracker[decision.key] = {
            "count": decision.sent_count,
            "last_sent_at": decision.sent_at,
            "last_reason": reason,
            "last_token_id": getattr(self.pending_token, "id", None),
        }

    def remaining_seconds(self, *, now: float) -> float:
        timeout_s = self._timeout_s_for_token(self.pending_token)
        return max(0.0, timeout_s - self.effective_elapsed_s(now=now))

    def effective_elapsed_s(self, *, now: float) -> float:
        if self.pending_token is None:
            return 0.0
        started_at = self.token_created_at
        if not isinstance(started_at, (int, float)):
            started_at = float(getattr(self.pending_token, "created_at", now))
        paused_s = float(self.paused_accum_s)
        if isinstance(self.pause_started_at, (int, float)) and self._pause_reason() is not None:
            paused_s += max(0.0, now - float(self.pause_started_at))
        return max(0.0, now - float(started_at) - paused_s)

    def _timeout_s_for_token(self, token: Any | None) -> float:
        if token is not None and getattr(token, "kind", "") == "research_permission":
            return self.research_permission_timeout_s
        return self.awaiting_decision_timeout_s

    def _pause_reason(self) -> str | None:
        if self.speech_active:
            return "speech_active"
        if self.asr_pending:
            return "asr_pending"
        return None

    def _refresh_pause(self, *, now: float) -> None:
        if self.pending_token is None:
            self.pause_started_at = None
            return
        if self._pause_reason() is None:
            if isinstance(self.pause_started_at, (int, float)):
                self.paused_accum_s += max(0.0, now - float(self.pause_started_at))
            self.pause_started_at = None
            return
        if self.pause_started_at is None:
            self.pause_started_at = float(now)

    def serialize_debug_state(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "token_id": getattr(self.pending_token, "id", None),
            "token_kind": getattr(self.pending_token, "kind", None),
            "token_created_at": self.token_created_at,
            "last_activity_at": self.last_activity_at,
            "speech_active": self.speech_active,
            "asr_pending": self.asr_pending,
            "paused_accum_s": self.paused_accum_s,
            "pause_started_at": self.pause_started_at,
            "reminder_tracker_keys": sorted(self.reminder_tracker.keys()),
        }
