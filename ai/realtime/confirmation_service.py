"""Confirmation lifecycle orchestration service for realtime flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai.realtime.confirmation import ConfirmationState, normalize_confirmation_decision


@dataclass(frozen=True)
class ConfirmationUserTextDecision:
    parsed_decision: str
    should_execute: bool = False
    should_reject: bool = False
    should_cancel_timeout: bool = False
    should_retry_prompt: bool = False
    retry_exhausted: bool = False
    retry_count: int = 0


@dataclass(frozen=True)
class ConfirmationTimeoutOutcome:
    expired: bool
    token: Any | None
    remaining_s: float


class ConfirmationService:
    """Single owner for confirmation token/state lifecycle mutations."""

    def __init__(self, *, awaiting_timeout_s: float, late_decision_grace_s: float) -> None:
        self.awaiting_timeout_s = max(0.0, float(awaiting_timeout_s))
        self.late_decision_grace_s = max(0.0, float(late_decision_grace_s))
        self.pending_token: Any | None = None
        self.state = ConfirmationState.IDLE
        self.last_closed: dict[str, Any] | None = None
        self.timeout_markers: dict[str, float] = {}
        self.timeout_causes: dict[str, str] = {}
        self.recent_outcomes: dict[str, dict[str, Any]] = {}

    def start_pending(self, token: Any, pending_action: Any, now: float) -> None:
        token.pending_action = pending_action
        token.created_at = float(now)
        self.pending_token = token
        self.state = ConfirmationState.PENDING_PROMPT

    def handle_user_text(self, text: str, now: float) -> ConfirmationUserTextDecision:
        _ = now
        token = self.pending_token
        decision = normalize_confirmation_decision(text)
        if token is None:
            return ConfirmationUserTextDecision(parsed_decision=decision)
        if decision == "yes":
            self.state = ConfirmationState.RESOLVING
            return ConfirmationUserTextDecision(parsed_decision=decision, should_execute=True)
        if decision in {"no", "cancel"}:
            self.state = ConfirmationState.RESOLVING
            return ConfirmationUserTextDecision(parsed_decision=decision, should_reject=True)
        token.retry_count += 1
        exhausted = token.retry_count > token.max_retries
        if exhausted:
            self.state = ConfirmationState.RESOLVING
        return ConfirmationUserTextDecision(
            parsed_decision=decision,
            should_retry_prompt=not exhausted,
            retry_exhausted=exhausted,
            retry_count=token.retry_count,
        )

    def on_timeout_tick(self, now: float) -> ConfirmationTimeoutOutcome:
        token = self.pending_token
        if token is None or self.state != ConfirmationState.AWAITING_DECISION:
            return ConfirmationTimeoutOutcome(expired=False, token=token, remaining_s=0.0)
        elapsed_s = max(0.0, float(now) - float(getattr(token, "created_at", now)))
        remaining_s = max(0.0, self.awaiting_timeout_s - elapsed_s)
        if remaining_s > 0.0:
            return ConfirmationTimeoutOutcome(expired=False, token=token, remaining_s=remaining_s)
        self.state = ConfirmationState.RESOLVING
        return ConfirmationTimeoutOutcome(expired=True, token=token, remaining_s=0.0)

    def close(self, *, outcome: str, now: float) -> Any | None:
        token = self.pending_token
        if token is None:
            return None
        self.state = ConfirmationState.COMPLETED
        self.last_closed = {
            "token": token,
            "token_id": getattr(token, "id", ""),
            "kind": getattr(token, "kind", ""),
            "outcome": outcome,
            "closed_at": float(now),
        }
        self.pending_token = None
        self.state = ConfirmationState.IDLE
        return token

    def build_prompt(
        self,
        *,
        action_summary: str,
        confirm_reason: str,
        dry_run_supported: bool,
        confirm_prompt: str | None,
    ) -> str:
        options = "Approve / Deny / Dry-run" if dry_run_supported else "Approve / Deny"
        options_suffix = f"options: {options}."
        if confirm_prompt:
            prompt_override = " ".join(str(confirm_prompt).split()).strip()
            if self._is_contract_compliant(prompt_override, options_suffix=options_suffix):
                return prompt_override
        summary = self._normalize_sentence(action_summary)
        reason = self._normalize_sentence(confirm_reason)
        return f"Action summary: {summary} Reason: {reason}; {options_suffix}"

    @staticmethod
    def _normalize_sentence(value: str | None) -> str:
        normalized = " ".join(str(value or "").split()).strip()
        if normalized.endswith("."):
            return normalized
        return f"{normalized}."

    @staticmethod
    def _is_contract_compliant(prompt_text: str, *, options_suffix: str) -> bool:
        parts = [part.strip() for part in prompt_text.split(" Reason: ", maxsplit=1)]
        if len(parts) != 2 or not parts[0].startswith("Action summary: "):
            return False
        reason_clause = parts[1]
        expected_suffix = f"; {options_suffix}"
        if not reason_clause.endswith(expected_suffix):
            return False
        action_body = parts[0][len("Action summary: ") :].strip()
        reason_body = reason_clause[: -len(expected_suffix)].strip()
        return bool(action_body.endswith(".") and reason_body.endswith("."))
