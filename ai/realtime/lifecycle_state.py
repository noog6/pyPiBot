"""Lifecycle state coordination helpers for RealtimeAPI."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from ai.realtime.types import CanonicalResponseState, UtteranceContext


class LifecycleStateCoordinator:
    """Centralizes canonical key, obligation, and release-gate lifecycle logic."""

    def canonical_utterance_key(self, *, run_id: str | None, turn_id: str, input_event_key: str | None) -> str:
        resolved_turn_id = str(turn_id or "").strip() or "turn-unknown"
        resolved_input_event_key = str(input_event_key or "").strip() or f"synthetic:{resolved_turn_id}"
        resolved_run_id = str(run_id or "").strip() or "run-unknown"
        return f"{resolved_run_id}:{resolved_turn_id}:{resolved_input_event_key}"

    def build_utterance_context(
        self,
        *,
        run_id: str | None,
        turn_id: str,
        input_event_key: str | None,
        utterance_seq: int,
    ) -> UtteranceContext:
        resolved_turn_id = str(turn_id or "").strip() or "turn-unknown"
        resolved_input_event_key = str(input_event_key or "").strip()
        return UtteranceContext(
            turn_id=resolved_turn_id,
            input_event_key=resolved_input_event_key,
            canonical_key=self.canonical_utterance_key(
                run_id=run_id,
                turn_id=resolved_turn_id,
                input_event_key=resolved_input_event_key,
            ),
            utterance_seq=int(utterance_seq or 0),
        )

    def active_input_event_key_for_turn(self, bindings: dict[str, str], *, turn_id: str) -> str:
        return str(bindings.get(turn_id) or "").strip()

    def bind_active_input_event_key_for_turn(
        self,
        bindings: dict[str, str],
        *,
        turn_id: str,
        input_event_key: str | None,
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip()
        normalized_key = str(input_event_key or "").strip()
        if normalized_turn_id and normalized_key:
            bindings[normalized_turn_id] = normalized_key

    def response_obligation_key(self, *, run_id: str | None, turn_id: str, input_event_key: str | None) -> str:
        return self.canonical_utterance_key(run_id=run_id, turn_id=turn_id, input_event_key=input_event_key)

    def transition_response_obligation(
        self,
        *,
        state_store: dict[str, CanonicalResponseState],
        obligation_key: str,
        turn_id: str,
        input_event_key: str | None,
        source: str | None,
        present: bool,
    ) -> tuple[CanonicalResponseState, bool]:
        prior_state = state_store.get(obligation_key)
        if not isinstance(prior_state, CanonicalResponseState):
            prior_state = CanonicalResponseState()
        next_state = replace(prior_state)
        normalized_input_event_key = str(input_event_key or "").strip()
        normalized_turn_id = str(turn_id or "").strip()
        if normalized_turn_id:
            next_state.turn_id = normalized_turn_id
        if normalized_input_event_key:
            next_state.input_event_key = normalized_input_event_key

        obligation_present_before = bool(prior_state.obligation_present)
        if present:
            next_state.obligation_present = True
            next_state.origin = str(source or "").strip() or next_state.origin
            next_state.obligation = {
                "turn_id": turn_id,
                "input_event_key": normalized_input_event_key,
                "source": source,
                "created_at": time.monotonic(),
            }
        else:
            next_state.obligation_present = False
            next_state.obligation = None

        state_store[obligation_key] = next_state
        return next_state, obligation_present_before

    def is_user_confirmation_trigger(self, trigger: str, metadata: dict[str, Any]) -> bool:
        normalized = str(trigger).strip().lower()
        if normalized == "text_message":
            return True
        source = str(metadata.get("source", "")).strip().lower()
        return source in {"user_audio", "user_text", "voice_confirmation", "text_confirmation"}

    def can_release_queued_response_create(
        self,
        *,
        has_active_confirmation_token: bool,
        is_awaiting_confirmation_phase: bool,
        trigger: str,
        metadata: dict[str, Any],
    ) -> bool:
        if not has_active_confirmation_token and not is_awaiting_confirmation_phase:
            return True
        if self.is_user_confirmation_trigger(trigger, metadata):
            return True
        approval_flow = str(metadata.get("approval_flow", "")).strip().lower()
        return approval_flow in {"true", "1", "yes"}
