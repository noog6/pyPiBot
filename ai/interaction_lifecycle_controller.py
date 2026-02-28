"""Deterministic interaction lifecycle controller keyed by canonical utterance key."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InteractionLifecycleState(str, Enum):
    NEW = "new"
    SERVER_AUTO_CREATED = "server_auto_created"
    CANCELLED = "cancelled"
    AUDIO_STARTED = "audio_started"
    DONE = "done"
    REPLACED = "replaced"


class LifecycleDecisionAction(str, Enum):
    ALLOW = "allow"
    CANCEL = "cancel"
    DEFER = "defer"


@dataclass(frozen=True)
class LifecycleDecision:
    action: LifecycleDecisionAction
    reason: str


@dataclass
class _LifecycleRecord:
    state: InteractionLifecycleState = InteractionLifecycleState.NEW
    transcript_final_seen: bool = False


class InteractionLifecycleController:
    """Tracks canonical interaction lifecycle transitions and emits deterministic decisions."""

    def __init__(self) -> None:
        self._records: dict[str, _LifecycleRecord] = {}

    def _record_for(self, canonical_key: str) -> _LifecycleRecord:
        key = str(canonical_key or "").strip()
        if not key:
            return _LifecycleRecord()
        existing = self._records.get(key)
        if existing is None:
            existing = _LifecycleRecord()
            self._records[key] = existing
        return existing

    def state_for(self, canonical_key: str) -> InteractionLifecycleState:
        return self._record_for(canonical_key).state

    def on_transcript_final(self, canonical_key: str) -> None:
        record = self._record_for(canonical_key)
        record.transcript_final_seen = True

    def decide_response_create_allow(self, canonical_key: str, *, origin: str) -> LifecycleDecision:
        record = self._record_for(canonical_key)
        if record.state in {
            InteractionLifecycleState.CANCELLED,
            InteractionLifecycleState.DONE,
            InteractionLifecycleState.REPLACED,
            InteractionLifecycleState.SERVER_AUTO_CREATED,
        }:
            return LifecycleDecision(LifecycleDecisionAction.CANCEL, f"state={record.state.value}")
        normalized_origin = str(origin or "").strip().lower()
        if normalized_origin == "server_auto" and not record.transcript_final_seen:
            return LifecycleDecision(
                LifecycleDecisionAction.DEFER,
                "awaiting_transcript_final",
            )
        return LifecycleDecision(LifecycleDecisionAction.ALLOW, "state=new")

    def on_response_created(self, canonical_key: str, *, origin: str) -> LifecycleDecision:
        allow = self.decide_response_create_allow(canonical_key, origin=origin)
        if allow.action is not LifecycleDecisionAction.ALLOW:
            return allow
        record = self._record_for(canonical_key)
        record.state = InteractionLifecycleState.SERVER_AUTO_CREATED
        return LifecycleDecision(LifecycleDecisionAction.ALLOW, "transitioned=server_auto_created")

    def on_audio_delta(self, canonical_key: str) -> LifecycleDecision:
        record = self._record_for(canonical_key)
        if record.state in {InteractionLifecycleState.CANCELLED, InteractionLifecycleState.REPLACED}:
            return LifecycleDecision(LifecycleDecisionAction.CANCEL, f"state={record.state.value}")
        if record.state == InteractionLifecycleState.DONE:
            return LifecycleDecision(LifecycleDecisionAction.CANCEL, "state=done")
        if record.state in {
            InteractionLifecycleState.NEW,
            InteractionLifecycleState.SERVER_AUTO_CREATED,
        }:
            record.state = InteractionLifecycleState.AUDIO_STARTED
            return LifecycleDecision(LifecycleDecisionAction.ALLOW, "transitioned=audio_started")
        if record.state == InteractionLifecycleState.AUDIO_STARTED:
            return LifecycleDecision(LifecycleDecisionAction.ALLOW, "state=audio_started")
        record.state = InteractionLifecycleState.AUDIO_STARTED
        return LifecycleDecision(LifecycleDecisionAction.ALLOW, "transitioned=audio_started")

    def on_response_done(self, canonical_key: str) -> None:
        record = self._record_for(canonical_key)
        record.state = InteractionLifecycleState.DONE

    def on_cancel_sent(self, canonical_key: str) -> None:
        record = self._record_for(canonical_key)
        record.state = InteractionLifecycleState.CANCELLED

    def on_replaced(self, old_canonical_key: str, new_canonical_key: str) -> None:
        old_record = self._record_for(old_canonical_key)
        old_record.state = InteractionLifecycleState.REPLACED
        new_record = self._record_for(new_canonical_key)
        if old_record.transcript_final_seen:
            new_record.transcript_final_seen = True

    def audio_started(self, canonical_key: str) -> bool:
        state = self.state_for(canonical_key)
        return state == InteractionLifecycleState.AUDIO_STARTED
