"""Shared types for realtime collaborators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UtteranceContext:
    turn_id: str
    input_event_key: str
    canonical_key: str
    utterance_seq: int


@dataclass
class CanonicalResponseState:
    created: bool = False
    audio_started: bool = False
    deliverable_observed: bool = False
    deliverable_class: str = "unknown"
    done: bool = False
    cancel_sent: bool = False
    origin: str = "unknown"
    response_id: str = ""
    obligation_present: bool = False
    input_event_key: str = ""
    turn_id: str = ""
    obligation: dict[str, Any] | None = None
    blocked_reason: str = ""


@dataclass
class PendingResponseCreate:
    websocket: Any
    event: dict[str, Any]
    origin: str
    turn_id: str
    created_at: float
    reason: str
    record_ai_call: bool = False
    debug_context: dict[str, Any] | None = None
    memory_brief_note: str | None = None
    queued_reminder_key: str | None = None
    enqueued_done_serial: int = 0
    enqueue_seq: int = 0


@dataclass
class ActiveResponseLifecycle:
    response_id: str | None = None
    origin: str = "unknown"
    input_event_key: str | None = None
    canonical_key: str | None = None
    consumes_canonical_slot: bool = True
    confirmation_guarded: bool = False
    preference_guarded: bool = False
