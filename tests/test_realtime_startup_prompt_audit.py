"""Tests for startup prompt lifecycle audit anchors."""

from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.response_terminal_handlers import ResponseTerminalHandlers
from ai.realtime_api import InteractionState, RealtimeAPI


class _StateManager:
    def __init__(self) -> None:
        self.state = InteractionState.IDLE

    def update_state(self, state: InteractionState, _reason: str) -> None:
        self.state = state


class _TransitionDecision:
    def __init__(self) -> None:
        self.allow_response_transition = True
        self.close_reason = "none"
        self.recover_mic = False


class _Transport:
    async def send_json(self, _websocket: object, _event: dict) -> None:
        return None


def test_startup_prompt_audit_emitters_are_single_transition(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn-startup"

    logs: list[str] = []
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: logs.append(message % args))

    api._emit_startup_prompt_dispatched(turn_id="turn-startup", input_event_key="evt-startup")
    api._emit_startup_prompt_dispatched(turn_id="turn-startup", input_event_key="evt-startup")
    api._emit_startup_prompt_bound(
        turn_id="turn-startup",
        input_event_key="evt-startup",
        canonical_key="run-test:turn-startup:evt-startup",
        reason="response_created",
    )
    api._emit_startup_prompt_bound(
        turn_id="turn-startup",
        input_event_key="evt-startup",
        canonical_key="run-test:turn-startup:evt-startup",
        reason="response_created",
    )
    api._emit_startup_prompt_terminal(
        terminal_state="completed",
        reason="response_done",
        turn_id="turn-startup",
        input_event_key="evt-startup",
        canonical_key="run-test:turn-startup:evt-startup",
    )
    api._emit_startup_prompt_terminal(
        terminal_state="skipped",
        reason="cancelled_before_done",
    )

    assert sum(1 for line in logs if line.startswith("startup_prompt_dispatched")) == 1
    assert sum(1 for line in logs if line.startswith("startup_prompt_bound")) == 1
    assert sum(1 for line in logs if line.startswith("startup_prompt_terminal")) == 1


def test_send_initial_prompts_emits_terminal_when_response_create_not_sent(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.prompts = ["hello startup"]
    api._current_response_turn_id = "turn-startup"
    api._current_input_event_key = "evt-startup"
    api._current_turn_id_or_unknown = lambda: "turn-startup"
    api._current_run_id = lambda: "run-test"
    api._record_user_input = lambda *_args, **_kwargs: None
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._consume_pending_memory_brief_note = lambda: None
    api._allow_ai_call = lambda _reason: True

    async def _false(*_args, **_kwargs) -> bool:
        return False

    api._maybe_process_research_intent = _false
    api._send_response_create = _false
    api._get_or_create_transport = lambda: _Transport()

    terminal_calls: list[tuple[str, str]] = []

    def _capture_terminal(*, terminal_state: str, reason: str, **_kwargs) -> None:
        terminal_calls.append((terminal_state, reason))

    api._emit_startup_prompt_terminal = _capture_terminal

    asyncio.run(api.send_initial_prompts(websocket=object()))

    assert terminal_calls == [("skipped", "cancelled_before_done")]


def test_response_completed_emits_startup_terminal_for_prompt_origin() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._active_response_id = "resp-1"
    api._active_response_origin = "prompt"
    api._active_response_input_event_key = ""
    api._active_response_consumes_canonical_slot = False
    api._active_server_auto_input_event_key = None
    api._active_response_confirmation_guarded = False
    api._active_response_preference_guarded = False
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._response_in_flight = False
    api.response_in_progress = False
    api.rate_limits = {}
    api._current_response_turn_id = "turn-startup"
    api._current_turn_id_or_unknown = lambda: "turn-startup"
    api._response_id_from_event = lambda event: str((event or {}).get("response", {}).get("id") or "")
    api._cancel_micro_ack = lambda **_kwargs: None
    api.state_manager = _StateManager()
    api._clear_cancelled_response_tracking = lambda _response_id: None
    api._emit_preference_recall_skip_trace_if_needed = lambda **_kwargs: None
    api._log_turn_conversation_efficiency = lambda **_kwargs: None
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key or 'none'}"
    api._build_confirmation_transition_decision = lambda **_kwargs: _TransitionDecision()
    api._confirmation_hold_components = lambda: (False, False, None, False)
    api.orchestration_state = type("_Orch", (), {"phase": None, "transition": lambda self, *_a, **_k: None})()
    api._maybe_enqueue_reflection = lambda *_args, **_kwargs: None

    async def _noop_drain(*_args, **_kwargs) -> None:
        return None

    api._drain_response_create_queue = _noop_drain
    api._response_delivery_state = lambda **_kwargs: "created"

    terminal_calls: list[tuple[str, str]] = []
    api._emit_startup_prompt_terminal = lambda *, terminal_state, reason, **_kwargs: terminal_calls.append((terminal_state, reason))

    handler = ResponseTerminalHandlers(api)
    asyncio.run(handler.handle_response_completed({"type": "response.completed", "response": {"id": "resp-1"}}))

    assert terminal_calls == [("completed", "response_done")]


def test_startup_prompt_superseded_reason_vocabulary() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._current_turn_id_or_unknown = lambda: "turn-startup"
    api._current_run_id = lambda: "run-test"

    terminal_calls: list[tuple[str, str]] = []
    api._emit_startup_prompt_terminal = lambda *, terminal_state, reason, **_kwargs: terminal_calls.append((terminal_state, reason))

    api._emit_startup_prompt_dispatched(turn_id="turn-startup", input_event_key="evt-startup")
    api._maybe_mark_startup_prompt_superseded(source="text_message")

    assert terminal_calls == [("superseded", "first_live_user_turn_source=text_message")]
