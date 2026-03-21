"""Regression coverage for turn-level response guards."""

from __future__ import annotations

import asyncio
import time
import sys
import types
from dataclasses import replace

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from collections import deque

from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime.types import CanonicalResponseState
from ai.embodiment_policy import EmbodimentActionType, EmbodimentDecision, EmbodimentPolicy
from ai.governance_spine import GovernanceDecision
from ai.opportunistic_arbitration import OpportunisticActionCandidate
from ai.realtime_api import PendingResponseCreate, RealtimeAPI
from core.logging import logger
from interaction import InteractionState


class _FakeStateManager:
    state = None

    def update_state(self, state, _reason: str) -> None:
        self.state = state


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = "turn_1"
    api._current_input_event_key = None
    api._synthetic_input_event_counter = 0
    api._response_created_canonical_keys = set()
    api._empty_response_retry_canonical_keys = set()
    api._response_delivery_ledger = {}
    api._empty_response_retry_counts = {}
    api._empty_response_retry_fallback_emitted = set()
    api._empty_response_retry_max_attempts = 2
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._pending_response_create = None
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_schedule_logged_turn_ids = set()
    api._conversation_efficiency_by_turn = {}
    api._conversation_efficiency_logged_turns = set()
    api._silent_turn_incident_count = 0
    api._turn_diagnostic_timestamps = {}
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._preference_recall_locked_input_event_keys = set()
    api._active_response_confirmation_guarded = False
    api._active_response_preference_guarded = False
    api._pending_confirmation_token = None
    api._pending_action = None
    api.orchestration_state = types.SimpleNamespace(phase=None)
    api._active_server_auto_input_event_key = None
    api._active_input_event_key_by_turn_id = {}
    api._curiosity_anchor_decay_window_s = 10.0
    api._curiosity_anchor_max_entries = 3
    api._curiosity_anchor_stats_by_anchor = {}
    api._curiosity_surface_max_turns = 2
    api._curiosity_surface_candidate_by_turn_id = {}
    api._response_create_runtime = ResponseCreateRuntime(api)
    api._current_run_id = lambda: "run-395"
    api._extract_confirmation_reminder_dedupe_key = lambda event: None
    api._sync_pending_response_create_queue = lambda: None
    api._mark_transcript_response_outcome = lambda **kwargs: None
    api._can_release_queued_response_create = lambda trigger, metadata: True
    api.websocket = None

    async def _noop_async(*_args, **_kwargs):
        return None

    api._enqueue_response_done_reflection = _noop_async
    api._emit_preference_recall_skip_trace_if_needed = lambda **kwargs: None
    api.state_manager = _FakeStateManager()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._turn_contracts_by_turn_id = {}
    api._turn_contract_fallback = None
    api._last_interaction_state = InteractionState.IDLE
    api._embodiment_policy = EmbodimentPolicy()
    api._last_gesture_time = 0.0
    api._gesture_global_cooldown_s = 0.0
    api._gesture_last_fired = {}
    api._gesture_cooldowns_s = {}
    api.rate_limits = {}
    api._last_response_metadata = {}
    return api


def test_turn_contract_parse_detects_exact_phrase_and_no_gesture() -> None:
    api = _make_api()

    contract = api._parse_turn_contract_from_text(
        'Say exactly "Ready. Run 556 online." Do not call tools or gesture.'
    )

    assert contract["exact_phrase"] == "Ready. Run 556 online."
    assert contract["no_tools"] is True
    assert contract["no_gesture"] is True


def test_turn_contract_blocks_state_gesture_cues() -> None:
    api = _make_api()
    api._last_interaction_state = InteractionState.SPEAKING
    api._update_turn_contract_from_input("Say exactly 'Ready.' Do not call tools or gesture.", source="startup_prompt")

    api._handle_state_gesture(InteractionState.IDLE)

    assert api._last_interaction_state == InteractionState.IDLE


def test_turn_contract_blocks_gesture_tool_calls() -> None:
    api = _make_api()
    api._update_turn_contract_from_input("Do not gesture.", source="input_audio_transcription")

    assert api._turn_contract_blocks_tool_call(tool_name="gesture_attention_snap") is True
    assert api._turn_contract_blocks_tool_call(tool_name="perform_research") is False


def test_turn_contract_allows_explicit_requested_gesture_even_when_no_gesture_present() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Do one attention snap, then do not gesture.",
        source="input_audio_transcription",
    )

    assert api._turn_contract_blocks_tool_call(tool_name="gesture_attention_snap") is False
    assert api._turn_contract_blocks_tool_call(tool_name="gesture_nod") is True


def test_turn_contract_exact_phrase_repair_scheduled_when_parent_missing_phrase() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Theo, do one attention snap, then say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""
    sent_events: list[dict[str, object]] = []

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    scheduled = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )

    assert scheduled is True
    assert len(sent_events) == 1
    metadata = sent_events[0]["event"]["response"]["metadata"]
    assert metadata["turn_contract_exact_phrase_repair"] == "true"
    assert metadata["input_event_key"] == "item_1:exact_phrase_repair"
    assert (
        sent_events[0]["event"]["response"]["instructions"]
        == "Exact phrase repair mode. Speak exactly this sentence and nothing else: 'Sentinel Theo online.'."
    )


def test_turn_contract_required_phrase_repair_uses_exact_phrase_instruction() -> None:
    api = _make_api()
    api._turn_contracts_by_turn_id = {
        "turn_1": {
            "required_phrase": "Ready!",
            "exact_phrase": "",
            "exact_phrase_repair_scheduled": False,
        }
    }
    api._assistant_reply_accum = ""
    sent_events: list[dict[str, object]] = []

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    scheduled = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )

    assert scheduled is True
    assert sent_events[0]["event"]["response"]["instructions"] == (
        "Exact phrase repair mode. Speak exactly this sentence and nothing else: 'Ready!'."
    )


def test_turn_contract_exact_phrase_repair_counts_queued_pending_as_scheduled() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""

    async def _queue_but_do_not_send(_websocket, event, **_kwargs):
        api._pending_response_create = PendingResponseCreate(
            websocket=None,
            event=event,
            origin="assistant_message",
            turn_id="turn_1",
            created_at=time.monotonic(),
            reason="audio_playback_busy",
            record_ai_call=False,
            debug_context=None,
            memory_brief_note=None,
            queued_reminder_key=None,
            enqueued_done_serial=0,
            enqueue_seq=1,
        )
        return False

    api._send_response_create = _queue_but_do_not_send

    scheduled = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )

    assert scheduled is True
    assert api._turn_contracts_by_turn_id["turn_1"]["exact_phrase_repair_scheduled"] is True


def test_turn_contract_exact_phrase_repair_schedules_at_most_once() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""
    sent_events: list[dict[str, object]] = []

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    first = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )
    second = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )

    assert first is True
    assert second is False
    assert len(sent_events) == 1




def test_turn_contract_prompt_contract_adopted_from_unknown_turn() -> None:
    api = _make_api()
    api._current_response_turn_id = ""

    api._update_turn_contract_from_input(
        "Say exactly 'Ready. Run 556 online.' Do not call tools or gesture.",
        source="startup_prompt",
    )
    api._adopt_unknown_turn_contract(turn_id="turn_1")

    assert "turn-unknown" not in api._turn_contracts_by_turn_id
    assert api._turn_contract_blocks_gesture_cues(turn_id="turn_1") is True
    assert api._turn_contract_exact_phrase(turn_id="turn_1") == "Ready. Run 556 online."


def test_turn_contract_parse_detects_required_phrase_without_exactly() -> None:
    api = _make_api()

    contract = api._parse_turn_contract_from_text(
        "Do one attention snap, then say, Sentinel Theo online."
    )

    assert contract["exact_phrase"] == ""
    assert contract["required_phrase"] == "Sentinel Theo online."




def test_turn_contract_parse_does_not_map_generic_release_words_to_attention_release() -> None:
    api = _make_api()

    release_answer = api._parse_turn_contract_from_text("Please release the answer once it's ready.")
    release_resources = api._parse_turn_contract_from_text("Before shutdown, release resources cleanly.")

    assert "gesture_attention_release" not in release_answer["explicit_gesture_tools"]
    assert "gesture_attention_release" not in release_resources["explicit_gesture_tools"]


def test_response_done_decision_holds_terminal_selection_when_required_phrase_open() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Do one attention snap, then say, Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""

    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="server_auto",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is False
    assert reason == "exact_phrase_obligation_open"


def test_turn_contract_required_phrase_not_open_when_already_spoken() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Do one attention snap, then say, Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = "Sentinel Theo online."

    assert api._turn_contract_exact_phrase_open(turn_id="turn_1") is False


def test_startup_required_phrase_normalization_variants_satisfy_contract() -> None:
    variants = ["ready", "Ready!", " Ready! ", "Ready."]

    for spoken in variants:
        api = _make_api()
        api._update_turn_contract_from_input("Say Ready!", source="startup_prompt")
        api._terminal_response_text_by_response_id = {"resp_ready": spoken}

        assert api._turn_contract_satisfied_by_text(
            phrase="Ready",
            mode="include",
            spoken_text=spoken,
        ) is True
        assert api._turn_contract_exact_phrase_open(turn_id="turn_1", response_id="resp_ready") is False


def test_startup_required_phrase_uses_per_response_reply_text_when_terminal_text_missing() -> None:
    api = _make_api()
    api._update_turn_contract_from_input("Say Ready!", source="startup_prompt")
    api._assistant_reply_accum = ""
    api.assistant_reply = ""
    api._assistant_reply_by_response_id = {"resp_ready": "Ready!"}

    assert api._turn_contract_exact_phrase_open(turn_id="turn_1", response_id="resp_ready") is False


def test_startup_required_phrase_remains_open_for_unsatisfied_transcript() -> None:
    api = _make_api()
    api._update_turn_contract_from_input("Say Ready!", source="startup_prompt")
    api._terminal_response_text_by_response_id = {"resp_hello": "Hello there."}

    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="prompt",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_prompt_1"),
        transcript_final_seen=True,
        input_event_key="synthetic_prompt_1",
        response_id="resp_hello",
    )

    assert selected is False
    assert reason == "exact_phrase_obligation_open"


def test_startup_required_phrase_uses_terminal_transcript_after_reply_buffer_clears() -> None:
    api = _make_api()
    api._update_turn_contract_from_input("Say Ready!", source="startup_prompt")
    api._assistant_reply_accum = ""
    api.assistant_reply = ""
    api._terminal_response_text_by_response_id = {"resp_ready": "Ready!"}

    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="prompt",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_prompt_1"),
        transcript_final_seen=True,
        input_event_key="synthetic_prompt_1",
        response_id="resp_ready",
    )

    assert selected is True
    assert reason == "normal"
    assert api._turn_contract_required_phrase(turn_id="turn_1") == ""


def test_turn_contract_exact_phrase_not_open_when_already_spoken() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = "Sentinel Theo online."

    assert api._turn_contract_exact_phrase_open(turn_id="turn_1") is False
    assert api._turn_contract_exact_phrase(turn_id="turn_1") == ""


def test_response_done_decision_holds_terminal_selection_when_exact_phrase_open() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Do one attention snap, then say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""

    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="server_auto",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is False
    assert reason == "exact_phrase_obligation_open"


def test_empty_response_done_schedules_single_retry_for_prompt_origin() -> None:
    api = _make_api()
    sent_events: list[dict[str, object]] = []
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )
    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert len(sent_events) == 1
    metadata = sent_events[0]["event"]["response"]["metadata"]
    assert metadata["retry_reason"] == "empty_response_done"
    assert metadata["idempotency_key"].startswith("empty_response_done:")



def test_empty_response_retry_uses_origin_canonical_key_and_exhausts_with_terminal_fallback() -> None:
    api = _make_api()
    api._empty_response_retry_max_attempts = 1
    sent_events: list[dict[str, object]] = []
    fallback_messages: list[tuple[str, bool]] = []
    terminal_markers: list[str] = []

    origin_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")
    retry_canonical_key = api._canonical_utterance_key(
        turn_id="turn_1",
        input_event_key="input_evt_1__empty_retry",
    )

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    async def _capture_assistant_message(message, _websocket, *, speak=True, **kwargs):
        fallback_messages.append((message, speak))

    def _capture_log_lifecycle_event(**kwargs):
        terminal_markers.append(str(kwargs.get("decision") or ""))

    api._send_response_create = _capture_send_response_create
    api.send_assistant_message = _capture_assistant_message
    api._log_lifecycle_event = _capture_log_lifecycle_event
    api._debug_dump_canonical_key_timeline = lambda **kwargs: None

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=origin_canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )
    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=retry_canonical_key,
            input_event_key="input_evt_1__empty_retry",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert len(sent_events) == 1
    assert api._empty_response_retry_counts[origin_canonical_key] == 1
    assert fallback_messages == [("Sorry—I’m having trouble generating a response right now.", False)]
    assert "transition_terminal:empty_response_retry_exhausted" in terminal_markers
    assert api._response_delivery_state(turn_id="turn_1", input_event_key="input_evt_1") == "done"

def test_empty_response_done_retry_skipped_for_cancelled_terminal_state() -> None:
    api = _make_api()
    sent_events: list[dict[str, object]] = []
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="cancelled",
        )
    )

    assert sent_events == []


def test_response_create_guard_blocks_duplicate_for_same_canonical_key() -> None:
    api = _make_api()
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_prompt_1")
    api._response_created_canonical_keys.add(canonical_key)

    blocked = api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "synthetic_prompt_1"}}},
        origin="injection",
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert blocked is False
    assert api._pending_response_create is None


def test_response_create_correlation_adds_non_unknown_input_event_key() -> None:
    api = _make_api()
    event = {"type": "response.create", "response": {"metadata": {}}}

    input_event_key = api._ensure_response_create_correlation(
        response_create_event=event,
        origin="prompt",
        turn_id="turn_1",
    )

    assert input_event_key.startswith("synthetic_prompt_")
    assert event["response"]["metadata"]["input_event_key"] == input_event_key


def test_clear_all_pending_response_creates_clears_pending_and_queue() -> None:
    api = _make_api()
    api._pending_response_create = object()
    api._response_create_queue.append({"event": {"type": "response.create"}})

    api._clear_all_pending_response_creates(reason="talk_over_abort")

    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []


def test_response_create_guard_blocks_default_response_while_preference_lock_active() -> None:
    api = _make_api()
    api._preference_recall_locked_input_event_keys.add("input_evt_1")

    blocked = api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert blocked is False
    assert api._pending_response_create is None


def test_audio_playback_busy_retry_dropped_after_delivery() -> None:
    api = _make_api()
    api._audio_playback_busy = True
    api._set_response_delivery_state(
        turn_id="turn_1",
        input_event_key="input_evt_1",
        state="delivered",
    )

    scheduled = api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        reason="audio_playback_busy",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert scheduled is False
    assert api._pending_response_create is None


def test_audio_playback_busy_retry_enqueues_once_before_delivery() -> None:
    api = _make_api()
    api._audio_playback_busy = True

    api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        reason="audio_playback_busy",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    send_attempts: list[str] = []

    async def _fake_send_response_create(*args, **kwargs):
        send_attempts.append("sent")
        return True

    api._send_response_create = _fake_send_response_create
    api._audio_playback_busy = False
    asyncio.run(api._drain_response_create_queue())

    assert send_attempts == ["sent"]
    assert api._pending_response_create is None


def test_run_411_watchdog_skips_audio_busy_retry_after_delivered_answer(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []
    emitted_response_creates: list[dict[str, object]] = []
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_1", state="delivered")

    async def _capture_send_response_create(*_args, **kwargs):
        emitted_response_creates.append(kwargs)
        return True

    api._send_response_create = _capture_send_response_create

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_1",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert not any(
        "response_not_scheduled" in entry
        and "input_event_key=input_evt_1" in entry
        and "reason=already_handled" in entry
        for entry in info_logs
    )
    assert any(
        "response_not_scheduled" in entry
        and "input_event_key=input_evt_1" in entry
        and "reason=already_handled" in entry
        for entry in debug_logs
    )
    assert not any(
        "response_create_scheduled" in entry and "reason=audio_playback_busy" in entry
        for entry in info_logs
    )


def test_watchdog_audio_busy_after_done_does_not_emit_duplicate_create_or_micro_ack(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []
    emitted_response_creates: list[dict[str, object]] = []
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)

    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_done")
    api._response_created_canonical_keys.add(canonical_key)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_done", state="done")
    initial_assistant_response_count = len(api._response_created_canonical_keys)

    async def _capture_send_response_create(*_args, **kwargs):
        emitted_response_creates.append(kwargs)
        return True

    api._send_response_create = _capture_send_response_create

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_done",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert not any("micro_ack_emitted" in entry for entry in info_logs)
    assert len(api._response_created_canonical_keys) == initial_assistant_response_count
    assert not any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_done" in entry
        for entry in info_logs
    )
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_done" in entry
        for entry in debug_logs
    )


def test_watchdog_audio_busy_terminal_state_ignores_pending_for_other_canonical_key(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []
    emitted_response_creates: list[dict[str, object]] = []
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_target", state="done")
    api._pending_response_create = PendingResponseCreate(
        websocket=None,
        event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_other"}}},
        origin="assistant_message",
        turn_id="turn_1",
        created_at=0.0,
        reason="audio_playback_busy",
    )

    async def _capture_send_response_create(*_args, **kwargs):
        emitted_response_creates.append(kwargs)
        return True

    api._send_response_create = _capture_send_response_create

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_target",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert api._pending_response_create is not None
    assert not any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_target" in entry
        for entry in info_logs
    )
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_target" in entry
        for entry in debug_logs
    )


def test_drain_response_create_queue_drops_terminal_canonical_key(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    transcript_outcomes: list[dict[str, object]] = []
    send_attempts: list[str] = []

    api._mark_transcript_response_outcome = lambda **kwargs: transcript_outcomes.append(kwargs)
    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    turn_id = "turn_1"
    input_event_key = "input_evt_terminal"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._lifecycle_controller().on_response_done(canonical_key)

    api._response_create_queue.append(
        {
            "websocket": None,
            "event": {"type": "response.create", "response": {"metadata": {"input_event_key": input_event_key}}},
            "origin": "assistant_message",
            "turn_id": turn_id,
            "record_ai_call": False,
        }
    )

    async def _fake_send_response_create(*_args, **_kwargs):
        send_attempts.append("sent")
        return True

    api._send_response_create = _fake_send_response_create
    asyncio.run(api._drain_response_create_queue())

    assert send_attempts == []
    assert list(api._response_create_queue) == []
    assert any("response_dropped_terminal_state" in entry for entry in info_logs)
    assert transcript_outcomes == [
        {
            "input_event_key": input_event_key,
            "turn_id": turn_id,
            "outcome": "response_not_scheduled",
            "reason": "canonical_delivery_terminal_state",
            "details": "canonical delivery terminal state origin=assistant_message prior_state=done canonical_key=run-395:turn_1:input_evt_terminal origin_canonical_key=run-395:turn_1:input_evt_terminal",
        }
    ]


def test_empty_response_done_retry_terminal_guard_uses_retry_canonical_key() -> None:
    api = _make_api()
    turn_id = "turn_1"
    origin_input_event_key = "input_evt_1"
    retry_input_event_key = f"{origin_input_event_key}__empty_retry"
    metadata = {"input_event_key": retry_input_event_key, "retry_reason": "empty_response_done"}

    origin_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=origin_input_event_key)
    retry_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=retry_input_event_key)
    api._lifecycle_controller().on_response_done(origin_canonical_key)

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key=retry_input_event_key,
        origin="prompt",
        response_metadata=metadata,
    )

    assert dropped is False
    assert api._lifecycle_controller().state_for(origin_canonical_key).value == "done"
    assert api._lifecycle_controller().state_for(retry_canonical_key).value == "new"


def test_empty_response_done_retry_terminal_guard_drops_when_retry_canonical_terminal() -> None:
    api = _make_api()
    turn_id = "turn_1"
    origin_input_event_key = "input_evt_1"
    retry_input_event_key = f"{origin_input_event_key}__empty_retry"
    metadata = {"input_event_key": retry_input_event_key, "retry_reason": "empty_response_done"}
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)

    retry_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=retry_input_event_key)
    api._lifecycle_controller().on_response_done(retry_canonical_key)

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key=retry_input_event_key,
        origin="prompt",
        response_metadata=metadata,
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "canonical_delivery_terminal_state"
    assert "origin_canonical_key=run-395:turn_1:input_evt_1" in outcomes[0]["details"]


def test_watchdog_micro_ack_not_scheduled_when_terminal_and_idle() -> None:
    api = _make_api()
    scheduled: list[dict[str, object]] = []

    class _Manager:
        def maybe_schedule(self, **kwargs):
            scheduled.append(kwargs)

        def cancel(self, **_kwargs):
            return None

    api._micro_ack_manager = _Manager()
    api.loop = object()
    api._pending_micro_ack_by_turn_channel = {}
    api._micro_ack_near_ready_suppress_ms = 0
    api._micro_ack_correlation_metadata = lambda: {}
    api._active_input_event_key_by_turn = {"turn_1": "input_evt_terminal"}
    api.state_manager.state = InteractionState.IDLE
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_terminal", state="done")

    api._maybe_schedule_micro_ack(
        turn_id="turn_1",
        category=api._micro_ack_category_for_reason("watchdog_audio_playback_busy"),
        channel="voice",
        reason="watchdog_audio_playback_busy",
    )

    assert scheduled == []


def test_response_text_delta_without_bound_response_does_not_enter_speaking() -> None:
    api = _make_api()
    transitions: list[tuple[InteractionState, str]] = []

    class _StateManager:
        def __init__(self) -> None:
            self.state = InteractionState.IDLE

        def update_state(self, state, reason):
            transitions.append((state, reason))
            self.state = state

    api.state_manager = _StateManager()
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **_kwargs: None
    api._mark_first_assistant_utterance_observed_if_needed = lambda _text: None
    api._append_assistant_reply_text = lambda *args, **kwargs: None
    api._response_status_by_id = {}
    api._active_response_id = None

    asyncio.run(
        api._handle_event_legacy(
            {"type": "response.text.delta", "response_id": "resp_terminal", "delta": "late delta"},
            websocket=None,
        )
    )

    assert transitions == []
    assert api.state_manager.state == InteractionState.IDLE


def test_empty_transcript_blocked_state_rejects_followup_response_scheduling() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)

    api._set_response_delivery_state(
        turn_id="turn_1",
        input_event_key="input_evt_empty",
        state="blocked_empty_transcript",
    )

    dropped = api._drop_response_create_for_terminal_state(
        turn_id="turn_1",
        input_event_key="input_evt_empty",
        origin="upgraded_response",
        response_metadata={"input_event_key": "input_evt_empty"},
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "empty_transcript_blocked"


def test_empty_transcript_blocked_state_rejects_injection_response_attempts() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)
    api._set_response_delivery_state(
        turn_id="turn_1",
        input_event_key="input_evt_empty",
        state="blocked_empty_transcript",
    )

    for origin in ("imu_injection", "battery_injection", "system_context"):
        dropped = api._drop_response_create_for_terminal_state(
            turn_id="turn_1",
            input_event_key="input_evt_empty",
            origin=origin,
            response_metadata={
                "input_event_key": "input_evt_empty",
                "create_response": "true",
            },
        )
        assert dropped is True

    assert len(outcomes) == 3
    assert all(entry["reason"] == "empty_transcript_blocked" for entry in outcomes)


def test_tool_followup_done_clears_stale_pending_assistant_create() -> None:
    api = _make_api()
    input_event_key = "tool:call_123"
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=input_event_key)
    api._pending_response_create = PendingResponseCreate(
        websocket=None,
        event={
            "type": "response.create",
            "response": {"metadata": {"turn_id": "turn_1", "input_event_key": input_event_key}},
        },
        origin="assistant_message",
        turn_id="turn_1",
        created_at=0.0,
        reason="audio_playback_busy",
    )

    api._set_tool_followup_state(canonical_key=canonical_key, state="done", reason="response_done")

    assert api._pending_response_create is None


def test_tool_followup_done_clears_stale_assistant_queue_residue_only() -> None:
    api = _make_api()
    input_event_key = "tool:call_123"
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=input_event_key)
    api._response_create_queue.append(
        {
            "websocket": None,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"input_event_key": input_event_key}},
            },
            "origin": "assistant_message",
            "turn_id": "turn_1",
            "record_ai_call": False,
        }
    )
    api._response_create_queue.append(
        {
            "websocket": None,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"input_event_key": input_event_key}},
            },
            "origin": "tool_output",
            "turn_id": "turn_1",
            "record_ai_call": False,
        }
    )

    api._set_tool_followup_state(canonical_key=canonical_key, state="done", reason="response_done")

    assert len(api._response_create_queue) == 1
    assert api._response_create_queue[0]["origin"] == "tool_output"


def test_tool_followup_cleanup_ignores_non_tool_canonical_keys() -> None:
    api = _make_api()
    input_event_key = "input_evt_1"
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=input_event_key)
    api._pending_response_create = PendingResponseCreate(
        websocket=None,
        event={
            "type": "response.create",
            "response": {"metadata": {"turn_id": "turn_1", "input_event_key": input_event_key}},
        },
        origin="assistant_message",
        turn_id="turn_1",
        created_at=0.0,
        reason="audio_playback_busy",
    )

    api._set_tool_followup_state(canonical_key=canonical_key, state="done", reason="response_done")

    assert api._pending_response_create is not None


def test_empty_retry_create_dropped_after_same_turn_final_deliverable() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)
    turn_id = "turn_1"
    final_input_event_key = "tool:call_123"
    final_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=final_input_event_key)

    api._canonical_response_state_store()[final_canonical_key] = CanonicalResponseState(
        turn_id=turn_id,
        input_event_key=final_input_event_key,
        origin="tool_output",
        deliverable_class="final",
    )

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key="synthetic_server_auto_3__empty_retry",
        origin="server_auto",
        response_metadata={
            "input_event_key": "synthetic_server_auto_3__empty_retry",
            "retry_reason": "empty_response_done",
        },
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "canonical_delivery_terminal_state"


def test_empty_retry_create_dropped_after_transcript_final_supersedes_provisional_lineage() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)
    turn_id = "turn_1"
    api._invalidate_superseded_empty_retry_lineage_for_turn(
        turn_id=turn_id,
        superseded_input_event_key="synthetic_server_auto_3",
        reason="transcript_final_handoff",
    )

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key="synthetic_server_auto_3__empty_retry",
        origin="server_auto",
        response_metadata={
            "input_event_key": "synthetic_server_auto_3__empty_retry",
            "retry_reason": "empty_response_done",
        },
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "canonical_delivery_terminal_state"


def test_empty_retry_create_not_dropped_without_same_turn_final_deliverable() -> None:
    api = _make_api()
    turn_id = "turn_1"

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key="synthetic_server_auto_3__empty_retry",
        origin="server_auto",
        response_metadata={
            "input_event_key": "synthetic_server_auto_3__empty_retry",
            "retry_reason": "empty_response_done",
        },
    )

    assert dropped is False



def test_empty_retry_not_scheduled_after_same_turn_final_deliverable() -> None:
    api = _make_api()
    turn_id = "turn_1"
    final_input_event_key = "tool:call_123"
    final_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=final_input_event_key)
    sent_events: list[dict[str, object]] = []

    api._canonical_response_state_store()[final_canonical_key] = CanonicalResponseState(
        turn_id=turn_id,
        input_event_key=final_input_event_key,
        origin="tool_output",
        deliverable_class="final",
    )

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id=turn_id,
            canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="synthetic_server_auto_3"),
            input_event_key="synthetic_server_auto_3",
            origin="server_auto",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []


def test_playback_complete_drain_drops_same_turn_empty_retry_after_final_deliverable() -> None:
    api = _make_api()
    ws = object()
    sent: list[dict[str, object]] = []

    async def _capture_send(*_args, **_kwargs):
        sent.append({"sent": True})
        return True

    api._send_response_create = _capture_send
    api._active_response_id = None
    api._active_response_origin = "unknown"

    turn_id = "turn_1"
    final_input_event_key = "tool:call_123"
    final_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=final_input_event_key)
    api._canonical_response_state_store()[final_canonical_key] = CanonicalResponseState(
        turn_id=turn_id,
        input_event_key=final_input_event_key,
        origin="tool_output",
        deliverable_class="final",
    )

    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event={
            "type": "response.create",
            "response": {
                "metadata": {
                    "turn_id": turn_id,
                    "input_event_key": "synthetic_server_auto_3__empty_retry",
                    "retry_reason": "empty_response_done",
                }
            },
        },
        origin="server_auto",
        turn_id=turn_id,
        created_at=0.0,
        reason="awaiting_transcript_final",
    )

    asyncio.run(api._drain_response_create_queue(source_trigger="playback_complete"))

    assert sent == []
    assert api._pending_response_create is None


def test_response_created_discards_invalidated_empty_retry_before_activation() -> None:
    api = _make_api()
    cancelled: list[dict[str, str]] = []
    sent_events: list[dict[str, object]] = []

    async def _async_capture(_websocket, event):
        sent_events.append(event)

    api._consume_response_origin = lambda _event: "server_auto"
    api._set_active_response_state = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("stale empty retry should not become active")
    )
    api._quarantine_cancelled_response_id = lambda **kwargs: cancelled.append(kwargs)
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._invalidate_superseded_empty_retry_lineage_for_turn(
        turn_id="turn_1",
        superseded_input_event_key="synthetic_server_auto_3",
        reason="transcript_final_handoff",
    )

    api._get_or_create_transport = lambda: types.SimpleNamespace(send_json=_async_capture)

    asyncio.run(
        api._handle_response_created_event(
            {
                "response": {
                    "id": "resp-stale-retry",
                    "metadata": {
                        "turn_id": "turn_1",
                        "input_event_key": "synthetic_server_auto_3__empty_retry",
                    },
                }
            },
            object(),
        )
    )

    assert cancelled == [
        {
            "response_id": "resp-stale-retry",
            "turn_id": "turn_1",
            "input_event_key": "synthetic_server_auto_3__empty_retry",
            "origin": "server_auto",
            "reason": "stale_empty_retry_lineage",
        }
    ]
    assert sent_events == [{"type": "response.cancel"}]


def test_invalidated_empty_retry_store_clears_on_turn_contender_cleanup() -> None:
    api = _make_api()
    api._invalidate_superseded_empty_retry_lineage_for_turn(
        turn_id="turn_cleanup",
        superseded_input_event_key="synthetic_server_auto_cleanup",
        reason="transcript_final_handoff",
    )

    api._clear_pending_response_contenders(
        turn_id="turn_cleanup",
        input_event_key="item_cleanup",
        reason="turn_complete",
    )

    assert (
        api._is_invalidated_empty_retry_lineage(
            turn_id="turn_cleanup",
            input_event_key="synthetic_server_auto_cleanup__empty_retry",
        )
        is False
    )


def test_invalidated_empty_retry_store_prunes_oldest_stale_turns() -> None:
    api = _make_api()
    api._invalidated_empty_retry_lineage_max_turns = 2
    api._active_input_event_key_by_turn_id = {"turn_keep": "item_keep"}

    api._invalidate_superseded_empty_retry_lineage_for_turn(
        turn_id="turn_oldest",
        superseded_input_event_key="synthetic_server_auto_oldest",
        reason="transcript_final_handoff",
    )
    api._invalidate_superseded_empty_retry_lineage_for_turn(
        turn_id="turn_keep",
        superseded_input_event_key="synthetic_server_auto_keep",
        reason="transcript_final_handoff",
    )
    api._invalidate_superseded_empty_retry_lineage_for_turn(
        turn_id="turn_newest",
        superseded_input_event_key="synthetic_server_auto_newest",
        reason="transcript_final_handoff",
    )

    assert (
        api._is_invalidated_empty_retry_lineage(
            turn_id="turn_oldest",
            input_event_key="synthetic_server_auto_oldest__empty_retry",
        )
        is False
    )
    assert (
        api._is_invalidated_empty_retry_lineage(
            turn_id="turn_keep",
            input_event_key="synthetic_server_auto_keep__empty_retry",
        )
        is True
    )
    assert (
        api._is_invalidated_empty_retry_lineage(
            turn_id="turn_newest",
            input_event_key="synthetic_server_auto_newest__empty_retry",
        )
        is True
    )


def test_response_path_candidate_envelope_preserves_same_turn_owner_drop() -> None:
    api = _make_api()
    runtime = api._response_create_runtime
    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=time.monotonic(),
    )
    prepared_snapshot = replace(
        prepared_snapshot,
        same_turn_owner_reason="tool_followup_owned",
        same_turn_owner_present=True,
        response_in_flight=True,
        audio_playback_busy=True,
        single_flight_block_reason="already_created",
    )

    decision = runtime.decide_response_create_action(prepared_snapshot)

    assert decision.action.value == "DROP"
    assert decision.reason_code == "same_turn_already_owned"
    assert decision.selected_candidate_id == "same_turn_owner"


def test_response_path_candidate_envelope_preserves_lineage_and_terminal_precedence() -> None:
    api = _make_api()
    runtime = api._response_create_runtime
    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=time.monotonic(),
    )
    prepared_snapshot = replace(
        prepared_snapshot,
        lineage_allowed=False,
        lineage_reason="tool_parent_missing",
        terminal_state_blocked=True,
        response_in_flight=True,
    )

    decision = runtime.decide_response_create_action(prepared_snapshot)

    assert decision.action.value == "BLOCK"
    assert decision.reason_code == "tool_parent_missing"
    assert decision.selected_candidate_id == "tool_lineage_guard"
    assert decision.blocked_by_terminal_state is False


def test_mark_transcript_response_outcome_demotes_intentional_non_action_reasons(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []

    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

    api._mark_transcript_response_outcome(
        input_event_key="input_evt_same_turn",
        turn_id="turn_1",
        outcome="response_not_scheduled",
        reason="same_turn_already_owned",
        details="owner=tool_followup_owned",
    )
    api._mark_transcript_response_outcome(
        input_event_key="input_evt_active",
        turn_id="turn_1",
        outcome="response_not_scheduled",
        reason="active_response_in_flight",
        details="active_origin=tool_output",
    )
    api._mark_transcript_response_outcome(
        input_event_key="input_evt_anomaly",
        turn_id="turn_1",
        outcome="response_not_scheduled",
        reason="already_handled",
        details="canonical_delivery_state=done",
    )

    assert not any("reason=same_turn_already_owned" in entry for entry in info_logs)
    assert not any("reason=active_response_in_flight" in entry for entry in info_logs)
    assert not any("reason=already_handled" in entry for entry in info_logs)
    assert any("reason=same_turn_already_owned" in entry for entry in debug_logs)
    assert any("reason=active_response_in_flight" in entry for entry in debug_logs)
    assert any("reason=already_handled" in entry for entry in debug_logs)


def test_curiosity_surface_block_reason_prefers_confirmation_and_obligation() -> None:
    api = _make_api()
    api._response_obligations = {"k": {"state": "open"}}
    api._pending_confirmation_token = object()

    reason = api._curiosity_surface_block_reason()

    assert reason == "confirmation_pending"


def test_curiosity_surface_block_reason_listening_suppresses() -> None:
    api = _make_api()
    api.state_manager.state = InteractionState.LISTENING

    reason = api._curiosity_surface_block_reason()

    assert reason == "suppressed_listening"


def test_curiosity_surface_block_reason_priority_obligation_over_busy() -> None:
    api = _make_api()
    api._response_obligations = {"k": {"state": "open"}}
    api._response_in_flight = True

    reason = api._curiosity_surface_block_reason()

    assert reason == "obligation_open"


def test_curiosity_surface_block_decision_returns_envelope() -> None:
    api = _make_api()
    api._response_obligations = {"k": {"state": "open"}}

    decision = api._curiosity_surface_block_decision()

    assert decision is not None
    assert decision.decision == "defer"
    assert decision.reason_code == "obligation_open"
    assert decision.subsystem == "curiosity"


def test_curiosity_anchor_repetition_count_decays_and_is_bounded() -> None:
    api = _make_api()

    assert api._curiosity_anchor_repetition_count(topic_anchor="battery", now=1.0) == 1
    assert api._curiosity_anchor_repetition_count(topic_anchor="battery", now=5.0) == 2
    assert api._curiosity_anchor_repetition_count(topic_anchor="battery", now=20.5) == 1

    api._curiosity_anchor_repetition_count(topic_anchor="a", now=21.0)
    api._curiosity_anchor_repetition_count(topic_anchor="b", now=22.0)
    api._curiosity_anchor_repetition_count(topic_anchor="c", now=23.0)
    api._curiosity_anchor_repetition_count(topic_anchor="d", now=24.0)

    assert len(api._curiosity_anchor_stats_by_anchor) <= api._curiosity_anchor_max_entries


def test_curiosity_surface_candidate_pruned_on_turn_complete_and_max_size() -> None:
    api = _make_api()
    api._curiosity_surface_candidate_by_turn_id = {
        "turn_1": {"created_at": 1.0},
        "turn_2": {"created_at": 2.0},
        "turn_3": {"created_at": 3.0},
    }

    api._prune_curiosity_surface_candidates(completed_turn_id="turn_2")

    assert "turn_2" not in api._curiosity_surface_candidate_by_turn_id
    assert len(api._curiosity_surface_candidate_by_turn_id) <= api._curiosity_surface_max_turns






def test_curiosity_surface_block_defer_decision_sets_expiry_fields() -> None:
    api = _make_api()
    api._response_obligations = {"k": {"state": "open"}}

    decision = api._curiosity_surface_block_decision()

    assert decision is not None
    assert decision.decision == "defer"
    assert decision.ttl_s == 1.0
    assert decision.expires_at is not None
    issued_at = float(decision.metadata.get("issued_at_monotonic_s", 0.0))
    assert issued_at > 0.0
    assert decision.expires_at == issued_at + decision.ttl_s


def test_curiosity_surface_stale_defer_decision_no_longer_blocks(monkeypatch) -> None:
    api = _make_api()

    class _Candidate:
        source = "conversation"
        reason_code = "curiosity_repeat_topic"
        score = 0.8
        dedupe_key = "topic:battery"
        suggested_followup = "Ask about battery trend"
        created_at = 10.0

    class _Decision:
        outcome = "surface"
        reason = "surface_threshold"

    class _Engine:
        candidate_ttl_s = 120.0

        def build_conversation_candidate(self, **_kwargs):
            return _Candidate()

        def evaluate(self, **kwargs):
            assert kwargs["arbitration_block_reason"] is None
            return _Decision()

    stale_logs: list[str] = []

    def _capture_debug(msg, *args, **_kwargs):
        rendered = msg % args if args else msg
        if rendered.startswith("curiosity_surface_governance_stale_ignored"):
            stale_logs.append(rendered)

    monkeypatch.setattr(logger, "debug", _capture_debug)
    api._curiosity_engine = _Engine()
    api._response_obligations = {"k": {"state": "open"}}
    api._arbitrate_opportunistic_surface = lambda **_kwargs: types.SimpleNamespace(
        selected_action_kind="curiosity_surface",
        selected_source="curiosity_engine",
        reason_code="arbitration_selected",
        selected_native_reason_code="curiosity_repeat_topic",
        is_opportunistic=True,
        suppressed_or_deferred=(),
    )
    api._embodiment_policy.decide_state_cue = lambda **_kwargs: EmbodimentDecision(
        action=EmbodimentActionType.NONE,
        reason="attention_continuity_hold",
    )

    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now)
    stale_decision = api._curiosity_surface_block_decision()
    monkeypatch.setattr(api, "_curiosity_surface_block_decision", lambda: stale_decision)
    monkeypatch.setattr(api, "_is_curiosity_defer_decision_fresh", lambda _decision: False)

    api._evaluate_curiosity_from_trust_snapshot(
        turn_id="turn_1",
        input_event_key="item_1",
        snapshot={"topic_anchors": ["battery"], "word_count": 4},
    )

    assert stale_logs
    stale_log = stale_logs[0]
    assert "run_id=" in stale_log
    assert "turn_id=" in stale_log
    assert "subsystem=curiosity" in stale_log
    assert "decision=defer" in stale_log
    assert "reason_code=obligation_open" in stale_log
    assert "priority=" in stale_log
    assert "expires_at=" in stale_log


def test_curiosity_surface_fresh_defer_decision_still_blocks(monkeypatch) -> None:
    api = _make_api()
    debug_logs: list[str] = []
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **_kwargs: debug_logs.append(msg % args if args else msg))

    class _Candidate:
        source = "conversation"
        reason_code = "curiosity_repeat_topic"
        score = 0.8
        dedupe_key = "topic:battery"
        suggested_followup = "Ask about battery trend"
        created_at = 10.0

    class _Decision:
        outcome = "ignore"
        reason = "arbitration_blocked"

    class _Engine:
        candidate_ttl_s = 120.0

        def build_conversation_candidate(self, **_kwargs):
            return _Candidate()

        def evaluate(self, **kwargs):
            assert kwargs["arbitration_block_reason"] == "obligation_open"
            return _Decision()

    api._curiosity_engine = _Engine()
    api._response_obligations = {"k": {"state": "open"}}

    api._evaluate_curiosity_from_trust_snapshot(
        turn_id="turn_1",
        input_event_key="item_1",
        snapshot={"topic_anchors": ["battery"], "word_count": 4},
    )

    assert api._curiosity_surface_candidate_by_turn_id == {}
    governance_logs = [entry for entry in debug_logs if entry.startswith("curiosity_surface_governance ")]
    assert governance_logs
    rendered = governance_logs[-1]
    assert "run_id=run-395" in rendered
    assert "turn_id=turn_1" in rendered
    assert "subsystem=curiosity" in rendered
    assert "decision=defer" in rendered
    assert "reason_code=obligation_open" in rendered
    assert "priority=" in rendered
    assert "obligation_count=1" in rendered

def test_curiosity_surface_seam_passes_curiosity_and_embodiment_governance_inputs() -> None:
    api = _make_api()

    class _Candidate:
        source = "conversation"
        reason_code = "curiosity_repeat_topic"
        score = 0.8
        dedupe_key = "topic:battery"
        suggested_followup = "Ask about battery trend"
        created_at = 10.0

    class _Decision:
        outcome = "surface"
        reason = "surface_threshold"

    class _Engine:
        candidate_ttl_s = 120.0

        def build_conversation_candidate(self, **_kwargs):
            return _Candidate()

        def evaluate(self, **_kwargs):
            return _Decision()

    captured: dict[str, object] = {}

    def _capture_arbitration(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            selected_action_kind="curiosity_surface",
            selected_source="curiosity_engine",
            reason_code="arbitration_selected",
            selected_native_reason_code="curiosity_repeat_topic",
            is_opportunistic=True,
            suppressed_or_deferred=(),
        )

    def _embodiment_none(**_kwargs):
        return EmbodimentDecision(
            action=EmbodimentActionType.NONE,
            reason="attention_continuity_hold",
        )

    api._curiosity_engine = _Engine()
    api._arbitrate_opportunistic_surface = _capture_arbitration
    api._embodiment_policy.decide_state_cue = _embodiment_none

    api._evaluate_curiosity_from_trust_snapshot(
        turn_id="turn_1",
        input_event_key="item_1",
        snapshot={"topic_anchors": ["battery"], "word_count": 4},
    )

    curiosity_governance = captured.get("candidate_curiosity_governance")
    embodiment_governance = captured.get("candidate_embodiment_governance")
    assert curiosity_governance is not None
    assert embodiment_governance is not None
    assert curiosity_governance.subsystem == "curiosity"
    assert curiosity_governance.decision == "allow"
    assert embodiment_governance.subsystem == "embodiment"
    assert embodiment_governance.reason_code == "attention_continuity_hold"



def test_opportunistic_arbitration_logs_governance_envelopes_and_winner(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []

    def _capture_info(msg, *args, **_kwargs):
        rendered = msg % args if args else msg
        if rendered.startswith("opportunistic_arbitration_seam"):
            info_logs.append(rendered)

    monkeypatch.setattr(logger, "info", _capture_info)

    result = api._arbitrate_opportunistic_surface(
        candidate_curiosity=OpportunisticActionCandidate(
            action_kind="curiosity_surface",
            source="curiosity_engine",
            priority=60,
            reason_code="curiosity_repeat_topic",
            opportunistic=True,
            ttl_s=120.0,
        ),
        candidate_curiosity_governance=GovernanceDecision(decision="allow", reason_code="curiosity_repeat_topic", subsystem="curiosity", priority=60),
        candidate_embodiment_flourish=OpportunisticActionCandidate(
            action_kind="embodiment_flourish",
            source="embodiment_policy",
            priority=30,
            reason_code="attention_continuity_hold",
            opportunistic=True,
            ttl_s=None,
        ),
        candidate_embodiment_governance=GovernanceDecision(
            decision="defer",
            reason_code="attention_continuity_hold",
            subsystem="embodiment",
            priority=40,
        ),
    )

    assert result.selected_action_kind == "curiosity_surface"
    assert info_logs
    assert "governance_inputs=" in info_logs[0]
    assert "selected_kind=" in info_logs[0]

def test_curiosity_surface_path_runs_seam_arbitration_and_suppresses_when_obligation_open() -> None:
    api = _make_api()
    api._response_obligations = {
        api._response_obligation_key(turn_id="turn_1", input_event_key="item_1"): {"state": "open"}
    }

    class _Candidate:
        source = "conversation"
        reason_code = "curiosity_repeat_topic"
        score = 0.8
        dedupe_key = "topic:battery"
        suggested_followup = "Ask about battery trend"
        created_at = 10.0

    class _Decision:
        outcome = "surface"
        reason = "surface_threshold"

    class _Engine:
        candidate_ttl_s = 120.0

        def build_conversation_candidate(self, **_kwargs):
            return _Candidate()

        def evaluate(self, **_kwargs):
            return _Decision()

    api._curiosity_engine = _Engine()

    api._evaluate_curiosity_from_trust_snapshot(
        turn_id="turn_1",
        input_event_key="item_1",
        snapshot={"topic_anchors": ["battery"], "word_count": 4},
    )

    assert api._curiosity_surface_candidate_by_turn_id == {}


def test_curiosity_surface_seam_ignores_unrelated_obligation_on_other_turn() -> None:
    api = _make_api()
    api._response_obligations = {
        api._response_obligation_key(turn_id="turn_9", input_event_key="item_9"): {"state": "open"}
    }

    class _Candidate:
        source = "conversation"
        reason_code = "curiosity_repeat_topic"
        score = 0.8
        dedupe_key = "topic:battery"
        suggested_followup = "Ask about battery trend"
        created_at = 10.0

    class _Decision:
        outcome = "surface"
        reason = "surface_threshold"

    class _Engine:
        candidate_ttl_s = 120.0

        def build_conversation_candidate(self, **_kwargs):
            return _Candidate()

        def evaluate(self, **_kwargs):
            return _Decision()

    api._curiosity_engine = _Engine()

    api._evaluate_curiosity_from_trust_snapshot(
        turn_id="turn_1",
        input_event_key="item_1",
        snapshot={"topic_anchors": ["battery"], "word_count": 4},
    )

    assert "turn_1" in api._curiosity_surface_candidate_by_turn_id


def test_response_done_decision_marks_provisional_server_auto_pre_final_as_non_deliverable() -> None:
    api = _make_api()
    api._is_empty_response_done = lambda **_kwargs: False

    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="server_auto",
        delivery_state_before_done="done",
        active_response_was_provisional=True,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_server_auto_1"),
        transcript_final_seen=False,
    )

    assert selected is False
    assert reason == "provisional_server_auto_awaiting_transcript_final"


def test_response_done_deliverable_decision_remains_tuple_compatible_over_pure_seam() -> None:
    api = _make_api()

    decision = api._response_done_deliverable_arbitration(
        turn_id="turn_1",
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_1"),
        transcript_final_seen=True,
    )
    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert decision.selected is True
    assert decision.reason_code == "normal"
    assert decision.selected_candidate_id == "terminal_selected"
    assert (selected, reason) == (decision.selected, decision.reason_code)


def test_response_done_decision_rejects_assistant_message_when_tool_followup_pending() -> None:
    api = _make_api()
    turn_id = "turn_1"
    api._tool_followup_state_by_canonical_key = {
        f"{api._current_run_id()}:{turn_id}:item_1:tool:call_1": "created",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is False
    assert reason == "tool_followup_precedence"


def test_response_done_decision_allows_tool_output_when_tool_followup_pending() -> None:
    api = _make_api()
    turn_id = "turn_1"
    api._tool_followup_state_by_canonical_key = {
        f"{api._current_run_id()}:{turn_id}:item_1:tool:call_1": "created",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="tool_output",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is True
    assert reason == "normal"



def test_response_done_decision_rejects_upgraded_response_when_tool_followup_pending() -> None:
    api = _make_api()
    turn_id = "turn_1"
    api._tool_followup_state_by_canonical_key = {
        f"{api._current_run_id()}:{turn_id}:item_1:tool:call_1": "created",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="upgraded_response",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is False
    assert reason == "tool_followup_precedence"


def test_response_done_decision_allows_upgraded_response_when_only_status_gesture_followup_pending() -> None:
    api = _make_api()
    turn_id = "turn_1"
    canonical_key = f"{api._current_run_id()}:{turn_id}:tool:call_1"
    api._tool_followup_state_by_canonical_key = {canonical_key: "created"}
    api._record_tool_followup_metadata(
        canonical_key=canonical_key,
        metadata={
            "tool_name": "gesture_curious_tilt",
            "tool_followup_status_only": "true",
        },
    )

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="upgraded_response",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is True
    assert reason == "normal"


def test_response_done_decision_keeps_precedence_for_non_status_tool_followup() -> None:
    api = _make_api()
    turn_id = "turn_1"
    canonical_key = f"{api._current_run_id()}:{turn_id}:tool:call_1"
    api._tool_followup_state_by_canonical_key = {canonical_key: "created"}
    api._record_tool_followup_metadata(
        canonical_key=canonical_key,
        metadata={
            "tool_name": "perform_research",
            "tool_followup_status_only": "true",
        },
    )

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="upgraded_response",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="item_1"),
        transcript_final_seen=True,
    )

    assert selected is False
    assert reason == "tool_followup_precedence"


def test_visual_snapshot_tag_allows_upgraded_response_without_harness_semantic_policing() -> None:
    api = _make_api()
    turn_id = "turn_2"
    input_event_key = "item_visual"
    response_id = "resp_visual_generic"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": True,
            "transcript_text": "do you see it now",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "it looks ready for use and you can start your project",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="upgraded_response",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
        transcript_final_seen=True,
        input_event_key=input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_visual_snapshot_tag_allows_unrelated_assistant_message_without_harness_semantic_policing() -> None:
    api = _make_api()
    turn_id = "turn_2"
    input_event_key = "item_visual"
    response_id = "resp_visual_chat"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": True,
            "transcript_text": "what do you see in my hand",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "we can work on a project together later",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=f"{input_event_key}:clarify"),
        transcript_final_seen=True,
        input_event_key=f"{input_event_key}:clarify",
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_visual_snapshot_tag_allows_visual_unavailable_fallback_terminal_selection() -> None:
    api = _make_api()
    turn_id = "turn_2"
    input_event_key = "item_visual"
    response_id = "resp_visual_fallback"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": True,
            "transcript_text": "what do you see in my hand",
        }
    }
    api._response_trace_context_by_id = {
        response_id: {
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "The camera is on, but I don’t have a fresh frame yet. Want me to take a new look now?",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=f"{input_event_key}:clarify"),
        transcript_final_seen=True,
        input_event_key=f"{input_event_key}:clarify",
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_visual_snapshot_tag_allows_visual_unavailable_fallback_from_trace_context_recording() -> None:
    api = _make_api()
    turn_id = "turn_2"
    input_event_key = "item_visual"
    response_id = "resp_visual_fallback_trace"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": True,
            "transcript_text": "what do you see in my hand",
        }
    }
    api._record_response_trace_context(
        response_id,
        turn_id=turn_id,
        input_event_key=f"{input_event_key}:clarify",
        canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=f"{input_event_key}:clarify"),
        origin="assistant_message",
        trigger="asr_verify_on_risk",
        reason="visual_unavailable",
    )
    api._terminal_response_text_by_response_id = {
        response_id: "I can’t see right now. Want me to take a quick look with the camera?",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=f"{input_event_key}:clarify"),
        transcript_final_seen=True,
        input_event_key=f"{input_event_key}:clarify",
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_visual_snapshot_tag_allows_generic_assistant_fallback_when_trace_marks_visual_unavailable() -> None:
    api = _make_api()
    turn_id = "turn_2"
    input_event_key = "item_visual"
    response_id = "resp_visual_fallback_generic"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": True,
            "transcript_text": "what do you see in my hand",
        }
    }
    api._record_response_trace_context(
        response_id,
        turn_id=turn_id,
        input_event_key=f"{input_event_key}:clarify",
        canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=f"{input_event_key}:clarify"),
        origin="assistant_message",
        trigger="asr_verify_on_risk",
        reason="visual_unavailable",
    )
    api._terminal_response_text_by_response_id = {
        response_id: "We can check together when you're ready. Just let me know, and I'll refresh the view.",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="assistant_message",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=f"{input_event_key}:clarify"),
        transcript_final_seen=True,
        input_event_key=f"{input_event_key}:clarify",
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_non_visual_snapshot_tag_remains_unaffected_by_visual_snapshot_classification() -> None:
    api = _make_api()
    turn_id = "turn_2"
    input_event_key = "item_non_visual"
    response_id = "resp_non_visual"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": False,
            "transcript_text": "tell me a joke",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "Here is a joke for you.",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="upgraded_response",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
        transcript_final_seen=True,
        input_event_key=input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_visual_snapshot_tag_allows_valid_replacement_when_response_is_grounded() -> None:
    api = _make_api()
    turn_id = "turn_3"
    input_event_key = "item_visual"
    response_id = "resp_visual_valid_replacement"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": True,
            "transcript_text": "do you see it now",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "I can see it now.",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="upgraded_response",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
        transcript_final_seen=True,
        input_event_key=input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_visual_snapshot_tag_tool_output_uses_parent_context_for_terminal_selection() -> None:
    api = _make_api()
    turn_id = "turn_5"
    parent_input_event_key = "item_visual_parent"
    tool_input_event_key = "tool:call_123"
    response_id = "resp_tool_visual"
    api._active_input_event_key_by_turn_id[turn_id] = tool_input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        parent_input_event_key: {
            "visual_question": True,
            "transcript_text": "can you see it now in my hand",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "The tool in your hand is clearly visible now.",
    }
    api._record_response_trace_context(
        response_id,
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key=parent_input_event_key,
    )

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="tool_output",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key),
        transcript_final_seen=True,
        input_event_key=tool_input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_non_visual_tool_output_remains_unaffected_by_visual_snapshot_classification() -> None:
    api = _make_api()
    turn_id = "turn_6"
    input_event_key = "item_non_visual"
    response_id = "resp_tool_non_visual"
    api._active_input_event_key_by_turn_id[turn_id] = input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        input_event_key: {
            "visual_question": False,
            "transcript_text": "gesture complete",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "Done. Gesture completed.",
    }

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="tool_output",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
        transcript_final_seen=True,
        input_event_key=input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_descriptive_turn_rejects_gesture_only_tool_output_terminal_selection() -> None:
    api = _make_api()
    turn_id = "turn_7"
    parent_input_event_key = "item_descriptive_parent"
    tool_input_event_key = "tool:call_777"
    response_id = "resp_tool_gesture_only"
    api._active_input_event_key_by_turn_id[turn_id] = tool_input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        parent_input_event_key: {
            "visual_question": False,
            "transcript_text": "I'd like you to tell me what it is.",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "Just nodded to confirm that for you.",
    }
    api._record_response_trace_context(
        response_id,
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key=parent_input_event_key,
    )

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="tool_output",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key),
        transcript_final_seen=True,
        input_event_key=tool_input_event_key,
        response_id=response_id,
    )

    assert selected is False
    assert reason == "tool_output_descriptive_gesture_only"


def test_descriptive_turn_allows_substantive_tool_output_terminal_selection() -> None:
    api = _make_api()
    turn_id = "turn_8"
    parent_input_event_key = "item_descriptive_parent"
    tool_input_event_key = "tool:call_888"
    response_id = "resp_tool_substantive"
    api._active_input_event_key_by_turn_id[turn_id] = tool_input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        parent_input_event_key: {
            "visual_question": False,
            "transcript_text": "Tell me what it is.",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "It looks like a small handheld device with a nozzle.",
    }
    api._record_response_trace_context(
        response_id,
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key=parent_input_event_key,
    )

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="tool_output",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key),
        transcript_final_seen=True,
        input_event_key=tool_input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"


def test_non_descriptive_turn_allows_confirmation_style_tool_output_terminal_selection() -> None:
    api = _make_api()
    turn_id = "turn_9"
    parent_input_event_key = "item_non_descriptive_parent"
    tool_input_event_key = "tool:call_999"
    response_id = "resp_tool_confirmation_ok"
    api._active_input_event_key_by_turn_id[turn_id] = tool_input_event_key
    api._utterance_trust_snapshot_by_input_event_key = {
        parent_input_event_key: {
            "visual_question": False,
            "transcript_text": "Please nod if you got that.",
        }
    }
    api._terminal_response_text_by_response_id = {
        response_id: "Confirmed.",
    }
    api._record_response_trace_context(
        response_id,
        turn_id=turn_id,
        input_event_key=tool_input_event_key,
        origin="tool_output",
        parent_turn_id=turn_id,
        parent_input_event_key=parent_input_event_key,
    )

    selected, reason = api._response_done_deliverable_decision(
        turn_id=turn_id,
        origin="tool_output",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key),
        transcript_final_seen=True,
        input_event_key=tool_input_event_key,
        response_id=response_id,
    )

    assert selected is True
    assert reason == "normal"
