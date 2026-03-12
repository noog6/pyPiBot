from __future__ import annotations

import base64
import asyncio
from collections import deque
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import PendingServerAutoResponse, RealtimeAPI


class _Transport:

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, _websocket, event: dict[str, object]) -> None:
        self.sent.append(event)


def _build_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._pending_server_auto_response_by_turn_id = {}
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._cancelled_response_timing_by_id = {}
    api._response_status_by_id = {}
    api._audio_accum = bytearray()
    api._audio_accum_response_id = None
    api.audio_player = None
    api._current_run_id = lambda: "run-464"
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._response_in_flight = True
    api.response_in_progress = True
    api._active_response_id = "resp-server-auto"
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-464:{turn_id}:{input_event_key}"
    api._current_utterance_seq = lambda: 2
    api._canonical_response_state_by_key = {}
    api._canonical_response_state_store = lambda: api._canonical_response_state_by_key
    api._sync_legacy_response_state_mirrors = lambda: None
    api._debug_assert_canonical_state_invariants = lambda **_kwargs: None
    from ai.interaction_lifecycle_controller import InteractionLifecycleController
    lifecycle_controller = InteractionLifecycleController()
    api._lifecycle_controller = lambda: lifecycle_controller
    api._cancel_micro_ack = lambda **_kwargs: None
    api._canonical_first_audio_started = lambda _canonical_key: True
    return api


def test_cancel_and_replace_server_auto_on_transcript_final_includes_preference_context() -> None:
    api = _build_api_stub()
    transport = _Transport()
    replacement_calls: list[dict[str, object]] = []

    api._pending_server_auto_response_by_turn_id["turn_2"] = PendingServerAutoResponse(
        turn_id="turn_2",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_2:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: {
        "hit": True,
        "prompt_note": "Preference recall context: user's preferred editor is Vim.",
    }

    async def _fake_send_response_create(_websocket, event, **kwargs):
        replacement_calls.append({"event": event, **kwargs})
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_2",
            input_event_key="item_abc",
            origin_label="upgraded_response",
        )
    )

    assert replaced is True
    assert transport.sent[0] == {"type": "response.cancel", "response_id": "resp-server-auto"}
    assert replacement_calls
    metadata = replacement_calls[0]["event"]["response"]["metadata"]
    assert metadata["input_event_key"] == "item_abc"
    assert metadata["safety_override"] == "true"
    assert replacement_calls[0]["origin"] == "upgraded_response"
    assert "Vim" in str(replacement_calls[0]["memory_brief_note"])
    assert api._is_cancelled_response_event({"response_id": "resp-server-auto"}) is True
    assert "resp-server-auto" in api._suppressed_audio_response_ids
    timing = api._cancelled_response_timing_by_id.get("resp-server-auto")
    assert isinstance(timing, dict)
    assert timing.get("cancel_issued_at") is not None
    assert timing.get("first_audio_delta_seen_at") is None
    assert timing.get("output_audio_done_at") is None
    assert api._is_cancelled_response_event({"response_id": "resp-replacement"}) is False






def test_cancel_and_replace_server_auto_on_transcript_final_tags_memory_intent_subtype() -> None:
    api = _build_api_stub()
    transport = _Transport()
    replacement_calls: list[dict[str, object]] = []

    api._pending_server_auto_response_by_turn_id["turn_2"] = PendingServerAutoResponse(
        turn_id="turn_2",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_2:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None

    async def _fake_send_response_create(_websocket, event, **kwargs):
        replacement_calls.append({"event": event, **kwargs})
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_2",
            input_event_key="item_abc",
            origin_label="upgraded_response",
            memory_intent_subtype="general_memory",
        )
    )

    assert replaced is True
    assert replacement_calls
    metadata = replacement_calls[0]["event"]["response"]["metadata"]
    assert metadata["memory_intent_subtype"] == "general_memory"


def test_transcript_upgrade_memory_intent_adds_observation_exclusion_guardrail() -> None:
    api = _build_api_stub()
    api._response_create_debug_trace = False
    api._last_response_create_ts = None
    api._extract_response_create_metadata = lambda event: event.setdefault("response", {}).setdefault("metadata", {})
    api._resolve_response_create_turn_id = lambda **_kwargs: "turn_9"
    api._ensure_response_create_correlation = lambda **_kwargs: "item_9"
    api._bind_active_input_event_key_for_turn = lambda **_kwargs: None
    api._utterance_context_scope = lambda **_kwargs: __import__("contextlib").nullcontext(type("Ctx", (), {"turn_id": "turn_9", "input_event_key": "item_9", "canonical_key": "run-471:turn_9:item_9"})())
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None
    api._canonical_utterance_key = lambda **_kwargs: "run-471:turn_9:item_9"
    api._active_input_event_key_for_turn = lambda _turn_id: "item_9"
    api._single_flight_block_reason = lambda **_kwargs: ""
    api._preference_recall_suppressed_turns = set()
    api._is_preference_recall_lock_blocked = lambda **_kwargs: False
    api._drop_response_create_for_terminal_state = lambda **_kwargs: False
    api._lifecycle_policy = lambda: __import__("ai.interaction_lifecycle_policy", fromlist=["InteractionLifecyclePolicy"]).InteractionLifecyclePolicy()
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._response_consumes_canonical_slot = lambda _metadata: True
    api._response_is_explicit_multipart = lambda _metadata: False
    api._response_has_safety_override = lambda _event: True
    api._response_created_canonical_keys = set()
    api._is_response_already_delivered = lambda **_kwargs: False
    api._record_duplicate_create_attempt = lambda **_kwargs: None
    api._log_response_create_blocked = lambda **_kwargs: None
    api._drop_suppressed_scheduled_response_creates = lambda **_kwargs: None
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._response_schedule_logged_turn_ids = set()
    api._turn_diagnostic_timestamps = {}
    api._resptrace_suppression_reason = lambda **_kwargs: ""
    api._response_create_queue = []
    api._pending_server_auto_input_event_keys = []
    api._response_obligations = {}
    api._response_obligation_key = lambda **_kwargs: "obl"
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_args, **_kwargs: __import__("asyncio").sleep(0))})()
    api._set_response_delivery_state = lambda **_kwargs: None
    api._canonical_first_audio_started = lambda _canonical_key: True

    from ai.realtime.response_create_runtime import ResponseCreateRuntime

    runtime = ResponseCreateRuntime(api)
    ws = object()
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_9",
                "input_event_key": "item_9",
                "transcript_upgrade_replacement": "true",
                "safety_override": "true",
                "memory_intent_subtype": "general_memory",
            }
        },
    }

    sent = asyncio.run(runtime.send_response_create(ws, event, origin="upgraded_response"))
    assert sent is True
    instructions = str(event["response"].get("instructions") or "")
    assert "Memory-intent response mode" in instructions
    assert "Do not add passive scene or environment narration" in instructions

def test_cancel_and_replace_clears_old_text_buffer_and_rejects_stale_deltas() -> None:
    api = _build_api_stub()
    transport = _Transport()

    api.assistant_reply = "I don't have that"
    api._assistant_reply_accum = "I don't have that"
    api._assistant_reply_response_id = "resp-server-auto"

    api._pending_server_auto_response_by_turn_id["turn_2"] = PendingServerAutoResponse(
        turn_id="turn_2",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_2:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None

    async def _fake_send_response_create(_websocket, _event, **_kwargs):
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_2",
            input_event_key="item_abc",
            origin_label="upgraded_response",
        )
    )

    assert replaced is True
    assert api.assistant_reply == ""
    assert api._assistant_reply_accum == ""
    assert api._assistant_reply_response_id is None

    asyncio.run(
        api._handle_event_legacy(
            {"type": "response.text.delta", "response_id": "resp-server-auto", "delta": "that"},
            websocket=None,
        )
    )

    assert api.assistant_reply == ""
    assert api._assistant_reply_accum == ""

    api._active_response_id = "resp-replacement"
    api._append_assistant_reply_text("Your favorite editor is Vim.", response_id="resp-replacement")

    assert api.assistant_reply == "Your favorite editor is Vim."
    assert api._assistant_reply_accum == "Your favorite editor is Vim."
    assert api._assistant_reply_response_id == "resp-replacement"

def test_transcript_final_without_pending_server_auto_does_not_cancel_or_replace() -> None:
    api = _build_api_stub()
    transport = _Transport()
    replacement_calls: list[dict[str, object]] = []

    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None

    async def _fake_send_response_create(_websocket, event, **kwargs):
        replacement_calls.append({"event": event, **kwargs})
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_3",
            input_event_key="item_333",
            origin_label="upgraded_response",
        )
    )

    assert replaced is False
    assert transport.sent == []
    assert replacement_calls == []


def test_cancel_and_replace_returns_false_when_replacement_blocked() -> None:
    api = _build_api_stub()
    transport = _Transport()
    cancel_reasons: list[str] = []

    api._pending_server_auto_response_by_turn_id["turn_4"] = PendingServerAutoResponse(
        turn_id="turn_4",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_4:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **kwargs: cancel_reasons.append(str(kwargs.get("reason") or ""))

    async def _blocked_send_response_create(_websocket, _event, **_kwargs):
        return False

    api._send_response_create = _blocked_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_4",
            input_event_key="item_444",
            origin_label="upgraded_response",
        )
    )

    assert replaced is False
    assert transport.sent[0] == {"type": "response.cancel", "response_id": "resp-server-auto"}
    assert cancel_reasons == ["upgrade_selected", "upgrade_blocked"]


def test_upgrade_fallback_keeps_server_auto_when_audio_started() -> None:
    api = _build_api_stub()
    pending = PendingServerAutoResponse(
        turn_id="turn_5",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_5:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._canonical_first_audio_started = lambda _canonical_key: True

    assert api.should_cancel_and_replace(
        server_auto_state=pending,
        transcript_final_state={"turn_id": "turn_5", "input_event_key": "item_555"},
        pref_ctx_state=None,
    ) is False


def test_upgrade_replacement_allowed_when_audio_started() -> None:
    api = _build_api_stub()
    api._response_create_runtime = None
    api._current_run_id = lambda: "run-471"
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._extract_response_create_metadata = lambda event: event["response"].get("metadata", {})
    api._resolve_response_create_turn_id = lambda **_kwargs: "turn_9"
    api._ensure_response_create_correlation = lambda **_kwargs: "item_9"
    api._bind_active_input_event_key_for_turn = lambda **_kwargs: None
    api._utterance_context_scope = lambda **_kwargs: __import__("contextlib").nullcontext(type("Ctx", (), {"turn_id": "turn_9", "input_event_key": "item_9", "canonical_key": "run-471:turn_9:item_9"})())
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None
    api._canonical_utterance_key = lambda **_kwargs: "run-471:turn_9:item_9"
    api._active_input_event_key_for_turn = lambda _turn_id: "item_9"
    api._single_flight_block_reason = lambda **_kwargs: ""
    api._preference_recall_suppressed_turns = set()
    api._is_preference_recall_lock_blocked = lambda **_kwargs: False
    api._drop_response_create_for_terminal_state = lambda **_kwargs: False
    api._lifecycle_policy = lambda: __import__("ai.interaction_lifecycle_policy", fromlist=["InteractionLifecyclePolicy"]).InteractionLifecyclePolicy()
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._response_consumes_canonical_slot = lambda _metadata: True
    api._response_is_explicit_multipart = lambda _metadata: False
    api._response_has_safety_override = lambda _event: True
    api._response_created_canonical_keys = set()
    api._is_response_already_delivered = lambda **_kwargs: False
    api._record_duplicate_create_attempt = lambda **_kwargs: None
    api._log_response_create_blocked = lambda **_kwargs: None
    api._drop_suppressed_scheduled_response_creates = lambda **_kwargs: None
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._response_schedule_logged_turn_ids = set()
    api._turn_diagnostic_timestamps = {}
    api._resptrace_suppression_reason = lambda **_kwargs: ""
    api._response_create_queue = []
    api._pending_server_auto_input_event_keys = []
    api._response_obligations = {}
    api._response_obligation_key = lambda **_kwargs: "obl"
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_args, **_kwargs: __import__("asyncio").sleep(0))})()
    api._set_response_delivery_state = lambda **_kwargs: None
    api._canonical_first_audio_started = lambda _canonical_key: True

    from ai.realtime.response_create_runtime import ResponseCreateRuntime

    runtime = ResponseCreateRuntime(api)
    ws = object()
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_9", "input_event_key": "item_9", "transcript_upgrade_replacement": "true", "safety_override": "true"}},
    }

    sent = asyncio.run(runtime.send_response_create(ws, event, origin="upgraded_response"))
    assert sent is True


def test_terminal_state_is_per_canonical_key_not_turn_id() -> None:
    api = _build_api_stub()
    api._canonical_response_state_by_key = {}
    api._canonical_response_state_store = lambda: api._canonical_response_state_by_key
    api._sync_legacy_response_state_mirrors = lambda: None
    api._debug_assert_canonical_state_invariants = lambda **_kwargs: None

    cancelled_key = "run-464:turn_8:synthetic_server_auto_1"
    replacement_key = "run-464:turn_8:item_real"
    api._mark_canonical_cancelled_for_upgrade(
        canonical_key=cancelled_key,
        turn_id="turn_8",
        response_id="resp-old",
    )
    api._clear_canonical_terminal_delivery_state(canonical_key=replacement_key)

    assert api._drop_response_create_for_terminal_state(
        turn_id="turn_8",
        input_event_key="item_real",
        origin="upgraded_response",
        response_metadata={"transcript_upgrade_replacement": "true"},
    ) is False




def test_server_auto_answer_verdict_does_not_emit_cancel() -> None:
    api = _build_api_stub()
    api._active_response_origin = "server_auto"
    api._active_response_input_event_key = "item_1"
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._response_gating_verdict_by_input_event_key = {
        "run-464:turn_1:item_1": type("V", (), {"action": "ANSWER"})(),
    }
    api._audio_playback_busy = False
    api._is_active_response_guarded = lambda: False
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._canonical_lifecycle_state = lambda _key: {"cancel_requested_pre_audio": True}
    api._active_response_canonical_key = "run-464:turn_1:item_1"
    api._active_response_id = "resp-server-auto"
    sent = []
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda _ws, event: sent.append(event) or __import__("asyncio").sleep(0))})()

    asyncio.run(api._handle_response_output_audio_delta_event({"response_id": "resp-server-auto", "delta": ""}, object()))

    assert sent == []

def test_server_auto_audio_does_not_start_before_gating() -> None:
    api = _build_api_stub()
    api._active_response_origin = "server_auto"
    api._active_response_input_event_key = "item_1"
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._response_gating_verdict_by_input_event_key = {
        "run-464:turn_1:item_1": type("V", (), {"action": "CLARIFY"})(),
    }
    api._audio_playback_busy = False
    api._is_active_response_guarded = lambda: False
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._canonical_lifecycle_state = lambda _key: {}
    api._active_response_canonical_key = "run-464:turn_1:item_1"
    api._lifecycle_controller = lambda: type("LC", (), {"on_audio_delta": staticmethod(lambda *_a, **_k: type("D", (), {"action": __import__("ai.interaction_lifecycle_controller", fromlist=["LifecycleDecisionAction"]).LifecycleDecisionAction.ALLOW, "reason": "ok"})())})()
    api._log_lifecycle_event = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **_kwargs: None
    api._canonical_response_state_mutate = lambda **_kwargs: None
    sent = []
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda _ws, event: sent.append(event) or __import__("asyncio").sleep(0))})()

    asyncio.run(api._handle_response_output_audio_delta_event({"response_id": "resp-server-auto", "delta": ""}, object()))

    assert sent == [{"type": "response.cancel", "response_id": "resp-server-auto"}]
    assert api._audio_playback_busy is False


def test_cancel_and_replace_conversation_item_added_uses_output_item_response_mapping(monkeypatch) -> None:
    api = _build_api_stub()
    api._response_trace_context_by_id = {}
    api._stale_response_map = {}
    api._stale_response_map_ttl_s = 15.0
    api._stale_response_ids_set = set()
    api._stale_response_drop_window_by_id = {}
    api._stale_response_drop_window_s = 3.0
    api._response_id_by_output_item_id = {}
    api._active_response_id = None
    api._active_response_origin = "unknown"
    api._active_response_input_event_key = None
    api._active_response_canonical_key = None
    api._lifecycle_trace_item_added_unknown_events = deque()
    api._lifecycle_trace_item_added_unknown_threshold = 3
    api._lifecycle_trace_item_added_unknown_window_s = 10.0
    api._lifecycle_trace_item_added_unknown_cooldown_s = 30.0
    api._lifecycle_trace_item_added_unknown_last_escalation_ts = 0.0
    api._lifecycle_trace_item_added_unknown_debug = True
    api._lifecycle_trace_transcript_delta_state = {}
    api._lifecycle_trace_transcript_delta_sample_n = 20
    api._lifecycle_trace_transcript_delta_inactivity_ms = 750
    api._current_turn_id_or_unknown = lambda: "turn_2"
    api._active_input_event_key_by_turn_id = {"turn_2": "item_new"}

    info_logs: list[str] = []

    async def _noop_handle_event_legacy(_event, _websocket):
        return None

    api._handle_event_legacy = _noop_handle_event_legacy

    monkeypatch.setattr("ai.realtime_api.logger.info", lambda message, *args: info_logs.append(message % args))

    asyncio.run(
        api._handle_response_lifecycle_event(
            {
                "type": "response.output_item.added",
                "response_id": "resp-server-auto",
                "item": {"id": "item_assistant_1", "type": "message", "role": "assistant"},
            },
            websocket=None,
        )
    )

    api._mark_pending_server_auto_response_cancelled(turn_id="turn_2", reason="transcript_final_upgrade")

    asyncio.run(
        api._handle_response_lifecycle_event(
            {
                "type": "conversation.item.added",
                "item": {"id": "item_assistant_1", "type": "message", "role": "assistant"},
            },
            websocket=None,
        )
    )

    assert not any("response_lifecycle_trace_unknown_item_added_spike" in line for line in info_logs)
    assert len(api._lifecycle_trace_item_added_unknown_events) == 0


def test_late_cancelled_output_audio_done_is_suppressed_before_lifecycle_trace() -> None:
    api = _build_api_stub()
    api._active_response_id = "resp-server-auto"
    api._active_response_origin = "unknown"
    api._active_response_input_event_key = None
    api._active_response_canonical_key = None
    api._response_trace_context_by_id = {}
    api._stale_response_map = {}
    api._stale_response_map_ttl_s = 15.0
    api._stale_response_ids_set = set()
    api._stale_response_drop_window_by_id = {}
    api._stale_response_drop_window_s = 3.0

    api._pending_server_auto_response_by_turn_id["turn_2"] = PendingServerAutoResponse(
        turn_id="turn_2",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_2:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._active_input_event_key_by_turn_id = {"turn_2": "item_abc"}

    api._mark_pending_server_auto_response_cancelled(turn_id="turn_2", reason="transcript_final_upgrade")
    api._active_response_id = None

    observed: list[tuple[str, str]] = []
    api._record_cancelled_audio_race_transition = (
        lambda **kwargs: observed.append((str(kwargs.get("response_id")), str(kwargs.get("event_type"))))
    )
    api._emit_response_lifecycle_trace = lambda **_kwargs: observed.append(("trace", "emitted"))

    asyncio.run(
        api._handle_response_lifecycle_event(
            {"type": "response.output_audio.done", "response_id": "resp-server-auto"},
            websocket=object(),
        )
    )

    assert observed == [("resp-server-auto", "response.output_audio.done")]


def test_empty_transcript_cancelled_response_late_audio_done_is_suppressed() -> None:
    api = _build_api_stub()
    api._active_response_id = "resp-empty"
    api._pending_server_auto_response_by_turn_id["turn_3"] = PendingServerAutoResponse(
        turn_id="turn_3",
        response_id="resp-empty",
        canonical_key="run-464:turn_3:synthetic_server_auto_3",
        created_at_ms=1,
        active=True,
    )
    api._active_input_event_key_by_turn_id = {"turn_3": "item_empty"}
    api._mark_pending_server_auto_response_cancelled(turn_id="turn_3", reason="empty_transcript")
    api._active_response_id = None

    observed: list[str] = []
    api._record_cancelled_audio_race_transition = lambda **_kwargs: observed.append("race")
    api._emit_response_lifecycle_trace = lambda **_kwargs: observed.append("trace")

    asyncio.run(
        api._handle_response_lifecycle_event(
            {"type": "response.output_audio.done", "response_id": "resp-empty"},
            websocket=object(),
        )
    )

    assert observed == ["race"]


class _StateManagerStub:

    def __init__(self) -> None:
        self.state = "thinking"
        self.transitions: list[tuple[str, str]] = []

    def update_state(self, next_state, reason: str) -> None:
        value = getattr(next_state, "value", str(next_state))
        self.state = value
        self.transitions.append((value, reason))


def test_cancelled_response_late_audio_and_transcript_deltas_are_sunk() -> None:
    api = _build_api_stub()
    transport = _Transport()

    api.state_manager = _StateManagerStub()
    api._is_active_response_guarded = lambda: False
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._record_cancelled_audio_race_transition = lambda **_kwargs: None
    api._active_response_origin = "assistant_message"
    api._active_response_input_event_key = "item_new"
    api._active_response_canonical_key = "run-464:turn_2:item_new"
    api._current_turn_id_or_unknown = lambda: "turn_2"
    api._get_response_gating_verdict = lambda **_kwargs: None

    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._assistant_reply_response_id = None

    api._pending_server_auto_response_by_turn_id["turn_2"] = PendingServerAutoResponse(
        turn_id="turn_2",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_2:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None

    async def _fake_send_response_create(_websocket, _event, **_kwargs):
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_2",
            input_event_key="item_new",
            origin_label="upgraded_response",
        )
    )

    assert replaced is True
    assert api._response_status("resp-server-auto") == "cancelled"

    audio_delta_event = {
        "type": "response.output_audio.delta",
        "response_id": "resp-server-auto",
        "delta": base64.b64encode(b"stale-audio").decode("utf-8"),
    }
    asyncio.run(api._handle_response_output_audio_delta_event(audio_delta_event, websocket=None))

    stale_transcript_event = {
        "type": "response.output_audio_transcript.delta",
        "response_id": "resp-server-auto",
        "delta": " stale transcript",
    }
    asyncio.run(api._handle_event_legacy(stale_transcript_event, websocket=None))

    assert api._audio_accum == bytearray()
    assert api.assistant_reply == ""
    assert api._assistant_reply_accum == ""
    assert api.state_manager.state == "thinking"
    assert api.state_manager.transitions == []


def test_mark_pending_server_auto_response_cancelled_logs_empty_transcript_label(monkeypatch) -> None:
    api = _build_api_stub()
    api._pending_server_auto_response_by_turn_id["turn_9"] = PendingServerAutoResponse(
        turn_id="turn_9",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_9:item_9",
        created_at_ms=1,
        active=True,
    )
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._mark_pending_server_auto_response_cancelled(turn_id="turn_9", reason="empty_transcript")

    assert any("server_auto_cancelled_for_empty_transcript" in line for line in captured)
    assert any("pending_owner_response_id=resp-server-auto" in line for line in captured)


def test_upgrade_flow_snapshot_log_includes_pending_owner_response_id(monkeypatch) -> None:
    api = _build_api_stub()
    api._active_response_id = "resp-42"
    api._pending_server_auto_response_by_turn_id["turn_42"] = PendingServerAutoResponse(
        turn_id="turn_42",
        response_id="resp-42",
        canonical_key="run-464:turn_42:synthetic_server_auto_1",
        created_at_ms=1,
        active=True,
    )
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None
    api._get_or_create_transport = lambda: _Transport()
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    async def _fake_send_response_create(_websocket, _event, **_kwargs):
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_42",
            input_event_key="item_42",
            origin_label="upgraded_response",
        )
    )

    assert replaced is True
    assert any("upgrade_flow_snapshot" in line and "pending_owner_response_id=resp-42" in line for line in captured)


def test_upgrade_chain_id_propagates_into_replacement_metadata() -> None:
    api = _build_api_stub()
    transport = _Transport()
    replacement_calls: list[dict[str, object]] = []

    api._pending_server_auto_response_by_turn_id["turn_8"] = PendingServerAutoResponse(
        turn_id="turn_8",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_8:synthetic_server_auto_8",
        created_at_ms=1,
        active=True,
        upgrade_chain_id="run-464:turn_8:synthetic_server_auto_8",
    )
    api._record_response_trace_context(
        "resp-server-auto",
        turn_id="turn_8",
        input_event_key="synthetic_server_auto_8",
        canonical_key="run-464:turn_8:synthetic_server_auto_8",
        origin="server_auto",
        upgrade_chain_id="run-464:turn_8:synthetic_server_auto_8",
    )
    api._get_or_create_transport = lambda: transport
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None

    async def _fake_send_response_create(_websocket, event, **kwargs):
        replacement_calls.append({"event": event, **kwargs})
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_8",
            input_event_key="item_888",
            origin_label="upgraded_response",
        )
    )

    assert replaced is True
    metadata = replacement_calls[0]["event"]["response"]["metadata"]
    assert metadata["upgrade_chain_id"] == "run-464:turn_8:synthetic_server_auto_8"


def test_mark_pending_server_auto_response_replaced_is_idempotent_after_upgrade(monkeypatch) -> None:
    api = _build_api_stub()
    api._pending_server_auto_response_by_turn_id["turn_20"] = PendingServerAutoResponse(
        turn_id="turn_20",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_20:synthetic_server_auto_20",
        created_at_ms=1,
        active=False,
        cancelled_for_upgrade=True,
        upgrade_chain_id="run-464:turn_20:synthetic_server_auto_20",
    )
    api._active_response_id = "resp-replacement"
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._mark_pending_server_auto_response_replaced(turn_id="turn_20")

    pending = api._pending_server_auto_response_for_turn(turn_id="turn_20")
    assert pending is not None
    assert pending.active is False
    assert pending.cancelled_for_upgrade is False
    assert not any("pending_server_auto_mutation_rejected" in line for line in captured)


def test_mark_pending_server_auto_response_replaced_ignores_duplicate(monkeypatch) -> None:
    api = _build_api_stub()
    api._pending_server_auto_response_by_turn_id["turn_21"] = PendingServerAutoResponse(
        turn_id="turn_21",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_21:synthetic_server_auto_21",
        created_at_ms=1,
        active=False,
        cancelled_for_upgrade=False,
        upgrade_chain_id="run-464:turn_21:synthetic_server_auto_21",
    )
    api._active_response_id = "resp-replacement"
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._mark_pending_server_auto_response_replaced(turn_id="turn_21")

    assert any(
        "pending_server_auto_mutation_ignored" in line and "reason=idempotent_duplicate" in line
        for line in captured
    )
    assert not any("pending_server_auto_mutation_rejected" in line for line in captured)


def test_mark_pending_server_auto_response_replaced_rejects_lineage_mismatch(monkeypatch) -> None:
    api = _build_api_stub()
    pending = PendingServerAutoResponse(
        turn_id="turn_22",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_22:synthetic_server_auto_22",
        created_at_ms=1,
        active=True,
        upgrade_chain_id="chain-good",
    )
    api._active_response_id = "resp-replacement"
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    allowed = api._pending_server_auto_response_mutation_allowed(
        pending=pending,
        turn_id="turn_22",
        mutation="replaced",
        expected_response_id="resp-server-auto",
        expected_upgrade_chain_id="chain-other",
    )

    assert allowed is False
    assert any(
        "pending_server_auto_mutation_rejected" in line and "reason=lineage_mismatch" in line
        for line in captured
    )


def test_provisional_server_auto_tool_call_is_deferred_before_transcript_final(monkeypatch) -> None:
    api = _build_api_stub()
    api._active_response_origin = "server_auto"
    api._current_response_turn_id = "turn_3"
    api._active_response_id = "resp-server-auto"
    api._active_server_auto_input_event_key = "synthetic_server_auto_3"
    api.function_call = {"name": "gesture_look_around", "call_id": "call-provisional"}
    api.function_call_args = "{}"

    captured: dict[str, str] = {}

    async def _fake_noop(_websocket, **kwargs):
        captured.update({k: str(v) for k, v in kwargs.items()})

    monkeypatch.setattr(api, "_send_noop_tool_output", _fake_noop)

    api._server_auto_pre_audio_hold_active = lambda **_kwargs: True

    asyncio.run(api.handle_function_call({}, object()))

    assert captured["reason"] == "provisional_server_auto_pre_audio_hold"
    assert captured["status"] == "deferred"
    assert api.function_call is None
    assert api.function_call_args == ""


def test_transcript_final_handoff_invalidates_provisional_tool_followup_lineage() -> None:
    api = _build_api_stub()
    api._pending_response_create = types.SimpleNamespace(
        event={
            "type": "response.create",
            "response": {
                "metadata": {
                    "turn_id": "turn_3",
                    "parent_turn_id": "turn_3",
                    "input_event_key": "tool:call_pending",
                    "parent_input_event_key": "synthetic_server_auto_3",
                    "tool_followup": "true",
                }
            },
        },
        turn_id="turn_3",
    )
    api._response_create_queue = deque(
        [
            {
                "turn_id": "turn_3",
                "event": {
                    "type": "response.create",
                    "response": {
                        "metadata": {
                            "turn_id": "turn_3",
                            "parent_turn_id": "turn_3",
                            "input_event_key": "tool:call_queued",
                            "parent_input_event_key": "synthetic_server_auto_3",
                            "tool_followup": "true",
                        }
                    },
                },
            }
        ]
    )
    api._sync_pending_response_create_queue = lambda: None

    api._invalidate_provisional_tool_followups_for_turn(
        turn_id="turn_3",
        provisional_parent_input_event_key="synthetic_server_auto_3",
        reason="transcript_final_handoff",
    )

    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []
    assert api._tool_followup_state(canonical_key="run-464:turn_3:tool:call_pending") == "dropped"
    assert api._tool_followup_state(canonical_key="run-464:turn_3:tool:call_queued") == "dropped"


def test_cancel_and_replace_allows_post_done_lineage_when_active_response_cleared(monkeypatch) -> None:
    api = _build_api_stub()
    transport = _Transport()
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._active_response_id = None
    api._pending_server_auto_response_by_turn_id["turn_30"] = PendingServerAutoResponse(
        turn_id="turn_30",
        response_id="resp-server-auto",
        canonical_key="run-464:turn_30:synthetic_server_auto_30",
        created_at_ms=1,
        active=True,
        upgrade_chain_id="run-464:turn_30:synthetic_server_auto_30",
    )
    api._peek_pending_preference_memory_context_payload = lambda **_kwargs: None
    api._get_or_create_transport = lambda: transport

    async def _fake_send_response_create(_websocket, _event, **_kwargs):
        return True

    api._send_response_create = _fake_send_response_create

    replaced = asyncio.run(
        api._cancel_and_replace_pending_server_auto_on_transcript_final(
            websocket=object(),
            turn_id="turn_30",
            input_event_key="item_30",
            origin_label="upgraded_response",
        )
    )

    assert replaced is True
    assert any("upgrade_flow_snapshot" in line and "action=cancel_and_replace" in line for line in captured)
    assert not any("pending_server_auto_mutation_rejected" in line for line in captured)


def test_distinct_info_tool_followup_not_suppressed_after_canonical_ownership() -> None:
    api = _build_api_stub()
    should_drop, _parent_entry, reason = api._should_suppress_queued_tool_followup_release(
        response_metadata={
            "tool_followup_suppress_if_parent_covered": "true",
            "tool_name": "gesture_look_around",
            "tool_result_has_distinct_info": "true",
        },
        blocked_by_response_id="resp-parent",
    )

    assert should_drop is False
    assert reason == "distinct_info"


def test_transcript_buffer_retains_prefix_per_response_id_across_active_response_shift() -> None:
    api = _build_api_stub()
    api._active_response_id = "resp-new"
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._assistant_reply_response_id = None

    api._append_assistant_reply_text("Leading prefix ", response_id="resp-old")
    api._append_assistant_reply_text("continues.", response_id="resp-old")

    assert api.assistant_reply == ""
    assert api._assistant_reply_accum == ""
    assert api._assistant_reply_text_for_response("resp-old") == "Leading prefix continues."


def test_stale_suppression_does_not_drop_surviving_response_prefix() -> None:
    api = _build_api_stub()

    api._active_response_id = "resp-final"
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._assistant_reply_response_id = None
    api._append_assistant_reply_text("The front has ", response_id="resp-final")

    api._stale_response_ids().add("resp-stale")
    stale_event = {
        "type": "response.output_audio_transcript.delta",
        "response_id": "resp-stale",
        "delta": "old text",
    }
    asyncio.run(api._handle_event_legacy(stale_event, websocket=None))

    assert api._assistant_reply_text_for_response("resp-final") == "The front has "


def test_audio_transcript_done_reconstructs_full_sentence_from_multiple_deltas() -> None:
    api = _build_api_stub()
    api._active_response_id = "resp-final"
    api._assistant_reply_response_id = "resp-final"

    events = [
        {"type": "response.output_audio_transcript.delta", "response_id": "resp-final", "delta": "I can see "},
        {"type": "response.output_audio_transcript.delta", "response_id": "resp-final", "delta": "a shelf "},
        {"type": "response.output_audio_transcript.delta", "response_id": "resp-final", "delta": "of games."},
    ]
    for event in events:
        asyncio.run(api._handle_event_legacy(event, websocket=None))

    assert api._assistant_reply_text_for_response("resp-final") == "I can see a shelf of games."
