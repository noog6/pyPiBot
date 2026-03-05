from __future__ import annotations

import asyncio
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
