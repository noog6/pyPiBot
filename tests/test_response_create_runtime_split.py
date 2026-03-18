from __future__ import annotations

import asyncio
import json
from collections import deque
import os
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.interaction_lifecycle_policy import ResponseCreateDecisionAction
from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime.transport import RealtimeTransport
from ai.realtime_api import RealtimeAPI


class _Ws:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.events.append(json.loads(payload))


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._transport = RealtimeTransport(
        connect_fn=lambda *args, **kwargs: None,
        validate_outbound_endpoint=lambda _url: None,
    )
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._active_response_id = None
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._response_create_queue = deque()
    api._pending_response_create = None
    api._response_create_turn_counter = 0
    api._current_response_turn_id = None
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._record_ai_call = lambda: None
    api._track_outgoing_event = lambda *args, **kwargs: None
    api._response_created_canonical_keys = set()
    api._preference_recall_suppressed_turns = set()
    api._response_create_runtime = ResponseCreateRuntime(api)
    return api


def test_prepare_response_create_snapshot_captures_expected_facts_and_mutations() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    expected_canonical_key = api._canonical_utterance_key(turn_id="turn_ctx", input_event_key="item_ctx")
    api._response_created_canonical_keys = {expected_canonical_key}
    api._preference_recall_suppressed_turns = {"turn_ctx"}
    api._peek_pending_preference_memory_context_payload = lambda *, turn_id, input_event_key: {
        "prompt_note": "Preference recall context for this SAME response: Top recalled value: Vim",
        "hit": True,
        "source": "memory",
    }

    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_ctx",
                "input_event_key": "item_ctx",
                "memory_intent_subtype": "preference_recall",
            }
        },
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=123.0,
    )

    assert prepared_snapshot.turn_id == "turn_ctx"
    assert prepared_snapshot.input_event_key == "item_ctx"
    assert prepared_snapshot.canonical_key == expected_canonical_key
    assert prepared_snapshot.effective_memory_note and "Vim" in prepared_snapshot.effective_memory_note
    assert prepared_snapshot.had_pending_preference_context is True
    assert prepared_snapshot.created_keys == {expected_canonical_key}
    assert prepared_snapshot.suppression_turns == {"turn_ctx"}
    assert "Memory-intent response mode" in event["response"]["instructions"]


def test_decide_response_create_action_uses_prepared_snapshot_for_server_auto_defer() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime

    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_srv", "input_event_key": "synthetic_server_auto_7"}},
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="server_auto",
        utterance_context=None,
        memory_brief_note=None,
        now=321.0,
    )
    decision = runtime.decide_response_create_action(prepared_snapshot)

    assert decision.action is ResponseCreateDecisionAction.SCHEDULE
    assert decision.reason_code == "awaiting_transcript_final"
    assert decision.queue_reason == "awaiting_transcript_final"


def test_send_response_create_executes_side_effects_after_direct_send_decision() -> None:
    api = _make_api_stub()
    ws = _Ws()

    sent = asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create", "response": {"metadata": {"turn_id": "turn_send", "input_event_key": "item_send"}}},
            origin="assistant_message",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert sent is True
    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]
    assert api._last_response_create_ts is not None
    assert api._response_in_flight is True
    assert api._is_response_already_delivered(turn_id="turn_send", input_event_key="item_send") is False
