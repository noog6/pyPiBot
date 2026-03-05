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
