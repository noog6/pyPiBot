"""Tests for memory brief injection sequencing in realtime responses."""

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
    api._realtime_model = "gpt-realtime"
    api._response_create_runtime = ResponseCreateRuntime(api)
    api._transcript_final_missing_for_turn = lambda *, turn_id, input_event_key: False
    return api


def test_send_response_create_injects_memory_brief_before_response() -> None:
    api = _make_api_stub()
    ws = _Ws()

    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="injection",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]





def test_send_response_create_injects_preference_memory_context_before_response() -> None:
    api = _make_api_stub()
    ws = _Ws()

    api._consume_pending_preference_memory_context_note = lambda *, turn_id, input_event_key: (
        "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: Vim"
    )

    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="assistant_message",
        )
    )

    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]
    assert "Vim" in ws.events[0]["item"]["content"][0]["text"]


def test_send_response_create_consumes_preference_context_only_once_per_canonical_key() -> None:
    api = _make_api_stub()
    ws = _Ws()
    consumed: list[tuple[str, str]] = []

    def _consume(*, turn_id: str, input_event_key: str) -> str | None:
        consumed.append((turn_id, input_event_key))
        if len(consumed) == 1:
            return "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: Vim"
        return None

    api._consume_pending_preference_memory_context_note = _consume

    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_1", "input_event_key": "item_1"}},
    }

    first_sent = asyncio.run(
        api._send_response_create(
            ws,
            event,
            origin="assistant_message",
        )
    )
    second_sent = asyncio.run(
        api._send_response_create(
            ws,
            event,
            origin="assistant_message",
        )
    )

    assert first_sent is True
    assert second_sent is False
    assert consumed == [("turn_1", "item_1")]
    assert [item["type"] for item in ws.events].count("conversation.item.create") == 1
    assert [item["type"] for item in ws.events].count("response.create") == 1




def test_pending_preference_context_resolves_from_response_owner_map() -> None:
    api = _make_api_stub()
    api._response_id_by_canonical_key = {
        api._canonical_utterance_key(turn_id="turn_owner", input_event_key="item_owner"): "resp_owner"
    }
    payload = {"prompt_note": "owner note", "hit": True, "returned_count": 1}
    api._set_pending_preference_memory_context(
        turn_id="turn_owner",
        input_event_key="item_owner",
        memory_context=payload,
    )

    resolved = api._peek_pending_preference_memory_context_payload(
        turn_id="turn_owner",
        input_event_key="item_owner",
    )

    assert isinstance(resolved, dict)
    assert resolved["prompt_note"] == "owner note"
    assert api._pending_preference_memory_context_response_id_by_turn_id["turn_owner"] == "resp_owner"
    assert api._pending_preference_memory_context_by_response_id["resp_owner"]["payload"]["prompt_note"] == "owner note"


def test_pending_preference_context_rejects_mismatched_owner_record() -> None:
    api = _make_api_stub()
    canonical_key = api._canonical_utterance_key(turn_id="turn_a", input_event_key="item_a")
    api._pending_preference_memory_context_by_canonical_key = {
        canonical_key: {
            "payload": {"prompt_note": "wrong owner", "hit": True},
            "owner_turn_id": "turn_b",
            "owner_canonical_key": canonical_key,
            "owner_response_id": "",
        }
    }

    resolved = api._peek_pending_preference_memory_context_payload(
        turn_id="turn_a",
        input_event_key="item_a",
    )

    assert resolved is None


def test_send_response_create_preserves_brief_when_deferred() -> None:
    api = _make_api_stub()
    api._response_in_flight = True
    ws = _Ws()

    sent = asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="injection",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert sent is False
    assert len(api._response_create_queue) == 1
    assert api._response_create_queue[0]["memory_brief_note"] == "Turn memory brief: ..."


def test_initialize_session_injects_startup_memory_digest_note() -> None:
    api = _make_api_stub()
    api.profile_manager = type("P", (), {"get_profile_context": lambda self: type("Ctx", (), {"to_instruction_block": lambda self: "profile"})()})()
    api._vad_turn_detection = {
        "profile": "default",
        "threshold": 0.2,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 900,
        "create_response": True,
        "interrupt_response": True,
    }
    api._build_startup_memory_digest_note = lambda: "Startup memory digest: ..."

    ws = _Ws()
    asyncio.run(api.initialize_session(ws))

    assert ws.events[0]["type"] == "session.update"
    assert ws.events[1]["type"] == "conversation.item.create"


def test_server_auto_response_create_consumes_preference_context_from_turn_scope() -> None:
    api = _make_api_stub()
    ws = _Ws()

    canonical_key = api._canonical_utterance_key(turn_id="turn_4", input_event_key="item_editor")
    payload = {
        "prompt_note": "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: Vim",
        "hit": True,
    }
    api._pending_preference_memory_context_by_canonical_key = {canonical_key: dict(payload)}
    api._pending_preference_memory_context_by_turn_id = {"turn_4": dict(payload)}

    asyncio.run(
        api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": "turn_4",
                        "input_event_key": "item_editor",
                    }
                },
            },
            origin="server_auto",
        )
    )

    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]
    assert "Vim" in ws.events[0]["item"]["content"][0]["text"]
    assert len([event for event in ws.events if event["type"] == "response.create"]) == 1


def test_server_auto_response_prefers_turn_scoped_hit_after_transition_replaced() -> None:
    api = _make_api_stub()
    ws = _Ws()

    synthetic_canonical_key = api._canonical_utterance_key(turn_id="turn_7", input_event_key="synthetic_server_auto_1")
    item_canonical_key = api._canonical_utterance_key(turn_id="turn_7", input_event_key="item_ABC")
    api._active_server_auto_input_event_key = "synthetic_server_auto_1"
    api._active_input_event_key_by_turn_id = {"turn_7": "item_ABC"}

    api._pending_preference_memory_context_by_canonical_key = {
        synthetic_canonical_key: {
            "prompt_note": "Preference recall context for this SAME response: no stored preference matched this query.",
            "hit": False,
            "returned_count": 0,
        },
        item_canonical_key: {
            "prompt_note": "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: preferred editor is Vim",
            "hit": True,
            "returned_count": 2,
        },
    }
    api._pending_preference_memory_context_by_turn_id = {
        "turn_7": {
            "prompt_note": "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: preferred editor is Vim",
            "hit": True,
            "returned_count": 2,
        }
    }

    asyncio.run(
        api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": "turn_7",
                        "input_event_key": "item_ABC",
                    }
                },
            },
            origin="server_auto",
        )
    )

    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]
    assert "preferred editor is Vim" in ws.events[0]["item"]["content"][0]["text"]
    assert len([event for event in ws.events if event["type"] == "response.create"]) == 1


def test_upgraded_server_auto_replacement_keeps_preference_context_for_final_response() -> None:
    api = _make_api_stub()
    ws = _Ws()

    final_input_event_key = "item_editor"
    initial_input_event_key = "synthetic_server_auto_1"
    canonical_key = api._canonical_utterance_key(turn_id="turn_9", input_event_key=final_input_event_key)
    payload = {
        "source": "preference_recall",
        "prompt_note": "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: Vim",
        "hit": True,
        "returned_count": 1,
    }
    api._pending_preference_memory_context_by_canonical_key = {canonical_key: dict(payload)}
    api._pending_preference_memory_context_by_turn_id = {"turn_9": dict(payload)}

    api._response_in_flight = True
    scheduled = asyncio.run(
        api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": "turn_9",
                        "input_event_key": initial_input_event_key,
                    }
                },
            },
            origin="server_auto",
        )
    )

    assert scheduled is False
    assert len(api._response_create_queue) == 1

    # Simulate transition replacement dropping the queued server_auto create.
    api._response_create_queue.clear()

    api._response_in_flight = False
    sent = asyncio.run(
        api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {
                    "metadata": {
                        "turn_id": "turn_9",
                        "input_event_key": final_input_event_key,
                        "canonical_key": canonical_key,
                    }
                },
            },
            origin="server_auto",
        )
    )

    assert sent is True
    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]
    assert "Vim" in ws.events[0]["item"]["content"][0]["text"]
    assert ws.events[1]["response"]["metadata"]["canonical_key"] == canonical_key
    assert len([event for event in ws.events if event["type"] == "response.create"]) == 1


def _capture_memory_audit_logs(monkeypatch) -> list[str]:
    captured: list[str] = []

    def _capture(message, *args, **kwargs) -> None:
        rendered = message % args if args else str(message)
        if "memory_usage_audit_" in rendered:
            captured.append(rendered)

    monkeypatch.setattr("ai.realtime_api.logger.debug", _capture)
    return captured


def test_memory_usage_audit_emits_for_turn_memory_brief_injection(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _Ws()
    logs = _capture_memory_audit_logs(monkeypatch)

    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="injection",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert any("memory_usage_audit_started" in entry for entry in logs)
    assert any("injection_types=turn_brief" in entry for entry in logs)


def test_memory_usage_audit_emits_for_preference_context_injection(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _Ws()
    logs = _capture_memory_audit_logs(monkeypatch)
    api._consume_pending_preference_memory_context_note = lambda *, turn_id, input_event_key: (
        "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: Vim"
    )

    asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create"},
            origin="assistant_message",
        )
    )

    assert any("memory_usage_audit_started" in entry for entry in logs)
    assert any("injection_types=preference_context" in entry for entry in logs)


def test_memory_usage_audit_marks_preference_backed_turn_brief_with_preference_context(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _Ws()
    _capture_memory_audit_logs(monkeypatch)
    note_text = (
        "Preference recall context for this SAME response: matched stored preference(s). Top recalled value: Vim"
    )
    api._peek_pending_preference_memory_context_payload = lambda *, turn_id, input_event_key: {
        "prompt_note": note_text,
        "hit": True,
    }

    asyncio.run(
        api._send_response_create(
            ws,
            {
                "type": "response.create",
                "response": {"metadata": {"turn_id": "turn_pref", "input_event_key": "item_pref"}},
            },
            origin="upgraded_response",
            memory_brief_note=note_text,
        )
    )

    turn_key = api._memory_usage_audit_turn_key(turn_id="turn_pref", input_event_key="item_pref")
    stored = api._memory_usage_audit_store()[turn_key]
    assert stored["injection_types"] == ["turn_brief", "preference_context"]


def test_memory_usage_audit_finalizes_non_injected_turn(monkeypatch) -> None:
    api = _make_api_stub()
    logs = _capture_memory_audit_logs(monkeypatch)

    api._finalize_memory_usage_audit(
        turn_id="turn_no_memory",
        input_event_key="item_no_memory",
        selected_response_id="resp_1",
        selected_canonical_key="turn_no_memory:item_no_memory",
        close_reason="response_done_received",
        final_assistant_text="Here is a direct answer.",
    )

    assert any("memory_usage_audit_finalized" in entry for entry in logs)
    assert any("memory_injected=false" in entry for entry in logs)
    assert any("memory_usage_outcome=none" in entry for entry in logs)


def test_memory_usage_audit_detects_explicit_memory_wording_signal(monkeypatch) -> None:
    api = _make_api_stub()
    logs = _capture_memory_audit_logs(monkeypatch)
    api._record_memory_usage_audit_injection(
        turn_id="turn_signal",
        input_event_key="item_signal",
        injection_type="turn_brief",
    )

    api._finalize_memory_usage_audit(
        turn_id="turn_signal",
        input_event_key="item_signal",
        selected_response_id="resp_signal",
        selected_canonical_key="turn_signal:item_signal",
        close_reason="response_done_received",
        final_assistant_text="I remember you said Vim is your preference.",
    )

    assert any("memory_reference_marker_present=true" in entry for entry in logs)
    assert any("memory_usage_outcome=injected_possible_signal" in entry for entry in logs)


def test_memory_usage_audit_finalizes_without_explicit_memory_wording(monkeypatch) -> None:
    api = _make_api_stub()
    logs = _capture_memory_audit_logs(monkeypatch)
    api._record_memory_usage_audit_injection(
        turn_id="turn_no_marker",
        input_event_key="item_no_marker",
        injection_type="preference_context",
    )

    api._finalize_memory_usage_audit(
        turn_id="turn_no_marker",
        input_event_key="item_no_marker",
        selected_response_id="resp_no_marker",
        selected_canonical_key="turn_no_marker:item_no_marker",
        close_reason="response_done_received",
        final_assistant_text="Vim should work well for this workflow.",
    )

    assert any("memory_injected=true" in entry for entry in logs)
    assert any("memory_reference_marker_present=false" in entry for entry in logs)
    assert any("memory_usage_outcome=injected_no_signal" in entry for entry in logs)


def test_memory_usage_audit_combines_turn_brief_and_preference_context_once(monkeypatch) -> None:
    api = _make_api_stub()
    logs = _capture_memory_audit_logs(monkeypatch)
    turn_id = "turn_combo"
    input_event_key = "item_combo"
    turn_key = api._memory_usage_audit_turn_key(turn_id=turn_id, input_event_key=input_event_key)

    api._record_memory_usage_audit_injection(
        turn_id=turn_id,
        input_event_key=input_event_key,
        injection_type="turn_brief",
    )
    api._record_memory_usage_audit_injection(
        turn_id=turn_id,
        input_event_key=input_event_key,
        injection_type="preference_context",
    )
    api._record_memory_usage_audit_injection(
        turn_id=turn_id,
        input_event_key=input_event_key,
        injection_type="preference_context",
    )

    stored = api._memory_usage_audit_store()[turn_key]
    assert stored["injection_types"] == ["turn_brief", "preference_context"]
    assert len([entry for entry in logs if "memory_usage_audit_started" in entry]) == 1

    api._finalize_memory_usage_audit(
        turn_id=turn_id,
        input_event_key=input_event_key,
        selected_response_id="resp_combo",
        selected_canonical_key=f"{turn_id}:{input_event_key}",
        close_reason="response_done_received",
        final_assistant_text="Done.",
    )

    assert turn_key not in api._memory_usage_audit_store()
    assert any(
        "memory_usage_audit_finalized" in entry
        and "injection_types=turn_brief,preference_context" in entry
        and "memory_injected=true" in entry
        for entry in logs
    )


def test_initialize_session_startup_digest_memory_usage_audit_is_separate(monkeypatch) -> None:
    api = _make_api_stub()
    logs = _capture_memory_audit_logs(monkeypatch)
    api.profile_manager = type("P", (), {"get_profile_context": lambda self: type("Ctx", (), {"to_instruction_block": lambda self: "profile"})()})()
    api._vad_turn_detection = {
        "profile": "default",
        "threshold": 0.2,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 900,
        "create_response": True,
        "interrupt_response": True,
    }
    api._build_startup_memory_digest_note = lambda: "Startup memory digest: ..."

    ws = _Ws()
    asyncio.run(api.initialize_session(ws))

    assert any("memory_usage_audit_started" in entry for entry in logs)
    assert any("telemetry_scope=startup" in entry for entry in logs)
    assert any("injection_types=startup_digest" in entry for entry in logs)
    assert any("memory_usage_audit_finalized" in entry for entry in logs)
    assert any("close_reason=startup_digest_injected" in entry for entry in logs)
