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

    async def send_json(self, _ws, event: dict[str, object]) -> None:
        self.sent.append(event)


def test_clarify_replaces_answer_no_mixed_output() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api.camera_controller = None
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {
        "turn-1": PendingServerAutoResponse(
            turn_id="turn-1",
            response_id="resp-old",
            canonical_key="run-1:turn-1:evt-1",
            created_at_ms=1,
            active=True,
        )
    }
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-1:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    transport = _Transport()
    api._get_or_create_transport = lambda: transport
    sent_messages: list[tuple[str, dict[str, str]]] = []

    async def _send_assistant_message(msg: str, _ws, *, response_metadata=None, **_kwargs):
        sent_messages.append((msg, response_metadata or {}))

    api.send_assistant_message = _send_assistant_message
    api._current_run_id = lambda: "run-1"
    api.assistant_reply = "I see blue pants"
    api._assistant_reply_accum = "I see blue pants"

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what color pants am i wearing",
            websocket=object(),
            turn_id="turn-1",
            input_event_key="evt-1",
            snapshot={"run_id": "run-1", "asr_confidence": 0.9},
        )
    )

    assert clarified is True
    assert transport.sent == [{"type": "response.cancel", "response_id": "resp-old"}]
    assert sent_messages
    msg, metadata = sent_messages[0]
    assert msg == "I can’t see right now. Want me to take a quick look with the camera?"
    assert "Actually, I can see" not in msg
    assert "gray" not in msg.lower()
    assert metadata["input_event_key"] == "evt-1:clarify"
    assert metadata["clarify_mode"] == "bounded"


def test_visual_unavailable_message_normalizer_blocks_vision_claims_without_camera_tool_result() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._tool_call_records = []
    api._current_turn_id_or_unknown = lambda: "turn-3"

    sanitized = api._normalize_verify_clarify_message(
        message=(
            "I can’t see right now. Want me to take a quick look with the camera? "
            "Actually, I can see that your pants look gray."
        ),
        metadata={
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
            "turn_id": "turn-3",
        },
    )

    assert sanitized == "I can’t see right now. Want me to take a quick look with the camera?"


def test_visual_unavailable_message_normalizer_respects_camera_active_truth() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._tool_call_records = []
    api._current_turn_id_or_unknown = lambda: "turn-3"
    api.get_vision_state = lambda: {
        "available": False,
        "can_capture": True,
        "camera_active": True,
        "queued_frame_count": 0,
    }

    sanitized = api._normalize_verify_clarify_message(
        message="I can’t see right now. Want me to take a quick look with the camera?",
        metadata={
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
            "turn_id": "turn-3",
        },
    )

    assert sanitized == "The camera is on, but I don’t have a fresh frame yet. Want me to take a new look now?"


def test_camera_active_visual_unavailable_clarify_message_is_truthful_and_exact() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api.camera_controller = type("Camera", (), {"_pending_images": [], "is_vision_loop_alive": lambda self: True})()
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-1:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: _Transport()
    api._current_run_id = lambda: "run-1"
    api._tool_call_records = []
    api._fresh_look_enabled = True
    api._fresh_look_cooldown_s = 0.0
    api._fresh_look_wait_timeout_s = 0.01
    api._last_fresh_look_request_at_monotonic = None
    api._last_fresh_look_completed_at_monotonic = None
    api._fresh_look_by_turn_id = {}
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True
    sent_messages: list[str] = []

    async def _send_assistant_message(msg: str, _ws, *, response_metadata=None, **_kwargs):
        sent_messages.append(msg)

    api.send_assistant_message = _send_assistant_message

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="tell me what you see in front of you",
            websocket=object(),
            turn_id="turn-9",
            input_event_key="evt-9",
            snapshot={"run_id": "run-1", "asr_confidence": 0.9},
        )
    )

    assert clarified is True
    assert sent_messages == [
        "I tried to take a fresh look, but the frame is still pending. I can answer from recent context or retry."
    ]
    assert "turn on the camera" not in sent_messages[0].lower()
