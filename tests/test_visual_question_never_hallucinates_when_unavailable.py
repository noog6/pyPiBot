from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI


def test_no_hallucinated_vision_when_unavailable() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api.camera_controller = None
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-1:{turn_id}:{input_event_key}"
    api._current_run_id = lambda: "run-1"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_args, **_kwargs: asyncio.sleep(0))})()

    captured: list[str] = []

    async def _send_assistant_message(msg: str, *_args, **_kwargs):
        captured.append(msg)

    api.send_assistant_message = _send_assistant_message
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-2",
            input_event_key="evt-2",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    assert captured
    assert "I see" not in captured[0]
    assert "shirt" not in captured[0].lower()


def test_visual_question_with_capturable_but_stale_frame_still_clarifies() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api.camera_controller = type("Camera", (), {"_pending_images": []})()
    api._last_vision_frame_sent_at_monotonic = 1.0
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-1:{turn_id}:{input_event_key}"
    api._current_run_id = lambda: "run-1"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_args, **_kwargs: asyncio.sleep(0))})()

    captured: list[str] = []

    async def _send_assistant_message(msg: str, *_args, **_kwargs):
        captured.append(msg)

    api.send_assistant_message = _send_assistant_message
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-2",
            input_event_key="evt-2",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    verdict = api._response_gating_verdict_by_input_event_key.get("run-1:turn-2:evt-2")
    assert verdict is not None
    assert verdict.action == "CLARIFY"
    assert verdict.reason == "visual_unavailable"
    assert captured == ["I can’t see right now. Want me to take a quick look with the camera?"]
