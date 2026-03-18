from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

import ai.realtime_api as realtime_api_module
from ai.realtime_api import RealtimeAPI


def _build_verify_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._asr_verify_visual_recent_window_ms = 5000
    api._asr_verify_visual_recent_window_catalog_only_ms = 12000
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-1:{turn_id}:{input_event_key}"
    api._current_run_id = lambda: "run-1"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_args, **_kwargs: asyncio.sleep(0))})()
    api.send_assistant_message = lambda *_args, **_kwargs: asyncio.sleep(0)
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    return api


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
    assert captured == ["The camera is on, but I don’t have a fresh frame yet. Want me to take a new look now?"]


def test_visual_question_with_camera_active_and_queued_frame_mentions_pending_fresh_frame() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6

    class _Camera:
        _pending_images = [object()]

        @staticmethod
        def is_vision_loop_alive() -> bool:
            return True

    api.camera_controller = _Camera()
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
            transcript="what do you see now",
            websocket=object(),
            turn_id="turn-3",
            input_event_key="evt-3",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    assert captured == ["The camera is on, but I’m still waiting for a fresh frame to finish processing."]


def test_compound_motion_visual_request_with_active_camera_does_not_clarify_visual_unavailable() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6

    class _Camera:
        _pending_images = []

        @staticmethod
        def is_vision_loop_alive() -> bool:
            return True

    api.camera_controller = _Camera()
    api._last_vision_frame_sent_at_monotonic = 1.0
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-609:{turn_id}:{input_event_key}"
    api._current_run_id = lambda: "run-609"

    sent: list[str] = []

    async def _send_assistant_message(msg: str, *_args, **_kwargs):
        sent.append(msg)

    api.send_assistant_message = _send_assistant_message
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="move back to center and tell me what you see in front of you",
            websocket=object(),
            turn_id="turn-2",
            input_event_key="evt-2",
            snapshot={"run_id": "run-609", "asr_confidence": 0.95},
        )
    )

    assert clarified is False
    assert sent == []
    verdict = api._response_gating_verdict_by_input_event_key.get("run-609:turn-2:evt-2")
    assert verdict is not None
    assert verdict.action == "ANSWER"
    assert verdict.reason == "verify_clear"


def test_catalog_only_with_recent_frame_inside_catalog_window_does_not_clarify_visual_unavailable() -> None:
    api = _build_verify_api()
    api._image_response_mode = "catalog_only"
    api._image_response_enabled = False
    api.get_vision_state = lambda: {
        "available": True,
        "can_capture": True,
        "last_frame_age_ms": 7000,
        "queued_frame_count": 0,
    }

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-5",
            input_event_key="evt-5",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    assert clarified is False
    verdict = api._response_gating_verdict_by_input_event_key.get("run-1:turn-5:evt-5")
    assert verdict is not None
    assert verdict.action == "ANSWER"
    assert verdict.reason == "verify_clear"


def test_catalog_only_with_frame_beyond_catalog_window_still_clarifies_visual_unavailable() -> None:
    api = _build_verify_api()
    api._image_response_mode = "catalog_only"
    api._image_response_enabled = False
    api.get_vision_state = lambda: {
        "available": True,
        "can_capture": True,
        "last_frame_age_ms": 13000,
        "queued_frame_count": 0,
        "camera_active": True,
    }

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-6",
            input_event_key="evt-6",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    assert clarified is True
    verdict = api._response_gating_verdict_by_input_event_key.get("run-1:turn-6:evt-6")
    assert verdict is not None
    assert verdict.action == "CLARIFY"
    assert verdict.reason == "visual_unavailable"


def test_non_catalog_mode_uses_normal_visual_recent_threshold() -> None:
    api = _build_verify_api()
    api._image_response_mode = "respond"
    api._image_response_enabled = True
    api.get_vision_state = lambda: {
        "available": True,
        "can_capture": True,
        "last_frame_age_ms": 7000,
        "queued_frame_count": 0,
        "camera_active": True,
    }

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-7",
            input_event_key="evt-7",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    assert clarified is True
    verdict = api._response_gating_verdict_by_input_event_key.get("run-1:turn-7:evt-7")
    assert verdict is not None
    assert verdict.action == "CLARIFY"
    assert verdict.reason == "visual_unavailable"


def test_non_visual_verify_behavior_unchanged_by_visual_freshness_mode() -> None:
    api = _build_verify_api()
    api._image_response_mode = "catalog_only"
    api._image_response_enabled = False
    api.get_vision_state = lambda: {
        "available": True,
        "can_capture": True,
        "last_frame_age_ms": 7000,
        "queued_frame_count": 0,
    }

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="set a timer",
            websocket=object(),
            turn_id="turn-8",
            input_event_key="evt-8",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    assert clarified is False
    verdict = api._response_gating_verdict_by_input_event_key.get("run-1:turn-8:evt-8")
    assert verdict is not None
    assert verdict.action == "ANSWER"
    assert verdict.reason == "verify_clear"


def test_visual_availability_eval_log_reports_selected_freshness_window_by_mode(monkeypatch) -> None:
    api = _build_verify_api()

    info_messages: list[str] = []

    def _capture_info(msg: str, *args, **_kwargs) -> None:
        info_messages.append(msg % args if args else msg)

    monkeypatch.setattr(realtime_api_module.logger, "info", _capture_info)

    api._image_response_mode = "respond"
    api._image_response_enabled = True
    api.get_vision_state = lambda: {
        "available": True,
        "can_capture": True,
        "last_frame_age_ms": 4000,
        "queued_frame_count": 0,
    }
    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-log-1",
            input_event_key="evt-log-1",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    api._image_response_mode = "catalog_only"
    api._image_response_enabled = False
    api.get_vision_state = lambda: {
        "available": True,
        "can_capture": True,
        "last_frame_age_ms": 7000,
        "queued_frame_count": 0,
    }
    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see",
            websocket=object(),
            turn_id="turn-log-2",
            input_event_key="evt-log-2",
            snapshot={"run_id": "run-1", "asr_confidence": 0.95},
        )
    )

    eval_messages = [msg for msg in info_messages if "visual_availability_eval" in msg]
    assert any(
        "turn_id=turn-log-1" in msg and "freshness_window_ms=5000" in msg for msg in eval_messages
    ) and any(
        "turn_id=turn-log-2" in msg and "freshness_window_ms=12000" in msg for msg in eval_messages
    )
