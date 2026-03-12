from __future__ import annotations

import asyncio
import sys
import time
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.asr_trust import is_current_visual_question
from ai.realtime_api import RealtimeAPI


class _CameraStub:
    def __init__(self, *, pending_images: list[object] | None = None, alive: bool = True) -> None:
        self._pending_images = pending_images or []
        self._alive = alive

    def is_vision_loop_alive(self) -> bool:
        return self._alive


def _build_api(*, camera: object | None = None) -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.camera_controller = camera
    api._fresh_look_enabled = True
    api._fresh_look_cooldown_s = 10.0
    api._fresh_look_wait_timeout_s = 0.15
    api._last_fresh_look_request_at_monotonic = None
    api._last_fresh_look_completed_at_monotonic = None
    api._fresh_look_by_turn_id = {}
    api._current_run_id = lambda: "run-test"
    api._last_vision_frame_sent_at_monotonic = None
    return api


def test_current_visual_question_detection() -> None:
    assert is_current_visual_question("what do you see right now")
    assert is_current_visual_question("can you check what's on the shelf now")
    assert not is_current_visual_question("what did you see earlier")




def test_current_visual_question_detection_covers_natural_phrasing() -> None:
    assert is_current_visual_question("tell me what you see in front of you")
    assert is_current_visual_question("let me know what you see in front of you")
    assert is_current_visual_question("what are you looking at right now")
    assert is_current_visual_question("can you look and tell me what's there")
    assert is_current_visual_question("look at what's in front of you and describe it")


def test_mixed_motion_and_visual_utterance_triggers_fresh_look() -> None:
    assert is_current_visual_question("go back to center and tell me what you see in front of you")


def test_current_visual_question_detection_covers_conditional_see_phrasing() -> None:
    assert is_current_visual_question("look back at center and then tell me if you see a can of pop in your field of view")
    assert is_current_visual_question("tell me if you see a can of pop")
    assert is_current_visual_question("do you see a can of pop in your field of view")
    assert is_current_visual_question("can you check whether there is a can of pop in front of you")


def test_non_current_visual_question_detection_stays_false() -> None:
    assert not is_current_visual_question("what did you see earlier")
    assert not is_current_visual_question("what do you remember seeing last night")
    assert not is_current_visual_question("tell me if you saw a can of pop earlier")

def test_non_current_visual_question_does_not_trigger_fresh_look(monkeypatch) -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    called = []

    async def _capture(**_kwargs):
        called.append(True)
        return {}

    api._attempt_fresh_look_for_turn = _capture
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    api.send_assistant_message = _send_assistant_message

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what did you see before",
            websocket=object(),
            turn_id="turn-1",
            input_event_key="evt-1",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert called == []


def test_verify_on_risk_runs_fresh_look_before_visual_unavailable_clarify() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    call_order: list[str] = []

    async def _capture(**_kwargs):
        call_order.append("fresh_look")
        return {"requested": True, "completed": False, "blocked_reason": "timeout"}

    async def _send_assistant_message(*_args, **_kwargs):
        call_order.append("clarify")
        return None

    api._attempt_fresh_look_for_turn = _capture
    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="go back to center and tell me what you see in front of you",
            websocket=object(),
            turn_id="turn-7",
            input_event_key="evt-7",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert call_order == ["fresh_look", "clarify"]


def test_fresh_look_respects_cooldown() -> None:
    api = _build_api(camera=_CameraStub())
    api._last_fresh_look_request_at_monotonic = time.monotonic()
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-2")

    assert not allowed
    assert reason == "cooldown"


def test_fresh_look_respects_busy_state() -> None:
    api = _build_api(camera=_CameraStub())
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-2")

    assert not allowed
    assert reason == "injection_not_ready"




def test_get_vision_state_uses_camera_instance_fallback_when_controller_unset() -> None:
    api = _build_api(camera=None)
    api.camera_instance = _CameraStub(pending_images=[], alive=True)
    api._last_vision_frame_sent_at_monotonic = time.monotonic()

    state = api.get_vision_state()

    assert state["can_capture"] is True
    assert state["camera_active"] is True
    assert state["available"] is True


def test_fresh_look_gating_uses_camera_instance_fallback_when_controller_unset() -> None:
    api = _build_api(camera=None)
    api.camera_instance = _CameraStub(pending_images=[])
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-fallback")

    assert not allowed
    assert reason == "injection_not_ready"

def test_fresh_look_alive_camera_not_marked_camera_unavailable() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-camera-alive")

    assert not allowed
    assert reason != "camera_missing"
    assert reason != "camera_loop_inactive"
    assert reason == "injection_not_ready"


def test_successful_fresh_look_marks_current_provenance() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-3"))

    assert state["requested"] is True
    assert state["completed"] is True
    assert state["visual_answer_mode"] == "fresh_current"
    assert state["last_fresh_capture_age_ms"] == 0


def test_visual_answer_provenance_ambient_recent_when_queue_present() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    state = api.get_vision_state()

    mode = api._classify_visual_answer_provenance(turn_id="turn-ambient", vision_state=state)

    assert mode == "ambient_recent"




def test_get_vision_state_reports_camera_alive_and_available_when_recent_frame_exists() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._last_vision_frame_sent_at_monotonic = time.monotonic()

    state = api.get_vision_state()

    assert state["can_capture"] is True
    assert state["camera_active"] is True
    assert state["available"] is True

def test_camera_active_clarify_never_uses_camera_unavailable_message() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-clarify"))
    message = api._visual_unavailable_clarify_text_for_turn(
        turn_id="turn-clarify",
        vision_state=api.get_vision_state(),
    )

    assert state["blocked_reason"] == "injection_not_ready"
    assert "camera isn’t available" not in message


def test_mixed_motion_visual_prompt_preserves_truthful_ambient_provenance() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    api._tool_call_records = [{"turn_id": "turn-mixed", "name": "gesture_look_center"}]
    api._fresh_look_state_for_turn(turn_id="turn-mixed")["completed"] = False

    provenance = api._classify_visual_answer_provenance(
        turn_id="turn-mixed",
        vision_state=api.get_vision_state(),
    )

    assert provenance == "ambient_recent"


def test_bounded_visual_clarify_never_uses_camera_unavailable_when_injection_not_ready() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False
    asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-bounded"))

    final_text = api._normalize_verify_clarify_message(
        message="I can’t take a fresh look right now because the camera isn’t available.",
        metadata={
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
            "clarify_mode": "bounded",
            "turn_id": "turn-bounded",
        },
    )

    assert "camera isn’t available" not in final_text
    assert "vision is busy" in final_text


def test_failed_fresh_look_returns_truthful_timeout_wording() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-4"))
    message = api._visual_unavailable_clarify_text_for_turn(
        turn_id="turn-4",
        vision_state=api.get_vision_state(),
    )

    assert state["blocked_reason"] == "timeout"
    assert "I tried to take a fresh look" in message


def test_repeated_look_now_requests_do_not_spam_captures() -> None:
    camera = _CameraStub(pending_images=[object()])
    api = _build_api(camera=camera)
    api._fresh_look_cooldown_s = 30.0
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    first = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-5"))
    camera._pending_images = []
    second = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-6"))

    assert first["completed"] is True
    assert second["completed"] is False
    assert second["blocked_reason"] == "cooldown"
